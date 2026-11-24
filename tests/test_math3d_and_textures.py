from __future__ import annotations

import math
import struct

from glbdiff.math3d import (
    IDENTITY_MAT4,
    mat4_from_trs,
    mat4_mul,
    mat4_transform_point,
    merge_aabbs,
    transform_aabb,
)
from glbdiff.models import AABB
from glbdiff.textures import sniff_image_header


def test_identity_and_trs_translation_scale() -> None:
    p = (1.0, 2.0, 3.0)
    assert mat4_transform_point(IDENTITY_MAT4, p) == p

    m = mat4_from_trs((10.0, -5.0, 2.0), (0.0, 0.0, 0.0, 1.0), (2.0, 3.0, 4.0))
    out = mat4_transform_point(m, p)
    assert math.isclose(out[0], 12.0, abs_tol=1e-6)
    assert math.isclose(out[1], 1.0, abs_tol=1e-6)
    assert math.isclose(out[2], 14.0, abs_tol=1e-6)


def test_trs_quaternion_rotation_and_zero_norm_fallback() -> None:
    # 90 deg rotation around Z axis: (x=1, y=0, z=0) -> (0, 1, 0)
    s2 = math.sqrt(0.5)
    # Non-unit quaternion scaled by 3.0 to verify normalization
    m = mat4_from_trs((0.0, 0.0, 0.0), (0.0, 0.0, 3.0 * s2, 3.0 * s2), (1.0, 1.0, 1.0))
    rx, ry, rz = mat4_transform_point(m, (1.0, 0.0, 0.0))
    assert math.isclose(rx, 0.0, abs_tol=1e-6)
    assert math.isclose(ry, 1.0, abs_tol=1e-6)
    assert math.isclose(rz, 0.0, abs_tol=1e-6)

    # Zero quaternion falls back to identity rotation
    m_zero = mat4_from_trs((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert mat4_transform_point(m_zero, (1.0, 0.0, 0.0)) == (2.0, 2.0, 3.0)


def test_mat4_mul_composition() -> None:
    t = mat4_from_trs((5.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
    s2 = math.sqrt(0.5)
    r = mat4_from_trs((0.0, 0.0, 0.0), (0.0, 0.0, s2, s2), (1.0, 1.0, 1.0))
    # Rotate first, then translate by (5, 0, 0): (1,0,0) -> (0,1,0) -> (5,1,0)
    tr = mat4_mul(t, r)
    out = mat4_transform_point(tr, (1.0, 0.0, 0.0))
    assert math.isclose(out[0], 5.0, abs_tol=1e-6)
    assert math.isclose(out[1], 1.0, abs_tol=1e-6)
    assert math.isclose(out[2], 0.0, abs_tol=1e-6)


def test_transform_and_merge_aabb() -> None:
    box = AABB(min_pt=(-1.0, -1.0, -1.0), max_pt=(1.0, 1.0, 1.0))
    # 45 deg around Z expands X and Y extents from [-1, 1] to [-sqrt(2), sqrt(2)]
    half = math.radians(22.5)
    m = mat4_from_trs((10.0, 0.0, 0.0), (0.0, 0.0, math.sin(half), math.cos(half)), (1.0, 1.0, 1.0))
    tbox = transform_aabb(m, box)
    s2 = math.sqrt(2.0)
    assert math.isclose(tbox.min_pt[0], 10.0 - s2, abs_tol=1e-6)
    assert math.isclose(tbox.max_pt[0], 10.0 + s2, abs_tol=1e-6)
    assert math.isclose(tbox.min_pt[1], -s2, abs_tol=1e-6)
    assert math.isclose(tbox.max_pt[1], s2, abs_tol=1e-6)
    assert math.isclose(tbox.min_pt[2], -1.0, abs_tol=1e-6)
    assert math.isclose(tbox.max_pt[2], 1.0, abs_tol=1e-6)

    assert merge_aabbs([]) is None
    merged = merge_aabbs([box, tbox])
    assert merged is not None
    assert math.isclose(merged.min_pt[0], -1.0, abs_tol=1e-6)
    assert math.isclose(merged.max_pt[0], 10.0 + s2, abs_tol=1e-6)


def test_sniff_png() -> None:
    header = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 1920, 1080)
    assert sniff_image_header(header) == ("image/png", 1920, 1080)


def test_sniff_jpeg() -> None:
    # SOI + APP0 (length 16) + fill 0xFF + SOF0 (1280x720)
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + (b"\x00" * 14)
    sof0 = b"\xff\xff\xc0" + struct.pack(">HBHH", 11, 8, 720, 1280) + (b"\x00" * 4)
    jpeg = b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"
    assert sniff_image_header(jpeg) == ("image/jpeg", 1280, 720)

    # Progressive SOF2
    sof2 = b"\xff\xd8\xff\xc2" + struct.pack(">HBHH", 7, 8, 512, 1024)
    assert sniff_image_header(sof2) == ("image/jpeg", 1024, 512)

    # Fallback when SOS precedes SOF / missing SOF
    no_sof = b"\xff\xd8\xff\xda\x00\x08\x00\x00\x00\x00\x00\x00"
    assert sniff_image_header(no_sof) == ("image/jpeg", None, None)


def test_sniff_webp_variants() -> None:
    # VP8X: width-1 and height-1 stored as 24-bit little-endian at offsets 24..27 and 27..30
    w_m1 = (640 - 1).to_bytes(3, "little")
    h_m1 = (480 - 1).to_bytes(3, "little")
    vp8x = b"RIFF\x00\x00\x00\x00WEBPVP8X" + (b"\x00" * 8) + w_m1 + h_m1
    assert sniff_image_header(vp8x) == ("image/webp", 640, 480)

    # VP8L: signature byte 0x2F at offset 20, packed 14-bit (w-1) | ((h-1) << 14) at offset 21
    packed = (256 - 1) | ((128 - 1) << 14)
    vp8l = b"RIFF\x00\x00\x00\x00WEBPVP8L" + struct.pack("<I", 10) + b"\x2f" + struct.pack("<I", packed)
    assert sniff_image_header(vp8l) == ("image/webp", 256, 128)

    # VP8 lossy: frame tag + start code 9d 01 2a at offset 23..26, little-endian u16 w, h at 26..30
    vp8 = (
        b"RIFF\x00\x00\x00\x00WEBPVP8 "
        + struct.pack("<I", 10)
        + b"\x00\x00\x00"
        + b"\x9d\x01\x2a"
        + struct.pack("<HH", 1024, 768)
    )
    assert sniff_image_header(vp8) == ("image/webp", 1024, 768)


def test_sniff_ktx2_and_unknown() -> None:
    ktx2 = (
        b"\xabKTX 20\xbb\r\n\x1a\n"
        + struct.pack("<IIII", 37, 1, 2048, 1024)
    )
    assert sniff_image_header(ktx2) == ("image/ktx2", 2048, 1024)
    assert sniff_image_header(b"not-an-image") == ("application/octet-stream", None, None)
    assert sniff_image_header(b"") == ("application/octet-stream", None, None)
