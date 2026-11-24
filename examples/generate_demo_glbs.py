from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any


def _make_png(width: int = 64, height: int = 64) -> bytes:
    def _chunk(ctype: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanline = b"\x00" + (b"\x38\xbd\xf8" * width)
    idat = zlib.compress(scanline * height)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _pack_glb(doc: dict[str, Any], bin_data: bytes) -> bytes:
    json_bytes = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "
    padded_bin = bytearray(bin_data)
    while len(padded_bin) % 4 != 0:
        padded_bin.append(0)
    chunks = (
        struct.pack("<II", len(json_bytes), 0x4E4F534A)
        + json_bytes
        + struct.pack("<II", len(padded_bin), 0x004E4942)
        + bytes(padded_bin)
    )
    return struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks


def _build_character_glb(
    *,
    extra_subdivisions: bool,
    alpha_mode: str,
    run_duration: float,
    include_sword: bool,
) -> bytes:
    buf = bytearray()
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []

    def add_bv(data: bytes) -> int:
        while len(buf) % 4 != 0:
            buf.append(0)
        off = len(buf)
        buf.extend(data)
        buffer_views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data)})
        return len(buffer_views) - 1

    def add_acc(
        bv_idx: int,
        comp: int,
        count: int,
        atype: str,
        mn: list[float] | None = None,
        mx: list[float] | None = None,
    ) -> int:
        acc: dict[str, Any] = {
            "bufferView": bv_idx,
            "componentType": comp,
            "count": count,
            "type": atype,
        }
        if mn is not None:
            acc["min"] = mn
        if mx is not None:
            acc["max"] = mx
        accessors.append(acc)
        return len(accessors) - 1

    # Torso + Head prism geometry (two stacked boxes / diamond prism)
    positions = [
        # Base torso box (0..7)
        (-0.35, 0.0, -0.2), (0.35, 0.0, -0.2), (0.35, 1.0, -0.2), (-0.35, 1.0, -0.2),
        (-0.35, 0.0, 0.2), (0.35, 0.0, 0.2), (0.35, 1.0, 0.2), (-0.35, 1.0, 0.2),
        # Head box (8..15)
        (-0.22, 1.05, -0.22), (0.22, 1.05, -0.22), (0.22, 1.55, -0.22), (-0.22, 1.55, -0.22),
        (-0.22, 1.05, 0.22), (0.22, 1.05, 0.22), (0.22, 1.55, 0.22), (-0.22, 1.55, 0.22),
    ]
    indices = [
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7,
        0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2,
        8, 9, 10, 8, 10, 11, 12, 14, 13, 12, 15, 14,
        8, 12, 13, 8, 13, 9, 11, 10, 14, 11, 14, 15,
        8, 11, 15, 8, 15, 12, 9, 13, 14, 9, 14, 10,
    ]
    if extra_subdivisions:
        # Add shoulder pads / chest plate triangles to raise triangle count > 20%
        base_v = len(positions)
        positions.extend(
            [
                (-0.6, 0.8, -0.22), (-0.35, 0.8, -0.22), (-0.35, 1.05, 0.22), (-0.6, 1.05, 0.22),
                (0.35, 0.8, -0.22), (0.6, 0.8, -0.22), (0.6, 1.05, 0.22), (0.35, 1.05, 0.22),
            ]
        )
        indices.extend(
            [
                base_v + 0, base_v + 1, base_v + 2, base_v + 0, base_v + 2, base_v + 3,
                base_v + 4, base_v + 5, base_v + 6, base_v + 4, base_v + 6, base_v + 7,
                base_v + 0, base_v + 3, base_v + 1, base_v + 4, base_v + 7, base_v + 5,
                base_v + 1, base_v + 4, base_v + 2, base_v + 2, base_v + 4, base_v + 7,
            ]
        )

    uvs = [(float(i % 2), float((i // 2) % 2)) for i in range(len(positions))]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]

    pos_acc = add_acc(
        add_bv(b"".join(struct.pack("<fff", *p) for p in positions)),
        5126,
        len(positions),
        "VEC3",
        [min(xs), min(ys), min(zs)],
        [max(xs), max(ys), max(zs)],
    )
    uv_acc = add_acc(add_bv(b"".join(struct.pack("<ff", *u) for u in uvs)), 5126, len(uvs), "VEC2")
    idx_acc = add_acc(
        add_bv(b"".join(struct.pack("<H", idx) for idx in indices)),
        5123,
        len(indices),
        "SCALAR",
    )

    png_bv = add_bv(_make_png(64, 64))
    images = [{"name": "HeroArmorTex", "bufferView": png_bv, "mimeType": "image/png"}]
    textures = [{"name": "HeroArmorTex", "source": 0}]
    materials = [
        {
            "name": "HeroArmor",
            "alphaMode": alpha_mode,
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.4,
                "roughnessFactor": 0.6,
            },
        }
    ]
    meshes = [
        {
            "name": "CharacterBody",
            "primitives": [
                {
                    "attributes": {"POSITION": pos_acc, "TEXCOORD_0": uv_acc},
                    "indices": idx_acc,
                    "material": 0,
                }
            ],
        }
    ]

    # 2-bone rig: RootPelvis -> UpperSpine
    nodes: list[dict[str, Any]] = [
        {"name": "RootPelvis", "translation": [0.0, 0.6, 0.0], "children": [1]},
        {"name": "UpperSpine", "translation": [0.0, 0.65, 0.0]},
        {"name": "CharacterNode", "mesh": 0, "skin": 0},
    ]
    scene_nodes = [0, 2]

    ident = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    ibm_acc = add_acc(add_bv(struct.pack("<16f", *ident) * 2), 5126, 2, "MAT4")
    skins = [{"name": "CharacterRig", "joints": [0, 1], "inverseBindMatrices": ibm_acc}]

    if include_sword:
        sw_pos = [
            (0.55, 0.4, 0.0), (0.65, 0.4, 0.0), (0.6, 1.35, 0.0),
            (0.55, 0.4, 0.05), (0.65, 0.4, 0.05), (0.6, 1.35, 0.05),
        ]
        sw_idx = [0, 1, 2, 3, 5, 4, 0, 3, 2, 1, 4, 2]
        sw_pos_acc = add_acc(
            add_bv(b"".join(struct.pack("<fff", *p) for p in sw_pos)),
            5126,
            len(sw_pos),
            "VEC3",
            [0.55, 0.4, 0.0],
            [0.65, 1.35, 0.05],
        )
        sw_idx_acc = add_acc(
            add_bv(b"".join(struct.pack("<H", idx) for idx in sw_idx)),
            5123,
            len(sw_idx),
            "SCALAR",
        )
        meshes.append(
            {
                "name": "SwordAttachment",
                "primitives": [{"attributes": {"POSITION": sw_pos_acc}, "indices": sw_idx_acc}],
            }
        )
        nodes.append({"name": "SwordNode", "mesh": 1})
        scene_nodes.append(len(nodes) - 1)

    def _make_anim(name: str, dur: float) -> dict[str, Any]:
        in_acc = add_acc(add_bv(struct.pack("<2f", 0.0, dur)), 5126, 2, "SCALAR", [0.0], [dur])
        out_acc = add_acc(
            add_bv(struct.pack("<8f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.2588, 0.0, 0.9659)),
            5126,
            2,
            "VEC4",
        )
        return {
            "name": name,
            "samplers": [{"input": in_acc, "output": out_acc, "interpolation": "LINEAR"}],
            "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}}],
        }

    animations = [_make_anim("Idle", 1.2), _make_anim("Run", run_duration)]

    doc = {
        "asset": {"version": "2.0", "generator": "glbdiff-demo-generator"},
        "scene": 0,
        "scenes": [{"nodes": scene_nodes}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "textures": textures,
        "images": images,
        "skins": skins,
        "animations": animations,
        "buffers": [{"byteLength": len(buf)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    return _pack_glb(doc, bytes(buf))


def main() -> None:
    samples_dir = Path(__file__).resolve().parent / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    v1 = _build_character_glb(
        extra_subdivisions=False,
        alpha_mode="OPAQUE",
        run_duration=0.8,
        include_sword=False,
    )
    v2 = _build_character_glb(
        extra_subdivisions=True,
        alpha_mode="BLEND",
        run_duration=1.1,
        include_sword=True,
    )

    p1 = samples_dir / "character_v1.glb"
    p2 = samples_dir / "character_v2.glb"
    p1.write_bytes(v1)
    p2.write_bytes(v2)
    print(f"Generated {p1} ({len(v1)} bytes) and {p2} ({len(v2)} bytes)")


if __name__ == "__main__":
    main()
