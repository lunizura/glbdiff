from __future__ import annotations

import struct


def sniff_image_header(data: bytes) -> tuple[str, int | None, int | None]:
    """Sniff MIME type and pixel dimensions (width, height) from image header bytes."""
    n = len(data)

    # PNG: 8-byte signature + length(4) + "IHDR"(4) + width(4) + height(4)
    if n >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack_from(">II", data, 16)
        return ("image/png", w, h)

    # JPEG: SOI 0xFF 0xD8 followed by marker segments until SOF0..SOF3 or SOS
    if n >= 4 and data[:2] == b"\xff\xd8":
        pos = 2
        while pos + 2 <= n:
            if data[pos] != 0xFF:
                break
            while pos < n and data[pos] == 0xFF:
                pos += 1
            if pos >= n:
                break
            mk = data[pos]
            pos += 1
            if mk in (0xD8, 0xD9) or (0xD0 <= mk <= 0xD7) or mk == 0x01:
                continue
            if mk == 0xDA:
                break
            if pos + 2 > n:
                break
            seg_len = struct.unpack_from(">H", data, pos)[0]
            if seg_len < 2 or pos + seg_len > n:
                break
            if mk in (0xC0, 0xC1, 0xC2, 0xC3) and seg_len >= 7:
                h, w = struct.unpack_from(">HH", data, pos + 3)
                return ("image/jpeg", w, h)
            pos += seg_len
        return ("image/jpeg", None, None)

    # WebP: RIFF....WEBP + VP8X / VP8L / VP8
    if n >= 16 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk_fourcc = data[12:16]
        if chunk_fourcc == b"VP8X" and n >= 30:
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
            return ("image/webp", w, h)
        if chunk_fourcc == b"VP8L" and n >= 25 and data[20] == 0x2F:
            bits = struct.unpack_from("<I", data, 21)[0]
            w = (bits & 0x3FFF) + 1
            h = ((bits >> 14) & 0x3FFF) + 1
            return ("image/webp", w, h)
        if chunk_fourcc == b"VP8 " and n >= 30 and data[23:26] == b"\x9d\x01\x2a":
            w, h = struct.unpack_from("<HH", data, 26)
            return ("image/webp", w & 0x3FFF, h & 0x3FFF)
        return ("image/webp", None, None)

    # KTX2: 12-byte magic + vkFormat(4) + typeSize(4) + pixelWidth(4) + pixelHeight(4)
    if n >= 28 and data[:12] == b"\xabKTX 20\xbb\r\n\x1a\n":
        w, h = struct.unpack_from("<II", data, 20)
        return ("image/ktx2", w, h)

    return ("application/octet-stream", None, None)
