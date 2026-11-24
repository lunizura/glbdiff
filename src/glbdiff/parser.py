from __future__ import annotations

import base64
import json
import math
import struct
import urllib.parse
from pathlib import Path, PureWindowsPath
from typing import Any

from glbdiff.errors import GlbParseError
from glbdiff.math3d import (
    IDENTITY_MAT4,
    mat4_from_trs,
    mat4_mul,
    mat4_transform_point,
    merge_aabbs,
    transform_aabb,
)
from glbdiff.models import (
    AABB,
    AnimationSummary,
    AssetSummary,
    JointInfo,
    MaterialSummary,
    MeshSummary,
    SkinSummary,
    TextureSummary,
    Vec3,
    WireframePreview,
)
from glbdiff.textures import sniff_image_header

_GLB_MAGIC = 0x46546C67  # b"glTF"
_GLB_CHUNK_JSON = 0x4E4F534A  # b"JSON"
_GLB_CHUNK_BIN = 0x004E4942  # b"BIN\x00"

_COMP_FORMATS: dict[int, tuple[str, int]] = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}

_TYPE_COUNTS: dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _as_dict(val: Any) -> dict[str, Any]:
    return val if isinstance(val, dict) else {}


def _as_list(val: Any) -> list[Any]:
    return val if isinstance(val, list) else []


def _disambiguate_names(items: list[dict[str, Any]], kind: str) -> list[str]:
    counts: dict[str, int] = {}
    used: set[str] = set()
    names: list[str] = []
    for idx, item in enumerate(items):
        raw = item.get("name") if isinstance(item, dict) else None
        base = raw.strip() if isinstance(raw, str) and raw.strip() else f"<unnamed_{kind}_{idx}>"
        occ = counts.get(base, 0)
        counts[base] = occ + 1
        candidate = base if occ == 0 else f"{base}#{idx}"
        while candidate in used:
            candidate = f"{candidate}_{idx}"
        used.add(candidate)
        names.append(candidate)
    return names


def _decode_data_uri(uri: str) -> bytes:
    if "," not in uri:
        raise GlbParseError("Malformed data URI without comma separator")
    header, payload = uri.split(",", 1)
    if ";base64" in header.lower():
        try:
            return base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise GlbParseError(f"Invalid base64 data URI: {exc}") from exc
    return urllib.parse.unquote_to_bytes(payload)


def _resolve_external_uri(base_dir: Path, uri: str) -> bytes:
    if not uri or not uri.strip():
        raise GlbParseError("Empty external resource URI")
    parsed = urllib.parse.urlsplit(uri)
    if parsed.scheme or uri.startswith("/") or uri.startswith("\\"):
        raise GlbParseError(f"Unsafe external buffer URI scheme or absolute path: {uri}")
    rel_uri = urllib.parse.unquote(uri)
    rel_parsed = urllib.parse.urlsplit(rel_uri)
    rel_path = Path(rel_uri)
    win_path = PureWindowsPath(rel_uri)
    if (
        not rel_uri.strip()
        or "\x00" in rel_uri
        or bool(rel_parsed.scheme)
        or rel_path.is_absolute()
        or win_path.is_absolute()
        or bool(rel_path.drive)
        or bool(win_path.drive)
        or rel_uri.startswith("/")
        or rel_uri.startswith("\\")
    ):
        raise GlbParseError(f"Unsafe external buffer URI traversal: {uri}")
    resolved_base = base_dir.resolve()
    target = (resolved_base / rel_uri).resolve()
    if target == resolved_base or not target.is_relative_to(resolved_base):
        raise GlbParseError(f"Unsafe external buffer URI traversal: {uri}")
    try:
        return target.read_bytes()
    except OSError as exc:
        raise GlbParseError(f"Failed to read external resource '{uri}': {exc}") from exc


def _normalize_int(val: int, comp_type: int) -> float:
    if comp_type == 5120:
        return max(val / 127.0, -1.0)
    if comp_type == 5121:
        return val / 255.0
    if comp_type == 5122:
        return max(val / 32767.0, -1.0)
    if comp_type == 5123:
        return val / 65535.0
    if comp_type == 5125:
        return val / 4294967295.0
    return float(val)


def _read_accessor(
    doc: dict[str, Any],
    buffers: list[bytes],
    acc_idx: int,
) -> list[tuple[float | int, ...]]:
    accessors = _as_list(doc.get("accessors"))
    if not (0 <= acc_idx < len(accessors)):
        raise GlbParseError(f"Accessor index out of range: {acc_idx}")
    acc = _as_dict(accessors[acc_idx])
    comp_type = acc.get("componentType")
    if comp_type not in _COMP_FORMATS:
        raise GlbParseError(f"Unsupported accessor componentType: {comp_type}")
    fmt_char, comp_size = _COMP_FORMATS[comp_type]
    acc_type = acc.get("type")
    if acc_type not in _TYPE_COUNTS:
        raise GlbParseError(f"Unsupported accessor type: {acc_type}")
    n_comps = _TYPE_COUNTS[acc_type]
    count = int(acc.get("count", 0))
    if count < 0:
        raise GlbParseError(f"Negative accessor count: {count}")

    bv_idx = acc.get("bufferView")
    if bv_idx is None:
        zero: float | int = 0.0 if comp_type == 5126 or acc.get("normalized") else 0
        return [(zero,) * n_comps for _ in range(count)]

    buffer_views = _as_list(doc.get("bufferViews"))
    if isinstance(bv_idx, bool) or not isinstance(bv_idx, int) or not (0 <= bv_idx < len(buffer_views)):
        raise GlbParseError(f"bufferView index out of range: {bv_idx}")
    bv = _as_dict(buffer_views[bv_idx])
    buf_idx = bv.get("buffer")
    if isinstance(buf_idx, bool) or not isinstance(buf_idx, int) or not (0 <= buf_idx < len(buffers)):
        raise GlbParseError(f"buffer index out of range: {buf_idx}")
    buf = buffers[buf_idx]

    bv_offset = int(bv.get("byteOffset", 0))
    acc_offset = int(acc.get("byteOffset", 0))
    if bv_offset < 0 or acc_offset < 0:
        raise GlbParseError("Negative bufferView or accessor byteOffset")

    elem_size = comp_size * n_comps
    raw_stride = bv.get("byteStride")
    stride = elem_size if raw_stride is None else int(raw_stride)
    if stride < elem_size:
        raise GlbParseError(f"Invalid byteStride {stride} < element size {elem_size}")

    normalized = bool(acc.get("normalized", False))
    unpack_fmt = f"<{n_comps}{fmt_char}"
    out: list[tuple[float | int, ...]] = []
    for i in range(count):
        base = bv_offset + acc_offset + i * stride
        if base + elem_size > len(buf):
            raise GlbParseError("Accessor read exceeds buffer bounds")
        vals = struct.unpack_from(unpack_fmt, buf, base)
        if normalized and comp_type != 5126:
            vals = tuple(_normalize_int(int(v), comp_type) for v in vals)
        out.append(vals)
    return out


def _triangle_indices_for_mode(idx_list: list[int], mode: int) -> list[tuple[int, int, int]]:
    tris: list[tuple[int, int, int]] = []
    if mode == 4:
        for k in range(0, len(idx_list) - 2, 3):
            tris.append((idx_list[k], idx_list[k + 1], idx_list[k + 2]))
    elif mode == 5:
        for k in range(len(idx_list) - 2):
            if k % 2 == 0:
                tris.append((idx_list[k], idx_list[k + 1], idx_list[k + 2]))
            else:
                tris.append((idx_list[k + 1], idx_list[k], idx_list[k + 2]))
    elif mode == 6:
        for k in range(1, len(idx_list) - 1):
            tris.append((idx_list[0], idx_list[k], idx_list[k + 1]))
    return tris


def _sample_mesh_edges(
    prim_edge_lists: list[list[tuple[Vec3, Vec3]]],
    budget: int,
) -> list[tuple[Vec3, Vec3]]:
    if budget <= 0:
        return []
    non_empty = [edges for edges in prim_edge_lists if edges]
    if not non_empty:
        return []
    total = sum(len(edges) for edges in non_empty)
    if total <= budget:
        out: list[tuple[Vec3, Vec3]] = []
        for edges in non_empty:
            out.extend(edges)
        return out

    n_prims = len(non_empty)
    out = []
    if budget <= n_prims:
        for i in range(budget):
            out.append(non_empty[i][0])
        return out

    alloc = [1] * n_prims
    rem = budget - n_prims
    caps = [len(edges) - 1 for edges in non_empty]
    total_cap = sum(caps)
    if total_cap > 0 and rem > 0:
        extra = [min(caps[i], (rem * caps[i]) // total_cap) for i in range(n_prims)]
        for i in range(n_prims):
            alloc[i] += extra[i]
        leftover = budget - sum(alloc)
        for i in range(n_prims):
            if leftover <= 0:
                break
            room = len(non_empty[i]) - alloc[i]
            if room > 0:
                take = min(room, leftover)
                alloc[i] += take
                leftover -= take

    for edges, quota in zip(non_empty, alloc):
        if quota >= len(edges):
            out.extend(edges)
        elif quota == 1:
            out.append(edges[0])
        else:
            step = (len(edges) - 1) / (quota - 1)
            for k in range(quota):
                out.append(edges[round(k * step)])
    return out


def _node_local_matrix(node: dict[str, Any]) -> tuple[float, ...]:
    mat = node.get("matrix")
    if mat is not None:
        if not isinstance(mat, list) or len(mat) != 16:
            raise GlbParseError("Node matrix must have 16 floats")
        return tuple(float(v) for v in mat)
    t_raw = node.get("translation", [0.0, 0.0, 0.0])
    r_raw = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    s_raw = node.get("scale", [1.0, 1.0, 1.0])
    if len(t_raw) != 3 or len(r_raw) != 4 or len(s_raw) != 3:
        raise GlbParseError("Invalid TRS vector dimensions on node")
    return mat4_from_trs(
        (float(t_raw[0]), float(t_raw[1]), float(t_raw[2])),
        (float(r_raw[0]), float(r_raw[1]), float(r_raw[2]), float(r_raw[3])),
        (float(s_raw[0]), float(s_raw[1]), float(s_raw[2])),
    )


def parse_asset(path: str | Path, max_wireframe_edges_per_mesh: int = 800) -> AssetSummary:
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise GlbParseError(f"Cannot read asset '{p}': {exc}") from exc

    file_size = len(raw)
    glb_bin_chunk: bytes | None = None
    doc: dict[str, Any] | None = None

    if (file_size >= 4 and raw[:4] == b"glTF") or p.suffix.lower() == ".glb":
        container_format: str = "glb"
        if file_size < 20:
            raise GlbParseError("GLB file too small (< 20 bytes)")
        magic, version, declared_len = struct.unpack_from("<III", raw, 0)
        if magic != _GLB_MAGIC:
            raise GlbParseError(f"Invalid GLB magic header: 0x{magic:08x}")
        if version != 2:
            raise GlbParseError(f"Unsupported GLB version: {version}")
        if declared_len != file_size:
            raise GlbParseError(
                f"GLB declared length ({declared_len}) does not match file size ({file_size})"
            )

        offset = 12
        chunk_idx = 0
        while offset < file_size:
            if offset + 8 > file_size:
                raise GlbParseError("Truncated GLB chunk header")
            chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
            offset += 8
            if offset + chunk_len > file_size:
                raise GlbParseError("GLB chunk extends beyond file bounds")
            chunk_data = raw[offset : offset + chunk_len]
            offset += chunk_len

            if chunk_idx == 0:
                if chunk_type != _GLB_CHUNK_JSON:
                    raise GlbParseError("GLB chunk 0 must be JSON")
                try:
                    doc = json.loads(chunk_data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise GlbParseError(f"Invalid GLB JSON chunk: {exc}") from exc
            elif chunk_type == _GLB_CHUNK_BIN:
                if glb_bin_chunk is not None:
                    raise GlbParseError("Multiple BIN chunks in GLB")
                glb_bin_chunk = chunk_data
            elif chunk_type == _GLB_CHUNK_JSON:
                raise GlbParseError("Duplicate JSON chunk in GLB")
            chunk_idx += 1

        if doc is None:
            raise GlbParseError("GLB contains no JSON chunk")
    else:
        container_format = "gltf"
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GlbParseError(f"Invalid glTF JSON: {exc}") from exc

    if not isinstance(doc, dict):
        raise GlbParseError("glTF root JSON must be an object")

    asset_obj = doc.get("asset", {})
    if not isinstance(asset_obj, dict):
        raise GlbParseError("glTF 'asset' field must be an object")
    gltf_version = str(asset_obj.get("version", ""))
    if not gltf_version.startswith("2."):
        raise GlbParseError(f"Unsupported glTF version: {gltf_version!r}")

    generator = str(asset_obj.get("generator") or "unknown")
    extensions_used = sorted(str(x) for x in _as_list(doc.get("extensionsUsed")))

    base_dir = p.parent.resolve()
    buffers: list[bytes] = []
    for idx, raw_buf_def in enumerate(_as_list(doc.get("buffers"))):
        buf_def = _as_dict(raw_buf_def)
        byte_len = int(buf_def.get("byteLength", 0))
        if byte_len < 0:
            raise GlbParseError(f"Negative buffer byteLength at index {idx}: {byte_len}")
        uri = buf_def.get("uri")
        if uri is None:
            if idx == 0 and container_format == "glb":
                if glb_bin_chunk is None or len(glb_bin_chunk) < byte_len:
                    raise GlbParseError("GLB BIN chunk missing or shorter than declared byteLength")
                buffers.append(glb_bin_chunk[:byte_len])
            else:
                raise GlbParseError(f"Buffer {idx} has no URI outside GLB chunk 0")
        elif isinstance(uri, str) and uri.startswith("data:"):
            data = _decode_data_uri(uri)
            if len(data) < byte_len:
                raise GlbParseError(f"Data-URI buffer {idx} shorter than declared byteLength")
            buffers.append(data[:byte_len] if byte_len > 0 else data)
        elif isinstance(uri, str):
            data = _resolve_external_uri(base_dir, uri)
            if len(data) < byte_len:
                raise GlbParseError(f"External buffer {idx} shorter than declared byteLength")
            buffers.append(data[:byte_len] if byte_len > 0 else data)
        else:
            raise GlbParseError(f"Invalid buffer URI type at index {idx}")

    materials_raw = [_as_dict(m) for m in _as_list(doc.get("materials"))]
    material_names = _disambiguate_names(materials_raw, "material")
    materials: list[MaterialSummary] = []
    for m_idx, mat_def in enumerate(materials_raw):
        pbr = _as_dict(mat_def.get("pbrMetallicRoughness"))
        bc_raw = pbr.get("baseColorFactor")
        bc = bc_raw if isinstance(bc_raw, list) else [1.0, 1.0, 1.0, 1.0]
        base_color = (
            float(bc[0]) if len(bc) > 0 else 1.0,
            float(bc[1]) if len(bc) > 1 else 1.0,
            float(bc[2]) if len(bc) > 2 else 1.0,
            float(bc[3]) if len(bc) > 3 else 1.0,
        )
        tex_refs: set[int] = set()
        for tex_obj in (
            pbr.get("baseColorTexture"),
            pbr.get("metallicRoughnessTexture"),
            mat_def.get("normalTexture"),
            mat_def.get("occlusionTexture"),
            mat_def.get("emissiveTexture"),
        ):
            if isinstance(tex_obj, dict) and isinstance(tex_obj.get("index"), int):
                tex_refs.add(tex_obj["index"])
        materials.append(
            MaterialSummary(
                name=material_names[m_idx],
                alpha_mode=str(mat_def.get("alphaMode") or "OPAQUE"),
                double_sided=bool(mat_def.get("doubleSided", False)),
                metallic_factor=float(
                    1.0 if pbr.get("metallicFactor") is None else pbr.get("metallicFactor")
                ),
                roughness_factor=float(
                    1.0 if pbr.get("roughnessFactor") is None else pbr.get("roughnessFactor")
                ),
                base_color=base_color,
                texture_count=len(tex_refs),
            )
        )

    images_raw = [_as_dict(im) for im in _as_list(doc.get("images"))]
    textures_raw = [_as_dict(tx) for tx in _as_list(doc.get("textures"))]
    buffer_views = [_as_dict(bv) for bv in _as_list(doc.get("bufferViews"))]

    def _load_image_bytes(img_def: dict[str, Any]) -> tuple[bytes, str | None]:
        declared_mime = img_def.get("mimeType")
        bv_idx = img_def.get("bufferView")
        if bv_idx is not None:
            if not isinstance(bv_idx, int) or not (0 <= bv_idx < len(buffer_views)):
                raise GlbParseError(f"Image bufferView out of range: {bv_idx}")
            bv = buffer_views[bv_idx]
            buf_idx = bv.get("buffer")
            if not isinstance(buf_idx, int) or not (0 <= buf_idx < len(buffers)):
                raise GlbParseError(f"Image buffer index out of range: {buf_idx}")
            off = int(bv.get("byteOffset", 0))
            length = int(bv.get("byteLength", 0))
            buf = buffers[buf_idx]
            if off < 0 or length < 0 or off + length > len(buf):
                raise GlbParseError("Image bufferView exceeds buffer bounds")
            return buf[off : off + length], declared_mime
        uri = img_def.get("uri")
        if isinstance(uri, str):
            if uri.startswith("data:"):
                return _decode_data_uri(uri), declared_mime
            return _resolve_external_uri(base_dir, uri), declared_mime
        return b"", declared_mime

    textures: list[TextureSummary] = []
    seen_image_bytes: dict[int, int] = {}
    if textures_raw:
        tex_names = _disambiguate_names(textures_raw, "texture")
        for t_idx, tex_def in enumerate(textures_raw):
            src_idx = tex_def.get("source")
            if src_idx is None:
                ktx_ext = _as_dict(_as_dict(tex_def.get("extensions")).get("KHR_texture_basisu"))
                src_idx = ktx_ext.get("source")
            if isinstance(src_idx, int) and 0 <= src_idx < len(images_raw):
                img_def = images_raw[src_idx]
                img_bytes, declared_mime = _load_image_bytes(img_def)
                mime, width, height = sniff_image_header(img_bytes)
                if mime == "application/octet-stream" and isinstance(declared_mime, str):
                    mime = declared_mime
                name = tex_names[t_idx]
                if name.startswith("<unnamed_texture_") and isinstance(img_def.get("name"), str):
                    if img_def["name"].strip():
                        name = img_def["name"].strip()
                seen_image_bytes[src_idx] = len(img_bytes)
                textures.append(
                    TextureSummary(
                        index=t_idx,
                        name=name,
                        mime_type=mime,
                        byte_length=len(img_bytes),
                        width=width,
                        height=height,
                    )
                )
            else:
                textures.append(
                    TextureSummary(
                        index=t_idx,
                        name=tex_names[t_idx],
                        mime_type="external",
                        byte_length=0,
                        width=None,
                        height=None,
                    )
                )
    elif images_raw:
        img_names = _disambiguate_names(images_raw, "texture")
        for i_idx, img_def in enumerate(images_raw):
            img_bytes, declared_mime = _load_image_bytes(img_def)
            mime, width, height = sniff_image_header(img_bytes)
            if mime == "application/octet-stream" and isinstance(declared_mime, str):
                mime = declared_mime
            seen_image_bytes[i_idx] = len(img_bytes)
            textures.append(
                TextureSummary(
                    index=i_idx,
                    name=img_names[i_idx],
                    mime_type=mime,
                    byte_length=len(img_bytes),
                    width=width,
                    height=height,
                )
            )
    total_texture_bytes = sum(seen_image_bytes.values())

    meshes_raw = [_as_dict(m) for m in _as_list(doc.get("meshes"))]
    mesh_names = _disambiguate_names(meshes_raw, "mesh")
    meshes: list[MeshSummary] = []
    mesh_local_edges_map: dict[int, list[tuple[Vec3, Vec3]]] = {}
    total_primitives = 0
    total_vertices = 0
    total_triangles = 0

    accessors = [_as_dict(a) for a in _as_list(doc.get("accessors"))]
    for m_idx, mesh_def in enumerate(meshes_raw):
        primitives = [_as_dict(pr) for pr in _as_list(mesh_def.get("primitives"))]
        m_verts = 0
        m_tris = 0
        attr_set: set[str] = set()
        mat_names_used: list[str] = []
        morph_count = 0
        is_draco = False
        prim_aabbs: list[AABB] = []
        prim_edge_lists: list[list[tuple[Vec3, Vec3]]] = []

        for prim in primitives:
            attrs = _as_dict(prim.get("attributes"))
            attr_set.update(str(k) for k in attrs.keys())
            targets = prim.get("targets")
            if isinstance(targets, list):
                morph_count = max(morph_count, len(targets))
            prim_exts = _as_dict(prim.get("extensions"))
            prim_draco = "KHR_draco_mesh_compression" in prim_exts
            if prim_draco:
                is_draco = True

            mat_idx = prim.get("material")
            if isinstance(mat_idx, int) and 0 <= mat_idx < len(material_names):
                mname = material_names[mat_idx]
                if mname not in mat_names_used:
                    mat_names_used.append(mname)

            positions: list[Vec3] = []
            prim_vcount = 0
            pos_idx = attrs.get("POSITION")
            if isinstance(pos_idx, int):
                if not (0 <= pos_idx < len(accessors)):
                    raise GlbParseError(f"POSITION accessor out of range: {pos_idx}")
                pos_acc = accessors[pos_idx]
                prim_vcount = int(pos_acc.get("count", 0))
                if prim_vcount < 0:
                    raise GlbParseError(f"Negative POSITION accessor count: {prim_vcount}")
                m_verts += prim_vcount
                if not prim_draco:
                    raw_pos = _read_accessor(doc, buffers, pos_idx)
                    positions = [(float(v[0]), float(v[1]), float(v[2])) for v in raw_pos]
                finite_pos = [
                    p0
                    for p0 in positions
                    if math.isfinite(p0[0]) and math.isfinite(p0[1]) and math.isfinite(p0[2])
                ]
                if finite_pos:
                    xs = [p0[0] for p0 in finite_pos]
                    ys = [p0[1] for p0 in finite_pos]
                    zs = [p0[2] for p0 in finite_pos]
                    prim_aabbs.append(
                        AABB(
                            min_pt=(min(xs), min(ys), min(zs)),
                            max_pt=(max(xs), max(ys), max(zs)),
                        )
                    )
                elif (
                    isinstance(pos_acc.get("min"), list)
                    and isinstance(pos_acc.get("max"), list)
                    and len(pos_acc["min"]) >= 3
                    and len(pos_acc["max"]) >= 3
                ):
                    mn = pos_acc["min"]
                    mx = pos_acc["max"]
                    mn_t = (float(mn[0]), float(mn[1]), float(mn[2]))
                    mx_t = (float(mx[0]), float(mx[1]), float(mx[2]))
                    if all(math.isfinite(c) for c in mn_t + mx_t):
                        prim_aabbs.append(AABB(min_pt=mn_t, max_pt=mx_t))

            mode_raw = prim.get("mode", 4)
            mode = 4 if mode_raw is None else int(mode_raw)
            indices_idx = prim.get("indices")
            idx_list: list[int] = []
            if isinstance(indices_idx, int):
                if not (0 <= indices_idx < len(accessors)):
                    raise GlbParseError(f"Indices accessor out of range: {indices_idx}")
                idx_acc = accessors[indices_idx]
                elem_count = int(idx_acc.get("count", 0))
                if elem_count < 0:
                    raise GlbParseError(f"Negative indices accessor count: {elem_count}")
                if not prim_draco:
                    idx_list = [int(v[0]) for v in _read_accessor(doc, buffers, indices_idx)]
            else:
                elem_count = prim_vcount
                if positions:
                    idx_list = list(range(len(positions)))

            if mode == 4:
                m_tris += elem_count // 3
            elif mode in (5, 6):
                m_tris += max(0, elem_count - 2)

            if positions and idx_list and max_wireframe_edges_per_mesh > 0:
                edge_set: set[tuple[int, int]] = set()
                prim_edges: list[tuple[Vec3, Vec3]] = []
                tris = _triangle_indices_for_mode(idx_list, mode)
                if len(tris) > max_wireframe_edges_per_mesh * 2:
                    step = (len(tris) - 1) / (max_wireframe_edges_per_mesh * 2 - 1)
                    tris = [tris[round(k * step)] for k in range(max_wireframe_edges_per_mesh * 2)]
                n_pos = len(positions)
                for a, b, c in tris:
                    if not (0 <= a < n_pos and 0 <= b < n_pos and 0 <= c < n_pos):
                        continue
                    for u, v in ((a, b), (b, c), (c, a)):
                        key = (u, v) if u < v else (v, u)
                        if key not in edge_set:
                            edge_set.add(key)
                            pu, pv = positions[u], positions[v]
                            if all(math.isfinite(coord) for coord in pu + pv):
                                prim_edges.append((pu, pv))
                prim_edge_lists.append(prim_edges)

        local_edges = _sample_mesh_edges(prim_edge_lists, max_wireframe_edges_per_mesh)
        local_aabb = merge_aabbs(prim_aabbs)
        mesh_local_edges_map[m_idx] = local_edges
        total_primitives += len(primitives)
        total_vertices += m_verts
        total_triangles += m_tris
        meshes.append(
            MeshSummary(
                name=mesh_names[m_idx],
                primitive_count=len(primitives),
                vertex_count=m_verts,
                triangle_count=m_tris,
                attributes=sorted(attr_set),
                morph_target_count=morph_count,
                material_names=mat_names_used,
                is_draco_compressed=is_draco,
                local_aabb=local_aabb,
            )
        )

    nodes_raw = [_as_dict(n) for n in _as_list(doc.get("nodes"))]
    node_names = _disambiguate_names(nodes_raw, "node")
    parent_of: dict[int, int] = {}
    for n_idx, node_def in enumerate(nodes_raw):
        for ch_idx in _as_list(node_def.get("children")):
            if not isinstance(ch_idx, int) or not (0 <= ch_idx < len(nodes_raw)):
                raise GlbParseError(f"Child node index out of range: {ch_idx}")
            if ch_idx in parent_of:
                raise GlbParseError(f"Node {ch_idx} has multiple parents")
            parent_of[ch_idx] = n_idx

    scenes_raw = [_as_dict(s) for s in _as_list(doc.get("scenes"))]
    if scenes_raw:
        scene_idx = int(0 if doc.get("scene") is None else doc.get("scene"))
        if not (0 <= scene_idx < len(scenes_raw)):
            raise GlbParseError(f"Default scene index out of range: {scene_idx}")
        scene_roots = [int(n) for n in _as_list(scenes_raw[scene_idx].get("nodes"))]
    else:
        scene_roots = [i for i in range(len(nodes_raw)) if i not in parent_of]

    world_matrices: dict[int, tuple[float, ...]] = {}
    world_aabbs: list[AABB] = []
    mesh_segments: list[dict[str, Any]] = []
    visited: set[int] = set()

    def _visit(node_idx: int, parent_mat: tuple[float, ...], stack: set[int], in_scene: bool) -> None:
        if not (0 <= node_idx < len(nodes_raw)):
            raise GlbParseError(f"Scene node index out of range: {node_idx}")
        if node_idx in stack:
            raise GlbParseError("Cyclic node hierarchy detected")
        stack.add(node_idx)
        visited.add(node_idx)

        node_def = nodes_raw[node_idx]
        local_mat = _node_local_matrix(node_def)
        world_mat = mat4_mul(parent_mat, local_mat)
        world_matrices[node_idx] = world_mat

        if in_scene and isinstance(node_def.get("mesh"), int):
            m_idx = node_def["mesh"]
            if not (0 <= m_idx < len(meshes)):
                raise GlbParseError(f"Node mesh index out of range: {m_idx}")
            m_summary = meshes[m_idx]
            if m_summary.local_aabb is not None:
                world_aabbs.append(transform_aabb(world_mat, m_summary.local_aabb))
            local_edges = mesh_local_edges_map.get(m_idx, [])
            if local_edges:
                world_edges = [
                    (mat4_transform_point(world_mat, p0), mat4_transform_point(world_mat, p1))
                    for p0, p1 in local_edges
                ]
                mesh_segments.append({"name": m_summary.name, "edges": world_edges})

        for ch_idx in _as_list(node_def.get("children")):
            _visit(ch_idx, world_mat, stack, in_scene)

        stack.remove(node_idx)

    for root_idx in scene_roots:
        _visit(root_idx, IDENTITY_MAT4, set(), True)

    # Also traverse any orphan/disconnected nodes to detect cycles and populate joint transforms
    orphan_roots = [i for i in range(len(nodes_raw)) if i not in visited and i not in parent_of]
    for root_idx in orphan_roots:
        _visit(root_idx, IDENTITY_MAT4, set(), False)
    for idx in range(len(nodes_raw)):
        if idx not in visited:
            _visit(idx, IDENTITY_MAT4, set(), False)

    world_aabb = merge_aabbs(world_aabbs)

    skins_raw = [_as_dict(s) for s in _as_list(doc.get("skins"))]
    skin_names = _disambiguate_names(skins_raw, "skin")
    skins: list[SkinSummary] = []
    bone_segments: list[tuple[Vec3, Vec3]] = []
    for s_idx, skin_def in enumerate(skins_raw):
        joint_indices = [int(j) for j in _as_list(skin_def.get("joints"))]
        joint_set = set(joint_indices)
        joints: list[JointInfo] = []
        for j_idx in joint_indices:
            if not (0 <= j_idx < len(nodes_raw)):
                raise GlbParseError(f"Skin joint node index out of range: {j_idx}")
            p_idx = parent_of.get(j_idx)
            p_name = node_names[p_idx] if (p_idx is not None and p_idx in joint_set) else None
            joints.append(JointInfo(node_index=j_idx, name=node_names[j_idx], parent_name=p_name))
            if (
                p_idx is not None
                and p_idx in joint_set
                and p_idx in world_matrices
                and j_idx in world_matrices
            ):
                p_pt = mat4_transform_point(world_matrices[p_idx], (0.0, 0.0, 0.0))
                c_pt = mat4_transform_point(world_matrices[j_idx], (0.0, 0.0, 0.0))
                bone_segments.append((p_pt, c_pt))

        skins.append(
            SkinSummary(
                name=skin_names[s_idx],
                joint_count=len(joints),
                joints=joints,
                has_inverse_bind_matrices=skin_def.get("inverseBindMatrices") is not None,
            )
        )

    anims_raw = [_as_dict(a) for a in _as_list(doc.get("animations"))]
    anim_names = _disambiguate_names(anims_raw, "animation")
    animations: list[AnimationSummary] = []
    for a_idx, anim_def in enumerate(anims_raw):
        channels = [_as_dict(ch) for ch in _as_list(anim_def.get("channels"))]
        samplers = [_as_dict(sp) for sp in _as_list(anim_def.get("samplers"))]
        target_paths = sorted(
            {
                str(ch.get("target", {}).get("path"))
                for ch in channels
                if isinstance(ch.get("target"), dict) and ch["target"].get("path")
            }
        )
        duration_sec = 0.0
        for samp in samplers:
            in_idx = samp.get("input")
            if isinstance(in_idx, int) and 0 <= in_idx < len(accessors):
                in_acc = accessors[in_idx]
                acc_max = in_acc.get("max")
                if isinstance(acc_max, list) and acc_max:
                    duration_sec = max(duration_sec, float(acc_max[0]))
                else:
                    ts = [float(v[0]) for v in _read_accessor(doc, buffers, in_idx)]
                    if ts:
                        duration_sec = max(duration_sec, max(ts))

        animations.append(
            AnimationSummary(
                name=anim_names[a_idx],
                channel_count=len(channels),
                sampler_count=len(samplers),
                duration_sec=duration_sec,
                target_paths=target_paths,
            )
        )

    return AssetSummary(
        path=str(p),
        container_format=container_format,  # type: ignore[arg-type]
        file_size_bytes=file_size,
        generator=generator,
        extensions_used=extensions_used,
        node_count=len(nodes_raw),
        mesh_count=len(meshes),
        total_primitives=total_primitives,
        total_vertices=total_vertices,
        total_triangles=total_triangles,
        total_texture_bytes=total_texture_bytes,
        world_aabb=world_aabb,
        meshes=meshes,
        materials=materials,
        textures=textures,
        skins=skins,
        animations=animations,
        preview=WireframePreview(
            mesh_segments=mesh_segments,
            bone_segments=bone_segments,
            world_aabb=world_aabb,
        ),
    )
