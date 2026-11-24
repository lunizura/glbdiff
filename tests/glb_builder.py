from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any


def make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(ihdr))
        + b"IHDR"
        + ihdr
        + b"\x00\x00\x00\x00"
        + struct.pack(">I", 0)
        + b"IEND\xae\x42\x60\x82"
    )


def make_jpeg_bytes(width: int = 128, height: int = 128) -> bytes:
    sof0 = b"\xff\xc0" + struct.pack(">HBHH", 11, 8, height, width) + (b"\x00" * 4)
    return b"\xff\xd8" + sof0 + b"\xff\xd9"


def make_webp_vp8x_bytes(width: int = 256, height: int = 256) -> bytes:
    w_m1 = (width - 1).to_bytes(3, "little")
    h_m1 = (height - 1).to_bytes(3, "little")
    return b"RIFF\x20\x00\x00\x00WEBPVP8X" + (b"\x00" * 8) + w_m1 + h_m1


def make_ktx2_bytes(width: int = 512, height: int = 512) -> bytes:
    return b"\xabKTX 20\xbb\r\n\x1a\n" + struct.pack("<IIII", 37, 1, width, height)


def pack_glb(doc: dict[str, Any], bin_data: bytes = b"") -> bytes:
    json_bytes = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "

    chunks = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    if bin_data:
        padded_bin = bin_data
        while len(padded_bin) % 4 != 0:
            padded_bin += b"\x00"
        chunks += struct.pack("<II", len(padded_bin), 0x004E4942) + padded_bin

    total_len = 12 + len(chunks)
    header = struct.pack("<III", 0x46546C67, 2, total_len)
    return header + chunks


class GltfSceneBuilder:
    """Deterministic builder for synthetic glTF 2.0 scenes (.glb and .gltf)."""

    def __init__(self, generator: str = "glbdiff-test-builder") -> None:
        self.bin_buf = bytearray()
        self.buffer_views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []
        self.meshes: list[dict[str, Any]] = []
        self.materials: list[dict[str, Any]] = []
        self.textures: list[dict[str, Any]] = []
        self.images: list[dict[str, Any]] = []
        self.nodes: list[dict[str, Any]] = []
        self.skins: list[dict[str, Any]] = []
        self.animations: list[dict[str, Any]] = []
        self.scene_nodes: list[int] = []
        self.extensions_used: list[str] = []
        self.generator = generator

    def _align4(self) -> None:
        while len(self.bin_buf) % 4 != 0:
            self.bin_buf.append(0)

    def append_bytes(self, data: bytes, byte_stride: int | None = None) -> int:
        self._align4()
        offset = len(self.bin_buf)
        self.bin_buf.extend(data)
        bv: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data),
        }
        if byte_stride is not None:
            bv["byteStride"] = byte_stride
        self.buffer_views.append(bv)
        return len(self.buffer_views) - 1

    def add_accessor(
        self,
        bv_idx: int | None,
        component_type: int,
        count: int,
        acc_type: str,
        byte_offset: int = 0,
        min_vals: list[float] | None = None,
        max_vals: list[float] | None = None,
    ) -> int:
        acc: dict[str, Any] = {
            "componentType": component_type,
            "count": count,
            "type": acc_type,
        }
        if bv_idx is not None:
            acc["bufferView"] = bv_idx
            acc["byteOffset"] = byte_offset
        if min_vals is not None:
            acc["min"] = min_vals
        if max_vals is not None:
            acc["max"] = max_vals
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def add_material(
        self,
        name: str,
        alpha_mode: str = "OPAQUE",
        double_sided: bool = False,
        metallic: float = 0.0,
        roughness: float = 0.8,
        base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        base_color_tex: int | None = None,
    ) -> int:
        pbr: dict[str, Any] = {
            "baseColorFactor": list(base_color),
            "metallicFactor": metallic,
            "roughnessFactor": roughness,
        }
        if base_color_tex is not None:
            pbr["baseColorTexture"] = {"index": base_color_tex}
        mat: dict[str, Any] = {
            "name": name,
            "alphaMode": alpha_mode,
            "doubleSided": double_sided,
            "pbrMetallicRoughness": pbr,
        }
        self.materials.append(mat)
        return len(self.materials) - 1

    def add_embedded_texture(self, name: str, img_bytes: bytes, mime_type: str | None = None) -> int:
        bv_idx = self.append_bytes(img_bytes)
        img_def: dict[str, Any] = {"name": name, "bufferView": bv_idx}
        if mime_type is not None:
            img_def["mimeType"] = mime_type
        self.images.append(img_def)
        img_idx = len(self.images) - 1
        self.textures.append({"name": name, "source": img_idx})
        return len(self.textures) - 1

    def add_mesh(
        self,
        name: str,
        positions: list[tuple[float, float, float]],
        indices: list[int] | None = None,
        uvs: list[tuple[float, float]] | None = None,
        normals: list[tuple[float, float, float]] | None = None,
        material_idx: int | None = None,
        mode: int = 4,
        morph_targets: int = 0,
        interleaved: bool = False,
    ) -> int:
        attrs: dict[str, int] = {}
        if interleaved:
            # Interleave POSITION (12B) + NORMAL (12B) + optional UV (8B)
            has_uv = uvs is not None
            stride = 24 + (8 if has_uv else 0)
            raw = bytearray()
            for i, p in enumerate(positions):
                raw.extend(struct.pack("<fff", *p))
                n = normals[i] if normals else (0.0, 1.0, 0.0)
                raw.extend(struct.pack("<fff", *n))
                if has_uv and uvs is not None:
                    raw.extend(struct.pack("<ff", *uvs[i]))
            bv_idx = self.append_bytes(bytes(raw), byte_stride=stride)
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            zs = [p[2] for p in positions]
            attrs["POSITION"] = self.add_accessor(
                bv_idx, 5126, len(positions), "VEC3", 0,
                [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)],
            )
            attrs["NORMAL"] = self.add_accessor(bv_idx, 5126, len(positions), "VEC3", 12)
            if has_uv:
                attrs["TEXCOORD_0"] = self.add_accessor(bv_idx, 5126, len(positions), "VEC2", 24)
        else:
            pos_raw = b"".join(struct.pack("<fff", *p) for p in positions)
            pos_bv = self.append_bytes(pos_raw)
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            zs = [p[2] for p in positions]
            attrs["POSITION"] = self.add_accessor(
                pos_bv, 5126, len(positions), "VEC3", 0,
                [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)],
            )
            if normals is not None:
                norm_raw = b"".join(struct.pack("<fff", *n) for n in normals)
                norm_bv = self.append_bytes(norm_raw)
                attrs["NORMAL"] = self.add_accessor(norm_bv, 5126, len(normals), "VEC3")
            if uvs is not None:
                uv_raw = b"".join(struct.pack("<ff", *u) for u in uvs)
                uv_bv = self.append_bytes(uv_raw)
                attrs["TEXCOORD_0"] = self.add_accessor(uv_bv, 5126, len(uvs), "VEC2")

        prim: dict[str, Any] = {"attributes": attrs, "mode": mode}
        if indices is not None:
            idx_raw = b"".join(struct.pack("<H", i) for i in indices)
            idx_bv = self.append_bytes(idx_raw)
            prim["indices"] = self.add_accessor(idx_bv, 5123, len(indices), "SCALAR")
        if material_idx is not None:
            prim["material"] = material_idx
        if morph_targets > 0:
            prim["targets"] = [{"POSITION": attrs["POSITION"]} for _ in range(morph_targets)]

        self.meshes.append({"name": name, "primitives": [prim]})
        return len(self.meshes) - 1

    def add_node(
        self,
        name: str,
        mesh_idx: int | None = None,
        translation: tuple[float, float, float] | None = None,
        rotation: tuple[float, float, float, float] | None = None,
        scale: tuple[float, float, float] | None = None,
        matrix: list[float] | None = None,
        children: list[int] | None = None,
        skin_idx: int | None = None,
        add_to_scene: bool = False,
    ) -> int:
        node: dict[str, Any] = {"name": name}
        if mesh_idx is not None:
            node["mesh"] = mesh_idx
        if translation is not None:
            node["translation"] = list(translation)
        if rotation is not None:
            node["rotation"] = list(rotation)
        if scale is not None:
            node["scale"] = list(scale)
        if matrix is not None:
            node["matrix"] = list(matrix)
        if children:
            node["children"] = list(children)
        if skin_idx is not None:
            node["skin"] = skin_idx
        self.nodes.append(node)
        idx = len(self.nodes) - 1
        if add_to_scene:
            self.scene_nodes.append(idx)
        return idx

    def add_skin(self, name: str, joints: list[int], include_ibm: bool = True) -> int:
        skin: dict[str, Any] = {"name": name, "joints": list(joints)}
        if include_ibm and joints:
            ident = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]
            raw = b"".join(struct.pack("<16f", *ident) for _ in joints)
            bv = self.append_bytes(raw)
            skin["inverseBindMatrices"] = self.add_accessor(bv, 5126, len(joints), "MAT4")
        self.skins.append(skin)
        return len(self.skins) - 1

    def add_animation(
        self,
        name: str,
        node_idx: int,
        duration_sec: float,
        path: str = "rotation",
    ) -> int:
        times = [0.0, duration_sec]
        t_bv = self.append_bytes(struct.pack("<2f", *times))
        in_acc = self.add_accessor(t_bv, 5126, 2, "SCALAR", 0, [0.0], [duration_sec])
        if path == "rotation":
            vals = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            v_bv = self.append_bytes(struct.pack("<8f", *vals))
            out_acc = self.add_accessor(v_bv, 5126, 2, "VEC4")
        else:
            vals = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            v_bv = self.append_bytes(struct.pack("<6f", *vals))
            out_acc = self.add_accessor(v_bv, 5126, 2, "VEC3")

        self.animations.append(
            {
                "name": name,
                "samplers": [{"input": in_acc, "output": out_acc, "interpolation": "LINEAR"}],
                "channels": [{"sampler": 0, "target": {"node": node_idx, "path": path}}],
            }
        )
        return len(self.animations) - 1

    def to_doc(self, buffer_uri: str | None = None) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "asset": {"version": "2.0", "generator": self.generator},
            "scene": 0,
            "scenes": [{"nodes": self.scene_nodes}],
            "nodes": self.nodes,
            "meshes": self.meshes,
        }
        if self.extensions_used:
            doc["extensionsUsed"] = self.extensions_used
        if self.materials:
            doc["materials"] = self.materials
        if self.textures:
            doc["textures"] = self.textures
        if self.images:
            doc["images"] = self.images
        if self.skins:
            doc["skins"] = self.skins
        if self.animations:
            doc["animations"] = self.animations
        if self.bin_buf or self.buffer_views:
            buf_entry: dict[str, Any] = {"byteLength": len(self.bin_buf)}
            if buffer_uri is not None:
                buf_entry["uri"] = buffer_uri
            doc["buffers"] = [buf_entry]
        if self.buffer_views:
            doc["bufferViews"] = self.buffer_views
        if self.accessors:
            doc["accessors"] = self.accessors
        return doc

    def build_glb_bytes(self) -> bytes:
        return pack_glb(self.to_doc(), bytes(self.bin_buf))

    def write_glb(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.build_glb_bytes())
        return p

    def write_gltf_data_uri(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        b64 = base64.b64encode(bytes(self.bin_buf)).decode("ascii")
        uri = f"data:application/octet-stream;base64,{b64}"
        doc = self.to_doc(buffer_uri=uri)
        p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return p

    def write_gltf_external(self, gltf_path: str | Path, bin_name: str = "scene.bin") -> Path:
        p = Path(gltf_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bin_path = p.parent / bin_name
        bin_path.write_bytes(bytes(self.bin_buf))
        doc = self.to_doc(buffer_uri=bin_name)
        p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return p
