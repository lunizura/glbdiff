from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from glbdiff.errors import GlbParseError
from glbdiff.parser import parse_asset
from tests.glb_builder import (
    GltfSceneBuilder,
    make_jpeg_bytes,
    make_png_bytes,
    pack_glb,
)


def test_parse_glb_with_interleaved_stride_skin_anim_and_textures(tmp_path: Path) -> None:
    b = GltfSceneBuilder()
    tex_png = b.add_embedded_texture("Albedo", make_png_bytes(512, 512))
    tex_jpg = b.add_embedded_texture("Normal", make_jpeg_bytes(256, 256))
    mat_idx = b.add_material("HeroMat", alpha_mode="MASK", base_color_tex=tex_png)
    assert tex_jpg == 1

    # Single triangle with interleaved POSITION + NORMAL + TEXCOORD_0
    mesh_idx = b.add_mesh(
        "HeroMesh",
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
        indices=[0, 1, 2],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        material_idx=mat_idx,
        morph_targets=2,
        interleaved=True,
    )
    child_joint = b.add_node("Spine", translation=(0.0, 1.0, 0.0))
    root_joint = b.add_node("Root", translation=(0.0, 0.0, 0.0), children=[child_joint])
    skin_idx = b.add_skin("HeroRig", [root_joint, child_joint], include_ibm=True)
    b.add_node("HeroNode", mesh_idx=mesh_idx, skin_idx=skin_idx, scale=(2.0, 2.0, 2.0), add_to_scene=True)
    b.nodes.append  # sanity check
    b.scene_nodes.append(root_joint)
    b.add_animation("Idle", root_joint, duration_sec=1.5, path="rotation")

    glb_path = b.write_glb(tmp_path / "hero.glb")
    summary = parse_asset(glb_path)

    assert summary.container_format == "glb"
    assert summary.mesh_count == 1
    assert summary.total_vertices == 3
    assert summary.total_triangles == 1
    assert summary.meshes[0].attributes == ["NORMAL", "POSITION", "TEXCOORD_0"]
    assert summary.meshes[0].morph_target_count == 2
    assert summary.meshes[0].material_names == ["HeroMat"]
    assert summary.materials[0].alpha_mode == "MASK"
    assert summary.materials[0].texture_count == 1
    assert len(summary.textures) == 2
    assert summary.textures[0].mime_type == "image/png"
    assert (summary.textures[0].width, summary.textures[0].height) == (512, 512)
    assert summary.textures[1].mime_type == "image/jpeg"
    assert (summary.textures[1].width, summary.textures[1].height) == (256, 256)

    # World AABB scaled by 2.0: max becomes (2, 4, 0)
    assert summary.world_aabb is not None
    assert math.isclose(summary.world_aabb.max_pt[0], 2.0, abs_tol=1e-5)
    assert math.isclose(summary.world_aabb.max_pt[1], 4.0, abs_tol=1e-5)

    # Skin hierarchy & wireframe bones
    assert len(summary.skins) == 1
    assert summary.skins[0].joint_count == 2
    assert summary.skins[0].has_inverse_bind_matrices is True
    assert summary.skins[0].joints[0].parent_name is None
    assert summary.skins[0].joints[1].parent_name == "Root"
    assert len(summary.preview.bone_segments) == 1
    assert len(summary.preview.mesh_segments) == 1
    assert len(summary.preview.mesh_segments[0]["edges"]) == 3

    # Animations & textconv
    assert len(summary.animations) == 1
    assert math.isclose(summary.animations[0].duration_sec, 1.5, abs_tol=1e-5)
    assert summary.animations[0].target_paths == ["rotation"]
    tc = summary.to_textconv()
    assert "HeroMesh" in tc and "Spine <- Root" in tc


def test_parse_gltf_data_uri_and_triangle_strip_and_external_bin(tmp_path: Path) -> None:
    b = GltfSceneBuilder()
    # 4 vertices in TRIANGLE_STRIP -> 2 triangles
    b.add_mesh(
        "QuadStrip",
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)],
        indices=[0, 1, 2, 3],
        mode=5,
    )
    b.add_node("QuadNode", mesh_idx=0, add_to_scene=True)

    gltf_data = b.write_gltf_data_uri(tmp_path / "embedded.gltf")
    s1 = parse_asset(gltf_data)
    assert s1.container_format == "gltf"
    assert s1.total_triangles == 2
    assert len(s1.preview.mesh_segments[0]["edges"]) == 5  # 2 tris sharing diagonal = 5 unique edges

    gltf_ext = b.write_gltf_external(tmp_path / "ext.gltf", "ext.bin")
    s2 = parse_asset(gltf_ext)
    assert s2.container_format == "gltf"
    assert s2.total_triangles == 2


def test_parse_disambiguates_duplicate_and_empty_names(tmp_path: Path) -> None:
    b = GltfSceneBuilder()
    b.add_mesh("Cube", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [0, 1, 2])
    b.add_mesh("Cube", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [0, 1, 2])
    b.add_mesh("   ", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [0, 1, 2])
    b.add_node("N0", mesh_idx=0, add_to_scene=True)
    b.add_node("N1", mesh_idx=1, add_to_scene=True)
    b.add_node("N2", mesh_idx=2, add_to_scene=True)

    s = parse_asset(b.write_glb(tmp_path / "dup.glb"))
    assert [m.name for m in s.meshes] == ["Cube", "Cube#1", "<unnamed_mesh_2>"]


def test_path_traversal_uri_rejected(tmp_path: Path) -> None:
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"\x00" * 64)

    sub = tmp_path / "assets"
    sub.mkdir()
    gltf_path = sub / "evil.gltf"
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "../secret.bin", "byteLength": 36}],
    }
    gltf_path.write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(GlbParseError, match="traversal"):
        parse_asset(gltf_path)

    # Absolute / scheme URIs also rejected
    doc_http = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "https://example.com/model.bin", "byteLength": 36}],
    }
    gltf_path.write_text(json.dumps(doc_http), encoding="utf-8")
    with pytest.raises(GlbParseError, match="Unsafe external"):
        parse_asset(gltf_path)


def test_corrupt_glb_headers_and_cyclic_nodes_rejected(tmp_path: Path) -> None:
    # Bad magic
    bad_magic = tmp_path / "bad_magic.glb"
    bad_magic.write_bytes(b"NOPE" + struct.pack("<II", 2, 20) + b"\x00" * 8)
    with pytest.raises(GlbParseError, match="magic"):
        parse_asset(bad_magic)

    # Declared length mismatch
    bad_len = tmp_path / "bad_len.glb"
    bad_len.write_bytes(struct.pack("<III", 0x46546C67, 2, 999) + b"\x00" * 8)
    with pytest.raises(GlbParseError, match="declared length"):
        parse_asset(bad_len)

    # Chunk extends past EOF
    bad_chunk = tmp_path / "bad_chunk.glb"
    hdr = struct.pack("<III", 0x46546C67, 2, 24)
    chunk_hdr = struct.pack("<II", 500, 0x4E4F534A)
    bad_chunk.write_bytes(hdr + chunk_hdr + b"{}  ")
    with pytest.raises(GlbParseError, match="beyond file bounds"):
        parse_asset(bad_chunk)

    # Cyclic node hierarchy
    cyclic_doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "A", "children": [1]},
            {"name": "B", "children": [0]},
        ],
    }
    cyclic_glb = tmp_path / "cyclic.glb"
    cyclic_glb.write_bytes(pack_glb(cyclic_doc))
    with pytest.raises(GlbParseError, match="Cyclic"):
        parse_asset(cyclic_glb)


def test_draco_bufferless_non_indexed_fan_and_normalized_positions(tmp_path: Path) -> None:
    # 1. Draco compressed primitive with min/max fallback and no bufferView
    draco_doc = {
        "asset": {"version": "2.0"},
        "meshes": [
            {
                "name": "DracoMesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "indices": 2,
                        "extensions": {
                            "KHR_draco_mesh_compression": {
                                "bufferView": 0,
                                "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                            }
                        },
                    }
                ],
            }
        ],
        "accessors": [
            {
                "componentType": 5126,
                "count": 120,
                "type": "VEC3",
                "min": [-1.0, -2.0, -3.0],
                "max": [1.0, 2.0, 3.0],
            },
            {"componentType": 5126, "count": 120, "type": "VEC2"},
            {"componentType": 5123, "count": 300, "type": "SCALAR"},
        ],
        "nodes": [{"name": "DracoNode", "mesh": 0}],
        "scenes": [{"nodes": [0]}],
    }
    p_draco = tmp_path / "draco.glb"
    p_draco.write_bytes(pack_glb(draco_doc))
    s_draco = parse_asset(p_draco)
    assert s_draco.meshes[0].is_draco_compressed is True
    assert s_draco.total_vertices == 120
    assert s_draco.total_triangles == 100
    assert s_draco.world_aabb is not None
    assert s_draco.world_aabb.min_pt == (-1.0, -2.0, -3.0)
    assert s_draco.world_aabb.max_pt == (1.0, 2.0, 3.0)

    # 2. Bufferless zero-filled accessor fallback on non-Draco mesh
    bufless_doc = {
        "asset": {"version": "2.0"},
        "meshes": [{"name": "ZeroMesh", "primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"componentType": 5126, "count": 3, "type": "VEC3"}],
        "nodes": [{"name": "N", "mesh": 0}],
        "scenes": [{"nodes": [0]}],
    }
    p_bufless = tmp_path / "bufless.glb"
    p_bufless.write_bytes(pack_glb(bufless_doc))
    s_bufless = parse_asset(p_bufless)
    assert s_bufless.total_vertices == 3
    assert s_bufless.total_triangles == 1
    assert s_bufless.world_aabb is not None
    assert s_bufless.world_aabb.min_pt == (0.0, 0.0, 0.0)

    # 3. Non-indexed TRIANGLES, TRIANGLE_FAN (mode=6), POINTS (mode=0), and normalized SHORT
    b = GltfSceneBuilder()
    b.add_mesh(
        "NonIndexedTris",
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        indices=None,
        mode=4,
    )
    b.add_mesh(
        "FanMesh",
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        indices=[0, 1, 2, 3],
        mode=6,
    )
    b.add_mesh("PointsMesh", positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], mode=0)
    # Normalized SHORT VEC3
    norm_bv = b.append_bytes(struct.pack("<3h", 32767, 0, -32767) * 3)
    norm_acc = b.add_accessor(norm_bv, 5122, 3, "VEC3")
    b.accessors[norm_acc]["normalized"] = True
    b.meshes.append({"name": "QuantMesh", "primitives": [{"attributes": {"POSITION": norm_acc}}]})
    for idx in range(4):
        b.add_node(f"N{idx}", mesh_idx=idx, add_to_scene=True)

    s_mix = parse_asset(b.write_glb(tmp_path / "mix.glb"))
    assert s_mix.meshes[0].triangle_count == 1
    assert s_mix.meshes[1].triangle_count == 2
    assert s_mix.meshes[2].triangle_count == 0
    assert s_mix.meshes[3].local_aabb is not None
    assert math.isclose(s_mix.meshes[3].local_aabb.max_pt[0], 1.0, abs_tol=1e-4)
    assert math.isclose(s_mix.meshes[3].local_aabb.min_pt[2], -1.0, abs_tol=1e-4)


def test_null_json_fields_negative_lengths_drive_uri_and_multiprim_wireframe(
    tmp_path: Path,
) -> None:
    # 1. Explicit null JSON fields do not crash parser
    nulls_doc = {
        "asset": {"version": "2.0", "generator": None},
        "extensionsUsed": None,
        "materials": [{"name": "M", "pbrMetallicRoughness": None, "alphaMode": None}],
        "textures": [{"source": None, "extensions": None}],
        "meshes": [{"name": "Mesh", "primitives": [{"attributes": None, "extensions": None}]}],
    }
    p_nulls = tmp_path / "nulls.glb"
    p_nulls.write_bytes(pack_glb(nulls_doc))
    s_nulls = parse_asset(p_nulls)
    assert s_nulls.mesh_count == 1
    assert s_nulls.materials[0].alpha_mode == "OPAQUE"

    # 2. Negative buffer byteLength rejected
    neg_doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": -1}]}
    p_neg = tmp_path / "neg.glb"
    p_neg.write_bytes(pack_glb(neg_doc, b"\x00" * 16))
    with pytest.raises(GlbParseError, match="Negative buffer"):
        parse_asset(p_neg)

    # 3. URL-encoded Windows drive-relative URI rejected before filesystem access
    p_drive = tmp_path / "drive.gltf"
    p_drive.write_text(
        json.dumps({"asset": {"version": "2.0"}, "buffers": [{"uri": "%43:secret.bin", "byteLength": 4}]}),
        encoding="utf-8",
    )
    with pytest.raises(GlbParseError, match="Unsafe external"):
        parse_asset(p_drive)

    # 4. Multi-primitive wireframe edge budget keeps edges from later primitives
    b = GltfSceneBuilder()
    pos0: list[tuple[float, float, float]] = []
    idx0: list[int] = []
    for i in range(900):
        base = i * 3
        pos0.extend([(0.0, 0.0, float(i)), (1.0, 0.0, float(i)), (0.0, 1.0, float(i))])
        idx0.extend([base, base + 1, base + 2])
    bv_p0 = b.append_bytes(b"".join(struct.pack("<fff", *p) for p in pos0))
    bv_i0 = b.append_bytes(b"".join(struct.pack("<H", x) for x in idx0))
    ap0 = b.add_accessor(bv_p0, 5126, len(pos0), "VEC3")
    ai0 = b.add_accessor(bv_i0, 5123, len(idx0), "SCALAR")

    pos1 = [(10.0, 10.0, 0.0), (11.0, 10.0, 0.0), (10.0, 11.0, 0.0)]
    bv_p1 = b.append_bytes(b"".join(struct.pack("<fff", *p) for p in pos1))
    bv_i1 = b.append_bytes(struct.pack("<HHH", 0, 1, 2))
    ap1 = b.add_accessor(bv_p1, 5126, 3, "VEC3")
    ai1 = b.add_accessor(bv_i1, 5123, 3, "SCALAR")

    b.meshes.append(
        {
            "name": "MultiPrim",
            "primitives": [
                {"attributes": {"POSITION": ap0}, "indices": ai0},
                {"attributes": {"POSITION": ap1}, "indices": ai1},
            ],
        }
    )
    b.add_node("N", mesh_idx=0, add_to_scene=True)
    s_mp = parse_asset(b.write_glb(tmp_path / "multiprim.glb"), max_wireframe_edges_per_mesh=800)
    edges = s_mp.preview.mesh_segments[0]["edges"]
    assert len(edges) <= 800
    assert any(e[0][0] >= 9.0 for e in edges)
