from __future__ import annotations

from pathlib import Path

import pytest

from glbdiff.diff import diff_assets
from glbdiff.errors import PolicyConfigError
from glbdiff.parser import parse_asset
from glbdiff.policy import PolicyConfig, evaluate_policy
from tests.glb_builder import GltfSceneBuilder, make_png_bytes


def _make_base_hero(
    path: Path,
    *,
    scale: float = 1.0,
    include_uv: bool = True,
    morph_targets: int = 2,
    alpha_mode: str = "OPAQUE",
    joint_names: tuple[str, str, str] = ("Root", "Spine", "Head"),
    reparent_head_to_root: bool = False,
    include_head: bool = True,
    include_skin: bool = True,
    clips: tuple[tuple[str, float], ...] = (("Idle", 1.0), ("Run", 0.8)),
    tri_count: int = 1,
    tex_bytes: bytes | None = None,
) -> Path:
    b = GltfSceneBuilder()
    tex_idx = None
    if tex_bytes is not None:
        tex_idx = b.add_embedded_texture("Albedo", tex_bytes)
    mat_idx = b.add_material("HeroMat", alpha_mode=alpha_mode, base_color_tex=tex_idx)

    positions: list[tuple[float, float, float]] = []
    indices: list[int] = []
    uvs: list[tuple[float, float]] = []
    for t in range(tri_count):
        base = t * 3
        positions.extend([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, float(t) * 0.01)])
        indices.extend([base, base + 1, base + 2])
        uvs.extend([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])

    mesh_idx = b.add_mesh(
        "HeroMesh",
        positions=positions,
        indices=indices,
        uvs=uvs if include_uv else None,
        material_idx=mat_idx,
        morph_targets=morph_targets,
    )

    if include_skin:
        n_root = joint_names[0]
        n_spine = joint_names[1]
        n_head = joint_names[2]
        if include_head:
            head_idx = b.add_node(n_head, translation=(0.0, 0.5, 0.0))
            if reparent_head_to_root:
                spine_idx = b.add_node(n_spine, translation=(0.0, 1.0, 0.0))
                root_idx = b.add_node(n_root, children=[spine_idx, head_idx])
            else:
                spine_idx = b.add_node(n_spine, translation=(0.0, 1.0, 0.0), children=[head_idx])
                root_idx = b.add_node(n_root, children=[spine_idx])
            joints = [root_idx, spine_idx, head_idx]
        else:
            spine_idx = b.add_node(n_spine, translation=(0.0, 1.0, 0.0))
            root_idx = b.add_node(n_root, children=[spine_idx])
            joints = [root_idx, spine_idx]
        skin_idx = b.add_skin("HeroRig", joints)
        b.scene_nodes.append(root_idx)
    else:
        root_idx = b.add_node("DummyRoot", add_to_scene=True)
        skin_idx = None

    b.add_node(
        "HeroNode",
        mesh_idx=mesh_idx,
        skin_idx=skin_idx,
        scale=(scale, scale, scale),
        add_to_scene=True,
    )
    for clip_name, dur in clips:
        b.add_animation(clip_name, root_idx, duration_sec=dur)
    return b.write_glb(path)


def test_unit_scale_100x_trap_and_large_scale_drift(tmp_path: Path) -> None:
    p_old = _make_base_hero(tmp_path / "v1.glb", scale=1.0)
    p_100x = _make_base_hero(tmp_path / "v2_100x.glb", scale=100.0)
    p_001x = _make_base_hero(tmp_path / "v2_001x.glb", scale=0.01)
    p_drift = _make_base_hero(tmp_path / "v2_drift.glb", scale=3.5)

    s_old = parse_asset(p_old)

    rep_up = diff_assets(s_old, parse_asset(p_100x))
    assert rep_up.has_breaking
    assert any(f.code == "UNIT_SCALE_100X_SHIFT" and f.severity == "BREAKING" for f in rep_up.findings)

    rep_down = diff_assets(s_old, parse_asset(p_001x))
    assert any(f.code == "UNIT_SCALE_100X_SHIFT" and f.severity == "BREAKING" for f in rep_down.findings)

    rep_drift = diff_assets(s_old, parse_asset(p_drift))
    assert not rep_drift.has_breaking
    assert any(f.code == "LARGE_SCALE_DRIFT" and f.severity == "WARNING" for f in rep_drift.findings)


def test_rig_bone_removal_rename_reparent_and_skin_removed(tmp_path: Path) -> None:
    s_old = parse_asset(_make_base_hero(tmp_path / "v1.glb"))

    # Head joint removed
    rep_rem = diff_assets(s_old, parse_asset(_make_base_hero(tmp_path / "v_rem.glb", include_head=False)))
    assert any(f.code == "SKELETON_JOINT_REMOVED" and f.subject == "Head" for f in rep_rem.findings)

    # Spine renamed -> Spine missing AND Head reparented to Chest
    rep_ren = diff_assets(
        s_old,
        parse_asset(_make_base_hero(tmp_path / "v_ren.glb", joint_names=("Root", "Chest", "Head"))),
    )
    codes = {(f.code, f.subject) for f in rep_ren.findings}
    assert ("SKELETON_JOINT_REMOVED", "Spine") in codes
    assert ("SKELETON_JOINT_REPARENTED", "Head") in codes

    # Head reparented from Spine to Root
    rep_rep = diff_assets(
        s_old,
        parse_asset(_make_base_hero(tmp_path / "v_rep.glb", reparent_head_to_root=True)),
    )
    assert any(f.code == "SKELETON_JOINT_REPARENTED" and f.subject == "Head" for f in rep_rep.findings)

    # Entire skin removed
    rep_noskin = diff_assets(
        s_old,
        parse_asset(_make_base_hero(tmp_path / "v_noskin.glb", include_skin=False)),
    )
    assert any(f.code == "SKELETON_JOINT_REMOVED" and f.subject == "HeroRig" for f in rep_noskin.findings)


def test_dropped_uvs_morphs_clips_mesh_and_material_warnings(tmp_path: Path) -> None:
    s_old = parse_asset(
        _make_base_hero(
            tmp_path / "v1.glb",
            tex_bytes=make_png_bytes(64, 64) + b"\x00" * 100,
            tri_count=10,
        )
    )
    s_new = parse_asset(
        _make_base_hero(
            tmp_path / "v2.glb",
            include_uv=False,
            morph_targets=0,
            alpha_mode="BLEND",
            clips=(("Idle", 1.25),),  # Run removed, Idle duration changed +0.25s
            tri_count=15,  # +50% -> TRIANGLE_SPIKE
            tex_bytes=make_png_bytes(64, 64) + b"\x00" * 400,  # >25% -> TEXTURE_MEMORY_SPIKE
        )
    )

    rep = diff_assets(s_old, s_new)
    codes = {f.code: f.severity for f in rep.findings}
    assert codes["UV_CHANNEL_DROPPED"] == "BREAKING"
    assert codes["MORPH_TARGETS_DROPPED"] == "BREAKING"
    assert codes["ANIMATION_CLIP_REMOVED"] == "BREAKING"
    assert codes["ANIMATION_DURATION_CHANGED"] == "WARNING"
    assert codes["MATERIAL_ALPHA_MODE_CHANGED"] == "WARNING"
    assert codes["TRIANGLE_SPIKE"] == "WARNING"
    assert codes["TEXTURE_MEMORY_SPIKE"] == "WARNING"
    assert "ASSET_OPTIMIZED" not in codes

    # Also test MESH_REMOVED
    b_empty = GltfSceneBuilder()
    b_empty.add_node("Empty", add_to_scene=True)
    rep_mesh_rem = diff_assets(s_old, parse_asset(b_empty.write_glb(tmp_path / "empty.glb")))
    assert any(f.code == "MESH_REMOVED" and f.subject == "HeroMesh" for f in rep_mesh_rem.findings)

    # Ensure terminal text, markdown, and dict formatting work cleanly
    assert "UNIT_SCALE" not in rep.to_terminal_text()
    assert "UV_CHANNEL_DROPPED" in rep.to_markdown()
    assert rep.to_dict()["has_breaking"] is True


def test_asset_optimized_info_and_policy_gates(tmp_path: Path) -> None:
    s_old = parse_asset(_make_base_hero(tmp_path / "v1.glb", tri_count=20))
    s_opt = parse_asset(_make_base_hero(tmp_path / "v2_opt.glb", tri_count=10))
    rep_opt = diff_assets(s_old, s_opt)
    assert not rep_opt.has_breaking
    assert any(f.code == "ASSET_OPTIMIZED" and f.severity == "INFO" for f in rep_opt.findings)
    assert evaluate_policy(rep_opt, PolicyConfig()) == []

    # Triangle spike & size budget checks
    s_heavy = parse_asset(_make_base_hero(tmp_path / "v2_heavy.glb", tri_count=30))
    rep_heavy = diff_assets(s_old, s_heavy)
    assert evaluate_policy(rep_heavy, PolicyConfig(fail_on_warning=False)) == []
    assert len(evaluate_policy(rep_heavy, PolicyConfig(fail_on_warning=True))) == 1
    assert len(evaluate_policy(rep_heavy, PolicyConfig(max_tri_increase_pct=25.0))) == 1
    assert len(evaluate_policy(rep_heavy, PolicyConfig(max_triangles_abs=25))) == 1
    assert len(evaluate_policy(rep_heavy, PolicyConfig(max_file_size_increase_pct=1.0))) == 1
    assert len(evaluate_policy(rep_heavy, PolicyConfig(max_file_size_bytes_abs=100))) == 1

    # Breaking policy fail vs allow
    s_uv_drop = parse_asset(_make_base_hero(tmp_path / "v2_nouv.glb", include_uv=False, tri_count=20))
    rep_brk = diff_assets(s_old, s_uv_drop)
    assert len(evaluate_policy(rep_brk, PolicyConfig(fail_on_breaking=True))) == 1
    assert evaluate_policy(rep_brk, PolicyConfig(fail_on_breaking=False)) == []

    # Invalid threshold configs
    with pytest.raises(PolicyConfigError):
        PolicyConfig(max_tri_increase_pct=-1.0)
    with pytest.raises(PolicyConfigError):
        PolicyConfig(max_file_size_increase_pct=float("nan"))
    with pytest.raises(PolicyConfigError):
        PolicyConfig(max_triangles_abs=-5)
    with pytest.raises(PolicyConfigError):
        PolicyConfig(max_file_size_bytes_abs=-1)
