from __future__ import annotations

from glbdiff.models import (
    AABB,
    AssetSummary,
    DiagnosticFinding,
    DiffReport,
    MaterialSummary,
    MeshSummary,
    MetricDelta,
)


def _metric_delta(old_val: int | float, new_val: int | float) -> MetricDelta:
    delta = new_val - old_val
    if old_val == 0:
        pct_change: float | None = 0.0 if new_val == 0 else None
    else:
        pct_change = ((new_val - old_val) / old_val) * 100.0
    return MetricDelta(
        old_val=old_val,
        new_val=new_val,
        delta=delta,
        pct_change=pct_change,
    )


def _aabb_differs(a: AABB | None, b: AABB | None, tol: float = 1e-5) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    for i in range(3):
        if abs(a.min_pt[i] - b.min_pt[i]) > tol or abs(a.max_pt[i] - b.max_pt[i]) > tol:
            return True
    return False


def _mesh_modified(old_m: MeshSummary, new_m: MeshSummary) -> bool:
    return (
        old_m.primitive_count != new_m.primitive_count
        or old_m.vertex_count != new_m.vertex_count
        or old_m.triangle_count != new_m.triangle_count
        or old_m.attributes != new_m.attributes
        or old_m.morph_target_count != new_m.morph_target_count
        or old_m.material_names != new_m.material_names
        or old_m.is_draco_compressed != new_m.is_draco_compressed
        or _aabb_differs(old_m.local_aabb, new_m.local_aabb)
    )


def _material_modified(old_mat: MaterialSummary, new_mat: MaterialSummary) -> bool:
    if (
        old_mat.alpha_mode != new_mat.alpha_mode
        or old_mat.double_sided != new_mat.double_sided
        or old_mat.texture_count != new_mat.texture_count
        or abs(old_mat.metallic_factor - new_mat.metallic_factor) > 1e-5
        or abs(old_mat.roughness_factor - new_mat.roughness_factor) > 1e-5
    ):
        return True
    return any(abs(old_mat.base_color[i] - new_mat.base_color[i]) > 1e-5 for i in range(4))


def diff_assets(old: AssetSummary, new: AssetSummary) -> DiffReport:
    old_joint_count = sum(s.joint_count for s in old.skins)
    new_joint_count = sum(s.joint_count for s in new.skins)

    metrics: dict[str, MetricDelta] = {
        "file_size_bytes": _metric_delta(old.file_size_bytes, new.file_size_bytes),
        "total_vertices": _metric_delta(old.total_vertices, new.total_vertices),
        "total_triangles": _metric_delta(old.total_triangles, new.total_triangles),
        "total_primitives": _metric_delta(old.total_primitives, new.total_primitives),
        "total_texture_bytes": _metric_delta(old.total_texture_bytes, new.total_texture_bytes),
        "node_count": _metric_delta(old.node_count, new.node_count),
        "animation_count": _metric_delta(len(old.animations), len(new.animations)),
        "joint_count": _metric_delta(old_joint_count, new_joint_count),
    }

    scale_ratio: float | None = None
    if (
        old.world_aabb is not None
        and new.world_aabb is not None
        and old.world_aabb.diagonal > 1e-6
        and new.world_aabb.diagonal > 1e-6
    ):
        scale_ratio = new.world_aabb.diagonal / old.world_aabb.diagonal

    old_meshes = {m.name: m for m in old.meshes}
    new_meshes = {m.name: m for m in new.meshes}
    added_meshes = sorted(new_meshes.keys() - old_meshes.keys())
    removed_meshes = sorted(old_meshes.keys() - new_meshes.keys())
    modified_meshes = sorted(
        name for name in (old_meshes.keys() & new_meshes.keys())
        if _mesh_modified(old_meshes[name], new_meshes[name])
    )

    old_mats = {m.name: m for m in old.materials}
    new_mats = {m.name: m for m in new.materials}
    added_materials = sorted(new_mats.keys() - old_mats.keys())
    removed_materials = sorted(old_mats.keys() - new_mats.keys())
    modified_materials = sorted(
        name for name in (old_mats.keys() & new_mats.keys())
        if _material_modified(old_mats[name], new_mats[name])
    )

    old_anims = {a.name: a for a in old.animations}
    new_anims = {a.name: a for a in new.animations}
    added_animations = sorted(new_anims.keys() - old_anims.keys())
    removed_animations = sorted(old_anims.keys() - new_anims.keys())
    modified_animations = sorted(
        name
        for name in (old_anims.keys() & new_anims.keys())
        if (
            old_anims[name].channel_count != new_anims[name].channel_count
            or old_anims[name].sampler_count != new_anims[name].sampler_count
            or old_anims[name].target_paths != new_anims[name].target_paths
            or abs(old_anims[name].duration_sec - new_anims[name].duration_sec) > 1e-4
        )
    )

    findings: list[DiagnosticFinding] = []

    # 1 & 2: Scale checks
    if scale_ratio is not None:
        if (90.0 <= scale_ratio <= 110.0) or (0.009 <= scale_ratio <= 0.011):
            findings.append(
                DiagnosticFinding(
                    code="UNIT_SCALE_100X_SHIFT",
                    severity="BREAKING",
                    subject="world_aabb",
                    message=(
                        f"World bounding diagonal shifted by {scale_ratio:.4f}x "
                        "(likely meters/centimeters export unit mismatch)."
                    ),
                )
            )
        elif scale_ratio > 2.0 or scale_ratio < 0.5:
            findings.append(
                DiagnosticFinding(
                    code="LARGE_SCALE_DRIFT",
                    severity="WARNING",
                    subject="world_aabb",
                    message=f"World bounding diagonal drifted by {scale_ratio:.4f}x.",
                )
            )

    # 3 & 4: Skeleton joints & reparenting
    old_skins = {s.name: s for s in old.skins}
    new_skins = {s.name: s for s in new.skins}
    for s_name in sorted(old_skins.keys()):
        old_skin = old_skins[s_name]
        if s_name not in new_skins:
            findings.append(
                DiagnosticFinding(
                    code="SKELETON_JOINT_REMOVED",
                    severity="BREAKING",
                    subject=s_name,
                    message=f"Entire skin '{s_name}' ({old_skin.joint_count} joints) was removed.",
                )
            )
            continue

        new_skin = new_skins[s_name]
        old_joints = {j.name: j for j in old_skin.joints}
        new_joints = {j.name: j for j in new_skin.joints}
        for j_name in sorted(old_joints.keys() - new_joints.keys()):
            findings.append(
                DiagnosticFinding(
                    code="SKELETON_JOINT_REMOVED",
                    severity="BREAKING",
                    subject=j_name,
                    message=f"Joint '{j_name}' was removed from skin '{s_name}'.",
                )
            )
        for j_name in sorted(old_joints.keys() & new_joints.keys()):
            old_parent = old_joints[j_name].parent_name
            new_parent = new_joints[j_name].parent_name
            if old_parent != new_parent:
                findings.append(
                    DiagnosticFinding(
                        code="SKELETON_JOINT_REPARENTED",
                        severity="BREAKING",
                        subject=j_name,
                        message=(
                            f"Joint '{j_name}' in skin '{s_name}' reparented from "
                            f"'{old_parent}' to '{new_parent}'."
                        ),
                    )
                )

    # 5 & 6: Animations
    for a_name in removed_animations:
        findings.append(
            DiagnosticFinding(
                code="ANIMATION_CLIP_REMOVED",
                severity="BREAKING",
                subject=a_name,
                message=f"Animation clip '{a_name}' was removed.",
            )
        )
    for a_name in sorted(old_anims.keys() & new_anims.keys()):
        old_dur = old_anims[a_name].duration_sec
        new_dur = new_anims[a_name].duration_sec
        if abs(new_dur - old_dur) > 0.05:
            findings.append(
                DiagnosticFinding(
                    code="ANIMATION_DURATION_CHANGED",
                    severity="WARNING",
                    subject=a_name,
                    message=(
                        f"Animation clip '{a_name}' duration changed from "
                        f"{old_dur:.3f}s to {new_dur:.3f}s."
                    ),
                )
            )

    # 7, 8 & 9: Meshes (UVs, morph targets, removals)
    for m_name in sorted(old_meshes.keys() & new_meshes.keys()):
        old_m = old_meshes[m_name]
        new_m = new_meshes[m_name]
        old_uvs = {a for a in old_m.attributes if a.startswith("TEXCOORD_")}
        new_uvs = {a for a in new_m.attributes if a.startswith("TEXCOORD_")}
        for uv in sorted(old_uvs - new_uvs):
            findings.append(
                DiagnosticFinding(
                    code="UV_CHANNEL_DROPPED",
                    severity="BREAKING",
                    subject=m_name,
                    message=f"UV channel '{uv}' was dropped from mesh '{m_name}'.",
                )
            )
        if old_m.morph_target_count > new_m.morph_target_count:
            findings.append(
                DiagnosticFinding(
                    code="MORPH_TARGETS_DROPPED",
                    severity="BREAKING",
                    subject=m_name,
                    message=(
                        f"Mesh '{m_name}' morph targets dropped from "
                        f"{old_m.morph_target_count} to {new_m.morph_target_count}."
                    ),
                )
            )
    for m_name in removed_meshes:
        findings.append(
            DiagnosticFinding(
                code="MESH_REMOVED",
                severity="BREAKING",
                subject=m_name,
                message=f"Mesh '{m_name}' was removed.",
            )
        )

    # 10: Material alpha mode changes
    for mat_name in sorted(old_mats.keys() & new_mats.keys()):
        old_alpha = old_mats[mat_name].alpha_mode
        new_alpha = new_mats[mat_name].alpha_mode
        if old_alpha != new_alpha:
            findings.append(
                DiagnosticFinding(
                    code="MATERIAL_ALPHA_MODE_CHANGED",
                    severity="WARNING",
                    subject=mat_name,
                    message=(
                        f"Material '{mat_name}' alphaMode changed from "
                        f"'{old_alpha}' to '{new_alpha}'."
                    ),
                )
            )

    # 11: Triangle spike
    tri_pct = metrics["total_triangles"].pct_change
    if old.total_triangles > 0 and tri_pct is not None and tri_pct > 20.0:
        findings.append(
            DiagnosticFinding(
                code="TRIANGLE_SPIKE",
                severity="WARNING",
                subject="total_triangles",
                message=(
                    f"Triangle count increased by {tri_pct:.2f}% "
                    f"({old.total_triangles} -> {new.total_triangles})."
                ),
            )
        )

    # 12: Texture memory spike
    tex_pct = metrics["total_texture_bytes"].pct_change
    if old.total_texture_bytes > 0 and tex_pct is not None and tex_pct > 25.0:
        findings.append(
            DiagnosticFinding(
                code="TEXTURE_MEMORY_SPIKE",
                severity="WARNING",
                subject="total_texture_bytes",
                message=(
                    f"Texture payload increased by {tex_pct:.2f}% "
                    f"({old.total_texture_bytes} -> {new.total_texture_bytes} bytes)."
                ),
            )
        )

    # 13: Asset optimized
    size_pct = metrics["file_size_bytes"].pct_change
    has_breaking = any(f.severity == "BREAKING" for f in findings)
    if not has_breaking and (
        (tri_pct is not None and tri_pct <= -5.0)
        or (size_pct is not None and size_pct <= -5.0)
    ):
        parts: list[str] = []
        if tri_pct is not None and tri_pct <= -5.0:
            parts.append(f"triangles {tri_pct:.2f}%")
        if size_pct is not None and size_pct <= -5.0:
            parts.append(f"file size {size_pct:.2f}%")
        findings.append(
            DiagnosticFinding(
                code="ASSET_OPTIMIZED",
                severity="INFO",
                subject="asset",
                message=f"Asset optimized with zero breaking changes ({', '.join(parts)}).",
            )
        )

    return DiffReport(
        old_asset=old,
        new_asset=new,
        metrics=metrics,
        scale_ratio=scale_ratio,
        added_meshes=added_meshes,
        removed_meshes=removed_meshes,
        modified_meshes=modified_meshes,
        added_materials=added_materials,
        removed_materials=removed_materials,
        modified_materials=modified_materials,
        added_animations=added_animations,
        removed_animations=removed_animations,
        modified_animations=modified_animations,
        findings=findings,
    )
