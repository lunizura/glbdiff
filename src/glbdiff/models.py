from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["BREAKING", "WARNING", "INFO"]
Vec3 = tuple[float, float, float]


@dataclass
class AABB:
    min_pt: Vec3
    max_pt: Vec3

    @property
    def size(self) -> Vec3:
        return (
            self.max_pt[0] - self.min_pt[0],
            self.max_pt[1] - self.min_pt[1],
            self.max_pt[2] - self.min_pt[2],
        )

    @property
    def diagonal(self) -> float:
        sx, sy, sz = self.size
        return math.sqrt(sx * sx + sy * sy + sz * sz)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_pt": [self.min_pt[0], self.min_pt[1], self.min_pt[2]],
            "max_pt": [self.max_pt[0], self.max_pt[1], self.max_pt[2]],
            "size": [self.size[0], self.size[1], self.size[2]],
            "diagonal": round(self.diagonal, 6),
        }


@dataclass
class MeshSummary:
    name: str
    primitive_count: int
    vertex_count: int
    triangle_count: int
    attributes: list[str]
    morph_target_count: int
    material_names: list[str]
    is_draco_compressed: bool = False
    local_aabb: AABB | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "primitive_count": self.primitive_count,
            "vertex_count": self.vertex_count,
            "triangle_count": self.triangle_count,
            "attributes": list(self.attributes),
            "morph_target_count": self.morph_target_count,
            "material_names": list(self.material_names),
            "is_draco_compressed": self.is_draco_compressed,
            "local_aabb": self.local_aabb.to_dict() if self.local_aabb is not None else None,
        }


@dataclass
class MaterialSummary:
    name: str
    alpha_mode: str
    double_sided: bool
    metallic_factor: float
    roughness_factor: float
    base_color: tuple[float, float, float, float]
    texture_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alpha_mode": self.alpha_mode,
            "double_sided": self.double_sided,
            "metallic_factor": self.metallic_factor,
            "roughness_factor": self.roughness_factor,
            "base_color": [
                self.base_color[0],
                self.base_color[1],
                self.base_color[2],
                self.base_color[3],
            ],
            "texture_count": self.texture_count,
        }


@dataclass
class TextureSummary:
    index: int
    name: str
    mime_type: str
    byte_length: int
    width: int | None
    height: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "mime_type": self.mime_type,
            "byte_length": self.byte_length,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class JointInfo:
    node_index: int
    name: str
    parent_name: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_index": self.node_index,
            "name": self.name,
            "parent_name": self.parent_name,
        }


@dataclass
class SkinSummary:
    name: str
    joint_count: int
    joints: list[JointInfo]
    has_inverse_bind_matrices: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "joint_count": self.joint_count,
            "joints": [j.to_dict() for j in self.joints],
            "has_inverse_bind_matrices": self.has_inverse_bind_matrices,
        }


@dataclass
class AnimationSummary:
    name: str
    channel_count: int
    sampler_count: int
    duration_sec: float
    target_paths: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "channel_count": self.channel_count,
            "sampler_count": self.sampler_count,
            "duration_sec": round(self.duration_sec, 6),
            "target_paths": list(self.target_paths),
        }


@dataclass
class WireframePreview:
    mesh_segments: list[dict[str, Any]]
    bone_segments: list[tuple[Vec3, Vec3]]
    world_aabb: AABB | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mesh_segments": [
                {
                    "name": seg["name"],
                    "edges": [
                        [[p0[0], p0[1], p0[2]], [p1[0], p1[1], p1[2]]]
                        for p0, p1 in seg.get("edges", [])
                    ],
                }
                for seg in self.mesh_segments
            ],
            "bone_segments": [
                [[p0[0], p0[1], p0[2]], [p1[0], p1[1], p1[2]]]
                for p0, p1 in self.bone_segments
            ],
            "world_aabb": self.world_aabb.to_dict() if self.world_aabb is not None else None,
        }


@dataclass
class AssetSummary:
    path: str
    container_format: Literal["glb", "gltf"]
    file_size_bytes: int
    generator: str
    extensions_used: list[str]
    node_count: int
    mesh_count: int
    total_primitives: int
    total_vertices: int
    total_triangles: int
    total_texture_bytes: int
    world_aabb: AABB | None
    meshes: list[MeshSummary]
    materials: list[MaterialSummary]
    textures: list[TextureSummary]
    skins: list[SkinSummary]
    animations: list[AnimationSummary]
    preview: WireframePreview = field(default_factory=lambda: WireframePreview([], [], None))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "container_format": self.container_format,
            "file_size_bytes": self.file_size_bytes,
            "generator": self.generator,
            "extensions_used": list(self.extensions_used),
            "node_count": self.node_count,
            "mesh_count": self.mesh_count,
            "total_primitives": self.total_primitives,
            "total_vertices": self.total_vertices,
            "total_triangles": self.total_triangles,
            "total_texture_bytes": self.total_texture_bytes,
            "world_aabb": self.world_aabb.to_dict() if self.world_aabb is not None else None,
            "meshes": [m.to_dict() for m in self.meshes],
            "materials": [m.to_dict() for m in self.materials],
            "textures": [t.to_dict() for t in self.textures],
            "skins": [s.to_dict() for s in self.skins],
            "animations": [a.to_dict() for a in self.animations],
        }

    def to_textconv(self) -> str:
        lines: list[str] = [
            f"format: {self.container_format}",
            f"file_size_bytes: {self.file_size_bytes}",
            f"generator: {self.generator}",
            f"extensions_used: {', '.join(self.extensions_used) if self.extensions_used else '(none)'}",
            f"nodes: {self.node_count}",
            f"meshes: {self.mesh_count}",
            f"primitives: {self.total_primitives}",
            f"vertices: {self.total_vertices}",
            f"triangles: {self.total_triangles}",
            f"texture_bytes: {self.total_texture_bytes}",
        ]
        if self.world_aabb is not None:
            mn = ", ".join(f"{v:.4f}" for v in self.world_aabb.min_pt)
            mx = ", ".join(f"{v:.4f}" for v in self.world_aabb.max_pt)
            sz = ", ".join(f"{v:.4f}" for v in self.world_aabb.size)
            lines.append(
                f"world_aabb: min=({mn}) max=({mx}) size=({sz}) diag={self.world_aabb.diagonal:.4f}"
            )
        else:
            lines.append("world_aabb: (none)")

        lines.append(f"[meshes ({len(self.meshes)})]")
        for m in sorted(self.meshes, key=lambda item: item.name):
            attrs = ",".join(m.attributes) if m.attributes else "(none)"
            mats = ",".join(m.material_names) if m.material_names else "(default)"
            lines.append(
                f"  - {m.name}: prims={m.primitive_count} verts={m.vertex_count} "
                f"tris={m.triangle_count} attrs=[{attrs}] morphs={m.morph_target_count} "
                f"mats=[{mats}] draco={m.is_draco_compressed}"
            )

        lines.append(f"[materials ({len(self.materials)})]")
        for mat in sorted(self.materials, key=lambda item: item.name):
            bc = ",".join(f"{c:.3f}" for c in mat.base_color)
            lines.append(
                f"  - {mat.name}: alpha={mat.alpha_mode} double_sided={mat.double_sided} "
                f"metallic={mat.metallic_factor:.3f} roughness={mat.roughness_factor:.3f} "
                f"base_color=({bc}) textures={mat.texture_count}"
            )

        lines.append(f"[textures ({len(self.textures)})]")
        for tex in sorted(self.textures, key=lambda item: item.index):
            dims = f"{tex.width}x{tex.height}" if tex.width and tex.height else "unknown"
            lines.append(
                f"  - #{tex.index} {tex.name}: mime={tex.mime_type} dims={dims} bytes={tex.byte_length}"
            )

        lines.append(f"[skins ({len(self.skins)})]")
        for skin in sorted(self.skins, key=lambda item: item.name):
            lines.append(
                f"  - {skin.name}: joints={skin.joint_count} "
                f"ibm={skin.has_inverse_bind_matrices}"
            )
            for j in sorted(skin.joints, key=lambda item: item.name):
                parent = j.parent_name if j.parent_name is not None else "<root>"
                lines.append(f"      joint {j.name} <- {parent}")

        lines.append(f"[animations ({len(self.animations)})]")
        for anim in sorted(self.animations, key=lambda item: item.name):
            paths = ",".join(anim.target_paths) if anim.target_paths else "(none)"
            lines.append(
                f"  - {anim.name}: duration={anim.duration_sec:.4f}s "
                f"channels={anim.channel_count} samplers={anim.sampler_count} paths=[{paths}]"
            )

        return "\n".join(lines) + "\n"


@dataclass
class DiagnosticFinding:
    code: str
    severity: Severity
    subject: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "subject": self.subject,
            "message": self.message,
        }


@dataclass
class MetricDelta:
    old_val: int | float
    new_val: int | float
    delta: int | float
    pct_change: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_val": self.old_val,
            "new_val": self.new_val,
            "delta": self.delta,
            "pct_change": round(self.pct_change, 4) if self.pct_change is not None else None,
        }


def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "new"
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.2f}%"


def _fmt_delta(delta: int | float) -> str:
    if isinstance(delta, float):
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta:.3f}"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta}"


@dataclass
class DiffReport:
    old_asset: AssetSummary
    new_asset: AssetSummary
    metrics: dict[str, MetricDelta]
    scale_ratio: float | None
    added_meshes: list[str]
    removed_meshes: list[str]
    modified_meshes: list[str]
    added_materials: list[str]
    removed_materials: list[str]
    modified_materials: list[str]
    added_animations: list[str]
    removed_animations: list[str]
    modified_animations: list[str]
    findings: list[DiagnosticFinding]

    @property
    def has_breaking(self) -> bool:
        return any(f.severity == "BREAKING" for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_asset": self.old_asset.to_dict(),
            "new_asset": self.new_asset.to_dict(),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "scale_ratio": round(self.scale_ratio, 6) if self.scale_ratio is not None else None,
            "added_meshes": list(self.added_meshes),
            "removed_meshes": list(self.removed_meshes),
            "modified_meshes": list(self.modified_meshes),
            "added_materials": list(self.added_materials),
            "removed_materials": list(self.removed_materials),
            "modified_materials": list(self.modified_materials),
            "added_animations": list(self.added_animations),
            "removed_animations": list(self.removed_animations),
            "modified_animations": list(self.modified_animations),
            "has_breaking": self.has_breaking,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_terminal_text(self) -> str:
        lines: list[str] = [
            f"glbdiff: {self.old_asset.path} -> {self.new_asset.path}",
            f"  format: {self.old_asset.container_format} -> {self.new_asset.container_format}",
        ]
        if self.scale_ratio is not None:
            lines.append(f"  world_scale_ratio: {self.scale_ratio:.4f}x")
        else:
            lines.append("  world_scale_ratio: n/a")

        lines.append("Metrics:")
        for key, m in self.metrics.items():
            lines.append(
                f"  {key:<22} {m.old_val} -> {m.new_val} "
                f"({_fmt_delta(m.delta)}, {_fmt_pct(m.pct_change)})"
            )

        if self.findings:
            lines.append(f"Diagnostics ({len(self.findings)}):")
            for f in self.findings:
                lines.append(f"  [{f.severity}] {f.code} ({f.subject}): {f.message}")
        else:
            lines.append("Diagnostics (0): none")

        def _append_group(label: str, added: list[str], removed: list[str], modified: list[str]) -> None:
            if added or removed or modified:
                lines.append(f"{label}:")
                if added:
                    lines.append(f"  + added: {', '.join(added)}")
                if removed:
                    lines.append(f"  - removed: {', '.join(removed)}")
                if modified:
                    lines.append(f"  ~ modified: {', '.join(modified)}")

        _append_group("Meshes", self.added_meshes, self.removed_meshes, self.modified_meshes)
        _append_group(
            "Materials", self.added_materials, self.removed_materials, self.modified_materials
        )
        _append_group(
            "Animations", self.added_animations, self.removed_animations, self.modified_animations
        )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        breaking_cnt = sum(1 for f in self.findings if f.severity == "BREAKING")
        warn_cnt = sum(1 for f in self.findings if f.severity == "WARNING")
        info_cnt = sum(1 for f in self.findings if f.severity == "INFO")

        lines: list[str] = [
            "## glbdiff Report",
            "",
            f"- **Old:** `{self.old_asset.path}` ({self.old_asset.container_format})",
            f"- **New:** `{self.new_asset.path}` ({self.new_asset.container_format})",
            f"- **Findings:** {breaking_cnt} BREAKING, {warn_cnt} WARNING, {info_cnt} INFO",
        ]
        if self.scale_ratio is not None:
            lines.append(f"- **World Scale Ratio:** `{self.scale_ratio:.4f}x`")

        lines.extend(
            [
                "",
                "| Metric | Old | New | Delta | Change |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for key, m in self.metrics.items():
            lines.append(
                f"| `{key}` | {m.old_val} | {m.new_val} | {_fmt_delta(m.delta)} | {_fmt_pct(m.pct_change)} |"
            )

        if self.findings:
            lines.extend(["", "### Diagnostics", ""])
            for f in self.findings:
                lines.append(f"- **[{f.severity}] `{f.code}`** (`{f.subject}`): {f.message}")

        sections = [
            ("Meshes", self.added_meshes, self.removed_meshes, self.modified_meshes),
            ("Materials", self.added_materials, self.removed_materials, self.modified_materials),
            ("Animations", self.added_animations, self.removed_animations, self.modified_animations),
        ]
        for title, added, removed, modified in sections:
            if added or removed or modified:
                lines.extend(["", f"### {title}", ""])
                if added:
                    lines.append(f"- Added: {', '.join(f'`{x}`' for x in added)}")
                if removed:
                    lines.append(f"- Removed: {', '.join(f'`{x}`' for x in removed)}")
                if modified:
                    lines.append(f"- Modified: {', '.join(f'`{x}`' for x in modified)}")

        return "\n".join(lines)
