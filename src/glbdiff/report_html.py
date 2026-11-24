from __future__ import annotations

import html
import json
from typing import Any

from glbdiff.models import AssetSummary, DiffReport


def _safe_script_json(data: Any) -> str:
    raw = json.dumps(data, separators=(",", ":"))
    return raw.replace("</", "<\\/").replace("<!--", "<\\!--")


def _fmt_bytes(n: int) -> str:
    if abs(n) < 1024:
        return f"{n} B"
    if abs(n) < 1024 * 1024:
        return f"{n / 1024.0:.1f} KB"
    return f"{n / (1024.0 * 1024.0):.2f} MB"


def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "new"
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.2f}%"


def _build_scene_payload(
    asset: AssetSummary,
    added: set[str],
    removed: set[str],
    modified: set[str],
) -> dict[str, Any]:
    meshes_out: list[dict[str, Any]] = []
    for seg in asset.preview.mesh_segments:
        name = str(seg.get("name", ""))
        if name in removed:
            status = "removed"
        elif name in added:
            status = "added"
        elif name in modified:
            status = "modified"
        else:
            status = "unchanged"
        edges = [
            [
                [round(p0[0], 5), round(p0[1], 5), round(p0[2], 5)],
                [round(p1[0], 5), round(p1[1], 5), round(p1[2], 5)],
            ]
            for p0, p1 in seg.get("edges", [])
        ]
        meshes_out.append({"name": name, "status": status, "edges": edges})

    bones_out = [
        [
            [round(p0[0], 5), round(p0[1], 5), round(p0[2], 5)],
            [round(p1[0], 5), round(p1[1], 5), round(p1[2], 5)],
        ]
        for p0, p1 in asset.preview.bone_segments
    ]

    aabb_out = None
    if asset.preview.world_aabb is not None:
        aabb_out = {
            "min": [round(v, 5) for v in asset.preview.world_aabb.min_pt],
            "max": [round(v, 5) for v in asset.preview.world_aabb.max_pt],
        }

    return {
        "path": asset.path,
        "meshes": meshes_out,
        "bones": bones_out,
        "world_aabb": aabb_out,
    }


def generate_html_diff(report: DiffReport) -> str:
    added_meshes = set(report.added_meshes)
    removed_meshes = set(report.removed_meshes)
    modified_meshes = set(report.modified_meshes)

    before_scene = _build_scene_payload(
        report.old_asset, set(), removed_meshes, modified_meshes
    )
    after_scene = _build_scene_payload(
        report.new_asset, added_meshes, set(), modified_meshes
    )

    breaking_cnt = sum(1 for f in report.findings if f.severity == "BREAKING")
    warn_cnt = sum(1 for f in report.findings if f.severity == "WARNING")
    info_cnt = sum(1 for f in report.findings if f.severity == "INFO")

    size_m = report.metrics["file_size_bytes"]
    tri_m = report.metrics["total_triangles"]
    vert_m = report.metrics["total_vertices"]
    tex_m = report.metrics["total_texture_bytes"]
    scale_str = f"{report.scale_ratio:.4f}x" if report.scale_ratio is not None else "n/a"

    old_path_esc = html.escape(report.old_asset.path, quote=True)
    new_path_esc = html.escape(report.new_asset.path, quote=True)

    findings_rows = []
    for f in report.findings:
        sev_cls = f.severity.lower()
        findings_rows.append(
            "<tr>"
            f"<td><span class='badge badge-{sev_cls}'>{html.escape(f.severity)}</span></td>"
            f"<td><code>{html.escape(f.code)}</code></td>"
            f"<td><code>{html.escape(f.subject)}</code></td>"
            f"<td>{html.escape(f.message)}</td>"
            "</tr>"
        )
    if not findings_rows:
        findings_rows.append(
            "<tr><td colspan='4' class='muted'>No diagnostic findings detected.</td></tr>"
        )

    def _diff_rows(
        old_names: list[str],
        new_names: list[str],
        added: list[str],
        removed: list[str],
        modified: list[str],
        kind: str,
    ) -> str:
        all_names = sorted(set(old_names) | set(new_names))
        if not all_names:
            return f"<tr><td colspan='3' class='muted'>No {html.escape(kind)} present.</td></tr>"
        added_s, removed_s, modified_s = set(added), set(removed), set(modified)
        rows: list[str] = []
        for name in all_names:
            if name in added_s:
                status = "<span class='badge badge-added'>ADDED</span>"
            elif name in removed_s:
                status = "<span class='badge badge-breaking'>REMOVED</span>"
            elif name in modified_s:
                status = "<span class='badge badge-warning'>MODIFIED</span>"
            else:
                status = "<span class='badge badge-info'>UNCHANGED</span>"
            rows.append(
                f"<tr><td>{html.escape(kind)}</td><td><code>{html.escape(name)}</code></td>"
                f"<td>{status}</td></tr>"
            )
        return "\n".join(rows)

    structure_rows = "\n".join(
        [
            _diff_rows(
                [m.name for m in report.old_asset.meshes],
                [m.name for m in report.new_asset.meshes],
                report.added_meshes,
                report.removed_meshes,
                report.modified_meshes,
                "Mesh",
            ),
            _diff_rows(
                [m.name for m in report.old_asset.materials],
                [m.name for m in report.new_asset.materials],
                report.added_materials,
                report.removed_materials,
                report.modified_materials,
                "Material",
            ),
            _diff_rows(
                [a.name for a in report.old_asset.animations],
                [a.name for a in report.new_asset.animations],
                report.added_animations,
                report.removed_animations,
                report.modified_animations,
                "Animation",
            ),
        ]
    )

    before_json = _safe_script_json(before_scene)
    after_json = _safe_script_json(after_scene)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>glbdiff 3D Inspector — {old_path_esc} vs {new_path_esc}</title>
<style>
  :root {{
    --bg: #090d16;
    --panel: #111827;
    --border: #1e293b;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --added: #34d399;
    --warn: #fbbf24;
    --brk: #fb7185;
    --bone: #f472b6;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 24px;
    background: var(--bg);
    color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    line-height: 1.5;
  }}
  .container {{ max-width: 1360px; margin: 0 auto; }}
  header {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }}
  h1 {{ margin: 0; font-size: 1.35rem; letter-spacing: -0.02em; }}
  .paths {{ color: var(--muted); font-size: 0.9rem; font-family: ui-monospace, monospace; }}
  .badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .badge {{
    display: inline-block;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
  }}
  .badge-breaking {{ background: rgba(251, 113, 133, 0.16); color: var(--brk); border: 1px solid rgba(251, 113, 133, 0.4); }}
  .badge-warning {{ background: rgba(251, 191, 36, 0.16); color: var(--warn); border: 1px solid rgba(251, 191, 36, 0.4); }}
  .badge-info {{ background: rgba(56, 189, 248, 0.14); color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.35); }}
  .badge-added {{ background: rgba(52, 211, 153, 0.16); color: var(--added); border: 1px solid rgba(52, 211, 153, 0.4); }}
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
  }}
  .card-label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }}
  .card-value {{ font-size: 1.2rem; font-weight: 700; margin-top: 4px; }}
  .card-sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 2px; font-family: ui-monospace, monospace; }}
  .viewer-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 10px;
  }}
  .controls {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  button {{
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 0.85rem;
    cursor: pointer;
  }}
  button:hover {{ border-color: var(--accent); color: var(--accent); }}
  .legend {{ display: flex; gap: 12px; flex-wrap: wrap; font-size: 0.8rem; color: var(--muted); }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }}
  .canvases {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 24px;
  }}
  @media (max-width: 860px) {{ .canvases {{ grid-template-columns: 1fr; }} }}
  .viewport {{
    background: #060911;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }}
  .viewport-head {{
    padding: 8px 12px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
  }}
  canvas {{
    display: block;
    width: 100%;
    height: 380px;
    cursor: grab;
    touch-action: none;
  }}
  canvas:active {{ cursor: grabbing; }}
  .section {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 20px;
  }}
  h2 {{ margin: 0 0 12px 0; font-size: 1.05rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  code {{ font-family: ui-monospace, monospace; font-size: 0.84rem; }}
  .muted {{ color: var(--muted); }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>glbdiff Synchronized 3D Diff Inspector</h1>
      <div class="paths">{old_path_esc} &rarr; {new_path_esc}</div>
    </div>
    <div class="badges">
      <span class="badge badge-breaking">{breaking_cnt} BREAKING</span>
      <span class="badge badge-warning">{warn_cnt} WARNING</span>
      <span class="badge badge-info">{info_cnt} INFO</span>
    </div>
  </header>

  <section class="cards">
    <div class="card">
      <div class="card-label">File Size</div>
      <div class="card-value">{_fmt_bytes(int(size_m.new_val))}</div>
      <div class="card-sub">{_fmt_bytes(int(size_m.old_val))} &rarr; {_fmt_pct(size_m.pct_change)}</div>
    </div>
    <div class="card">
      <div class="card-label">Triangles</div>
      <div class="card-value">{tri_m.new_val}</div>
      <div class="card-sub">{tri_m.old_val} &rarr; {_fmt_pct(tri_m.pct_change)}</div>
    </div>
    <div class="card">
      <div class="card-label">Vertices</div>
      <div class="card-value">{vert_m.new_val}</div>
      <div class="card-sub">{vert_m.old_val} &rarr; {_fmt_pct(vert_m.pct_change)}</div>
    </div>
    <div class="card">
      <div class="card-label">Texture Payload</div>
      <div class="card-value">{_fmt_bytes(int(tex_m.new_val))}</div>
      <div class="card-sub">{_fmt_bytes(int(tex_m.old_val))} &rarr; {_fmt_pct(tex_m.pct_change)}</div>
    </div>
    <div class="card">
      <div class="card-label">World Scale</div>
      <div class="card-value">{scale_str}</div>
      <div class="card-sub">Bounding diagonal ratio</div>
    </div>
  </section>

  <div class="viewer-bar">
    <div class="controls">
      <button id="btn-reset" type="button">Reset Camera</button>
      <button id="btn-toggle" type="button">Toggle Bones / AABB</button>
    </div>
    <div class="legend">
      <span><i class="dot" style="background:#38bdf8"></i>Unchanged</span>
      <span><i class="dot" style="background:#34d399"></i>Added</span>
      <span><i class="dot" style="background:#fbbf24"></i>Modified</span>
      <span><i class="dot" style="background:#fb7185"></i>Removed</span>
      <span><i class="dot" style="background:#f472b6"></i>Bones</span>
      <span><i class="dot" style="background:#64748b"></i>World AABB</span>
    </div>
  </div>

  <section class="canvases">
    <div class="viewport">
      <div class="viewport-head"><span>Before</span><span class="muted">{old_path_esc}</span></div>
      <canvas id="canvas-before"></canvas>
    </div>
    <div class="viewport">
      <div class="viewport-head"><span>After</span><span class="muted">{new_path_esc}</span></div>
      <canvas id="canvas-after"></canvas>
    </div>
  </section>

  <section class="section">
    <h2>Diagnostics</h2>
    <table>
      <thead><tr><th>Severity</th><th>Code</th><th>Subject</th><th>Message</th></tr></thead>
      <tbody>
        {"".join(findings_rows)}
      </tbody>
    </table>
  </section>

  <section class="section">
    <h2>Scene Structure Diff (Meshes, Materials, Animations)</h2>
    <table>
      <thead><tr><th>Kind</th><th>Name</th><th>Status</th></tr></thead>
      <tbody>
        {structure_rows}
      </tbody>
    </table>
  </section>
</div>

<script>
const BEFORE_SCENE = {before_json};
const AFTER_SCENE = {after_json};

(function() {{
  const COLORS = {{
    unchanged: "#38bdf8",
    added: "#34d399",
    modified: "#fbbf24",
    removed: "#fb7185",
    bone: "#f472b6",
    aabb: "#64748b"
  }};

  const cam = {{ yaw: 0.65, pitch: 0.35, zoom: 1.0 }};
  let showHelpers = true;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  function aabbEdges(box) {{
    if (!box) return [];
    const [x0, y0, z0] = box.min;
    const [x1, y1, z1] = box.max;
    const c = [
      [x0,y0,z0], [x1,y0,z0], [x1,y1,z0], [x0,y1,z0],
      [x0,y0,z1], [x1,y0,z1], [x1,y1,z1], [x0,y1,z1]
    ];
    const idx = [
      [0,1],[1,2],[2,3],[3,0],
      [4,5],[5,6],[6,7],[7,4],
      [0,4],[1,5],[2,6],[3,7]
    ];
    return idx.map(([a, b]) => [c[a], c[b]]);
  }}

  function computeSharedBounds(scenes) {{
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    function expand(p) {{
      if (p[0] < minX) minX = p[0];
      if (p[1] < minY) minY = p[1];
      if (p[2] < minZ) minZ = p[2];
      if (p[0] > maxX) maxX = p[0];
      if (p[1] > maxY) maxY = p[1];
      if (p[2] > maxZ) maxZ = p[2];
    }}
    for (const sc of scenes) {{
      if (sc.world_aabb) {{ expand(sc.world_aabb.min); expand(sc.world_aabb.max); }}
      for (const m of sc.meshes) for (const e of m.edges) {{ expand(e[0]); expand(e[1]); }}
      for (const b of sc.bones) {{ expand(b[0]); expand(b[1]); }}
    }}
    if (!isFinite(minX)) return {{ center: [0, 0, 0], radius: 1.0 }};
    const cx = (minX + maxX) * 0.5;
    const cy = (minY + maxY) * 0.5;
    const cz = (minZ + maxZ) * 0.5;
    const dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
    const radius = Math.max(Math.sqrt(dx*dx + dy*dy + dz*dz) * 0.5, 1e-4);
    return {{ center: [cx, cy, cz], radius }};
  }}

  const bounds = computeSharedBounds([BEFORE_SCENE, AFTER_SCENE]);

  function project(pt, w, h) {{
    const cx = pt[0] - bounds.center[0];
    const cy = pt[1] - bounds.center[1];
    const cz = pt[2] - bounds.center[2];

    const cyaw = Math.cos(cam.yaw), syaw = Math.sin(cam.yaw);
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);

    const x1 = cx * cyaw - cz * syaw;
    const z1 = cx * syaw + cz * cyaw;
    const y2 = cy * cp - z1 * sp;
    const z2 = cy * sp + z1 * cp;

    const dist = bounds.radius * 3.0 / Math.max(cam.zoom, 0.15);
    const denom = Math.max(dist - z2, bounds.radius * 0.25);
    const scale = (Math.min(w, h) * 0.85 * cam.zoom) / (bounds.radius * 2.2);
    const persp = dist / denom;
    return [w * 0.5 + x1 * scale * persp, h * 0.5 - y2 * scale * persp];
  }}

  function drawScene(canvas, scene) {{
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(rect.width, 100);
    const h = Math.max(rect.height, 100);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {{
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }}
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#060911";
    ctx.fillRect(0, 0, w, h);

    if (showHelpers && scene.world_aabb) {{
      ctx.strokeStyle = COLORS.aabb;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (const [p0, p1] of aabbEdges(scene.world_aabb)) {{
        const a = project(p0, w, h);
        const b = project(p1, w, h);
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }}
      ctx.stroke();
    }}

    for (const mesh of scene.meshes) {{
      ctx.strokeStyle = COLORS[mesh.status] || COLORS.unchanged;
      ctx.lineWidth = 1.15;
      ctx.beginPath();
      for (const [p0, p1] of mesh.edges) {{
        const a = project(p0, w, h);
        const b = project(p1, w, h);
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }}
      ctx.stroke();
    }}

    if (showHelpers && scene.bones.length > 0) {{
      ctx.strokeStyle = COLORS.bone;
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      for (const [p0, p1] of scene.bones) {{
        const a = project(p0, w, h);
        const b = project(p1, w, h);
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }}
      ctx.stroke();
    }}
  }}

  const cBefore = document.getElementById("canvas-before");
  const cAfter = document.getElementById("canvas-after");

  function renderAll() {{
    drawScene(cBefore, BEFORE_SCENE);
    drawScene(cAfter, AFTER_SCENE);
  }}

  for (const c of [cBefore, cAfter]) {{
    c.addEventListener("pointerdown", (ev) => {{
      dragging = true;
      lastX = ev.clientX;
      lastY = ev.clientY;
      c.setPointerCapture(ev.pointerId);
    }});
    c.addEventListener("pointermove", (ev) => {{
      if (!dragging) return;
      const dx = ev.clientX - lastX;
      const dy = ev.clientY - lastY;
      lastX = ev.clientX;
      lastY = ev.clientY;
      cam.yaw += dx * 0.01;
      cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + dy * 0.01));
      renderAll();
    }});
    c.addEventListener("pointerup", () => {{ dragging = false; }});
    c.addEventListener("pointercancel", () => {{ dragging = false; }});
    c.addEventListener("wheel", (ev) => {{
      ev.preventDefault();
      const factor = ev.deltaY < 0 ? 1.1 : 0.9;
      cam.zoom = Math.max(0.2, Math.min(8.0, cam.zoom * factor));
      renderAll();
    }}, {{ passive: false }});
  }}

  document.getElementById("btn-reset").addEventListener("click", () => {{
    cam.yaw = 0.65;
    cam.pitch = 0.35;
    cam.zoom = 1.0;
    renderAll();
  }});
  document.getElementById("btn-toggle").addEventListener("click", () => {{
    showHelpers = !showHelpers;
    renderAll();
  }});
  window.addEventListener("resize", renderAll);
  renderAll();
}})();
</script>
</body>
</html>
"""
