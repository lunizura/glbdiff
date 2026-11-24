from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from glbdiff.cli import main
from tests.glb_builder import GltfSceneBuilder, make_png_bytes


def _make_cli_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    b1 = GltfSceneBuilder()
    t1 = b1.add_embedded_texture("Tex", make_png_bytes(64, 64))
    m1 = b1.add_material("Mat", alpha_mode="OPAQUE", base_color_tex=t1)
    mesh1 = b1.add_mesh(
        "Body",
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [0, 1, 2],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        material_idx=m1,
    )
    b1.add_node("Root", mesh_idx=mesh1, add_to_scene=True)
    p1 = b1.write_glb(tmp_path / "v1.glb")

    # Safe + warning (alphaMode BLEND, +100% tris)
    b2 = GltfSceneBuilder()
    t2 = b2.add_embedded_texture("Tex", make_png_bytes(64, 64))
    m2 = b2.add_material("Mat", alpha_mode="BLEND", base_color_tex=t2)
    mesh2 = b2.add_mesh(
        "Body",
        [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        ],
        [0, 1, 2, 3, 4, 5],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)] * 2,
        material_idx=m2,
    )
    b2.add_node("Root", mesh_idx=mesh2, add_to_scene=True)
    p2_warn = b2.write_glb(tmp_path / "v2_warn.glb")

    # Breaking (UV dropped)
    b3 = GltfSceneBuilder()
    m3 = b3.add_material("Mat", alpha_mode="OPAQUE")
    mesh3 = b3.add_mesh(
        "Body",
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [0, 1, 2],
        uvs=None,
        material_idx=m3,
    )
    b3.add_node("Root", mesh_idx=mesh3, add_to_scene=True)
    p3_brk = b3.write_glb(tmp_path / "v3_brk.glb")

    return p1, p2_warn, p3_brk


def test_cli_diff_formats_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p1, _, p3_brk = _make_cli_pair(tmp_path)

    # diff returns 0 even with breaking changes
    rc = main(["diff", str(p1), str(p3_brk), "--format", "json"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["has_breaking"] is True

    md_out = tmp_path / "diff.md"
    assert main(["diff", str(p1), str(p3_brk), "--format", "markdown", "-o", str(md_out)]) == 0
    assert "UV_CHANNEL_DROPPED" in md_out.read_text(encoding="utf-8")

    # Missing file -> exit 2
    assert main(["diff", str(p1), str(tmp_path / "missing.glb")]) == 2


def test_cli_check_exit_codes_0_1_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p1, p2_warn, p3_brk = _make_cli_pair(tmp_path)

    # Identical pass -> 0
    assert main(["check", str(p1), str(p1)]) == 0

    # Warning without --fail-on-warning -> 0; with --fail-on-warning -> 1
    assert main(["check", str(p1), str(p2_warn)]) == 0
    assert main(["check", str(p1), str(p2_warn), "--fail-on-warning"]) == 1
    assert main(["check", str(p1), str(p2_warn), "--max-tri-pct", "50"]) == 1
    capsys.readouterr()
    assert main(["check", str(p1), str(p2_warn), "--max-tris", "1", "--json"]) == 1
    json_res = json.loads(capsys.readouterr().out)
    assert json_res["passed"] is False
    assert len(json_res["violations"]) == 1

    # Breaking -> 1; with --allow-breaking -> 0
    assert main(["check", str(p1), str(p3_brk)]) == 1
    assert main(["check", str(p1), str(p3_brk), "--allow-breaking"]) == 0

    # Invalid policy threshold or missing file -> 2
    assert main(["check", str(p1), str(p2_warn), "--max-tri-pct", "-10"]) == 2
    assert main(["check", str(p1), str(tmp_path / "nope.glb")]) == 2


def test_cli_textconv_html_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    p1, p2_warn, p3_brk = _make_cli_pair(tmp_path)

    assert main(["textconv", str(p1)]) == 0
    tc = capsys.readouterr().out
    assert "triangles: 1" in tc
    assert "Body" in tc
    assert main(["textconv", str(tmp_path / "missing.glb")]) == 2

    html_file = tmp_path / "out" / "report.html"
    assert main(["html", str(p1), str(p2_warn), "-o", str(html_file)]) == 0
    html_text = html_file.read_text(encoding="utf-8")
    assert "canvas-before" in html_text and "canvas-after" in html_text

    sum_file = tmp_path / "out" / "summary.md"
    assert (
        main(["summary", str(p1), str(p3_brk), "--provider", "heuristic", "-o", str(sum_file)])
        == 0
    )
    assert "REQUEST CHANGES" in sum_file.read_text(encoding="utf-8")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert main(["summary", str(p1), str(p3_brk), "--provider", "gemini"]) == 2


def test_cli_git_setup_print_only_and_repo_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["git-setup", "--print-only"]) == 0
    printed = capsys.readouterr().out
    assert "*.glb diff=glb" in printed
    assert 'git config diff.glb.textconv "glbdiff textconv"' in printed

    # Outside a git repo -> exit 2
    non_repo = tmp_path / "non_repo"
    non_repo.mkdir()
    monkeypatch.chdir(non_repo)
    assert main(["git-setup"]) == 2

    # Inside an initialized git repo -> writes .gitattributes + git config
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True)
    monkeypatch.chdir(repo)
    assert main(["git-setup"]) == 0
    assert main(["git-setup"]) == 0  # idempotent

    attrs = (repo / ".gitattributes").read_text(encoding="utf-8")
    assert attrs.count("*.glb diff=glb") == 1
    assert attrs.count("*.gltf diff=glb") == 1
    cfg = subprocess.run(
        ["git", "config", "--get", "diff.glb.textconv"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert cfg.stdout.strip() == "glbdiff textconv"
