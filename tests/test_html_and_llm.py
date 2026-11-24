from __future__ import annotations

import io
import json
import re
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from glbdiff.diff import diff_assets
from glbdiff.errors import LLMClientError
from glbdiff.llm_client import (
    GEMINI_API_REVISION,
    GEMINI_INTERACTIONS_URL,
    ScriptedMockLLM,
    generate_pr_summary,
)
from glbdiff.parser import parse_asset
from glbdiff.report_html import generate_html_diff
from tests.glb_builder import GltfSceneBuilder


class _FakeHttpResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._raw = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def _make_sample_pair(tmp_path: Path, xss_name: bool = False) -> tuple[Any, Any]:
    mesh_name = "</script><script>alert('xss')</script>" if xss_name else "HeroBody"

    b1 = GltfSceneBuilder()
    m1 = b1.add_material("MatA", alpha_mode="OPAQUE")
    mesh1 = b1.add_mesh(
        mesh_name,
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        indices=[0, 1, 2],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        material_idx=m1,
    )
    j1 = b1.add_node("Spine", translation=(0.0, 1.0, 0.0))
    j0 = b1.add_node("Root", children=[j1])
    s1 = b1.add_skin("Rig", [j0, j1])
    b1.scene_nodes.append(j0)
    b1.add_node("Hero", mesh_idx=mesh1, skin_idx=s1, add_to_scene=True)
    b1.add_animation("Idle", j0, 1.0)

    b2 = GltfSceneBuilder()
    m2 = b2.add_material("MatA", alpha_mode="BLEND")
    mesh2 = b2.add_mesh(
        mesh_name,
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        indices=[0, 1, 2],
        uvs=None,  # UV_CHANNEL_DROPPED -> BREAKING
        material_idx=m2,
    )
    mesh_sword = b2.add_mesh(
        "Sword",
        positions=[(1.0, 0.0, 0.0), (1.5, 0.0, 0.0), (1.0, 1.5, 0.0)],
        indices=[0, 1, 2],
    )
    j1_b = b2.add_node("Spine", translation=(0.0, 1.0, 0.0))
    j0_b = b2.add_node("Root", children=[j1_b])
    s2 = b2.add_skin("Rig", [j0_b, j1_b])
    b2.scene_nodes.append(j0_b)
    b2.add_node("Hero", mesh_idx=mesh2, skin_idx=s2, add_to_scene=True)
    b2.add_node("SwordNode", mesh_idx=mesh_sword, add_to_scene=True)
    b2.add_animation("Idle", j0_b, 1.4)

    old_s = parse_asset(b1.write_glb(tmp_path / "v1.glb"))
    new_s = parse_asset(b2.write_glb(tmp_path / "v2.glb"))
    return old_s, new_s


def test_html_report_escaping_zero_cdn_and_wireframe_json(tmp_path: Path) -> None:
    old_s, new_s = _make_sample_pair(tmp_path, xss_name=True)
    report = diff_assets(old_s, new_s)
    html_doc = generate_html_diff(report)

    # Zero raw XSS payload and zero external CDNs
    assert "<script>alert('xss')</script>" not in html_doc
    assert "&lt;/script&gt;&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html_doc
    assert "<\\/script><script>" in html_doc  # escaped </ inside embedded JSON
    assert "unpkg" not in html_doc
    assert "jsdelivr" not in html_doc
    assert "cdnjs" not in html_doc

    # Dual canvases & controls present
    assert 'id="canvas-before"' in html_doc
    assert 'id="canvas-after"' in html_doc
    assert "Reset Camera" in html_doc
    assert "Toggle Bones / AABB" in html_doc

    # Parse embedded BEFORE_SCENE & AFTER_SCENE JSON
    m_before = re.search(r"const BEFORE_SCENE = (\{.*?\});\n", html_doc)
    m_after = re.search(r"const AFTER_SCENE = (\{.*?\});\n", html_doc)
    assert m_before is not None and m_after is not None

    before_data = json.loads(m_before.group(1).replace("<\\/", "</"))
    after_data = json.loads(m_after.group(1).replace("<\\/", "</"))

    assert len(before_data["meshes"]) == 1
    assert len(before_data["bones"]) == 1
    assert before_data["world_aabb"] is not None
    assert len(after_data["meshes"]) == 2
    statuses = {m["name"]: m["status"] for m in after_data["meshes"]}
    assert statuses["Sword"] == "added"
    assert statuses["</script><script>alert('xss')</script>"] == "modified"


def test_heuristic_and_mock_llm_summary(tmp_path: Path) -> None:
    old_s, new_s = _make_sample_pair(tmp_path)
    report = diff_assets(old_s, new_s)

    md = generate_pr_summary(report, provider_name="heuristic")
    assert "REQUEST CHANGES" in md
    assert "UV_CHANNEL_DROPPED" in md
    assert "MATERIAL_ALPHA_MODE_CHANGED" in md

    mock = ScriptedMockLLM("## Mock Summary\nLooks risky.")
    out = generate_pr_summary(report, provider=mock)
    assert out == "## Mock Summary\nLooks risky."
    assert len(mock.calls) == 1
    assert "UV_CHANNEL_DROPPED" in mock.calls[0][1]

    # Safe update heuristic verdict
    safe_report = diff_assets(old_s, old_s)
    safe_md = generate_pr_summary(safe_report, provider_name="heuristic")
    assert "APPROVE" in safe_md


def test_gemini_interactions_http_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_s, new_s = _make_sample_pair(tmp_path)
    report = diff_assets(old_s, new_s)

    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 30.0) -> _FakeHttpResponse:
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHttpResponse(
            {
                "id": "int_123",
                "steps": [
                    {"content": [{"type": "text", "text": "Gemini PR review: fix dropped UV."}]}
                ],
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = generate_pr_summary(report, provider_name="gemini", api_key="test-gem-key")
    assert out == "Gemini PR review: fix dropped UV."
    assert captured["url"] == GEMINI_INTERACTIONS_URL
    assert captured["method"] == "POST"
    assert captured["headers"]["x-goog-api-key"] == "test-gem-key"
    assert captured["headers"]["api-revision"] == GEMINI_API_REVISION
    assert captured["headers"]["content-type"] == "application/json"
    assert captured["body"]["model"] == "gemini-4-hydrogen"

    # Missing key fails cleanly without HTTP call
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMClientError, match="GEMINI_API_KEY"):
        generate_pr_summary(report, provider_name="gemini")


def test_openai_ollama_http_contracts_and_error_handling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_s, new_s = _make_sample_pair(tmp_path)
    report = diff_assets(old_s, new_s)

    captured: list[dict[str, Any]] = []

    def fake_urlopen(req: Any, timeout: float = 30.0) -> _FakeHttpResponse:
        captured.append(
            {
                "url": req.full_url,
                "headers": {k.lower(): v for k, v in req.header_items()},
                "body": json.loads(req.data.decode("utf-8")),
            }
        )
        return _FakeHttpResponse(
            {"choices": [{"message": {"content": "OpenAI-compat review ok."}}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    out_oa = generate_pr_summary(report, provider_name="openai", api_key="sk-test")
    assert out_oa == "OpenAI-compat review ok."
    assert captured[0]["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured[0]["headers"]["authorization"] == "Bearer sk-test"
    assert captured[0]["body"]["model"] == "gpt-4o-mini"

    out_ol = generate_pr_summary(report, provider_name="ollama")
    assert out_ol == "OpenAI-compat review ok."
    assert captured[1]["url"] == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in captured[1]["headers"]
    assert captured[1]["body"]["model"] == "qwen2.5-coder:7b"

    def fake_http_err(req: Any, timeout: float = 30.0) -> Any:
        raise urllib.error.HTTPError(
            req.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"rate limit")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_http_err)
    with pytest.raises(LLMClientError, match="429"):
        generate_pr_summary(report, provider_name="openai", api_key="sk-test")
