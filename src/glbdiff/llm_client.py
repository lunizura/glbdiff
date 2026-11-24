from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from glbdiff.errors import LLMClientError
from glbdiff.models import DiffReport

GEMINI_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
GEMINI_API_REVISION = "2026-05-20"
DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"


class LLMProvider(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class ScriptedMockLLM:
    """Deterministic offline mock provider for unit tests."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response_text


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float = 30.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        raise LLMClientError(f"LLM HTTP {exc.code} from {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMClientError(f"LLM network failure calling {url}: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMClientError(f"Invalid JSON response from LLM endpoint: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMClientError("LLM response JSON root must be an object")
    return parsed


@dataclass
class GeminiInteractionsProvider:
    model: str = DEFAULT_GEMINI_MODEL
    api_key: str | None = None
    endpoint: str = GEMINI_INTERACTIONS_URL
    timeout: float = 30.0
    _resolved_key: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key.strip():
            raise LLMClientError("GEMINI_API_KEY is required for GeminiInteractionsProvider")
        self._resolved_key = key.strip()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "x-goog-api-key": self._resolved_key,
            "Content-Type": "application/json",
            "Api-Revision": GEMINI_API_REVISION,
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "system_instruction": system_prompt,
            "input": user_prompt,
        }
        data = _post_json(self.endpoint, headers, payload, timeout=self.timeout)
        texts: list[str] = []
        steps = data.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                content = step.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        texts.append(block["text"])
        if not texts:
            raise LLMClientError("Gemini Interactions response contained no steps[].content[].text")
        return "\n".join(texts).strip()


@dataclass
class OpenAICompatibleProvider:
    model: str = DEFAULT_OPENAI_MODEL
    api_key: str | None = None
    endpoint: str = "https://api.openai.com/v1/chat/completions"
    require_api_key: bool = True
    timeout: float = 30.0
    _resolved_key: str | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        key = self.api_key or (os.environ.get("OPENAI_API_KEY") if self.require_api_key else None)
        if self.require_api_key and (not key or not key.strip()):
            raise LLMClientError("OPENAI_API_KEY is required for OpenAICompatibleProvider")
        self._resolved_key = key.strip() if key and key.strip() else None

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self._resolved_key:
            headers["Authorization"] = f"Bearer {self._resolved_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        data = _post_json(self.endpoint, headers, payload, timeout=self.timeout)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("OpenAI-compatible response missing choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMClientError("OpenAI-compatible response returned empty content")
        return content.strip()


def _heuristic_summary(report: DiffReport) -> str:
    breaking = [f for f in report.findings if f.severity == "BREAKING"]
    warnings = [f for f in report.findings if f.severity == "WARNING"]
    infos = [f for f in report.findings if f.severity == "INFO"]

    if breaking:
        verdict = "REQUEST CHANGES (Breaking 3D Asset Changes Detected)"
    elif warnings:
        verdict = "REVIEW RECOMMENDED (Performance or Material Warnings)"
    else:
        verdict = "APPROVE (Safe 3D Asset Update)"

    size_m = report.metrics["file_size_bytes"]
    tri_m = report.metrics["total_triangles"]
    vert_m = report.metrics["total_vertices"]
    tex_m = report.metrics["total_texture_bytes"]

    def _pct(val: float | None) -> str:
        if val is None:
            return "new"
        sign = "+" if val > 0 else ""
        return f"{sign}{val:.2f}%"

    lines: list[str] = [
        f"### PR 3D Asset Review: {verdict}",
        "",
        f"- **Asset:** `{report.old_asset.path}` -> `{report.new_asset.path}`",
        f"- **File Size:** {size_m.old_val} B -> {size_m.new_val} B ({_pct(size_m.pct_change)})",
        f"- **Triangles:** {tri_m.old_val} -> {tri_m.new_val} ({_pct(tri_m.pct_change)})",
        f"- **Vertices:** {vert_m.old_val} -> {vert_m.new_val} ({_pct(vert_m.pct_change)})",
        f"- **Texture Memory:** {tex_m.old_val} B -> {tex_m.new_val} B ({_pct(tex_m.pct_change)})",
    ]
    if report.scale_ratio is not None:
        lines.append(f"- **World Scale Ratio:** `{report.scale_ratio:.4f}x`")

    if breaking:
        lines.extend(["", "#### Breaking Changes (Must Fix Before Merge)"])
        for f in breaking:
            lines.append(f"- **`{f.code}`** (`{f.subject}`): {f.message}")

    if warnings:
        lines.extend(["", "#### Warnings (Verify In Engine)"])
        for f in warnings:
            lines.append(f"- **`{f.code}`** (`{f.subject}`): {f.message}")

    if infos:
        lines.extend(["", "#### Optimizations & Notes"])
        for f in infos:
            lines.append(f"- **`{f.code}`** (`{f.subject}`): {f.message}")

    changes: list[str] = []
    if report.added_meshes:
        changes.append(f"Added meshes: {', '.join(f'`{x}`' for x in report.added_meshes)}")
    if report.removed_meshes:
        changes.append(f"Removed meshes: {', '.join(f'`{x}`' for x in report.removed_meshes)}")
    if report.modified_meshes:
        changes.append(f"Modified meshes: {', '.join(f'`{x}`' for x in report.modified_meshes)}")
    if report.added_materials or report.removed_materials or report.modified_materials:
        changes.append(
            f"Materials (+{len(report.added_materials)} / -{len(report.removed_materials)} / "
            f"~{len(report.modified_materials)})"
        )
    if report.added_animations or report.removed_animations or report.modified_animations:
        changes.append(
            f"Animations (+{len(report.added_animations)} / -{len(report.removed_animations)} / "
            f"~{len(report.modified_animations)})"
        )
    if changes:
        lines.extend(["", "#### Structural Diff"])
        for c in changes:
            lines.append(f"- {c}")

    return "\n".join(lines)


def generate_pr_summary(
    report: DiffReport,
    provider_name: str = "heuristic",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    provider: LLMProvider | None = None,
) -> str:
    if provider is None:
        name = provider_name.lower().strip()
        if name == "heuristic":
            return _heuristic_summary(report)
        if name == "gemini":
            provider = GeminiInteractionsProvider(
                model=model or DEFAULT_GEMINI_MODEL,
                api_key=api_key,
                endpoint=base_url or GEMINI_INTERACTIONS_URL,
            )
        elif name == "openai":
            provider = OpenAICompatibleProvider(
                model=model or DEFAULT_OPENAI_MODEL,
                api_key=api_key,
                endpoint=base_url or "https://api.openai.com/v1/chat/completions",
                require_api_key=True,
            )
        elif name == "ollama":
            provider = OpenAICompatibleProvider(
                model=model or DEFAULT_OLLAMA_MODEL,
                api_key=api_key,
                endpoint=base_url or "http://localhost:11434/v1/chat/completions",
                require_api_key=False,
            )
        else:
            raise LLMClientError(f"Unsupported LLM provider: {provider_name!r}")

    system_prompt = (
        "You are a senior 3D technical artist reviewing a glTF/GLB asset change in a Pull Request. "
        "Summarize breaking risks (scale shifts, missing rig joints, dropped UVs/morphs/clips), "
        "budget impacts, and actionable recommendations concisely in Markdown."
    )
    compact_report = json.dumps(report.to_dict(), indent=2)
    user_prompt = f"Review this glbdiff JSON report and write a PR review comment:\n\n{compact_report}"
    return provider.complete(system_prompt, user_prompt)
