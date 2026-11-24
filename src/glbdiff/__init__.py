from __future__ import annotations

from glbdiff.diff import diff_assets
from glbdiff.errors import GlbDiffError, GlbParseError, LLMClientError, PolicyConfigError
from glbdiff.llm_client import ScriptedMockLLM, generate_pr_summary
from glbdiff.models import (
    AABB,
    AnimationSummary,
    AssetSummary,
    DiagnosticFinding,
    DiffReport,
    JointInfo,
    MaterialSummary,
    MeshSummary,
    MetricDelta,
    SkinSummary,
    TextureSummary,
    WireframePreview,
)
from glbdiff.parser import parse_asset
from glbdiff.policy import PolicyConfig, evaluate_policy
from glbdiff.report_html import generate_html_diff

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AABB",
    "AnimationSummary",
    "AssetSummary",
    "DiagnosticFinding",
    "DiffReport",
    "GlbDiffError",
    "GlbParseError",
    "JointInfo",
    "LLMClientError",
    "MaterialSummary",
    "MeshSummary",
    "MetricDelta",
    "PolicyConfig",
    "PolicyConfigError",
    "ScriptedMockLLM",
    "SkinSummary",
    "TextureSummary",
    "WireframePreview",
    "diff_assets",
    "evaluate_policy",
    "generate_html_diff",
    "generate_pr_summary",
    "parse_asset",
]
