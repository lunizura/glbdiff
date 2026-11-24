from __future__ import annotations


class GlbDiffError(Exception):
    """Base error for all glbdiff failures."""


class GlbParseError(GlbDiffError):
    """Raised on malformed, truncated, or unsafe .glb/.gltf files."""


class PolicyConfigError(GlbDiffError):
    """Raised on invalid CI budget threshold parameters."""


class LLMClientError(GlbDiffError):
    """Raised when a BYOK LLM provider request or response fails."""
