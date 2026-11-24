# glbdiff

[![CI](https://img.shields.io/badge/CI-Node%2024%20Actions-blue)](#github-actions-ci-budget-gate)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-3776ab)](pyproject.toml)
[![Dependencies](https://img.shields.io/badge/runtime%20dependencies-0-34d399)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Semantic diff tool, breaking-change detector, CI budget gate, Git `textconv` driver, BYOK PR summarizer, and offline dual-canvas 3D HTML diff inspector for Khronos glTF 2.0 (`.glb` and `.gltf`) assets.

`dependencies = []` — 100% Python 3.10+ standard library at runtime.

---

## Why `git diff` Fails on `.glb` Files

To Git, a `.glb` is an opaque binary blob (`Binary files a/hero.glb and b/hero.glb differ`). Standard Git diffs cannot tell you when a re-export silently:

- **Shifted world scale by 100x** (Blender/Maya meter vs. centimeter export mismatch).
- **Broke a character rig** (renamed, removed, or reparented skeleton joints).
- **Dropped `TEXCOORD_0` or morph targets** (breaking shaders and facial blendshapes).
- **Deleted an animation clip** or altered clip duration.
- **Switched a material from `OPAQUE` to `BLEND`** (causing alpha-sorting regressions).
- **Spiked triangle counts or embedded texture VRAM** past your runtime budget.

`glbdiff` parses the binary GLB chunks, interleaved `byteStride` vertex buffers, node transform hierarchy, skins, animations, and embedded PNG/JPEG/WebP/KTX2 image headers directly in pure Python.

---

## Architecture

| Module | Responsibility |
| --- | --- |
| `glbdiff.parser` | Binary `.glb` chunk reader & `.gltf` (data-URI / path-safe external `.bin`) scene & wireframe extractor |
| `glbdiff.math3d` | Column-major 4x4 matrix math, TRS-to-mat4, and 8-corner world AABB transforms |
| `glbdiff.textures` | Pure-stdlib header sniffer for PNG, JPEG, WebP (`VP8X`/`VP8L`/`VP8`), and KTX2 dimensions |
| `glbdiff.diff` | Semantic scene-graph diff & `BREAKING` / `WARNING` / `INFO` classifier |
| `glbdiff.policy` | CI budget gate evaluator (triangle/size deltas, absolute caps, strict warning/breaking gates) |
| `glbdiff.report_html` | Zero-CDN standalone dark-mode HTML5 viewer with synchronized dual-canvas 3D orbit/zoom |
| `glbdiff.llm_client` | Offline heuristic PR summary + zero-SDK HTTP clients for Gemini Interactions (`2026-05-20`), OpenAI, and Ollama |
| `glbdiff.cli` | 6 subcommands: `diff`, `check`, `textconv`, `git-setup`, `html`, `summary` |

---

## Breaking-Change & Diagnostic Rules

| Code | Severity | Trigger |
| --- | --- | --- |
| `UNIT_SCALE_100X_SHIFT` | `BREAKING` | World AABB diagonal ratio in `[90, 110]` or `[0.009, 0.011]` |
| `LARGE_SCALE_DRIFT` | `WARNING` | World AABB diagonal ratio `> 2.0` or `< 0.5` (non-100x) |
| `SKELETON_JOINT_REMOVED` | `BREAKING` | Joint missing from matching skin or entire skin removed |
| `SKELETON_JOINT_REPARENTED` | `BREAKING` | Joint changed `parent_name` within skeleton hierarchy |
| `ANIMATION_CLIP_REMOVED` | `BREAKING` | Animation clip present in old asset missing in new asset |
| `ANIMATION_DURATION_CHANGED` | `WARNING` | Matching animation duration delta `> 0.05s` |
| `UV_CHANNEL_DROPPED` | `BREAKING` | `TEXCOORD_*` attribute removed from a matching mesh |
| `MORPH_TARGETS_DROPPED` | `BREAKING` | Morph target count decreased on a matching mesh |
| `MESH_REMOVED` | `BREAKING` | Named mesh removed from asset |
| `MATERIAL_ALPHA_MODE_CHANGED` | `WARNING` | Matching material changed `alphaMode` (e.g. `OPAQUE` -> `BLEND`) |
| `TRIANGLE_SPIKE` | `WARNING` | Total triangles increased by `> 20.0%` |
| `TEXTURE_MEMORY_SPIKE` | `WARNING` | Embedded texture bytes increased by `> 25.0%` |
| `ASSET_OPTIMIZED` | `INFO` | Triangles or file size decreased by `>= 5.0%` with zero `BREAKING` findings |

---

## 30-Second Quickstart

```bash
pip install -e .

# Generate two sample rigged .glb assets in examples/samples/
python examples/generate_demo_glbs.py

# 1. Terminal semantic diff
glbdiff diff examples/samples/character_v1.glb examples/samples/character_v2.glb

# 2. CI budget gate (exit 0 = pass, 1 = policy fail, 2 = parse/config error)
glbdiff check examples/samples/character_v1.glb examples/samples/character_v2.glb --max-tri-pct 15

# 3. Synchronized dual-canvas 3D HTML diff inspector (zero CDN, works offline)
glbdiff html examples/samples/character_v1.glb examples/samples/character_v2.glb -o diff_report.html

# 4. Offline PR review summary
glbdiff summary examples/samples/character_v1.glb examples/samples/character_v2.glb --provider heuristic
```

---

## Native `git diff` Integration (`textconv`)

Configure `.gitattributes` and Git's `textconv` driver in one command:

```bash
glbdiff git-setup
```

Or inspect/apply manually:

```gitattributes
*.glb diff=glb
*.gltf diff=glb
```

```bash
git config diff.glb.textconv "glbdiff textconv"
git diff HEAD~1 -- assets/character.glb
```

---

## GitHub Actions CI Budget Gate

```yaml
- name: Check 3D asset budgets
  run: |
    glbdiff check origin/main_hero.glb assets/hero.glb \
      --max-tri-pct 10 \
      --max-size-pct 15 \
      --max-tris 50000
```

Exit codes for `glbdiff check`:
- `0`: All policy checks passed.
- `1`: One or more policy violations (breaking changes or budget exceeded).
- `2`: Invalid GLB/glTF input, path-traversal error, or invalid threshold configuration.

---

## BYOK LLM PR Summaries

Generate a deterministic Markdown review offline (`--provider heuristic`), or point to a model provider using standard library `urllib.request`:

```bash
# Offline deterministic summary (0 API keys)
glbdiff summary v1.glb v2.glb --provider heuristic -o pr_comment.md

# Gemini Interactions API (Api-Revision: 2026-05-20, default: gemini-4-hydrogen)
export GEMINI_API_KEY="..."
glbdiff summary v1.glb v2.glb --provider gemini

# OpenAI-compatible (default: gpt-4o-mini) or local Ollama (default: qwen2.5-coder:7b)
glbdiff summary v1.glb v2.glb --provider ollama --model qwen2.5-coder:7b
```
