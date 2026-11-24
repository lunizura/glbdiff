from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from glbdiff import __version__
from glbdiff.diff import diff_assets
from glbdiff.errors import GlbDiffError
from glbdiff.llm_client import generate_pr_summary
from glbdiff.parser import parse_asset
from glbdiff.policy import PolicyConfig, evaluate_policy
from glbdiff.report_html import generate_html_diff


def _write_output(path_str: str, content: str) -> None:
    out_path = Path(path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glbdiff",
        description="Zero-dependency semantic diff, CI budget gate, and 3D inspector for glTF 2.0.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. diff
    p_diff = sub.add_parser("diff", help="Show semantic diff between two .glb/.gltf assets.")
    p_diff.add_argument("old", help="Path to baseline .glb/.gltf file.")
    p_diff.add_argument("new", help="Path to updated .glb/.gltf file.")
    p_diff.add_argument(
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format (default: text).",
    )
    p_diff.add_argument("-o", "--output", default=None, help="Optional output file path.")

    # 2. check
    p_check = sub.add_parser("check", help="Evaluate CI budgets and breaking-change policy gate.")
    p_check.add_argument("old", help="Path to baseline .glb/.gltf file.")
    p_check.add_argument("new", help="Path to updated .glb/.gltf file.")
    p_check.add_argument("--max-tri-pct", type=float, default=None, help="Max triangle increase %%.")
    p_check.add_argument("--max-size-pct", type=float, default=None, help="Max file size increase %%.")
    p_check.add_argument("--max-tris", type=int, default=None, help="Max absolute triangle count.")
    p_check.add_argument("--max-bytes", type=int, default=None, help="Max absolute file size bytes.")
    p_check.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Fail CI gate on WARNING diagnostics.",
    )
    p_check.add_argument(
        "--allow-breaking",
        action="store_true",
        help="Do not fail CI gate on BREAKING diagnostics.",
    )
    p_check.add_argument("--json", action="store_true", help="Print machine-readable JSON result.")

    # 3. textconv
    p_tc = sub.add_parser("textconv", help="Print canonical deterministic text summary for Git.")
    p_tc.add_argument("asset", help="Path to .glb/.gltf file.")

    # 4. git-setup
    p_git = sub.add_parser(
        "git-setup",
        help="Configure .gitattributes and git diff.glb.textconv in the current repo.",
    )
    p_git.add_argument(
        "--print-only",
        action="store_true",
        help="Print configuration snippet without modifying git files.",
    )

    # 5. html
    p_html = sub.add_parser(
        "html",
        help="Generate standalone dual-canvas 3D HTML diff inspector.",
    )
    p_html.add_argument("old", help="Path to baseline .glb/.gltf file.")
    p_html.add_argument("new", help="Path to updated .glb/.gltf file.")
    p_html.add_argument("-o", "--output", required=True, help="Output HTML report path.")

    # 6. summary
    p_sum = sub.add_parser("summary", help="Generate Markdown PR review summary.")
    p_sum.add_argument("old", help="Path to baseline .glb/.gltf file.")
    p_sum.add_argument("new", help="Path to updated .glb/.gltf file.")
    p_sum.add_argument(
        "--provider",
        choices=["heuristic", "gemini", "openai", "ollama"],
        default="heuristic",
        help="Summary provider (default: heuristic).",
    )
    p_sum.add_argument("--model", default=None, help="Override provider model name.")
    p_sum.add_argument("-o", "--output", default=None, help="Optional output Markdown path.")

    return parser


def _cmd_diff(args: argparse.Namespace) -> int:
    old_asset = parse_asset(args.old)
    new_asset = parse_asset(args.new)
    report = diff_assets(old_asset, new_asset)

    if args.format == "json":
        rendered = json.dumps(report.to_dict(), indent=2)
    elif args.format == "markdown":
        rendered = report.to_markdown()
    else:
        rendered = report.to_terminal_text()

    if args.output:
        _write_output(args.output, rendered + "\n")
    else:
        print(rendered)
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    policy = PolicyConfig(
        fail_on_breaking=not args.allow_breaking,
        fail_on_warning=args.fail_on_warning,
        max_tri_increase_pct=args.max_tri_pct,
        max_file_size_increase_pct=args.max_size_pct,
        max_triangles_abs=args.max_tris,
        max_file_size_bytes_abs=args.max_bytes,
    )
    old_asset = parse_asset(args.old)
    new_asset = parse_asset(args.new)
    report = diff_assets(old_asset, new_asset)
    violations = evaluate_policy(report, policy)

    if args.json:
        payload = {
            "passed": len(violations) == 0,
            "violations": violations,
            "report": report.to_dict(),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(report.to_terminal_text())
        if violations:
            print("\nPolicy Check: FAILED")
            for v in violations:
                print(f"  - {v}")
        else:
            print("\nPolicy Check: PASSED")

    return 1 if violations else 0


def _cmd_textconv(args: argparse.Namespace) -> int:
    asset = parse_asset(args.asset)
    sys.stdout.write(asset.to_textconv())
    return 0


def _cmd_git_setup(args: argparse.Namespace) -> int:
    attr_lines = ["*.glb diff=glb", "*.gltf diff=glb"]
    if args.print_only:
        print("# Add to .gitattributes:")
        for line in attr_lines:
            print(line)
        print("\n# Run in your repository:")
        print('git config diff.glb.textconv "glbdiff textconv"')
        return 0

    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except OSError as exc:
        print(f"Error running git: {exc}", file=sys.stderr)
        return 2

    if res.returncode != 0 or not res.stdout.strip():
        print("Error: current directory is not inside a git repository.", file=sys.stderr)
        return 2

    repo_root = Path(res.stdout.strip())
    gitattributes = repo_root / ".gitattributes"
    existing_text = (
        gitattributes.read_text(encoding="utf-8") if gitattributes.exists() else ""
    )
    existing_lines = {ln.strip() for ln in existing_text.splitlines() if ln.strip()}
    to_add = [ln for ln in attr_lines if ln not in existing_lines]
    if to_add:
        prefix = "" if (not existing_text or existing_text.endswith("\n")) else "\n"
        new_text = existing_text + prefix + "\n".join(to_add) + "\n"
        gitattributes.write_text(new_text, encoding="utf-8")

    cfg = subprocess.run(
        ["git", "config", "diff.glb.textconv", "glbdiff textconv"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if cfg.returncode != 0:
        print(f"Error setting git config: {cfg.stderr.strip()}", file=sys.stderr)
        return 2

    print(f"Configured {gitattributes} and git diff.glb.textconv='glbdiff textconv'.")
    return 0


def _cmd_html(args: argparse.Namespace) -> int:
    old_asset = parse_asset(args.old)
    new_asset = parse_asset(args.new)
    report = diff_assets(old_asset, new_asset)
    html_doc = generate_html_diff(report)
    _write_output(args.output, html_doc)
    print(f"Wrote 3D HTML diff report to {args.output}")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    old_asset = parse_asset(args.old)
    new_asset = parse_asset(args.new)
    report = diff_assets(old_asset, new_asset)
    md = generate_pr_summary(report, provider_name=args.provider, model=args.model)
    if args.output:
        _write_output(args.output, md + "\n")
    else:
        print(md)
    return 0


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        enc = getattr(stream, "encoding", None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure) and isinstance(enc, str) and enc.lower() not in ("utf-8", "utf8"):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_utf8_stdio()
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "diff":
            return _cmd_diff(args)
        if args.command == "check":
            return _cmd_check(args)
        if args.command == "textconv":
            return _cmd_textconv(args)
        if args.command == "git-setup":
            return _cmd_git_setup(args)
        if args.command == "html":
            return _cmd_html(args)
        if args.command == "summary":
            return _cmd_summary(args)
        parser.print_help()
        return 2
    except (GlbDiffError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
