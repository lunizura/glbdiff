from __future__ import annotations

import math
from dataclasses import dataclass

from glbdiff.errors import PolicyConfigError
from glbdiff.models import DiffReport


@dataclass
class PolicyConfig:
    fail_on_breaking: bool = True
    fail_on_warning: bool = False
    max_tri_increase_pct: float | None = None
    max_file_size_increase_pct: float | None = None
    max_triangles_abs: int | None = None
    max_file_size_bytes_abs: int | None = None

    def __post_init__(self) -> None:
        if self.max_tri_increase_pct is not None:
            if not math.isfinite(self.max_tri_increase_pct) or self.max_tri_increase_pct < 0:
                raise PolicyConfigError("max_tri_increase_pct must be a non-negative finite number")
        if self.max_file_size_increase_pct is not None:
            if (
                not math.isfinite(self.max_file_size_increase_pct)
                or self.max_file_size_increase_pct < 0
            ):
                raise PolicyConfigError(
                    "max_file_size_increase_pct must be a non-negative finite number"
                )
        if self.max_triangles_abs is not None and self.max_triangles_abs < 0:
            raise PolicyConfigError("max_triangles_abs must be >= 0")
        if self.max_file_size_bytes_abs is not None and self.max_file_size_bytes_abs < 0:
            raise PolicyConfigError("max_file_size_bytes_abs must be >= 0")


def evaluate_policy(report: DiffReport, policy: PolicyConfig) -> list[str]:
    violations: list[str] = []

    if policy.fail_on_breaking and report.has_breaking:
        breaking = [f for f in report.findings if f.severity == "BREAKING"]
        codes = ", ".join(f"{f.code}({f.subject})" for f in breaking)
        violations.append(f"Breaking changes detected ({len(breaking)}): {codes}")

    if policy.fail_on_warning:
        warnings = [f for f in report.findings if f.severity == "WARNING"]
        if warnings:
            codes = ", ".join(f"{f.code}({f.subject})" for f in warnings)
            violations.append(f"Warnings detected under strict mode ({len(warnings)}): {codes}")

    if policy.max_tri_increase_pct is not None:
        tri_pct = report.metrics["total_triangles"].pct_change
        if tri_pct is not None and tri_pct > policy.max_tri_increase_pct:
            violations.append(
                f"Triangle count increase {tri_pct:.2f}% exceeds budget "
                f"{policy.max_tri_increase_pct:.2f}%"
            )

    if policy.max_file_size_increase_pct is not None:
        size_pct = report.metrics["file_size_bytes"].pct_change
        if size_pct is not None and size_pct > policy.max_file_size_increase_pct:
            violations.append(
                f"File size increase {size_pct:.2f}% exceeds budget "
                f"{policy.max_file_size_increase_pct:.2f}%"
            )

    if policy.max_triangles_abs is not None:
        if report.new_asset.total_triangles > policy.max_triangles_abs:
            violations.append(
                f"Total triangles {report.new_asset.total_triangles} exceeds absolute limit "
                f"{policy.max_triangles_abs}"
            )

    if policy.max_file_size_bytes_abs is not None:
        if report.new_asset.file_size_bytes > policy.max_file_size_bytes_abs:
            violations.append(
                f"File size {report.new_asset.file_size_bytes} bytes exceeds absolute limit "
                f"{policy.max_file_size_bytes_abs} bytes"
            )

    return violations
