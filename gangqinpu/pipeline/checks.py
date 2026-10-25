"""Per-sample record of filter decisions (``pipeline_checks.json``).

Each filter stores its outcome and evidence. A sample with any failed check is
rejected and skipped by later steps; the record is never deleted, so the
reason for every rejection remains inspectable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import CHECKS_FILE, read_json, to_jsonable, write_json


def load_checks(sample_dir: Path) -> dict[str, Any]:
    path = Path(sample_dir) / CHECKS_FILE
    return read_json(path) if path.is_file() else {}


def record_check(sample_dir: Path, name: str, passed: bool | None, details: dict[str, Any]) -> None:
    """Store one check. ``passed=None`` means the check could not be decided."""
    checks = load_checks(sample_dir)
    checks[name] = {"passed": passed, "details": to_jsonable(details)}
    write_json(Path(sample_dir) / CHECKS_FILE, checks)


def is_rejected(sample_dir: Path) -> bool:
    return any(check.get("passed") is False for check in load_checks(sample_dir).values())

