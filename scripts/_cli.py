"""Shared command-line plumbing for the per-sample pipeline steps."""
from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gangqinpu.pipeline.checks import load_checks  # noqa: E402
from gangqinpu.pipeline.common import list_sample_dirs  # noqa: E402


def sample_parser(description: str, workers: int = 1) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--samples-dir", type=Path, required=True,
                        help="Folder with one sub-folder per sample (score.musicxml, metadata.json, ...).")
    parser.add_argument("--audio-dir", type=Path,
                        help="Optional parallel tree <audio-dir>/<sample_id>/audio.* (default: inside each sample folder).")
    parser.add_argument("--ids", nargs="+", help="Process only these sample IDs.")
    parser.add_argument("--workers", type=int, default=workers, help=f"Parallel processes (default: {workers}).")
    parser.add_argument("--force", action="store_true", help="Recompute outputs that already exist.")
    return parser


def run_samples(args: argparse.Namespace, job: Callable[..., str], *job_args, recheck: tuple[str, ...] = ()) -> int:
    """Apply ``job(sample_dir, *job_args) -> status`` to every non-rejected sample."""
    def eligible(sample: Path) -> bool:
        checks = load_checks(sample)
        return not any(check.get("passed") is False
                       for name, check in checks.items()
                       if not (args.force and name in recheck))

    samples = [d for d in list_sample_dirs(args.samples_dir, args.ids) if eligible(d)]
    print(f"{len(samples)} samples to process", flush=True)
    counts: dict[str, int] = {}

    def report(sample: Path, status: str) -> None:
        counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
        print(f"[{sum(counts.values())}/{len(samples)}] {sample.name}: {status}", flush=True)

    if args.workers <= 1:
        for sample in samples:
            report(sample, _safe(job, sample, *job_args))
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            futures = {pool.submit(_safe, job, sample, *job_args): sample for sample in samples}
            for future in as_completed(futures):
                report(futures[future], future.result())
    print("summary:", counts, flush=True)
    return 1 if counts.get("error") or counts.get("undecided") else 0


def _safe(job: Callable[..., str], sample: Path, *job_args) -> str:
    try:
        return job(sample, *job_args)
    except Exception as exc:  # keep going; report the failing sample
        detail = traceback.format_exception_only(type(exc), exc)[-1].strip()
        return f"error: {detail}"
