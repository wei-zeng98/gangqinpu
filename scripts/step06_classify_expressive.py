#!/usr/bin/env python3
"""Step 6 - Filter 4: expressive-performance classification.

Uses the raw DualDTW alignment from step 4 (observed matches only, no
interpolated notes) to compute the eight score-informed timing/dynamics
features and applies the trained MLP. Samples with predicted expressive
probability < 0.5 are rejected. The result is stored under
``expressive_performance_classifier`` in ``alignment_metadata.json``.
"""
from __future__ import annotations

from pathlib import Path

from _cli import run_samples, sample_parser
from gangqinpu.pipeline.checks import load_checks, record_check
from gangqinpu.pipeline.common import (
    ALIGNMENT_METADATA, PERFORMANCE_CLEAN_MIDI, RAW_ALIGNMENT, SCORE_TABLE, read_json, work_path, write_json,
)

KEY = "expressive_performance_classifier"


def job(sample: Path, force: bool) -> str:
    from gangqinpu.classifier import classify_alignment
    from gangqinpu.pipeline.alignment import load_raw_alignment
    from gangqinpu.pipeline.performance import performance_notes_from_midi

    metadata = read_json(sample / ALIGNMENT_METADATA)
    cached = metadata.get(KEY, {})
    if (cached.get("label") == "expressive"
            and load_checks(sample).get("expressive", {}).get("passed") is True and not force):
        return "skipped: exists"
    score_notes = read_json(work_path(sample, SCORE_TABLE))["score_notes"]
    performance_notes = performance_notes_from_midi(sample / PERFORMANCE_CLEAN_MIDI)
    result = classify_alignment(score_notes, performance_notes, load_raw_alignment(sample / RAW_ALIGNMENT))
    metadata[KEY] = result
    write_json(sample / ALIGNMENT_METADATA, metadata)
    if "error" in result:  # undecidable; the sample stays pending
        record_check(sample, "expressive", None, result)
        return "undecided: " + result["error"]
    record_check(sample, "expressive", result["label"] == "expressive", result)
    return f"{result['label']}: p={result['proba_expressive']:.4f}"


def main() -> int:
    args = sample_parser(__doc__, workers=4).parse_args()
    return run_samples(args, job, args.force, recheck=("expressive",))


if __name__ == "__main__":
    raise SystemExit(main())
