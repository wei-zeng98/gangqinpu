#!/usr/bin/env python3
"""Step 4 - Note-level alignment (DualDTW) and refinement (RAScoP).

Writes the raw alignment ``performance_align.npz`` (DualDTW via Parangonar),
then refines it with RAScoP (hole filling, onset cleaning, interpolation of
missing performance notes; no performance synchronisation), writing
``score_refined.mid``, ``performance_refined.mid`` and
``performance_refined_align.npz``.

Filter 3 is applied here: the pair is rejected when the raw insertion rate
(unmatched performance notes / performance notes) or deletion rate
(unmatched score notes / score notes) exceeds 10 %.
"""
from __future__ import annotations

from pathlib import Path

from _cli import run_samples, sample_parser
from gangqinpu.pipeline.checks import load_checks, record_check
from gangqinpu.pipeline.common import (
    PERFORMANCE_CLEAN_MIDI, RAW_ALIGNMENT, REFINED_ALIGNMENT, REFINEMENT_INFO, SCORE_MIDI,
    SCORE_TABLE, SCORE_REFINED_MIDI, PERFORMANCE_REFINED_MIDI, read_json, work_path, write_json,
)

MAX_ALIGNMENT_ERROR_RATE = 0.10


def job(sample: Path, interpolate: bool, force: bool) -> str:
    from gangqinpu.pipeline.alignment import align_and_refine, alignment_rates
    from gangqinpu.pipeline.performance import performance_notes_from_midi

    outputs = [sample / name for name in (RAW_ALIGNMENT, REFINED_ALIGNMENT, SCORE_REFINED_MIDI, PERFORMANCE_REFINED_MIDI)]
    outputs.append(work_path(sample, REFINEMENT_INFO))
    if (not force and all(p.is_file() and p.stat().st_size for p in outputs)
            and load_checks(sample).get("alignment_error_rate", {}).get("passed") is not None):
        return "skipped: exists"
    score_notes = read_json(work_path(sample, SCORE_TABLE))["score_notes"]
    raw, info = align_and_refine(
        score_midi_path=sample / SCORE_MIDI,
        performance_clean_path=sample / PERFORMANCE_CLEAN_MIDI,
        score_notes=score_notes,
        raw_output_path=sample / RAW_ALIGNMENT,
        refined_output_path=sample / REFINED_ALIGNMENT,
        output_dir=sample,
        interpolate_missing=interpolate,
    )
    write_json(work_path(sample, REFINEMENT_INFO), info)

    performance_notes = performance_notes_from_midi(sample / PERFORMANCE_CLEAN_MIDI)
    rates = alignment_rates(raw, len(score_notes), len(performance_notes))
    passed = rates["insertion_rate"] <= MAX_ALIGNMENT_ERROR_RATE and rates["deletion_rate"] <= MAX_ALIGNMENT_ERROR_RATE
    record_check(sample, "alignment_error_rate", passed, {**rates, "maximum_rate": MAX_ALIGNMENT_ERROR_RATE})
    status = "passed" if passed else "rejected"
    return f"{status}: ins={rates['insertion_rate']:.3f} del={rates['deletion_rate']:.3f}"


def main() -> int:
    parser = sample_parser(__doc__, workers=4)
    parser.add_argument("--no-interpolate", action="store_true", help="Do not interpolate missing performance notes.")
    args = parser.parse_args()
    return run_samples(args, job, not args.no_interpolate, args.force, recheck=("alignment_error_rate", "expressive"))


if __name__ == "__main__":
    raise SystemExit(main())
