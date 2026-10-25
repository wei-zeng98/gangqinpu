#!/usr/bin/env python3
"""Step 5 - Alignment-derived beat/downbeat annotation and metadata.

Score beats (dotted-quarter beats in compound meters, pickups and inferred
overfull measures handled) are transferred to performance time: a beat with
matched notes within ±0.05 quarters takes their median performed onset (confidence
0.95, or 0.40 if only interpolated notes support it); other beats are
interpolated between onset anchors or extrapolated with the local tempo.
Beats inside alignment holes have their confidence multiplied by 0.35;
confidence < 0.35 is labelled ``bR`` (ambiguous).

Writes ``midi_score_annotations.txt``, ``performance_annotations.txt`` and
``alignment_metadata.json`` (the classifier result is added in step 6).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _cli import run_samples, sample_parser
from gangqinpu.pipeline.common import (
    ALIGNMENT_METADATA, PERFORMANCE_CLEAN_MIDI, PERFORMANCE_INFO, PERFORMANCE_REFINED_MIDI,
    RAW_ALIGNMENT, REFINED_ALIGNMENT, REFINEMENT_INFO, SCORE_MIDI,
    SCORE_REFINED_MIDI, SCORE_TABLE, TRANSCRIPTION_MIDI, read_json, release_relative,
    to_jsonable, work_path, write_json,
)


def job(sample: Path, force: bool) -> str:
    from gangqinpu.pipeline.alignment import ALIGNER_BACKEND, alignment_rates, load_raw_alignment, load_refined_alignment
    from gangqinpu.pipeline.annotations import write_annotations
    from gangqinpu.pipeline.performance import performance_notes_from_midi

    if (sample / ALIGNMENT_METADATA).is_file() and not force:
        return "skipped: exists"
    table = read_json(work_path(sample, SCORE_TABLE))
    score_notes, score_beats = table["score_notes"], table["score_beats"]
    performance_notes = performance_notes_from_midi(sample / PERFORMANCE_CLEAN_MIDI)
    raw = load_raw_alignment(sample / RAW_ALIGNMENT)
    refined = load_refined_alignment(sample / REFINED_ALIGNMENT)
    info = read_json(work_path(sample, REFINEMENT_INFO))
    refined["tempo_jump_count"] = info["tempo_jump_count"]

    paths = {
        "performance_midi": TRANSCRIPTION_MIDI,
        "score_MS_mid": SCORE_MIDI,
        "performance_clean_mid": PERFORMANCE_CLEAN_MIDI,
        "performance_align_npz": RAW_ALIGNMENT,
        "performance_refined_align_npz": REFINED_ALIGNMENT,
        "score_refined_mid": SCORE_REFINED_MIDI,
        "performance_refined_mid": PERFORMANCE_REFINED_MIDI,
    }
    metadata = write_annotations(sample, score_notes, performance_notes, score_beats, raw, refined,
                                 {key: release_relative(sample, name) for key, name in paths.items()})
    rates = alignment_rates(raw, len(score_notes), len(performance_notes))
    metadata.update(
        aligner_backend=ALIGNER_BACKEND,
        num_score_notes=len(score_notes),
        num_performance_notes=len(performance_notes),
        num_raw_performance_notes=read_json(work_path(sample, PERFORMANCE_INFO))["num_raw_notes"],
        rascop_recall=info["rascop_recall"],
        rascop_precision=info["rascop_precision"],
        rascop_metrics=info["rascop_metrics"],
        rascop_stage_times=info["rascop_stage_times"],
        num_interpolated_notes=int(np.asarray(refined["interpolated_mask"]).sum()),
        raw_alignment_insertion_rate=rates["insertion_rate"],
        raw_alignment_deletion_rate=rates["deletion_rate"],
    )
    write_json(sample / ALIGNMENT_METADATA, to_jsonable(metadata))
    return f"done: {len(metadata['performance_beats'])} beats, {len(metadata['performance_downbeats'])} downbeats"


def main() -> int:
    args = sample_parser(__doc__, workers=4).parse_args()
    return run_samples(args, job, args.force, recheck=("expressive",))


if __name__ == "__main__":
    raise SystemExit(main())
