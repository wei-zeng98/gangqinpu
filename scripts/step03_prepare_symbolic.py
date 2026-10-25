#!/usr/bin/env python3
"""Step 3 - Prepare both symbolic representations for alignment.

Score side: MusicXML is loaded with Partitura, repeats/voltas are unfolded,
grace notes and ornaments expanded and duplicates removed (SyMuPe
preprocessing); writes ``score_MS.mid`` plus the score note table and the
metrical beat grid (``.work/score_table.json``).

Performance side: ``audio_transkun.mid`` is merged to one track, duplicate and
implausibly short notes are removed and same-pitch overlaps truncated;
writes ``performance_clean.mid``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

from _cli import run_samples, sample_parser
from gangqinpu.pipeline.common import (
    PERFORMANCE_CLEAN_MIDI, PERFORMANCE_INFO, SCORE_MIDI, SCORE_MUSICXML, SCORE_TABLE,
    TRANSCRIPTION_MIDI, work_path, write_json,
)

# Converted scores trigger many harmless Partitura parser-recovery warnings.
warnings.filterwarnings("ignore", module=r"partitura(\..*)?")


def job(sample: Path, force: bool) -> str:
    from gangqinpu.pipeline.performance import clean_performance_midi
    from gangqinpu.pipeline.score import process_score

    outputs = [sample / SCORE_MIDI, sample / PERFORMANCE_CLEAN_MIDI,
               work_path(sample, SCORE_TABLE), work_path(sample, PERFORMANCE_INFO)]
    if all(p.is_file() for p in outputs) and not force:
        return "skipped: exists"
    score = process_score(sample / SCORE_MUSICXML, sample)
    write_json(work_path(sample, SCORE_TABLE),
               {"score_notes": score["score_notes"], "score_beats": score["score_beats"]})
    performance = clean_performance_midi(sample / TRANSCRIPTION_MIDI, sample)
    write_json(work_path(sample, PERFORMANCE_INFO),
               {"num_raw_notes": performance["num_raw_notes"], "num_clean_notes": performance["num_clean_notes"]})
    return f"done: {len(score['score_notes'])} score notes, {performance['num_clean_notes']} performance notes"


def main() -> int:
    args = sample_parser(__doc__, workers=4).parse_args()
    return run_samples(args, job, args.force, recheck=("alignment_error_rate", "expressive"))


if __name__ == "__main__":
    raise SystemExit(main())
