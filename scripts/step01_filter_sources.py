#!/usr/bin/env python3
"""Step 1 - Filters 1 and 2 on the source pair.

Filter 1: audio duration >= 60 s (ffprobe).
Filter 2: MusicXML score with >= 32 measures and piano-only sounding parts.

Writes the outcomes to ``pipeline_checks.json``; failing samples are skipped
by all later steps.
"""
from __future__ import annotations

from pathlib import Path

from _cli import run_samples, sample_parser
from gangqinpu.pipeline.checks import load_checks, record_check
from gangqinpu.pipeline.common import SCORE_MUSICXML, find_audio
from gangqinpu.pipeline.quality import MIN_AUDIO_SECONDS, MIN_BARS, audio_duration_seconds, inspect_musicxml


def job(sample: Path, audio_dir: Path | None, min_seconds: float, min_bars: int, force: bool) -> str:
    checks = load_checks(sample)
    if not force and {"audio_duration", "score_bar_count", "piano_only"} <= set(checks):
        return "skipped: already checked"
    audio = find_audio(sample, audio_dir)
    if audio is None:
        raise FileNotFoundError("no audio.* file")
    duration = audio_duration_seconds(audio)
    record_check(sample, "audio_duration", duration >= min_seconds,
                 {"duration_seconds": duration, "minimum_seconds": min_seconds})

    inspection = inspect_musicxml(sample / SCORE_MUSICXML)
    record_check(sample, "score_bar_count", inspection.bar_count >= min_bars,
                 {"bar_count": inspection.bar_count, "minimum_bars": min_bars})
    record_check(sample, "piano_only", inspection.piano_only, {"sounding_parts": inspection.parts})
    passed = duration >= min_seconds and inspection.bar_count >= min_bars and inspection.piano_only
    return "passed" if passed else "rejected"


def main() -> int:
    parser = sample_parser(__doc__, workers=4)
    parser.add_argument("--min-audio-seconds", type=float, default=MIN_AUDIO_SECONDS)
    parser.add_argument("--min-bars", type=int, default=MIN_BARS)
    args = parser.parse_args()
    return run_samples(args, job, args.audio_dir, args.min_audio_seconds, args.min_bars, args.force, recheck=("audio_duration", "score_bar_count", "piano_only", "alignment_error_rate", "expressive"))


if __name__ == "__main__":
    raise SystemExit(main())
