#!/usr/bin/env python3
"""Step 2 - Audio-to-MIDI transcription with Transkun v2 -> ``audio_transkun.mid``.

Compressed audio is decoded to mono 44.1 kHz WAV; existing WAV is reused.
Transkun uses its default pretrained 2.0 checkpoint (note offsets are not
extended by pedal).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from _cli import run_samples, sample_parser
from gangqinpu.pipeline.common import TRANSCRIPTION_MIDI, find_audio
from gangqinpu.pipeline.transcription import transcribe


def job(sample: Path, audio_dir: Path | None, device: str, tmp_dir: Path, force: bool) -> str:
    output = sample / TRANSCRIPTION_MIDI
    if output.is_file() and output.stat().st_size and not force:
        return "skipped: exists"
    audio = find_audio(sample, audio_dir)
    if audio is None:
        raise FileNotFoundError("no audio.* file")
    transcribe(audio, output, tmp_dir, device)
    return "done"


def main() -> int:
    parser = sample_parser(__doc__, workers=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N (default: auto).")
    parser.add_argument("--tmp-dir", type=Path, default=Path(tempfile.gettempdir()) / "gangqinpu-wav")
    args = parser.parse_args()
    return run_samples(args, job, args.audio_dir, args.device, args.tmp_dir, args.force, recheck=("alignment_error_rate", "expressive"))


if __name__ == "__main__":
    raise SystemExit(main())
