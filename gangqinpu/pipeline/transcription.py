"""Audio-to-MIDI transcription with Transkun v2."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


def find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("ffmpeg is required (system ffmpeg or `pip install imageio-ffmpeg`)") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def to_mono_wav(audio: Path, tmp_dir: Path) -> Path:
    """Decode to mono 44.1 kHz WAV, the input representation used for Transkun."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(audio.resolve()).encode("utf-8")).hexdigest()[:12]
    wav = tmp_dir / f"{audio.parent.name}-{digest}.wav"
    command = [find_ffmpeg(), "-y", "-v", "error", "-i", str(audio), "-ar", "44100", "-ac", "1", str(wav)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or f"ffmpeg exited {completed.returncode}")
    return wav


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def transcribe(audio: Path, output: Path, tmp_dir: Path, device: str = "auto") -> Path:
    """Run the Transkun v2 CLI; reuse WAV inputs and decode other formats first."""
    wav = audio if audio.suffix.lower() == ".wav" else to_mono_wav(audio, tmp_dir)
    try:
        command = [sys.executable, "-m", "transkun.transcribe", str(wav), str(output),
                   "--device", resolve_device(device)]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(completed.stdout.strip() + completed.stderr.strip()
                               or f"transkun exited {completed.returncode}")
    finally:
        if wav != audio:
            wav.unlink(missing_ok=True)
    return output
