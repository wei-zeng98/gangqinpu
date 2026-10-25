"""Shared types, constants and small helpers for the Gangqinpu pipeline."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

LABEL_BEAT = "b"
LABEL_DOWNBEAT = "db"
LABEL_AMBIGUOUS = "bR"

# Per-sample file names. Every step reads and writes inside one sample folder.
SCORE_MUSICXML = "score.musicxml"
AUDIO_STEM = "audio"
SOURCE_METADATA = "metadata.json"
TRANSCRIPTION_MIDI = "audio_transkun.mid"
SCORE_MIDI = "score_MS.mid"
PERFORMANCE_CLEAN_MIDI = "performance_clean.mid"
RAW_ALIGNMENT = "performance_align.npz"
REFINED_ALIGNMENT = "performance_refined_align.npz"
SCORE_REFINED_MIDI = "score_refined.mid"
PERFORMANCE_REFINED_MIDI = "performance_refined.mid"
SCORE_ANNOTATIONS = "midi_score_annotations.txt"
PERFORMANCE_ANNOTATIONS = "performance_annotations.txt"
ALIGNMENT_METADATA = "alignment_metadata.json"

# Pipeline bookkeeping; these files are not part of the released dataset.
CHECKS_FILE = "pipeline_checks.json"
WORK_DIR = ".work"
SCORE_TABLE = "score_table.json"
PERFORMANCE_INFO = "performance_info.json"
REFINEMENT_INFO = "refinement_info.json"

AUDIO_EXTENSIONS = (".mp3", ".m4a", ".wav", ".flac", ".ogg")


class ScoreNote(TypedDict, total=False):
    score_note_index: int
    pitch: int
    onset_quarter: float
    duration_quarter: float
    measure: int
    source_measure: int
    playthrough_measure_index: int
    beat_in_measure: float
    staff: int
    voice: int
    id: str
    onset_div: int
    duration_div: int
    divs_pq: int
    time_signature: str
    time_signature_inferred: bool
    key_signature: str
    key_fifths: int
    key_mode: int


class PerformanceNote(TypedDict, total=False):
    performance_note_index: int
    pitch: int
    onset_sec: float
    offset_sec: float
    duration_sec: float
    velocity: int


class ScoreBeat(TypedDict, total=False):
    score_beat_index: int
    onset_quarter: float
    measure: int
    beat_in_measure: float
    is_downbeat: bool
    time_signature: str
    key_fifths: int
    key_mode: int
    is_pickup: bool
    time_signature_inferred: bool
    score_time_sec: float
    time_sec: float
    label: str


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except Exception:
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return default if value is None else int(value)
    except Exception:
        return default


def sort_note_dicts(notes: list[dict], onset_key: str, duration_key: str) -> list[dict]:
    """Sort notes by onset, pitch and duration (the index order used everywhere)."""
    return sorted(
        notes,
        key=lambda note: (
            round(as_float(note.get(onset_key)), 9),
            as_int(note.get("pitch")),
            round(as_float(note.get(duration_key)), 9),
        ),
    )


def beat_key(time_sec: float) -> str:
    return f"{time_sec:.6f}".rstrip("0").rstrip(".")


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="." + path.name,
                                     suffix=".tmp", encoding="utf-8", delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def work_path(sample_dir: Path, name: str) -> Path:
    return Path(sample_dir) / WORK_DIR / name


def find_audio(sample_dir: Path, audio_dir: Path | None = None) -> Path | None:
    """Return ``audio.<ext>`` from the sample folder or a parallel audio tree."""
    sample_dir = Path(sample_dir)
    roots = [sample_dir]
    if audio_dir is not None:
        roots.insert(0, Path(audio_dir) / sample_dir.name)
    for root in roots:
        for ext in AUDIO_EXTENSIONS:
            path = root / f"{AUDIO_STEM}{ext}"
            if path.is_file() and path.stat().st_size:
                return path
    return None


def list_sample_dirs(samples_dir: Path, ids: list[str] | None = None) -> list[Path]:
    samples_dir = Path(samples_dir)
    if ids:
        return [samples_dir / sample_id for sample_id in ids]
    return sorted(
        path for path in samples_dir.iterdir()
        if path.is_dir() and not path.name.startswith((".", "_"))
    )


def release_relative(sample_dir: Path, name: str) -> str:
    """Path as stored in released JSON: ``<dataset folder>/<sample_id>/<file>``."""
    sample_dir = Path(sample_dir).resolve()
    return f"{sample_dir.parent.name}/{sample_dir.name}/{name}"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
