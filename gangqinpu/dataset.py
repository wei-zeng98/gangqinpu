"""Reader for the released Gangqinpu dataset.

Expected layout (the Hugging Face release, optionally plus recovered sources)::

    <root>/gangqinpu_meta_v1.0.0.csv
    <root>/gangqinpu_v1.0.0/<sample_id>/...       # released symbolic files
    <root>/gangqinpu_audio/<sample_id>/audio.<ext>   # optional, see tools/download_sources.py

Only ``numpy`` and ``symusic`` are needed. Note indices in the alignment
archives refer to track 0 of ``score_refined.mid`` / ``performance_refined.mid``
in the order returned by symusic, which is what :meth:`GangqinpuSample.score_notes`
and :meth:`GangqinpuSample.performance_notes` return.
"""
from __future__ import annotations

import csv
import json
from urllib.parse import urlsplit
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterator

import numpy as np

METADATA_CSV = "gangqinpu_meta_v1.0.0.csv"
INT_COLUMNS = {"note_count", "score_note_count", "score_bar_count", "annotation_beat_count", "annotation_downbeat_count"}
FLOAT_COLUMNS = {"audio_duration_seconds", "duration_seconds"}


def _source_asset_path(row: dict[str, Any], asset: str) -> Path:
    """Conventional optional recovery target; not a published manifest path."""
    if asset == "audio":
        suffix = Path(urlsplit(row.get("audio_url") or "").path).suffix.lower()
        if suffix not in (".mp3", ".m4a", ".wav", ".flac", ".ogg"):
            suffix = ".mp3"
        return Path("gangqinpu_audio") / row["sample_id"] / f"audio{suffix}"
    if asset == "ccmz":
        return Path(row["sample_dir"]) / "score.ccmz"
    raise ValueError("asset must be 'audio' or 'ccmz'")


def read_metadata(root: str | Path, source: str | None = "gangqinpu") -> list[dict[str, Any]]:
    """Rows of the metadata CSV (IDs kept as strings; empty cells become None)."""
    with (Path(root) / METADATA_CSV).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        if source is not None and row["source"] != source:
            continue
        for key, value in row.items():
            if value == "":
                row[key] = None
            elif key in INT_COLUMNS:
                row[key] = int(float(value))
            elif key in FLOAT_COLUMNS:
                row[key] = float(value)
        out.append(row)
    return out


NOTE_DTYPE = [("time", "f8"), ("duration", "f8"), ("pitch", "i2"), ("velocity", "i2")]


def _midi_notes(path: Path, merge_tracks: bool = False) -> np.ndarray:
    """Track-0 notes, or all non-drum tracks sorted by onset/pitch/duration."""
    from symusic import Score

    score = Score(str(path)).to("second")
    if not score.tracks:
        return np.zeros(0, dtype=NOTE_DTYPE)
    tracks = [t for t in score.tracks if not t.is_drum] if merge_tracks else score.tracks[:1]
    arrays = []
    for track in tracks:
        columns = track.notes.numpy()
        notes = np.zeros(len(columns["pitch"]), dtype=NOTE_DTYPE)
        for name, _ in NOTE_DTYPE:
            notes[name] = columns[name]
        arrays.append(notes)
    notes = np.concatenate(arrays) if arrays else np.zeros(0, dtype=NOTE_DTYPE)
    return notes[np.lexsort((notes["duration"], notes["pitch"], notes["time"]))] if merge_tracks else notes


@dataclass
class GangqinpuSample:
    root: Path
    row: dict[str, Any]

    # ------------------------------------------------------------ identifiers
    @property
    def sample_id(self) -> str:
        return self.row["sample_id"]

    @property
    def split(self) -> str:
        return self.row["split"]

    @property
    def genre(self) -> str:
        return self.row["genre"]

    def path(self, column: str) -> Path | None:
        """Manifest path relative to the dataset root; reference files are external."""
        value = self.row.get(column)
        return self.root / value if value else None

    @property
    def sample_dir(self) -> Path:
        return self.root / self.row["sample_dir"]

    def file(self, name: str) -> Path:
        return self.sample_dir / name

    @property
    def audio_path(self) -> Path | None:
        path = self.root / _source_asset_path(self.row, "audio")
        if path.is_file():
            return path
        for folder in (path.parent, self.sample_dir):
            for candidate in sorted(folder.glob("audio.*")):
                if candidate.suffix.lower() in (".mp3", ".m4a", ".wav", ".flac", ".ogg"):
                    return candidate
        return None

    @property
    def musicxml_path(self) -> Path | None:
        path = self.sample_dir / "score.musicxml"
        return path if path.is_file() else None

    # ------------------------------------------------------------- JSON files
    @cached_property
    def source_metadata(self) -> dict[str, Any]:
        return json.loads(self.file("metadata.json").read_text(encoding="utf-8"))

    @cached_property
    def annotations(self) -> dict[str, Any]:
        """``alignment_metadata.json``: beats, signatures, alignment statistics, classifier output."""
        return json.loads(self.file("alignment_metadata.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ beats
    def beats(self, min_confidence: float = 0.0) -> dict[str, np.ndarray]:
        """Performance-time beats (seconds) with downbeat flags and confidences.

        Uses ``performance_beat_details`` so every beat keeps its provenance.
        Beats are returned in score order; timestamps are not re-sorted.
        """
        details = [b for b in self.annotations["performance_beat_details"] if b["confidence"] >= min_confidence]
        return {
            "time": np.array([b["performance_time_sec"] for b in details], dtype=float),
            "is_downbeat": np.array([b["is_downbeat"] for b in details], dtype=bool),
            "confidence": np.array([b["confidence"] for b in details], dtype=float),
            "measure": np.array([b["measure"] for b in details], dtype=int),
            "label": np.array([b["label"] for b in details]),
            "method": np.array([b["method"] for b in details]),
        }

    def read_annotation_tsv(self, which: str = "performance") -> list[tuple[float, str]]:
        """ASAP-style TSV rows ``(time, label)``. Labels: b, db, bR, meter (e.g. ``4/4``), key fifths."""
        if which not in ("performance", "score"):
            raise ValueError("which must be 'performance' or 'score'")
        name = "performance_annotations.txt" if which == "performance" else "midi_score_annotations.txt"
        rows = []
        for line in self.file(name).read_text(encoding="utf-8").splitlines():
            start, _end, label = line.split("\t")
            rows.append((float(start), label))
        return rows

    # ------------------------------------------------------------------ notes
    def performance_notes(self, stage: str = "refined") -> np.ndarray:
        """Notes of ``performance_refined.mid`` (or ``clean`` / ``transkun`` stage)."""
        name = {"refined": "performance_refined.mid", "clean": "performance_clean.mid",
                "transkun": "audio_transkun.mid"}[stage]
        return _midi_notes(self.file(name))

    def score_notes(self, stage: str = "refined") -> np.ndarray:
        """Notes of ``score_refined.mid`` (or ``score_MS.mid`` with ``stage='ms'``); times in score-MIDI seconds."""
        if stage not in ("refined", "ms"):
            raise ValueError("stage must be 'refined' or 'ms'")
        return _midi_notes(self.file("score_refined.mid" if stage == "refined" else "score_MS.mid"),
                           merge_tracks=stage == "ms")

    def refined_alignment(self) -> dict[str, np.ndarray]:
        """Arrays of ``performance_refined_align.npz`` (indexed by score note)."""
        with np.load(self.file("performance_refined_align.npz"), allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def note_alignment(self, include_interpolated: bool = False) -> dict[str, np.ndarray]:
        """Matched (score note index, performance note index) pairs.

        Indices refer to :meth:`score_notes` and :meth:`performance_notes`.
        Unmatched score notes (index -1) are always dropped; notes interpolated
        by RAScoP are dropped unless ``include_interpolated`` is True.
        """
        data = self.refined_alignment()
        mapping = data["performance_note_indices"]
        keep = mapping >= 0
        if not include_interpolated:
            keep &= ~data["interpolated_mask"]
        score_index = np.flatnonzero(keep)
        return {
            "score_index": score_index,
            "performance_index": mapping[keep],
            "interpolated": data["interpolated_mask"][keep],
            "performance_onset": data["refined_performance_onsets"][keep],
        }


class GangqinpuDataset:
    """Iterable view of the released dataset, optionally filtered by split/genre."""

    def __init__(self, root: str | Path, split: str | list[str] | None = None,
                 genre: str | list[str] | None = None):
        self.root = Path(root)
        splits = {split} if isinstance(split, str) else set(split or [])
        genres = {genre} if isinstance(genre, str) else set(genre or [])
        self.rows = [r for r in read_metadata(self.root)
                     if (not splits or r["split"] in splits) and (not genres or r["genre"] in genres)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> GangqinpuSample:
        return GangqinpuSample(self.root, self.rows[index])

    def __iter__(self) -> Iterator[GangqinpuSample]:
        for row in self.rows:
            yield GangqinpuSample(self.root, row)

    def by_id(self, sample_id: str) -> GangqinpuSample:
        for row in self.rows:
            if row["sample_id"] == str(sample_id):
                return GangqinpuSample(self.root, row)
        raise KeyError(sample_id)
