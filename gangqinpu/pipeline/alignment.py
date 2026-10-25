"""Note-level score-performance alignment (DualDTW) and refinement (RAScoP).

Both stages use the SyMuPe implementations: ``ParangonarAligner`` wraps
Parangonar's DualDTW note matcher and ``RAScoP`` refines the alignment,
fills alignment holes and interpolates missing performance notes.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Parangonar imports matplotlib; give it a writable cache directory.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".cache", "matplotlib"))

from symusic import Score
from symupe.data.alignments import Alignment, ParangonarAligner, RAScoP, RAScoPConfig

from .common import (
    PERFORMANCE_REFINED_MIDI,
    SCORE_REFINED_MIDI,
    PerformanceNote,
    ScoreNote,
)
from .performance import performance_notes_from_midi

ALIGNER_BACKEND = "symupe_parangonar_aligner"


def _alignment_indices(alignment: Alignment) -> tuple[np.ndarray, ...]:
    """Split SyMuPe pairs into raw match, deletion and insertion index arrays."""
    matches: list[tuple[int, int]] = []
    deleted: list[int] = []
    inserted: list[int] = []
    for pair in alignment.pairs:
        score_note, perf_note = pair.score_note, pair.perf_note
        if score_note is not None and perf_note is not None:
            matches.append((int(score_note.idx), int(perf_note.idx)))
        elif score_note is not None:
            deleted.append(int(score_note.idx))
        elif perf_note is not None:
            inserted.append(int(perf_note.idx))
    matches = sorted(set(matches))
    matched_score = np.asarray([s for s, _ in matches], dtype=np.int64)
    matched_perf = np.asarray([p for _, p in matches], dtype=np.int64)
    deleted_score = np.asarray(sorted(set(deleted) - set(matched_score.tolist())), dtype=np.int64)
    inserted_perf = np.asarray(sorted(set(inserted) - set(matched_perf.tolist())), dtype=np.int64)
    return matched_score, matched_perf, deleted_score, inserted_perf


def _values(indices: Iterable[int], notes: list[dict], field: str, dtype: Any) -> np.ndarray:
    indices = list(indices)
    if any(i < 0 or i >= len(notes) for i in indices):
        raise ValueError("raw alignment contains an out-of-range note index")
    return np.asarray([notes[i][field] for i in indices], dtype=dtype)


def write_raw_alignment(
    output_path: Path,
    score_notes: list[ScoreNote],
    performance_notes: list[PerformanceNote],
    alignment: Alignment,
) -> dict[str, Any]:
    """Save the DualDTW result as ``performance_align.npz``."""
    matched_score, matched_perf, deleted_score, inserted_perf = _alignment_indices(alignment)
    np.savez_compressed(
        output_path,
        matched_score_indices=matched_score,
        matched_performance_indices=matched_perf,
        deleted_score_indices=deleted_score,
        inserted_performance_indices=inserted_perf,
        matched_score_pitches=_values(matched_score, score_notes, "pitch", np.int16),
        matched_performance_pitches=_values(matched_perf, performance_notes, "pitch", np.int16),
        matched_score_onsets=_values(matched_score, score_notes, "onset_quarter", float),
        matched_performance_onsets=_values(matched_perf, performance_notes, "onset_sec", float),
        deleted_score_pitches=_values(deleted_score, score_notes, "pitch", np.int16),
        deleted_score_onsets=_values(deleted_score, score_notes, "onset_quarter", float),
        inserted_performance_pitches=_values(inserted_perf, performance_notes, "pitch", np.int16),
        inserted_performance_onsets=_values(inserted_perf, performance_notes, "onset_sec", float),
    )
    return load_raw_alignment(output_path)


def load_raw_alignment(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "aligner_backend": ALIGNER_BACKEND,
            "matched_score_indices": data["matched_score_indices"],
            "matched_performance_indices": data["matched_performance_indices"],
            "deleted_score_indices": data["deleted_score_indices"],
            "inserted_performance_indices": data["inserted_performance_indices"],
        }


def alignment_rates(raw_alignment: dict[str, Any], score_count: int, performance_count: int) -> dict[str, float]:
    """Insertion/deletion rates of the raw alignment (before RAScoP interpolation)."""
    matched = len(raw_alignment["matched_score_indices"])
    inserted = len(raw_alignment["inserted_performance_indices"])
    deleted = len(raw_alignment["deleted_score_indices"])
    return {
        "matched_notes": matched,
        "insertion_rate": inserted / performance_count if performance_count else 1.0,
        "deletion_rate": deleted / score_count if score_count else 1.0,
    }


def tempo_jumps(score_notes: list[ScoreNote], refined_onsets: np.ndarray, interpolated_mask: np.ndarray) -> list[dict]:
    """Report (never alter) implausible tempi between neighbouring onset anchors."""
    grouped: dict[float, list[float]] = {}
    for index, note in enumerate(score_notes):
        if interpolated_mask[index] or refined_onsets[index] < 0:
            continue
        grouped.setdefault(round(float(note["onset_quarter"]), 8), []).append(float(refined_onsets[index]))
    anchors = [(onset, float(np.median(times))) for onset, times in sorted(grouped.items())]
    jumps = []
    for (left_onset, left_time), (right_onset, right_time) in zip(anchors, anchors[1:]):
        quarters, seconds = right_onset - left_onset, right_time - left_time
        if quarters <= 0 or seconds <= 0:
            continue
        bpm = 60.0 * quarters / seconds
        if bpm < 15.0 or bpm > 480.0:
            jumps.append({"left_score_onset_quarter": left_onset,
                          "right_score_onset_quarter": right_onset, "bpm": bpm})
    return jumps


def serialized_note_indices(in_memory: Score, serialized: Score) -> np.ndarray:
    """Map track-0 indices to the order obtained by reading the saved MIDI.

    RAScoP can move note onsets without sorting its note list. MIDI writing
    sorts events, so its input order is not a stable on-disk note index.
    """
    before = in_memory.tracks[0].notes.numpy()
    after = serialized.tracks[0].notes.numpy()
    # MIDI note-off events cannot preserve crossing same-pitch durations.
    # Onset, pitch and velocity identify the note-on event being indexed.
    fields = ("time", "pitch", "velocity")
    available = defaultdict(deque)
    for i in range(len(after["pitch"])):
        available[tuple(int(after[k][i]) for k in fields)].append(i)
    mapping = []
    for i in range(len(before["pitch"])):
        key = tuple(int(before[k][i]) for k in fields)
        if not available[key]:
            raise RuntimeError("MIDI serialization changed note-on events")
        mapping.append(available[key].popleft())
    if any(available.values()):
        raise RuntimeError("MIDI serialization changed note count")
    return np.asarray(mapping, dtype=np.int64)


def align_and_refine(
    score_midi_path: Path,
    performance_clean_path: Path,
    score_notes: list[ScoreNote],
    raw_output_path: Path,
    refined_output_path: Path,
    output_dir: Path,
    interpolate_missing: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run DualDTW + RAScoP; write raw/refined NPZ and refined MIDI files."""
    performance_notes = performance_notes_from_midi(performance_clean_path)
    raw_object, _paths, messages = ParangonarAligner().align(str(score_midi_path), str(performance_clean_path))
    if raw_object is None:
        raise RuntimeError("Parangonar alignment failed: " + ("; ".join(messages) if messages else "no detail"))
    raw_alignment = write_raw_alignment(raw_output_path, score_notes, performance_notes, raw_object)

    config = RAScoPConfig(
        score_holes=True,
        performance_holes=True,
        clean_onsets=True,
        interpolate_notes=interpolate_missing,
        synchronize_performance=False,
        min_recall=0.7,
        num_runs=1,
    )
    rascop = RAScoP(score_midi=str(score_midi_path), perf_midi=str(performance_clean_path),
                    alignment=raw_object, config=config, verbose=0)
    refined_object, metrics, stage_times = rascop()
    rascop.score_midi.dump_midi(str(Path(output_dir) / SCORE_REFINED_MIDI))
    rascop.perf_midi.dump_midi(str(Path(output_dir) / PERFORMANCE_REFINED_MIDI))

    # The refined score must keep the score-table note order, otherwise the
    # score-note indices below would not refer to the MusicXML-derived table.
    count = rascop.score_midi.note_num()
    if count != len(score_notes):
        raise RuntimeError(f"Refined score note count differs from score table ({count} != {len(score_notes)})")
    refined_pitches = rascop.score_midi.tracks[0].notes.numpy()["pitch"]
    if not np.array_equal(refined_pitches, np.asarray([n["pitch"] for n in score_notes], dtype=refined_pitches.dtype)):
        raise RuntimeError("RAScoP changed the score-note order")

    pair_to_score, _ = refined_object.match_with_midi(rascop.score_midi, is_score_midi=True)
    pair_to_perf, _ = refined_object.match_with_midi(rascop.perf_midi, is_score_midi=False)
    # RAScoP gives interpolated (Stage-I) performance notes negative IDs.
    pair_interpolated = np.asarray(
        [pair.perf_note is not None and pair.perf_note.idx < 0 for pair in refined_object.pairs], dtype=bool)
    score_to_perf = np.full(count, -1, dtype=np.int64)
    interpolated_mask = np.zeros(count, dtype=bool)
    for pair_index, score_index in enumerate(pair_to_score.astype(np.int64)):
        perf_index = int(pair_to_perf[pair_index])
        if score_index < 0 or perf_index < 0:
            continue
        score_to_perf[score_index] = perf_index
        interpolated_mask[score_index] = bool(pair_interpolated[pair_index])

    saved_score = Score(str(Path(output_dir) / SCORE_REFINED_MIDI))
    saved_performance = Score(str(Path(output_dir) / PERFORMANCE_REFINED_MIDI))
    score_order = serialized_note_indices(rascop.score_midi, saved_score)
    if not np.array_equal(score_order, np.arange(count)):
        raise RuntimeError("MIDI serialization changed the score-table note order")
    performance_order = serialized_note_indices(rascop.perf_midi, saved_performance)
    matched = score_to_perf >= 0
    score_to_perf[matched] = performance_order[score_to_perf[matched]]
    perf_seconds = saved_performance.to("second").tracks[0].notes.numpy()
    refined_onsets = np.full(count, -1.0, dtype=float)
    valid = score_to_perf >= 0
    refined_onsets[valid] = perf_seconds["time"][score_to_perf[valid]]
    real_mask = valid & ~interpolated_mask
    real_score = np.flatnonzero(real_mask).astype(np.int64)
    real_perf = score_to_perf[real_mask].astype(np.int64)

    metrics_payload = {str(k): v for k, v in asdict(metrics).items()}
    stage_times_payload = {str(k): v for k, v in asdict(stage_times).items()}
    recall = float(metrics.recall.cleaner if metrics.recall.cleaner is not None else metrics.recall.aligner or 0.0)
    precision = float(metrics.precision.cleaner if metrics.precision.cleaner is not None
                      else metrics.precision.aligner or 0.0)
    np.savez_compressed(
        refined_output_path,
        performance_note_indices=score_to_perf,
        interpolated_mask=interpolated_mask,
        score_note_source_indices=np.arange(count, dtype=np.int64),
        refined_performance_onsets=refined_onsets,
        # RAScoP fills holes internally; its interpolated notes mark them.
        alignment_hole_mask=interpolated_mask,
        real_matched_score_indices=real_score,
        real_matched_performance_indices=real_perf,
        rascop_recall=np.asarray(recall, dtype=float),
        rascop_precision=np.asarray(precision, dtype=float),
        rascop_metrics=np.asarray(json.dumps(metrics_payload, sort_keys=True)),
        rascop_stage_times=np.asarray(json.dumps(stage_times_payload, sort_keys=True)),
    )
    jumps = tempo_jumps(score_notes, refined_onsets, interpolated_mask)
    info = {
        "tempo_jump_count": len(jumps),
        "tempo_jumps": jumps,
        "rascop_recall": recall,
        "rascop_precision": precision,
        "rascop_metrics": metrics_payload,
        "rascop_stage_times": stage_times_payload,
        "num_interpolated_notes": int(interpolated_mask.sum()),
    }
    return raw_alignment, info


def load_refined_alignment(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in (
            "performance_note_indices", "interpolated_mask", "refined_performance_onsets",
            "score_note_source_indices", "alignment_hole_mask",
            "real_matched_score_indices", "real_matched_performance_indices")}
