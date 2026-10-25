"""Performance MIDI cleanup (``performance_clean.mid``) and note-table reading."""
from __future__ import annotations

from pathlib import Path

from symusic import Score
from symupe.data.midi import preprocess_midi

from .common import PERFORMANCE_CLEAN_MIDI, PerformanceNote, sort_note_dicts


def performance_notes_from_midi(midi_path: Path) -> list[PerformanceNote]:
    """Read track 0 of a single-track MIDI as notes sorted by onset/pitch/duration."""
    midi_sec = Score(str(midi_path)).to("second")
    if not midi_sec.tracks:
        return []
    notes_sec = midi_sec.tracks[0].notes.numpy()
    notes: list[PerformanceNote] = [
        {
            "pitch": int(notes_sec["pitch"][i]),
            "onset_sec": float(notes_sec["time"][i]),
            "offset_sec": float(notes_sec["time"][i] + notes_sec["duration"][i]),
            "duration_sec": float(notes_sec["duration"][i]),
            "velocity": int(notes_sec["velocity"][i]),
        }
        for i in range(len(notes_sec["time"]))
    ]
    notes = sort_note_dicts(notes, "onset_sec", "duration_sec")
    for index, note in enumerate(notes):
        note["performance_note_index"] = index
    return notes


def clean_performance_midi(performance_path: Path, output_dir: Path, min_duration_sec: float = 0.03) -> dict:
    """SyMuPe/PianoCoRe-compatible cleanup of a transcribed performance.

    Merges tracks, removes duplicate and implausibly short notes and truncates
    overlapping notes of the same pitch. The expressive time axis is kept.
    """
    output_path = Path(output_dir) / PERFORMANCE_CLEAN_MIDI
    raw_note_count = Score(str(performance_path)).note_num()
    cleaned = preprocess_midi(
        Score(str(performance_path)),
        to_single_track=True,
        clean_duplicates=True,
        cut_overlapped_notes=True,
        clean_short_notes=True,
        min_tick_shift=10,
        min_tick_duration=max(1, round(480 * min_duration_sec * 2)),
        target_ticks_per_quarter=480,
    )
    cleaned.dump_midi(str(output_path))
    notes = performance_notes_from_midi(output_path)
    return {
        "performance_clean_path": str(output_path),
        "performance_notes": notes,
        "num_raw_notes": raw_note_count,
        "num_clean_notes": len(notes),
    }
