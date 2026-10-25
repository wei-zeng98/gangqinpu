"""Beat/downbeat annotation transfer from score time to performance time.

A metrical grid extracted from the unfolded score is mapped to performance
seconds through the refined note correspondences (ASAP-style), with a heuristic
confidence per beat. Outputs follow ASAP's TSV and JSON conventions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import (
    LABEL_AMBIGUOUS,
    LABEL_BEAT,
    LABEL_DOWNBEAT,
    PERFORMANCE_ANNOTATIONS,
    SCORE_ANNOTATIONS,
    ScoreBeat,
    ScoreNote,
    as_float,
    beat_key,
)

OnsetAnchor = dict
PerformanceBeat = dict


def write_asap_tsv(path: Path, beats: list[dict], extra_labels: list[tuple[float, str]] | None = None) -> None:
    """Write ASAP-compatible beat, meter and key-signature rows (start, end, label)."""
    rows: list[tuple[float, str]] = []
    for beat in beats:
        time = as_float(beat.get("time_sec"))
        label = LABEL_DOWNBEAT if beat.get("is_downbeat") else beat.get("label", LABEL_BEAT)
        rows.append((time, str(label)))
    rows.extend(extra_labels or [])
    rows.sort(key=lambda row: (row[0], row[1]))
    lines = [f"{time:.6f}\t{time:.6f}\t{label}" for time, label in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _time_signature_value(time_signature: str) -> list:
    """Encode ASAP's meter value, including compound-meter beat counts."""
    numerator_text, denominator_text = time_signature.split("/", 1)
    numerator, denominator = int(numerator_text), int(denominator_text)
    beat_count = (
        numerator // 3
        if denominator == 8 and numerator > 3 and numerator % 3 == 0
        else numerator
    )
    return [time_signature, beat_count]


def _key_signature_value(beat: ScoreBeat) -> list:
    """Encode ASAP's [major-tonic, fifths] representation from MusicXML data."""
    fifths = int(beat.get("key_fifths", 0))
    return [(7 * fifths) % 12, fifths]


def _signature_changes(
    score_beats: list[ScoreBeat],
    performance_beats: list[PerformanceBeat],
) -> tuple[
    dict,
    dict,
    dict,
    dict,
    list[tuple[float, str]],
    list[tuple[float, str]],
]:
    """Build ASAP JSON maps and matching TSV rows for meter/key changes."""
    score_time_signatures: dict = {}
    performance_time_signatures: dict = {}
    score_key_signatures: dict = {}
    performance_key_signatures: dict = {}
    score_tsv: list[tuple[float, str]] = []
    performance_tsv: list[tuple[float, str]] = []
    previous_meter: str | None = None
    previous_key: tuple[int, int] | None = None

    for score_beat, performance_beat in zip(score_beats, performance_beats):
        score_time = float(score_beat["score_time_sec"])
        performance_time = float(performance_beat["performance_time_sec"])
        meter = str(score_beat["time_signature"])
        key = (int(score_beat.get("key_fifths", 0)), int(score_beat.get("key_mode", 0)))
        if meter != previous_meter:
            meter_value = _time_signature_value(meter)
            score_time_signatures[beat_key(score_time)] = meter_value
            performance_time_signatures[beat_key(performance_time)] = meter_value
            score_tsv.append((score_time, meter))
            performance_tsv.append((performance_time, meter))
            previous_meter = meter
        if key != previous_key:
            key_value = _key_signature_value(score_beat)
            score_key_signatures[beat_key(score_time)] = key_value
            performance_key_signatures[beat_key(performance_time)] = key_value
            score_tsv.append((score_time, str(key[0])))
            performance_tsv.append((performance_time, str(key[0])))
            previous_key = key
    return (
        score_time_signatures,
        performance_time_signatures,
        score_key_signatures,
        performance_key_signatures,
        score_tsv,
        performance_tsv,
    )


def _interp_from_anchors(
    score_onset: float, anchors: list[OnsetAnchor]
) -> tuple[float, str, float]:
    """Map a score beat with no note anchor through neighboring onset anchors."""
    if not anchors:
        return 0.0, "extrapolated", 0.0
    if score_onset <= anchors[0]["score_onset_quarter"]:
        return _extrapolate_anchor_time(score_onset, anchors, from_start=True)
    if score_onset >= anchors[-1]["score_onset_quarter"]:
        return _extrapolate_anchor_time(score_onset, anchors, from_start=False)
    for left, right in zip(anchors, anchors[1:]):
        if left["score_onset_quarter"] <= score_onset <= right["score_onset_quarter"]:
            span = right["score_onset_quarter"] - left["score_onset_quarter"]
            if span <= 1e-9:
                return float(left["performance_time_sec"]), "extrapolated", 0.25
            ratio = (score_onset - left["score_onset_quarter"]) / span
            time = left["performance_time_sec"] + ratio * (
                right["performance_time_sec"] - left["performance_time_sec"]
            )
            confidence = (
                min(
                    float(left.get("confidence", 0.5)),
                    float(right.get("confidence", 0.5)),
                )
                * 0.75
            )
            return float(time), "interpolated_between_anchors", confidence
    return float(anchors[-1]["performance_time_sec"]), "extrapolated", 0.25


def _extrapolate_anchor_time(
    score_onset: float,
    anchors: list[OnsetAnchor],
    *,
    from_start: bool,
) -> tuple[float, str, float]:
    """Continue the nearest reliable tempo instead of collapsing tail beats.

    A final partial measure can have fewer score beats. The score
    grid handles that case directly. If a remaining explicit beat still lies
    beyond the last note anchor, use the local anchor slope so multiple beats
    cannot be emitted at effectively the same performance instant.
    """
    ordered = anchors if from_start else list(reversed(anchors))
    edge = ordered[0]
    for neighbor in ordered[1:]:
        score_delta = float(neighbor["score_onset_quarter"]) - float(
            edge["score_onset_quarter"]
        )
        time_delta = float(neighbor["performance_time_sec"]) - float(
            edge["performance_time_sec"]
        )
        if abs(score_delta) > 1e-9 and time_delta / score_delta > 1e-9:
            seconds_per_quarter = time_delta / score_delta
            estimate = (
                float(edge["performance_time_sec"])
                + (score_onset - float(edge["score_onset_quarter"]))
                * seconds_per_quarter
            )
            return float(max(0.0, estimate)), "extrapolated", 0.25
    return float(edge["performance_time_sec"]), "extrapolated", 0.25


def build_onset_anchors(
    score_notes: list[ScoreNote],
    refined: dict,
) -> list[OnsetAnchor]:
    """Build the bridge from refined note alignment to beat/downbeat mapping."""
    onsets: dict[float, list[int]] = {}
    for idx, note in enumerate(score_notes):
        onsets.setdefault(round(float(note["onset_quarter"]), 8), []).append(idx)

    anchors: list[OnsetAnchor] = []
    interpolated_mask = refined["interpolated_mask"]
    refined_onsets = refined["refined_performance_onsets"]
    hole_mask = refined["alignment_hole_mask"]
    for score_onset, indices in sorted(onsets.items()):
        times = [
            float(refined_onsets[idx])
            for idx in indices
            if refined_onsets[idx] > 0 or score_onset == 0
        ]
        if not times:
            continue
        num_interpolated = int(sum(bool(interpolated_mask[idx]) for idx in indices))
        num_real = len(indices) - num_interpolated
        confidence = 0.95 if num_real else 0.35
        if num_real and num_interpolated:
            confidence = max(0.4, num_real / len(indices))
        if any(bool(hole_mask[idx]) for idx in indices):
            confidence *= 0.35
        anchors.append(
            {
                "score_onset_quarter": float(score_onset),
                "performance_time_sec": float(np.median(times)),
                "num_real_notes": num_real,
                "num_interpolated_notes": num_interpolated,
                "confidence": float(confidence),
            }
        )
    return anchors


def map_beats_to_performance(
    score_notes: list[ScoreNote],
    score_beats: list[ScoreBeat],
    refined: dict,
    anchors: list[OnsetAnchor],
    nearby_note_tolerance_quarter: float = 0.05,
) -> list[PerformanceBeat]:
    """Map ASAP-style score beats/downbeats into performance seconds."""
    refined_onsets = refined["refined_performance_onsets"]
    interpolated_mask = refined["interpolated_mask"]
    hole_mask = refined["alignment_hole_mask"]
    performance_beats: list[PerformanceBeat] = []

    for beat in score_beats:
        score_onset = float(beat["onset_quarter"])
        nearby = [
            idx
            for idx, note in enumerate(score_notes)
            if abs(float(note["onset_quarter"]) - score_onset)
            <= nearby_note_tolerance_quarter
        ]
        valid = [idx for idx in nearby if refined_onsets[idx] > 0 or score_onset == 0]
        inside_hole = any(bool(hole_mask[idx]) for idx in nearby) if nearby else False

        # Prefer notes at the beat onset. If no local note exists, use the
        # nearest score-onset anchors, as ASAP-style annotation transfer needs.
        if valid:
            time = float(np.median([refined_onsets[idx] for idx in valid]))
            num_real = sum(not bool(interpolated_mask[idx]) for idx in valid)
            confidence = 0.95 if num_real else 0.4
            method = "matched_anchor" if num_real else "interpolated_between_anchors"
        else:
            time, method, confidence = _interp_from_anchors(score_onset, anchors)

        if inside_hole:
            method = "inside_alignment_hole"
            confidence *= 0.35

        label = "db" if beat["is_downbeat"] else "b"
        if confidence < 0.35:
            label = LABEL_AMBIGUOUS
        performance_beats.append(
            {
                "measure": beat["measure"],
                "beat_in_measure": beat["beat_in_measure"],
                "score_onset_quarter": score_onset,
                "performance_time_sec": time,
                "time_sec": time,
                "is_downbeat": bool(beat["is_downbeat"]),
                "is_pickup": bool(beat.get("is_pickup", False)),
                "time_signature_inferred": bool(
                    beat.get("time_signature_inferred", False)
                ),
                "method": method,
                "confidence": float(confidence),
                "label": label,
            }
        )
    return performance_beats


def write_annotations(
    output_dir: Path,
    score_notes: list[ScoreNote],
    performance_notes: list[dict],
    score_beats: list[ScoreBeat],
    raw_alignment: dict,
    refined: dict,
    paths: dict[str, str],
) -> dict:
    """Write both ASAP TSV files and return the annotation part of the JSON.

    ``refined`` holds the refined-alignment arrays plus ``tempo_jump_count``.
    """
    output_dir = Path(output_dir)
    anchors = build_onset_anchors(score_notes, refined)
    performance_beats = map_beats_to_performance(
        score_notes, score_beats, refined, anchors
    )
    score_annotation_beats: list[ScoreBeat] = [
        {
            **beat,
            "time_sec": float(beat.get("score_time_sec", beat["onset_quarter"] * 0.5)),
            "label": "db" if beat["is_downbeat"] else "b",
        }
        for beat in score_beats
    ]

    (
        midi_score_time_signatures,
        performance_time_signatures,
        midi_score_key_signatures,
        performance_key_signatures,
        score_signature_rows,
        performance_signature_rows,
    ) = _signature_changes(score_annotation_beats, performance_beats)
    write_asap_tsv(
        output_dir / SCORE_ANNOTATIONS,
        score_annotation_beats,
        score_signature_rows,
    )
    write_asap_tsv(
        output_dir / PERFORMANCE_ANNOTATIONS,
        performance_beats,
        performance_signature_rows,
    )

    # Metadata mirrors ASAP fields and adds confidence scores for QA and review (like PianoCoRe/RAScoP)
    performance_times = [
        float(beat["performance_time_sec"]) for beat in performance_beats
    ]
    performance_downbeats = [
        float(beat["performance_time_sec"])
        for beat in performance_beats
        if beat["is_downbeat"]
    ]
    midi_score_times = [float(beat["time_sec"]) for beat in score_annotation_beats]
    midi_score_downbeats = [
        float(beat["time_sec"])
        for beat in score_annotation_beats
        if beat["is_downbeat"]
    ]
    downbeats_monotonic = all(
        b > a for a, b in zip(performance_downbeats, performance_downbeats[1:])
    )
    low_confidence = [
        beat
        for beat in performance_beats
        if float(beat["confidence"]) < 0.5 or beat["method"] == "inside_alignment_hole"
    ]
    low_confidence_downbeats = [beat for beat in low_confidence if beat["is_downbeat"]]
    inferred_meter_measures = {
        (int(beat["measure"]), float(beat["score_onset_quarter"]))
        for beat in performance_beats
        if beat.get("time_signature_inferred", False) and beat["is_downbeat"]
    }
    raw_match_rate = len(raw_alignment["matched_score_indices"]) / max(
        len(score_notes), 1
    )
    refined_real_match_rate = len(refined["real_matched_score_indices"]) / max(
        len(score_notes), 1
    )
    performance_usage_rate = len(
        set(raw_alignment["matched_performance_indices"].tolist())
    ) / max(len(performance_notes), 1)
    hole_mask = refined["alignment_hole_mask"]
    hole_lengths = _contiguous_true_lengths(hole_mask)
    metadata: dict = {
        "performance_beats": performance_times,
        "performance_downbeats": performance_downbeats,
        "performance_beats_type": {
            beat_key(float(beat["performance_time_sec"])): (
                "db" if beat["is_downbeat"] else beat.get("label", "b")
            )
            for beat in performance_beats
        },
        "midi_score_beats": midi_score_times,
        "midi_score_downbeats": midi_score_downbeats,
        "midi_score_beats_type": {
            beat_key(float(beat["time_sec"])): ("db" if beat["is_downbeat"] else "b")
            for beat in score_annotation_beats
        },
        "perf_time_signatures": performance_time_signatures,
        "perf_key_signatures": performance_key_signatures,
        "midi_score_time_signatures": midi_score_time_signatures,
        "midi_score_key_signatures": midi_score_key_signatures,
        "downbeats_score_map": [
            int(beat["measure"])
            - min(int(candidate["measure"]) for candidate in score_beats)
            for beat in performance_beats
            if beat["is_downbeat"]
        ],
        "score_and_performance_aligned": bool(
            downbeats_monotonic and refined_real_match_rate >= 0.70
        ),
        "raw_note_match_rate": raw_match_rate,
        "refined_note_match_rate": refined_real_match_rate,
        "performance_note_usage_rate": performance_usage_rate,
        "num_alignment_holes": len(hole_lengths),
        "largest_alignment_hole_length": max(hole_lengths) if hole_lengths else 0,
        "num_low_confidence_beats": len(low_confidence),
        "num_low_confidence_downbeats": len(low_confidence_downbeats),
        "num_inferred_time_signature_measures": len(inferred_meter_measures),
        "num_tempo_jumps": refined["tempo_jump_count"],
        "whether_downbeats_are_monotonic": downbeats_monotonic,
        "onset_anchors": anchors,
        "performance_beat_details": performance_beats,
        "paths": paths,
    }
    return metadata


def _contiguous_true_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in mask.tolist():
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths
