"""Score-informed timing and dynamics features for expressive-performance classification.

Features are computed from observed (matched, non-interpolated) note pairs of
a raw score-performance alignment. Explicitly flagged grace notes are excluded;
realized grace/ornament events in the construction score table are ordinary
notes because that table does not preserve grace flags. Notes sharing a notated score onset form a
group whose median performed onset defines its timing; a local linear
score-to-performance time mapping (window +/-4 quarter notes, widened until it
contains at least six onset positions) yields timing residuals.

Insertion/deletion rates are deliberately *not* features: for transcribed
audio they mostly reflect transcription or alignment errors rather than
expressiveness.
"""
from __future__ import annotations

import numpy as np

FEATURES = [
    "tempo_residual_std_ms",         # timing variability
    "chord_spread_med_ms",           # chord asynchrony
    "vel_std",                       # velocity variability
    "vel_unique",                    # velocity diversity
    "voicing_boost",                 # staff contrast
    "pitch_boost",                   # pitch contrast
    "tempo_residual_autocorr_lag1",  # timing correlation
    "onset_entropy",                 # timing entropy
]

MIN_MATCHED_NOTES = 80
MELODY_STAFF = 1                 # staff 1 = upper staff
TEMPO_FIT_WINDOW_QUARTERS = 4.0
TEMPO_FIT_MIN_POINTS = 6

TABLE_DTYPE = [
    ("onset_beat", "f8"), ("duration_beat", "f8"),
    ("voice", "i4"), ("staff", "i4"), ("pitch", "i4"),
    ("is_grace", "?"), ("is_downbeat", "?"),
    ("perf_onset", "f8"), ("perf_duration", "f8"), ("velocity", "f8"),
]


def matched_note_table(score_notes, performance_notes, matched_score, matched_performance) -> np.ndarray:
    """Join matched score/performance notes into one table sorted by score onset."""
    if len(matched_score) != len(matched_performance):
        raise ValueError("raw alignment has unequal score/performance match arrays")
    rows = []
    for score_index, performance_index in zip(matched_score, matched_performance):
        score_index, performance_index = int(score_index), int(performance_index)
        if not (0 <= score_index < len(score_notes) and 0 <= performance_index < len(performance_notes)):
            raise ValueError("raw alignment contains an out-of-range note index")
        s, p = score_notes[score_index], performance_notes[performance_index]
        rows.append((
            float(s["onset_quarter"]), float(s["duration_quarter"]),
            int(s.get("voice", -1)), int(s.get("staff", -1)), int(s["pitch"]),
            bool(s.get("is_grace", False)), bool(s.get("beat_in_measure", 0.0) == 1.0),
            float(p["onset_sec"]), float(p["duration_sec"]), float(p["velocity"]),
        ))
    table = np.array(rows, dtype=TABLE_DTYPE)
    return table[np.argsort(table["onset_beat"], kind="mergesort")]


def local_linear_fit(x: np.ndarray, y: np.ndarray, window: float) -> np.ndarray:
    """Predict y[i] from a linear fit over neighbours within x[i] +/- window."""
    pred = np.empty(len(x))
    for i in range(len(x)):
        w = window
        while True:
            lo = np.searchsorted(x, x[i] - w, side="left")
            hi = np.searchsorted(x, x[i] + w, side="right")
            if hi - lo >= TEMPO_FIT_MIN_POINTS or w > (x[-1] - x[0]):
                break
            w *= 2
        xi, yi = x[lo:hi], y[lo:hi]
        if len(xi) < 2 or xi.std() == 0:
            pred[i] = y[i]
            continue
        b, a = np.polyfit(xi, yi, 1)
        pred[i] = a + b * x[i]
    return pred


def autocorr_lag1(x: np.ndarray) -> float | None:
    if len(x) < 20:
        return None
    a, b = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def compute_features(table: np.ndarray) -> dict:
    """Return the eight model features (None when undefined) plus warnings."""
    r: dict = {"warnings": [], "n_matched": len(table)}
    if len(table) < MIN_MATCHED_NOTES:
        r["warnings"].append(f"too few matched notes ({len(table)}), result unreliable")
    real = table[~table["is_grace"]]

    # Onset groups: notes sharing a notated onset; median performed onset.
    beats_u, inv = np.unique(real["onset_beat"], return_inverse=True)
    group_sec = np.array([np.median(real["perf_onset"][inv == i]) for i in range(len(beats_u))])
    order = np.argsort(beats_u)
    beats_u, group_sec = beats_u[order], group_sec[order]

    # Timing: residuals from a local score-to-performance tempo fit.
    if len(beats_u) >= TEMPO_FIT_MIN_POINTS:
        residual_ms = (group_sec - local_linear_fit(beats_u, group_sec, TEMPO_FIT_WINDOW_QUARTERS)) * 1000
        r["tempo_residual_std_ms"] = round(float(residual_ms.std()), 1)
        lag1 = autocorr_lag1(residual_ms)
        r["tempo_residual_autocorr_lag1"] = None if lag1 is None else round(lag1, 3)
        _, counts = np.unique(np.round(residual_ms / 10.0).astype(int), return_counts=True)
        p = counts / counts.sum()
        r["onset_entropy"] = round(float(-(p * np.log2(p)).sum()), 3)
    else:
        r["warnings"].append("too few chord/note time points, tempo residual missing")
        r["tempo_residual_std_ms"] = r["tempo_residual_autocorr_lag1"] = r["onset_entropy"] = None

    # Chord asynchrony: onset range within notated simultaneities.
    chords = [real["perf_onset"][inv == i] for i in range(len(beats_u))]
    chords = [g for g in chords if len(g) >= 2]
    if len(chords) >= 15:
        r["chord_spread_med_ms"] = round(float(np.median([g.max() - g.min() for g in chords]) * 1000), 1)
    else:
        r["warnings"].append("too few notated chords, chord spread missing")
        r["chord_spread_med_ms"] = None

    # Groups containing both upper-staff and other-staff notes (only a warning
    # is emitted here; kept for consistency with the released annotations).
    both_staves = 0
    for i in range(len(beats_u)):
        staves = real["staff"][inv == i]
        if np.any(staves == MELODY_STAFF) and np.any(staves != MELODY_STAFF):
            both_staves += 1
    if both_staves < 15:
        r["warnings"].append("too few chords with simultaneous melody/accompaniment notes, melody lead missing")

    # Dynamics.
    vels = real["velocity"]
    r["vel_std"] = round(float(vels.std()), 1)
    r["vel_unique"] = int(len(set(vels.tolist())))
    upper = real["velocity"][real["staff"] == MELODY_STAFF]
    other = real["velocity"][real["staff"] != MELODY_STAFF]
    r["voicing_boost"] = round(float(upper.mean() - other.mean()), 1) if len(upper) > 10 and len(other) > 10 else None

    boosts = []
    for i in range(len(beats_u)):
        pitches, chord_vels = real["pitch"][inv == i], real["velocity"][inv == i]
        if len(pitches) < 2:
            continue
        top = np.argmax(pitches)
        boosts.append(chord_vels[top] - np.delete(chord_vels, top).mean())
    r["pitch_boost"] = round(float(np.mean(boosts)), 1) if len(boosts) >= 15 else None
    return r
