"""Expressive vs. non-expressive performance classifier (inference).

The model is a scikit-learn pipeline (StandardScaler + MLP with 32 and 16
hidden units) over the eight features in :mod:`gangqinpu.classifier.features`.
It predicts perceived expressiveness of timing and dynamics, *not* whether a
recording was produced by a human pianist.
"""
from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from .features import FEATURES, compute_features, matched_note_table

MODEL_PATH = Path(__file__).with_name("model.joblib")
THRESHOLD = 0.5


@lru_cache(maxsize=2)
def load_model(path: str | Path = MODEL_PATH):
    return joblib.load(path)


def classify_alignment(score_notes, performance_notes, raw_alignment, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Classify from an existing raw alignment (as produced by the pipeline)."""
    from ..pipeline.alignment import alignment_rates

    rates = alignment_rates(raw_alignment, len(score_notes), len(performance_notes))
    table = matched_note_table(score_notes, performance_notes,
                               raw_alignment["matched_score_indices"], raw_alignment["matched_performance_indices"])
    values = compute_features(table)
    missing = [name for name in FEATURES if values.get(name) is None]
    if missing:
        return {**rates, "error": f"missing features: {missing}", "n_matched": len(table)}
    proba = float(load_model(str(model_path)).predict_proba([[values[name] for name in FEATURES]])[0, 1])
    return {
        **rates,
        "label": "expressive" if proba >= THRESHOLD else "non-expressive",
        "proba_expressive": round(proba, 4),
        "n_matched": len(table),
        "features": {name: values[name] for name in FEATURES},
        "warnings": values["warnings"],
    }


def classify_pair(score_musicxml: Path, performance_midi: Path, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Classify one performance MIDI against its MusicXML score.

    Runs the same preprocessing as the dataset pipeline (score unfolding and
    score MIDI export, performance cleanup, DualDTW note alignment) in a
    temporary directory, then extracts features and applies the model.
    """
    from symupe.data.alignments import ParangonarAligner

    from ..pipeline.alignment import write_raw_alignment
    from ..pipeline.performance import clean_performance_midi
    from ..pipeline.score import process_score

    with tempfile.TemporaryDirectory(prefix="gangqinpu-cls-") as tmp:
        tmp_dir = Path(tmp)
        score = process_score(Path(score_musicxml), tmp_dir)
        performance = clean_performance_midi(Path(performance_midi), tmp_dir)
        alignment, _paths, messages = ParangonarAligner().align(
            score["score_midi_path"], performance["performance_clean_path"])
        if alignment is None:
            raise RuntimeError("alignment failed: " + "; ".join(messages or ["no detail"]))
        raw = write_raw_alignment(tmp_dir / "raw.npz", score["score_notes"],
                                  performance["performance_notes"], alignment)
        return classify_alignment(score["score_notes"], performance["performance_notes"], raw, model_path)
