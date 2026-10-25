#!/usr/bin/env python3
"""Minimal tour of the Gangqinpu release.

    python examples/read_dataset.py --root /path/to/gangqinpu   # folder with gangqinpu_meta_v1.0.0.csv
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gangqinpu import GangqinpuDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-id", default=None, help="Sample to inspect (default: first test sample).")
    args = parser.parse_args()

    # 1. Metadata: one row per score-performance pair, with fixed splits.
    dataset = GangqinpuDataset(args.root)
    print(f"{len(dataset)} samples")
    print("splits:", dict(collections.Counter(r["split"] for r in dataset.rows)))
    print("genres:", dict(collections.Counter(r["genre"] for r in dataset.rows).most_common()))
    hours = sum(r["audio_duration_seconds"] or 0 for r in dataset.rows) / 3600
    print(f"audio duration: {hours:.1f} h")

    test = GangqinpuDataset(args.root, split="test")
    sample = dataset.by_id(args.sample_id) if args.sample_id else test[0]
    print(f"\nsample {sample.sample_id}: {sample.row['title']!r} ({sample.genre}, {sample.split})")

    # 2. Symbolic files: refined performance MIDI and refined score MIDI.
    performance = sample.performance_notes()     # structured array: time, duration, pitch, velocity
    score = sample.score_notes()
    print(f"performance notes: {len(performance)}, score notes: {len(score)}")

    # 3. Note-level alignment (observed matches only; RAScoP-interpolated notes excluded).
    pairs = sample.note_alignment()
    print(f"observed note pairs: {len(pairs['score_index'])}")
    for s_idx, p_idx in list(zip(pairs["score_index"], pairs["performance_index"]))[:5]:
        s, p = score[s_idx], performance[p_idx]
        print(f"  score pitch {s['pitch']:3d} @ {s['time']:7.3f}s (score MIDI)  ->  "
              f"performed @ {p['time']:7.3f}s, velocity {p['velocity']}")
    ioi = np.diff(np.unique(performance["time"]))
    print(f"median performed inter-onset interval: {np.median(ioi) * 1000:.0f} ms")

    # 4. Beats and downbeats in performance (audio) time, with confidences.
    beats = sample.beats()
    reliable = beats["confidence"] >= 0.5
    print(f"beats: {len(beats['time'])} ({reliable.sum()} with confidence >= 0.5), "
          f"downbeats: {beats['is_downbeat'].sum()}")
    tsv = [(t, label) for t, label in sample.read_annotation_tsv() if label in ("b", "db", "bR")]
    print("first TSV beat events:", tsv[:4])

    # 5. Alignment statistics and the expressive-performance classifier output.
    ann = sample.annotations
    cls = ann["expressive_performance_classifier"]
    print(f"refined note match rate: {ann['refined_note_match_rate']:.3f}, "
          f"classifier: {cls['label']} (p={cls['proba_expressive']})")

    # 6. Source audio / MusicXML are not distributed; they exist only after recovery.
    print("audio:", sample.audio_path or "not downloaded (see tools/download_sources.py)")
    print("musicxml:", sample.musicxml_path or "not available (convert score.ccmz, see README)")


if __name__ == "__main__":
    main()
