#!/usr/bin/env python3
"""Classify a performance MIDI as expressive or non-expressive, given its score.

    python scripts/classify_expressive.py --score score.musicxml --performance performance.mid
    python scripts/classify_expressive.py --pairs pairs.csv --output results.csv   # columns: score,performance

The score must be MusicXML (staff and metrical information are used). The
performance can be any piano MIDI (e.g. a transcription). The pair is aligned
with the dataset pipeline's preprocessing and DualDTW, then the eight
timing/dynamics features are fed to the released MLP (gangqinpu/classifier/model.joblib).
The label describes perceived expressiveness of timing and dynamics; it does
not identify whether a recording was made by a human.

The classifier assumes the score and performance contain the same music. It
prints the raw insertion/deletion rates; the dataset pipeline only trusted
pairs where both are <= 10 %.
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import _cli  # noqa: F401
from gangqinpu.classifier import classify_pair

warnings.filterwarnings("ignore", module=r"partitura(\..*)?")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", type=Path, help="MusicXML score")
    parser.add_argument("--performance", type=Path, help="Performance MIDI")
    parser.add_argument("--pairs", type=Path, help="CSV with columns score,performance")
    parser.add_argument("--output", type=Path, help="CSV output for --pairs (default: print JSON lines)")
    parser.add_argument("--model", type=Path, help="Alternative model.joblib")
    args = parser.parse_args()
    if args.pairs and (args.score or args.performance):
        parser.error("--pairs cannot be combined with --score or --performance")
    if bool(args.score) != bool(args.performance):
        parser.error("--score and --performance must be provided together")
    if args.output and not args.pairs:
        parser.error("--output requires --pairs")
    kwargs = {"model_path": args.model} if args.model else {}

    if args.score and args.performance:
        result = classify_pair(args.score, args.performance, **kwargs)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if "error" in result else 0
    if not args.pairs:
        parser.error("give --score and --performance, or --pairs")
    with args.pairs.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"score", "performance"}.issubset(reader.fieldnames or []):
            parser.error("pairs CSV must contain score and performance columns")
        pairs = list(reader)
    if not pairs:
        parser.error("pairs CSV contains no data rows")
    results = []
    for pair in pairs:
        try:
            r = classify_pair(Path(pair["score"]), Path(pair["performance"]), **kwargs)
        except Exception as exc:
            r = {"error": f"{type(exc).__name__}: {exc}"}
        row = {"score": pair["score"], "performance": pair["performance"], "label": r.get("label", ""),
               "proba_expressive": r.get("proba_expressive", ""), "insertion_rate": r.get("insertion_rate", ""),
               "deletion_rate": r.get("deletion_rate", ""), "error": r.get("error", "")}
        results.append(row)
        if not args.output:
            print(json.dumps(row, ensure_ascii=False))
    if args.output:
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"wrote {len(results)} rows to {args.output}")
    return 1 if any(r["error"] for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
