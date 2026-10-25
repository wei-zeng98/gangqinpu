#!/usr/bin/env python3
"""Step 9 - Leakage-aware train/validation/test split.

Assigns whole split groups from step 8 (never splitting a group), balancing
source counts and genre labels (step 7). Optionally:

  --heldout-size N      fixed validation and test size for Gangqinpu
  --protect-test FILE   "source:sample_id" per line; their groups go to test
                        (e.g. the test pieces of an external benchmark)

Afterwards, exact duplicate audio files are checked across partitions: a
held-out sample whose audio is byte-identical to a training sample's audio is
excluded (split = ``excluded``); the training copy is kept.

Output: --output/splits.csv (source, sample_id, split, group_id) and
--output/split_summary.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import _cli  # noqa: F401
from gangqinpu.pipeline.common import find_audio, read_json, work_path
from gangqinpu.pipeline.splits import SPLITS, assign_groups

PRIMARY = "gangqinpu"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--groups", type=Path, required=True, help="groups.csv from step 8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heldout-size", type=int)
    parser.add_argument("--protect-test", type=Path)
    parser.add_argument("--manifest", type=Path, help="Reuse fixed splits from a released metadata CSV")
    args = parser.parse_args()

    with args.groups.open(newline="") as f:
        rows = list(csv.DictReader(f))
    groups: dict[str, dict] = collections.defaultdict(
        lambda: {"primary": 0, "reference": 0, "genres": collections.Counter()})
    for r in rows:
        g = groups[r["group_id"]]
        if r["source"] == PRIMARY:
            g["primary"] += 1
            genre_file = work_path(args.samples_dir / r["sample_id"], "genre.json")
            g["genres"][read_json(genre_file)["genre"] if genre_file.is_file() else "unknown"] += 1
        else:
            g["reference"] += 1

    protected: set[str] = set()
    if args.protect_test:
        wanted = {line.strip() for line in args.protect_test.read_text().splitlines() if line.strip()}
        protected = {r["group_id"] for r in rows if f"{r['source']}:{r['sample_id']}" in wanted}
    if args.manifest:
        if args.heldout_size is not None or args.protect_test:
            parser.error("--manifest cannot be combined with split-generation constraints")
        with args.manifest.open(encoding="utf-8-sig", newline="") as f:
            fixed = {(r["source"], r["sample_id"]): r for r in csv.DictReader(f)}
        group_splits = collections.defaultdict(set)
        for r in rows:
            entry = fixed.get((r["source"], r["sample_id"]))
            if entry is None or entry["split"] not in SPLITS:
                raise ValueError(f"missing fixed split: {r['source']}:{r['sample_id']}")
            r["split"] = entry["split"]
            group_splits[r["group_id"]].add(r["split"])
        if any(len(v) > 1 for v in group_splits.values()):
            raise ValueError("fixed manifest splits a detected group")
        report = {"method": "fixed_manifest", "manifest_sha256": sha256(args.manifest)}
    else:
        assignment, report = assign_groups(dict(groups), args.seed, args.heldout_size, protected)
        for r in rows:
            r["split"] = assignment[r["group_id"]]

    # Exact duplicate audio across partitions: keep the training copy.
    by_hash: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if r["source"] == PRIMARY:
            audio = find_audio(args.samples_dir / r["sample_id"], args.audio_dir)
            if audio is None:
                raise FileNotFoundError(f"{r['sample_id']}: audio is required for duplicate screening")
            by_hash[sha256(audio)].append(r)
    excluded = []
    for duplicates in by_hash.values():
        if len({r["split"] for r in duplicates}) > 1 and any(r["split"] == "train" for r in duplicates):
            if args.manifest:
                raise ValueError("source audio conflicts with the fixed manifest: duplicate across splits")
            for r in duplicates:
                if r["split"] != "train":
                    r["split"] = "excluded"
                    excluded.append(f"{r['source']}:{r['sample_id']}")

    if args.manifest:
        for r in rows:
            r["group_id"] = fixed[(r["source"], r["sample_id"])]["group_id"]
        fixed_groups = collections.defaultdict(set)
        for r in rows:
            if r["split"] in SPLITS:
                fixed_groups[r["group_id"]].add(r["split"])
        if any(len(v) > 1 for v in fixed_groups.values()):
            raise ValueError("fixed manifest has a group crossing splits")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "splits.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "sample_id", "split", "group_id"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in rows)
    counts = {src: {s: sum(r["source"] == src and r["split"] == s for r in rows) for s in SPLITS + ["excluded"]}
              for src in sorted({r["source"] for r in rows})}
    summary = {"seed": args.seed, "counts": counts, "excluded_duplicate_audio": excluded, "balance": report}
    (args.output / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"counts": counts, "excluded_duplicate_audio": excluded}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
