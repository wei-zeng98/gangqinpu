#!/usr/bin/env python3
"""Step 8 - Score-similarity grouping (optionally jointly with a reference corpus).

Groups every accepted sample by its ``score_MS.mid`` (see
gangqinpu/pipeline/grouping.py for the method). To prevent leakage when the
dataset is combined with another corpus such as ASAP, pass
``--reference-csv`` with columns ``source,sample_id,score_midi,work_id``;
reference scores are grouped jointly and all performances sharing a
``work_id`` stay in one group.

Outputs in --output:
  groups.csv           source, sample_id, score_group_id, group_id
  pair_evidence.jsonl  every verified candidate pair with its relation
  summary.json         counts
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import _cli  # noqa: F401
from gangqinpu.pipeline import grouping as G
from gangqinpu.pipeline.checks import is_rejected, load_checks
from gangqinpu.pipeline.common import SCORE_MIDI, list_sample_dirs

PRIMARY = "gangqinpu"
FAMILIES: list[dict] = []
REFERENCE_FAMILIES: set[int] = set()


def _verify(pair: tuple[int, int]):
    i, j = pair
    fn = G.verify_cross if (i in REFERENCE_FAMILIES or j in REFERENCE_FAMILIES) else G.verify
    return i, j, fn(FAMILIES[i], FAMILIES[j])


def _init(families, reference):
    global FAMILIES, REFERENCE_FAMILIES
    FAMILIES, REFERENCE_FAMILIES = families, reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, help="Optional: source,sample_id,score_midi,work_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    members = [
        {"source": PRIMARY, "sample_id": d.name, "path": str(d / SCORE_MIDI), "work_id": ""}
        for d in list_sample_dirs(args.samples_dir)
        if not is_rejected(d) and load_checks(d).get("expressive", {}).get("passed") is True
    ]
    if not members:
        raise SystemExit("No accepted primary samples; complete filters and classification first.")
    if args.reference_csv:
        with args.reference_csv.open(encoding="utf-8-sig", newline="") as f:
            members += [{"source": r["source"], "sample_id": r["sample_id"], "path": r["score_midi"],
                         "work_id": r.get("work_id", "")} for r in csv.DictReader(f)]
    paths = sorted({m["path"] for m in members})
    print(f"extracting {len(paths)} scores ({len(members)} samples)", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        extracted = dict(zip(paths, pool.map(G.extract, paths, chunksize=16)))
    failures = {p: e["error"] for p, e in extracted.items() if "error" in e}
    if failures:
        for path, error in list(failures.items())[:20]:
            print("unreadable:", path, error)
        raise SystemExit(f"{len(failures)} unreadable score MIDI files; fix or remove them first")

    # Transposition families are the unit of comparison.
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for m in members:
        by_family[extracted[m["path"]]["transposed_hash"]].append(m)
    keys = sorted(by_family)
    families = [extracted[by_family[k][0]["path"]] for k in keys]
    reference = {i for i, k in enumerate(keys) if any(m["source"] != PRIMARY for m in by_family[k])}
    print(f"{len(keys)} transposition families; retrieving candidates", flush=True)
    candidates = G.retrieve(families)
    print(f"verifying {len(candidates)} candidate pairs", flush=True)

    guard = G.UnionFind(len(keys))
    strong: set[tuple[int, int]] = set()
    relations: collections.Counter = collections.Counter()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "pair_evidence.jsonl").open("w") as out, \
            ProcessPoolExecutor(args.workers, initializer=_init, initargs=(families, reference)) as pool:
        for n, (i, j, result) in enumerate(pool.map(_verify, sorted(candidates), chunksize=8), 1):
            result.update(score_a=families[i]["path"], score_b=families[j]["path"], retrieval_cosine=candidates[(i, j)])
            out.write(json.dumps(result) + "\n")
            relations[result["relation"]] += 1
            if result["relation"] in G.SPLIT_RELATIONS:
                guard.join(i, j)
            if result["relation"] == "highly_similar":
                strong.add((i, j))
            if n % 5000 == 0:
                print(f"verified {n}/{len(candidates)}", flush=True)

    # Reference performances of the same work always stay together.
    work_families: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i, k in enumerate(keys):
        for m in by_family[k]:
            if m["work_id"]:
                work_families[(m["source"], m["work_id"])].append(i)
    for ids in work_families.values():
        for i in ids[1:]:
            guard.join(ids[0], i)

    # Score groups: complete-link over highly similar primary-corpus families.
    primary = [i for i, k in enumerate(keys) if any(m["source"] == PRIMARY for m in by_family[k])]
    local = {i: n for n, i in enumerate(primary)}
    edges = {(local[i], local[j]) for i, j in strong if i in local and j in local}
    score_group: dict[str, str] = {}
    for group in G.complete_link_groups(len(primary), edges):
        ids = [m["sample_id"] for n in group for m in by_family[keys[primary[n]]] if m["source"] == PRIMARY]
        name = G.stable_id("score_", ids)
        for sample_id in ids:
            score_group[sample_id] = name
    split_group: dict[tuple[str, str], str] = {}
    for group in guard.groups():
        ids = [(m["source"], m["sample_id"]) for i in group for m in by_family[keys[i]]]
        name = G.stable_id("joint_", [f"{s}:{x}" for s, x in ids])
        for item in ids:
            split_group[item] = name

    rows = [{"source": m["source"], "sample_id": m["sample_id"],
             "score_group_id": score_group.get(m["sample_id"], "") if m["source"] == PRIMARY else "",
             "group_id": split_group[(m["source"], m["sample_id"])]} for m in members]
    with (args.output / "groups.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "samples": len(members),
        "transposition_families": len(keys),
        "candidate_pairs": len(candidates),
        "relations": dict(relations),
        "score_groups": len(set(score_group.values())),
        "split_groups": len(set(split_group.values())),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
