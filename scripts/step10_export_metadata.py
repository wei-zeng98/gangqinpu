#!/usr/bin/env python3
"""Step 10 - Export the dataset manifest CSV (same columns as gangqinpu_meta_v1.0.0.csv).

One row per accepted Gangqinpu sample that has a split (excluded duplicates are
dropped). Paths are relative to the dataset root, i.e. the parent of
--samples-dir. ``audio_url`` / ``score_ccmz_url`` are filled from the required
--source-urls CSV (sample_id,audio_url,score_ccmz_url).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _cli  # noqa: F401
from gangqinpu.pipeline.checks import load_checks
from gangqinpu.pipeline.common import (
    ALIGNMENT_METADATA, PERFORMANCE_REFINED_MIDI, REFINED_ALIGNMENT, SCORE_MIDI,
    SOURCE_METADATA, read_json, work_path,
)

COLUMNS = [
    "sample_id", "source", "split", "group_id", "score_group_id", "title", "composer", "genre", "genre_source",
    "audio_duration_seconds", "duration_seconds", "note_count", "score_note_count", "score_bar_count",
    "annotation_beat_count", "annotation_downbeat_count", "sample_dir", "performance_midi_path",
    "score_midi_path", "alignment_path", "annotations_path",
    "metadata_path", "source_page_url", "audio_url", "score_ccmz_url",
]


def midi_stats(path: Path) -> tuple[int, float]:
    """Positive-duration non-drum note count and last note offset in seconds."""
    from symusic import Score
    score = Score(str(path), ttype="second")
    notes = [n for t in score.tracks if not t.is_drum for n in t.notes if n.duration > 0]
    if not notes:
        raise ValueError(f"MIDI has no positive-duration notes: {path.name}")
    return len(notes), max(n.time + n.duration for n in notes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True, help="splits.csv from step 9")
    parser.add_argument("--groups", type=Path, required=True, help="groups.csv from step 8")
    parser.add_argument("--source-urls", type=Path, required=True, help="Source URL CSV or the released metadata CSV")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-metadata", type=Path, help="Preserve ASAP reference rows from the released CSV")
    args = parser.parse_args()

    with args.splits.open(newline="") as f:
        splits = {r["sample_id"]: r for r in csv.DictReader(f) if r["source"] == "gangqinpu"}
    with args.groups.open(newline="") as f:
        score_groups = {r["sample_id"]: r["score_group_id"] for r in csv.DictReader(f) if r["source"] == "gangqinpu"}
    urls: dict[str, dict] = {}
    with args.source_urls.open(encoding="utf-8-sig", newline="") as f:
        urls = {r["sample_id"]: r for r in csv.DictReader(f) if r.get("source", "gangqinpu") == "gangqinpu"}

    dataset_root = args.samples_dir.resolve().parent
    if args.output.resolve().parent != dataset_root:
        parser.error("--output must be inside the dataset root (parent of --samples-dir)")
    root = args.samples_dir.resolve().name
    rows = []
    for sample_id, split in sorted(splits.items()):
        if split["split"] not in ("train", "val", "test"):
            continue
        sample = args.samples_dir / sample_id
        checks = load_checks(sample)
        required = ("audio_duration", "score_bar_count", "piano_only", "alignment_error_rate", "expressive")
        if any(checks.get(name, {}).get("passed") is not True for name in required):
            raise ValueError(f"{sample_id}: filters are incomplete or rejected")
        if not urls.get(sample_id, {}).get("audio_url") or not urls.get(sample_id, {}).get("score_ccmz_url"):
            raise ValueError(f"{sample_id}: missing source URLs")
        meta = read_json(sample / SOURCE_METADATA) if (sample / SOURCE_METADATA).is_file() else {}
        genre = read_json(work_path(sample, "genre.json"))
        annotations = read_json(sample / ALIGNMENT_METADATA)
        note_count, duration = midi_stats(sample / PERFORMANCE_REFINED_MIDI)
        score_note_count, _ = midi_stats(sample / SCORE_MIDI)
        rel = f"{root}/{sample_id}"
        rows.append({
            "sample_id": sample_id, "source": "gangqinpu", "split": split["split"], "group_id": split["group_id"],
            "score_group_id": score_groups.get(sample_id, ""), "title": meta.get("title", ""), "composer": "",
            "genre": genre["genre"], "genre_source": genre["genre_source"],
            "audio_duration_seconds": checks.get("audio_duration", {}).get("details", {}).get("duration_seconds", ""),
            "duration_seconds": duration, "note_count": note_count, "score_note_count": score_note_count,
            "score_bar_count": checks.get("score_bar_count", {}).get("details", {}).get("bar_count", ""),
            "annotation_beat_count": len(annotations["performance_beats"]),
            "annotation_downbeat_count": len(annotations["performance_downbeats"]),
            "sample_dir": rel,
            "performance_midi_path": f"{rel}/{PERFORMANCE_REFINED_MIDI}",
            "score_midi_path": f"{rel}/{SCORE_MIDI}",
            "alignment_path": f"{rel}/{REFINED_ALIGNMENT}",
            "annotations_path": f"{rel}/{ALIGNMENT_METADATA}",
            "metadata_path": f"{rel}/{SOURCE_METADATA}",
            "source_page_url": meta.get("detail_url", ""),
            "audio_url": urls.get(sample_id, {}).get("audio_url", ""),
            "score_ccmz_url": urls.get(sample_id, {}).get("score_ccmz_url", ""),
        })
    if args.reference_metadata:
        with args.reference_metadata.open(encoding="utf-8-sig", newline="") as f:
            references = [r for r in csv.DictReader(f) if r["source"] == "asap"]
        rows.extend({key: row.get(key, "") for key in COLUMNS} for row in references)
    groups = {}
    for row in rows:
        previous = groups.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"group crosses splits: {row['group_id']}")
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
