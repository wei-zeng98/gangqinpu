"""Native Hugging Face tables and optional lossless export of source files."""
from __future__ import annotations

import csv
from bisect import bisect_right
import hashlib
import math
import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .dataset import FLOAT_COLUMNS, INT_COLUMNS, METADATA_CSV, GangqinpuSample, read_metadata

SAMPLE_DIR = METADATA_CSV.replace("_meta_", "_").removesuffix(".csv")

METADATA_COLUMNS = (
    "sample_id", "source", "split", "group_id", "score_group_id", "title", "composer", "genre", "genre_source",
    "audio_duration_seconds", "duration_seconds", "note_count", "score_note_count", "score_bar_count",
    "annotation_beat_count", "annotation_downbeat_count", "sample_dir", "performance_midi_path", "score_midi_path",
    "alignment_path", "annotations_path", "metadata_path", "source_page_url", "audio_url", "score_ccmz_url",
)
FILES = (
    "metadata.json", "audio_transkun.mid", "performance_clean.mid", "score_MS.mid", "performance_align.npz",
    "performance_refined_align.npz", "score_refined.mid", "performance_refined.mid", "alignment_metadata.json",
    "performance_annotations.txt", "midi_score_annotations.txt",
)
INDEX_ARRAYS = ("performance_note_indices", "score_note_source_indices", "real_matched_score_indices",
                "real_matched_performance_indices")


def metadata_features():
    from datasets import Features, Value
    return Features({name: Value("int64" if name in INT_COLUMNS else "float64" if name in FLOAT_COLUMNS else "string")
                     for name in METADATA_COLUMNS})


def native_features():
    from datasets import Features, List, Value
    from .notation import notation_features
    note = {"time": Value("float64"), "duration": Value("float64"),
            "pitch": Value("int16"), "velocity": Value("int16")}
    score_note = {**note, "onset_quarter": Value("float64"), "duration_quarter": Value("float64")}
    alignment = {name: List(Value("int64")) for name in INDEX_ARRAYS}
    alignment.update({"interpolated_mask": List(Value("bool")), "alignment_hole_mask": List(Value("bool")),
                      "refined_performance_onsets": List(Value("float64")), "rascop_recall": Value("float64"),
                      "rascop_precision": Value("float64"), "rascop_metrics": Value("string"),
                      "rascop_stage_times": Value("string")})
    beat = {"measure": Value("int64"), "beat_in_measure": Value("float64"),
            "score_onset_quarter": Value("float64"), "performance_time_sec": Value("float64"),
            "time_sec": Value("float64"), "is_downbeat": Value("bool"), "is_pickup": Value("bool"),
            "time_signature_inferred": Value("bool"), "method": Value("string"),
            "confidence": Value("float64"), "label": Value("string"),
            "time_signature": Value("string"), "beats_per_bar": Value("int16"),
            "key_signature": {"fifths": Value("int16"), "major_tonic_pitch_class": Value("int16")}}
    beat["time_valid"] = Value("bool")
    return Features({**metadata_features(), "performance_notes": List(note), "score_notes": List(score_note),
                     "alignment": alignment, "beats": List(beat), "notation": notation_features(),
                     "performance_pedal": List({"time": Value("float64"), "value": Value("int16")})})


def original_features():
    from datasets import Features, Value
    return Features({"sample_id": Value("string"), "source": Value("string"), "split": Value("string"),
                     "sample_dir": Value("string"), "files": {name: Value("binary") for name in FILES}})


def _note_records(notes):
    return [{name: value.item() for name, value in zip(notes.dtype.names, row)} for row in notes]


def _score_note_records(sample: GangqinpuSample) -> list[dict]:
    """Seconds and quarter-note positions in the saved refined-MIDI note order."""
    from symusic import Score

    score = Score(str(sample.file("score_refined.mid")), ttype="tick")
    records = _note_records(sample.score_notes())
    if not score.tracks:
        return records
    ticks = score.tracks[0].notes.numpy()
    if len(records) != len(ticks["time"]):
        raise ValueError("score note order differs between time units")
    for index, record in enumerate(records):
        record["onset_quarter"] = float(ticks["time"][index]) / score.ticks_per_quarter
        record["duration_quarter"] = float(ticks["duration"][index]) / score.ticks_per_quarter
    return records


def _beat_records(sample: GangqinpuSample) -> list[dict]:
    """Attach active score signatures in score order, independent of performed time."""
    annotations = sample.annotations
    details = annotations["performance_beat_details"]
    times = annotations["midi_score_beats"]
    if len(details) != len(times):
        raise ValueError("score and performance beat counts differ")
    events = {"meter": {}, "key": {}}
    for time, label in sample.read_annotation_tsv("score"):
        kind = "meter" if "/" in label else "key" if label.lstrip("-").isdigit() else None
        if kind is not None:
            events[kind].setdefault(round(time, 6), set()).add(label)
    ordered = {kind: sorted(values) for kind, values in events.items()}

    def active(kind, time):
        index = bisect_right(ordered[kind], round(float(time), 6)) - 1
        if index < 0:
            return None
        values = events[kind][ordered[kind][index]]
        return next(iter(values)) if len(values) == 1 else None

    records = []
    for beat, time in zip(details, times):
        meter = active("meter", time)
        key = active("key", time)
        count = None
        if meter is not None:
            numerator, denominator = map(int, meter.split("/"))
            count = numerator // 3 if denominator == 8 and numerator > 3 and numerator % 3 == 0 else numerator
        fifths = int(key) if key is not None else None
        records.append({**beat, "time_signature": meter, "beats_per_bar": count,
                        "key_signature": None if fifths is None else
                        {"fifths": fifths, "major_tonic_pitch_class": (7 * fifths) % 12}})
    return records


def native_record(root: Path, row: dict) -> dict:
    """Read the refined MIDI pair in symusic order and preserve every source byte."""
    sample = GangqinpuSample(Path(root), row)
    expected = f"{SAMPLE_DIR}/{row['sample_id']}"
    if row["sample_dir"] != expected or not row["sample_id"].isdigit() or sample.sample_dir.is_symlink():
        raise ValueError("unexpected sample directory")
    files = {}
    for name in FILES:
        path = sample.file(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or unsafe release file: {name}")
        files[name] = path.read_bytes()
    alignment = {key: value.tolist() if value.ndim else value.item()
                 for key, value in sample.refined_alignment().items()}
    record = {**{key: row.get(key) for key in METADATA_COLUMNS},
            "performance_notes": _note_records(sample.performance_notes()),
            "score_notes": _score_note_records(sample),
            "alignment": alignment, "beats": _beat_records(sample), "files": files}
    from .notation import extract_notation, mark_beat_times, pedal_records
    record["notation"] = extract_notation(record, sample.file("score.musicxml"))
    record["performance_pedal"] = pedal_records(sample.sample_dir)
    mark_beat_times(record)
    return record


def _write_parquet(job):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root, path, rows, full = job
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    original = path.parent.parent / "originals" / path.name if full else None
    original_tmp = original.with_suffix(original.suffix + ".tmp") if full else None
    schema = (native_features() if full else metadata_features()).arrow_schema
    raw_writer = None
    try:
        if full:
            original.parent.mkdir(parents=True, exist_ok=True)
            raw_writer = pq.ParquetWriter(original_tmp, original_features().arrow_schema,
                                         compression="zstd", compression_level=3)
        with pq.ParquetWriter(temporary, schema, compression="zstd", compression_level=3,
                              use_dictionary=True) as writer:
            for start in range(0, len(rows), 16):
                batch = [native_record(Path(root), row) if full else {k: row.get(k) for k in METADATA_COLUMNS}
                         for row in rows[start:start + 16]]
                writer.write_table(pa.Table.from_pylist(batch, schema=schema), row_group_size=16)
                if full:
                    raw_writer.write_table(pa.Table.from_pylist(batch, schema=original_features().arrow_schema),
                                           row_group_size=16)
        if raw_writer is not None:
            raw_writer.close()
            raw_writer = None
        os.replace(temporary, path)
        if full:
            os.replace(original_tmp, original)
    finally:
        if raw_writer is not None:
            raw_writer.close()
        temporary.unlink(missing_ok=True)
        if original_tmp is not None:
            original_tmp.unlink(missing_ok=True)
    return [path, original] if full else [path]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pack_dataset(root: str | Path, output: str | Path, train_shards: int = 2, workers: int = 2):
    """Write native train/validation/test Parquet and external reference metadata."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if train_shards < 1 or workers < 1:
        raise ValueError("train_shards and workers must be positive")
    if root == output or root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError("source and upload directories must not contain each other")
    if any((output / folder).exists() and any((output / folder).iterdir()) for folder in ("data", "originals", "metadata")):
        raise ValueError("output data, originals and metadata directories must be empty")
    rows = read_metadata(root, source=None)
    samples = [row for row in rows if row["source"] == "gangqinpu"]
    if not samples or len({row['sample_id'] for row in samples}) != len(samples):
        raise ValueError("manifest must contain unique Gangqinpu samples")
    if any(row["source"] not in ("gangqinpu", "asap") or row["split"] not in ("train", "val", "test") for row in rows):
        raise ValueError("unsupported manifest source or split")
    jobs = []
    for split, label in [("train", "train"), ("val", "validation"), ("test", "test")]:
        selected = sorted((row for row in samples if row["split"] == split), key=lambda row: row["sample_id"])
        count = min(train_shards, len(selected)) if split == "train" else min(1, len(selected))
        size = math.ceil(len(selected) / count) if count else 1
        for index, start in enumerate(range(0, len(selected), size)):
            jobs.append((root, output / "data" / f"{label}-{index:05d}-of-{count:05d}.parquet",
                         selected[start:start + size], True))
        reference = [row for row in rows if row["source"] == "asap" and row["split"] == split]
        if reference:
            jobs.append((root, output / "metadata" / f"asap_{label}.parquet", reference, False))
    output.mkdir(parents=True, exist_ok=True)
    if workers == 1:
        paths = [path for job in jobs for path in _write_parquet(job)]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            paths = []
            for written in pool.map(_write_parquet, jobs):
                paths.extend(written)
                for path in written:
                    print(f"wrote {path.relative_to(output)}", flush=True)
    manifest = output / METADATA_CSV
    manifest.write_bytes((root / METADATA_CSV).read_bytes())
    paths = sorted([manifest, *paths])
    (output / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(output).as_posix()}\n" for p in paths))
    return paths


def _rows(dataset):
    if isinstance(dataset, Mapping):
        for split in dataset.values():
            yield from split
    else:
        yield from dataset


def export_original_files(originals, metadata, root: str | Path, references=None) -> int:
    """Restore byte-identical sample files and a manifest; references are optional.

    Join originals with the structured metadata by sample_id. Existing files
    must match. Export all splits plus reference metadata for a full inventory.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / METADATA_CSV
    if manifest.exists():
        raise FileExistsError(f"manifest already exists: {manifest}")
    by_id = {}
    for row in _rows(metadata):
        key = str(row["sample_id"])
        if row["source"] != "gangqinpu" or key in by_id:
            raise ValueError("metadata must contain unique Gangqinpu IDs")
        by_id[key] = {name: row.get(name) for name in METADATA_COLUMNS}
    rows, seen = [], set()
    for record in _rows(originals):
        sample_id = str(record["sample_id"])
        relative = Path(record["sample_dir"])
        if not sample_id.isdigit() or relative.as_posix() != f"{SAMPLE_DIR}/{sample_id}":
            raise ValueError("unsafe sample path")
        if sample_id in seen:
            raise ValueError("duplicate sample ID")
        seen.add(sample_id)
        if sample_id not in by_id:
            raise ValueError("original has no corresponding metadata")
        row = by_id[sample_id]
        if any(record[key] != row[key] for key in ("source", "split", "sample_dir")):
            raise ValueError("original and metadata identity disagree")
        if set(record["files"]) != set(FILES):
            raise ValueError("record must contain exactly the 11 release files")
        for name, content in record["files"].items():
            target = root / relative / name
            if root not in target.resolve().parents:
                raise ValueError("unsafe export target")
            if target.exists():
                if target.read_bytes() != content:
                    raise FileExistsError(f"existing file differs: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            temporary.write_bytes(content)
            os.replace(temporary, target)
        rows.append(row)
    if references is not None:
        for row in _rows(references):
            if row["source"] != "asap":
                raise ValueError("references must contain ASAP metadata only")
            rows.append({key: row.get(key) for key in METADATA_COLUMNS})
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(seen)
