#!/usr/bin/env python3
"""Offline additive rebuild of published Parquet; never uploads or changes splits.

Only structured labels and SHA256 provenance are emitted, never MusicXML bytes.
The old input shards are immutable. Existing output shards are not overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
import pyarrow.parquet as pq
from gangqinpu.hub import native_features, sha256
from gangqinpu.notation import extract_notation, mark_beat_times, pedal_records

SAMPLES = None


def initialize(samples):
    global SAMPLES
    SAMPLES = Path(samples)
    warnings.filterwarnings("ignore")


def expired(*args):
    raise TimeoutError("notation_extraction_timeout")


def enrich(record):
    xml = SAMPLES / record["sample_id"] / "score.musicxml"
    if not xml.is_file():
        raise FileNotFoundError(xml)
    source = "xml"
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(120)
    try:
        record["performance_pedal"] = pedal_records(xml.parent)
        notation = extract_notation(record, xml)
        record["notation"] = notation
        mark_beat_times(record)
        return record, dict(sample_id=record["sample_id"], source=source,
                            alignment_complete=notation["alignment_complete"], issues=notation["issues"])
    except Exception as exc:
        # Retain the complete raw inventory. An unresolved extraction is explicit,
        # never a guessed staff/grace label and never silently removed.
        issue = f"{type(exc).__name__}: {exc}"
        record["notation"] = dict(version=2, source_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
                                  midi_origin_offset_quarter=None, parts=[], notes=[], measures=[],
                                  alignment_complete=False, issues=["extraction_failed"])
        mark_beat_times(record)
        return record, dict(sample_id=record["sample_id"], source=source, alignment_complete=False,
                            issues=["extraction_failed"], error=issue)
    finally:
        signal.alarm(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--samples", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="diagnostic only; recorded in report")
    args=p.parse_args()
    if args.workers < 1 or args.limit < 0 or args.output.resolve() == args.input.resolve():
        p.error("invalid workers/limit or output equals input")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("output must be new or empty")
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"data").mkdir()
    schema=native_features().arrow_schema
    counts=Counter(); details=[]; written=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(args.samples,)) as pool:
        for source in sorted((args.input/"data").glob("*.parquet")):
            target=args.output/"data"/source.name
            seen=0
            with pq.ParquetWriter(str(target)+".tmp",schema,compression="zstd",compression_level=3) as writer:
                for batch in pq.ParquetFile(source).iter_batches(batch_size=64):
                    rows=batch.to_pylist()
                    if args.limit:
                        rows=rows[:max(0,args.limit-seen)]
                    if not rows:
                        break
                    results=list(pool.map(enrich,rows,chunksize=1))
                    writer.write_table(pa.Table.from_pylist([r for r,_ in results],schema=schema))
                    for row,detail in results:
                        details.append(detail); counts["samples"]+=1
                        counts["split_"+row["split"]]+=1
                        counts["alignment_complete"]+=detail["alignment_complete"]
                        counts["notation_notes"]+=len(row["notation"]["notes"])
                        counts["grace_notes"]+=sum(n["is_grace"] for n in row["notation"]["notes"])
                        for issue in detail["issues"]:counts[issue]+=1
                    seen+=len(rows)
                    if seen % 512 == 0:
                        print(json.dumps(dict(counts)),flush=True)
                    if args.limit and seen>=args.limit:
                        break
            os.replace(str(target)+".tmp",target)
            written.append(target)
            print(f"wrote {target.name}: {seen}",flush=True)
    for source in args.input.glob("*.csv"):
        shutil.copy2(source,args.output/source.name);written.append(args.output/source.name)
    for name in ["README.md","LICENSE"]:
        if (args.input/name).exists():
            shutil.copy2(args.input/name,args.output/name)
    report=dict(schema_version=2,diagnostic_subset=bool(args.limit),counts=dict(counts),
                input_sha256={f.name:sha256(f) for f in (args.input/"data").glob("*.parquet")},
                output_sha256={f.name:sha256(f) for f in written},
                details=details)
    (args.output/"build_report.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.output/"SHA256SUMS").write_text("".join(f"{sha256(f)}  {f.relative_to(args.output)}\n" for f in written))
    print(json.dumps(dict(counts),indent=2),flush=True)


if __name__=="__main__":
    main()
