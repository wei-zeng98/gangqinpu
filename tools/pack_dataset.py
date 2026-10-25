#!/usr/bin/env python3
"""Build native Hugging Face Parquet from the extracted Gangqinpu dataset.

Install the hub extra, then run:
    python tools/pack_dataset.py --root DATASET_ROOT --output UPLOAD_DIR

Recovered score.musicxml files are required to derive notation labels at build
time; they are never included in the output. Install the hub and notation extras.
The upload directory must be outside the source, with empty data/, originals/ and metadata/
subdirectories. README and LICENSE are maintained separately. The default
configuration contains decoded notes, alignment and beats; the originals
configuration independently preserves all 11 files per sample.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gangqinpu.dataset import METADATA_CSV as MANIFEST
from gangqinpu.hub import FILES, pack_dataset as pack, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-shards", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    pack(args.root, args.output, args.train_shards, args.workers)


if __name__ == "__main__":
    main()
