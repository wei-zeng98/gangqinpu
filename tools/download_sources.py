#!/usr/bin/env python3
"""Download the source audio and CCMZ scores listed in the metadata CSV.

Only the direct links stored in ``audio_url`` and ``score_ccmz_url`` are
fetched (no page parsing, no crawling). Recovery targets are derived from sample IDs and source URLs,
relative to the dataset root:

    gangqinpu_audio/<sample_id>/audio.<ext>
    gangqinpu_v1.0.0/<sample_id>/score.ccmz

Existing non-empty files are skipped, so the command can be re-run. Remote
files may change or disappear; availability is not guaranteed. Please respect
the source website's terms of use and keep the request rate low.

    python tools/download_sources.py --root /path/to/gangqinpu [--assets audio ccmz] [--ids 1000009 ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gangqinpu.dataset import _source_asset_path

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
ASSETS = {"audio": "audio_url", "ccmz": "score_ccmz_url"}


def download(url: str, dest: Path, referer: str, timeout: float, retries: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Referer": referer})
            with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as out:
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("server returned HTML instead of a media file")
                shutil.copyfileobj(response, out, 1 << 20)
            if not part.stat().st_size:
                raise ValueError("empty download")
            os.replace(part, dest)
            return
        except Exception:
            part.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(min(2 ** attempt, 10))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root containing gangqinpu_meta_v1.0.0.csv")
    parser.add_argument("--metadata", type=Path, help="Default: <root>/gangqinpu_meta_v1.0.0.csv")
    parser.add_argument("--assets", nargs="+", choices=sorted(ASSETS), default=sorted(ASSETS))
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between requests (default 1).")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    if args.retries < 1 or args.timeout <= 0 or args.interval < 0 or (args.limit is not None and args.limit < 1):
        parser.error("retries/timeout/limit must be positive and interval non-negative")
    root = args.root.resolve()
    with (args.metadata or root / "gangqinpu_meta_v1.0.0.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == "gangqinpu"]
    if args.ids:
        wanted = set(args.ids)
        rows = [r for r in rows if r["sample_id"] in wanted]
        missing = wanted - {r["sample_id"] for r in rows}
        if missing:
            parser.error("unknown sample IDs: " + ", ".join(sorted(missing)))
    rows = rows[:args.limit] if args.limit else rows

    counts = {"downloaded": 0, "existing": 0, "failed": 0}
    with (root / "download_log.jsonl").open("a", encoding="utf-8") as log:
        for n, row in enumerate(rows, 1):
            for asset in args.assets:
                url_column = ASSETS[asset]
                relative = _source_asset_path(row, asset)
                dest = (root / relative).resolve()
                if root not in dest.parents:
                    raise ValueError(f"unsafe recovery path: {relative}")
                if dest.is_file() and dest.stat().st_size:
                    counts["existing"] += 1
                    continue
                try:
                    download(row[url_column], dest, row["source_page_url"], args.timeout, args.retries)
                    status, error = "downloaded", ""
                except Exception as exc:
                    status, error = "failed", f"{type(exc).__name__}: {exc}"
                counts[status] += 1
                log.write(json.dumps({"sample_id": row["sample_id"], "asset": asset, "status": status,
                                      "url": row[url_column], "error": error}, ensure_ascii=False) + "\n")
                log.flush()
                time.sleep(args.interval)
            if n % 100 == 0 or n == len(rows):
                print(f"{n}/{len(rows)} samples {counts}", flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
