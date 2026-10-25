#!/usr/bin/env python3
"""Step 7 - Genre labelling of accepted samples (Qwen-Flash, then LAION-CLAP fallback).

--stage metadata : Qwen-Flash on ``metadata.json`` (needs DASHSCOPE_API_KEY).
--stage audio    : CLAP on samples still ``unknown`` (needs --clap-model, a GPU
                   is recommended; see tools/convert_clap_checkpoint.py).
--stage all      : both, in order.

Per-sample results go to ``.work/genre_metadata.json``, ``.work/genre_audio.json``
and the final ``.work/genre.json`` ({"genre", "genre_source"}), where
genre_source is ``metadata_qwen_flash``, ``audio_clap_heuristic`` or ``unknown``.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import _cli  # noqa: F401  (sets up the import path)
from gangqinpu.pipeline.checks import is_rejected, load_checks
from gangqinpu.pipeline.common import SOURCE_METADATA, find_audio, list_sample_dirs, read_json, work_path, write_json
from gangqinpu.pipeline import genre as G


def accepted_samples(args) -> list[Path]:
    return [d for d in list_sample_dirs(args.samples_dir, args.ids)
            if not is_rejected(d) and load_checks(d).get("expressive", {}).get("passed") is True]


def write_final(sample: Path) -> dict:
    metadata = read_json(work_path(sample, "genre_metadata.json"))
    audio_path = work_path(sample, "genre_audio.json")
    if metadata.get("category", "unknown") != "unknown":
        final = {"genre": metadata["category"], "genre_source": "metadata_qwen_flash"}
    elif audio_path.is_file() and read_json(audio_path).get("accepted"):
        final = {"genre": read_json(audio_path)["accepted_category"], "genre_source": "audio_clap_heuristic"}
    else:
        final = {"genre": "unknown", "genre_source": "unknown"}
    write_json(work_path(sample, "genre.json"), final)
    return final


def metadata_stage(samples: list[Path], workers: int, force: bool) -> None:
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    todo = [s for s in samples if force or not work_path(s, "genre_metadata.json").is_file()]
    if todo and not key:
        raise SystemExit("Set DASHSCOPE_API_KEY for the Qwen-Flash metadata stage.")
    print(f"[metadata] {len(todo)} samples to label", flush=True)
    for sample in todo:
        work_path(sample, "genre.json").unlink(missing_ok=True)
        work_path(sample, "genre_metadata.json").unlink(missing_ok=True)

    def label(sample: Path) -> dict:
        return G.classify_metadata(read_json(sample / SOURCE_METADATA), key)

    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(label, s): s for s in todo}
        for n, future in enumerate(as_completed(futures), 1):
            sample = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: metadata classification failed"}
            if "error" in result:
                print(f"[metadata] {sample.name}: error {result['error']} (will retry on next run)", flush=True)
                continue
            write_json(work_path(sample, "genre_metadata.json"), result)
            write_final(sample)
            if n % 100 == 0 or n == len(todo):
                print(f"[metadata] {n}/{len(todo)}", flush=True)


def audio_stage(samples: list[Path], audio_dir: Path | None, model_dir: Path, device: str,
                batch: int, force: bool) -> None:
    todo = [s for s in samples
            if work_path(s, "genre_metadata.json").is_file()
            and read_json(work_path(s, "genre_metadata.json")).get("category") == "unknown"
            and (force or not work_path(s, "genre_audio.json").is_file()
                 or "error" in read_json(work_path(s, "genre_audio.json")))]
    print(f"[audio] {len(todo)} metadata-unknown samples", flush=True)
    if not todo:
        return
    scorer = G.ClapGenreScorer(model_dir, device)
    for start in range(0, len(todo), batch):
        loaded = []
        for sample in todo[start:start + batch]:
            try:
                loaded.append((sample, G.load_clips(find_audio(sample, audio_dir))))
            except Exception as exc:
                print(f"[audio] {sample.name}: {type(exc).__name__}: {exc}", flush=True)
                write_json(work_path(sample, "genre_audio.json"), {"accepted": False, "error": str(exc)})
                work_path(sample, "genre.json").unlink(missing_ok=True)
        if not loaded:
            continue
        scores = scorer.score([clip for _, clips in loaded for clip in clips])
        offset = 0
        for sample, clips in loaded:
            result = G.decide_audio_genre(scores[offset:offset + len(clips)])
            offset += len(clips)
            write_json(work_path(sample, "genre_audio.json"), result)
            write_final(sample)
        print(f"[audio] {min(start + batch, len(todo))}/{len(todo)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--stage", choices=["metadata", "audio", "all"], default="all")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent API requests (metadata stage).")
    parser.add_argument("--clap-model", type=Path, help="Converted HF CLAP music model directory.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-samples", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.batch_samples < 1:
        parser.error("workers and batch-samples must be positive")
    if args.stage in ("audio", "all") and args.clap_model is None:
        parser.error("--clap-model is required for the audio stage")
    samples = accepted_samples(args)
    if not samples:
        raise SystemExit("No accepted samples; complete filters and classification first.")
    if args.stage in ("metadata", "all"):
        metadata_stage(samples, args.workers, args.force)
    if args.stage in ("audio", "all"):
        audio_stage(samples, args.audio_dir, args.clap_model, args.device, args.batch_samples, args.force)
    for sample in samples:
        metadata = work_path(sample, "genre_metadata.json")
        audio = work_path(sample, "genre_audio.json")
        if metadata.is_file() and (read_json(metadata).get("category") != "unknown"
                                  or args.stage == "metadata"
                                  or (audio.is_file() and "error" not in read_json(audio))):
            write_final(sample)
    missing = [s.name for s in samples if not work_path(s, "genre.json").is_file()]
    print(f"done; {len(missing)} accepted samples still without a genre label", flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
