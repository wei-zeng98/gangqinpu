"""Regression checks for serialized alignment indices and pipeline boundaries."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from symusic import Note, Score, Track

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gangqinpu.pipeline.alignment import serialized_note_indices
from gangqinpu.pipeline.common import read_json, work_path, write_json
import _cli
import step07_genre as genre


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PipelineTests(unittest.TestCase):
    def test_serialization_remaps_unsorted_notes(self):
        midi = Score(480)
        track = Track()
        track.notes = [Note(960, 120, 60, 80), Note(0, 120, 62, 70), Note(480, 120, 64, 60)]
        midi.tracks.append(track)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "performance.mid"
            midi.dump_midi(str(path))
            saved = Score(str(path))
            order = serialized_note_indices(midi, saved)
            self.assertEqual(order.tolist(), [2, 0, 1])
            for field in ("time", "pitch", "velocity"):
                np.testing.assert_array_equal(midi.tracks[0].notes.numpy()[field],
                                              saved.tracks[0].notes.numpy()[field][order])

    def test_note_off_repair_does_not_break_note_on_identity(self):
        midi = Score(480)
        track = Track()
        track.notes = [Note(0, 480, 60, 80), Note(60, 240, 60, 70)]
        midi.tracks.append(track)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "overlap.mid"
            midi.dump_midi(str(path))
            self.assertEqual(serialized_note_indices(midi, Score(str(path))).tolist(), [0, 1])

    def test_force_rechecks_current_stage_but_preserves_upstream_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "1"
            sample.mkdir()
            write_json(sample / "pipeline_checks.json", {"alignment_error_rate": {"passed": False}})
            args = _cli.sample_parser("test").parse_args(["--samples-dir", tmp, "--force"])
            seen = []
            _cli.run_samples(args, lambda p: seen.append(p) or "passed", recheck=("alignment_error_rate",))
            self.assertEqual(seen, [sample])
            write_json(sample / "pipeline_checks.json", {"piano_only": {"passed": False}})
            seen.clear()
            _cli.run_samples(args, lambda p: seen.append(p) or "passed", recheck=("alignment_error_rate",))
            self.assertEqual(seen, [])

    def test_metadata_api_failure_invalidates_stale_final_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp)
            write_json(sample / "metadata.json", {"title": "test"})
            write_json(work_path(sample, "genre.json"), {"genre": "pop"})
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test"}), \
                 patch.object(genre.G, "classify_metadata", return_value={"error": "HTTP 503"}):
                genre.metadata_stage([sample], 1, True)
            self.assertFalse(work_path(sample, "genre.json").exists())
            self.assertFalse(work_path(sample, "genre_metadata.json").exists())

    def test_audio_failure_is_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp)
            write_json(work_path(sample, "genre_metadata.json"), {"category": "unknown"})
            write_json(work_path(sample, "genre_audio.json"), {"accepted": False, "error": "decode"})
            with patch.object(genre.G, "ClapGenreScorer"), patch.object(genre.G, "load_clips", side_effect=ValueError("decode")) as decode:
                genre.audio_stage([sample], None, sample, "cpu", 1, False)
            self.assertEqual(decode.call_count, 1)
            self.assertFalse(work_path(sample, "genre.json").exists())

    def test_wrapper_exports_next_to_samples_and_passes_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "dataset" / "gangqinpu_v1.0.0"
            samples.mkdir(parents=True)
            urls = root / "urls.csv"
            urls.write_text("sample_id,audio_url,score_ccmz_url\n")
            fake = root / "python"
            fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n")
            fake.chmod(0o755)
            env = {**os.environ, "PYTHON": str(fake), "CALL_LOG": str(root / "calls")}
            subprocess.run(["bash", str(ROOT / "scripts/run_pipeline.sh"), str(samples), str(root / "audio"),
                            str(root / "reports"), str(urls), "--force"], env=env, check=True)
            calls = (root / "calls").read_text().splitlines()
            self.assertEqual(len(calls), 10)
            self.assertIn(f"--source-urls {urls}", calls[-1])
            self.assertIn(f"--output {samples.parent / 'gangqinpu_meta_v1.0.0.csv'}", calls[-1])
            self.assertNotIn("--force", calls[-2])

    def test_fixed_manifest_preserves_partitions_and_rejects_group_conflicts(self):
        import step09_split as split
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples"
            for sample_id in ("1", "2"):
                sample = samples / sample_id
                sample.mkdir(parents=True)
                (sample / "audio.mp3").write_bytes(sample_id.encode())
            groups = root / "groups.csv"
            groups.write_text("source,sample_id,group_id\ngangqinpu,1,new1\ngangqinpu,2,new2\n")
            manifest = root / "manifest.csv"
            manifest.write_text("source,sample_id,group_id,split\ngangqinpu,1,official1,train\ngangqinpu,2,official2,val\n")
            args = ["split", "--samples-dir", str(samples), "--groups", str(groups),
                    "--manifest", str(manifest), "--output", str(root / "out")]
            with patch.object(sys, "argv", args):
                self.assertEqual(split.main(), 0)
            with (root / "out/splits.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([(r["group_id"], r["split"]) for r in rows],
                             [("official1", "train"), ("official2", "val")])
            groups.write_text("source,sample_id,group_id\ngangqinpu,1,same\ngangqinpu,2,same\n")
            with patch.object(sys, "argv", args), self.assertRaises(ValueError):
                split.main()

    def test_export_is_readable_and_preserves_source_urls(self):
        import step10_export_metadata as export
        from gangqinpu import GangqinpuDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "gangqinpu_v1.0.0" / "1"
            sample.mkdir(parents=True)
            write_json(sample / "pipeline_checks.json", {
                "expressive": {"passed": True},
                "piano_only": {"passed": True},
                "alignment_error_rate": {"passed": True},
                "audio_duration": {"passed": True, "details": {"duration_seconds": 100}},
                "score_bar_count": {"passed": True, "details": {"bar_count": 32}}})
            write_json(sample / "metadata.json", {"title": "test", "detail_url": "https://example.org/1"})
            write_json(work_path(sample, "genre.json"), {"genre": "pop", "genre_source": "metadata_qwen_flash"})
            write_json(sample / "alignment_metadata.json", {"performance_beats": [0], "performance_downbeats": [0]})
            midi = Score(480)
            track = Track()
            track.notes = [Note(0, 480, 60, 80)]
            midi.tracks.append(track)
            for name in ("performance_refined.mid", "score_MS.mid"):
                midi.dump_midi(str(sample / name))
            (root / "splits.csv").write_text("sample_id,source,split,group_id\n1,gangqinpu,train,g1\n")
            (root / "groups.csv").write_text("sample_id,source,score_group_id\n1,gangqinpu,s1\n")
            (root / "urls.csv").write_text("sample_id,audio_url,score_ccmz_url\n1,https://example.org/audio,https://example.org/score\n")
            argv = ["export", "--samples-dir", str(sample.parent), "--splits", str(root / "splits.csv"),
                    "--groups", str(root / "groups.csv"), "--source-urls", str(root / "urls.csv"),
                    "--output", str(root / "gangqinpu_meta_v1.0.0.csv")]
            with patch.object(sys, "argv", argv):
                self.assertEqual(export.main(), 0)
            dataset = GangqinpuDataset(root)
            self.assertEqual(dataset[0].row["audio_url"], "https://example.org/audio")
            self.assertTrue(dataset[0].path("performance_midi_path").is_file())
            (sample / "audio.m4a").write_bytes(b"source audio")
            with patch.object(sys, "argv", argv):
                self.assertEqual(export.main(), 0)
            self.assertNotIn("audio_path", GangqinpuDataset(root)[0].row)
            self.assertEqual(GangqinpuDataset(root)[0].audio_path.name, "audio.m4a")
            self.assertNotIn("score_ccmz_path", export.COLUMNS)
            self.assertNotIn("score_musicxml_path", export.COLUMNS)

    def test_undecided_classifier_is_retried_instead_of_cached_as_success(self):
        import step06_classify_expressive as stage
        import gangqinpu.classifier as classifier
        import gangqinpu.pipeline.alignment as alignment
        import gangqinpu.pipeline.performance as performance
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp)
            write_json(sample / "alignment_metadata.json", {stage.KEY: {"error": "missing features"}})
            write_json(sample / "pipeline_checks.json", {"expressive": {"passed": None}})
            write_json(work_path(sample, "score_table.json"), {"score_notes": []})
            with patch.object(classifier, "classify_alignment", return_value={"error": "missing features"}) as predict, \
                 patch.object(alignment, "load_raw_alignment", return_value={}), \
                 patch.object(performance, "performance_notes_from_midi", return_value=[]):
                self.assertTrue(stage.job(sample, False).startswith("undecided:"))
            self.assertEqual(predict.call_count, 1)

    def test_partial_alignment_outputs_are_not_treated_as_complete_cache(self):
        import step04_align as stage
        import gangqinpu.pipeline.alignment as alignment
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp)
            (sample / "performance_refined_align.npz").write_bytes(b"partial")
            write_json(work_path(sample, "refinement_info.json"), {})
            write_json(work_path(sample, "score_table.json"), {"score_notes": []})
            with patch.object(alignment, "align_and_refine", side_effect=RuntimeError("recomputed")) as recompute, \
                 self.assertRaisesRegex(RuntimeError, "recomputed"):
                stage.job(sample, True, False)
            self.assertEqual(recompute.call_count, 1)

    def test_score_clock_respects_explicit_zero_tick_tempo_and_changes(self):
        import mido
        from gangqinpu.pipeline.score import midi_quarter_to_seconds
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "score.mid"
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.extend([mido.MetaMessage("set_tempo", tempo=400000, time=0),
                          mido.MetaMessage("set_tempo", tempo=1000000, time=480)])
            midi.save(path)
            np.testing.assert_allclose(midi_quarter_to_seconds(path, [2, 0, 1]), [1.4, 0, .4])

    def test_virtual_pickup_origin_is_separate_from_saved_midi_time(self):
        from gangqinpu.pipeline.score import score_origin_offset
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "score.mid"
            midi = Score(480)
            track = Track()
            track.notes = [Note(0, 120, 60, 80), Note(480, 120, 62, 80)]
            midi.tracks.append(track)
            midi.dump_midi(str(path))
            notes = [{"pitch": 60, "onset_quarter": 1}, {"pitch": 62, "onset_quarter": 2}]
            self.assertEqual(score_origin_offset(path, notes), 1)

    def test_reader_merges_both_score_staves(self):
        from gangqinpu.dataset import _midi_notes, GangqinpuSample
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi = Score(480)
            for pitch in (72, 48):
                track = Track()
                track.notes = [Note(0, 480, pitch, 80)]
                midi.tracks.append(track)
            midi.dump_midi(str(root / "score_MS.mid"))
            self.assertEqual(len(_midi_notes(root / "score_MS.mid")), 1)
            self.assertEqual(_midi_notes(root / "score_MS.mid", merge_tracks=True)["pitch"].tolist(), [48, 72])
            sample = GangqinpuSample(root, {"sample_dir": "."})
            self.assertEqual(len(sample.score_notes("ms")), 2)
            with self.assertRaises(ValueError):
                sample.score_notes("invalid")

    def test_raw_alignment_rejects_out_of_range_diagnostics(self):
        from gangqinpu.pipeline.alignment import _values
        with self.assertRaises(ValueError):
            _values([0, 3], [{"pitch": 60}], "pitch", np.int16)

    def test_classifier_cli_propagates_pair_errors_and_rejects_empty_csv(self):
        import classify_expressive as cli
        with tempfile.TemporaryDirectory() as tmp:
            pairs = Path(tmp) / "pairs.csv"
            pairs.write_text("score,performance\n")
            with patch.object(sys, "argv", ["classify", "--pairs", str(pairs)]), self.assertRaises(SystemExit):
                cli.main()
            pairs.write_text("score,performance\nscore.musicxml,performance.mid\n")
            with patch.object(sys, "argv", ["classify", "--pairs", str(pairs)]), \
                 patch.object(cli, "classify_pair", return_value={"error": "missing features"}):
                self.assertEqual(cli.main(), 1)

    def test_download_rejects_zero_retries(self):
        tool = load_tool("download_sources")
        with patch.object(sys, "argv", ["download", "--root", ".", "--retries", "0"]), self.assertRaises(SystemExit):
            tool.main()

    def test_source_recovery_without_unpublished_path_columns(self):
        tool = load_tool("download_sources")
        from gangqinpu import GangqinpuDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "gangqinpu_meta_v1.0.0.csv"
            manifest.write_text("sample_id,source,split,sample_dir,audio_url,score_ccmz_url,source_page_url\n"
                                "1,gangqinpu,train,gangqinpu_v1.0.0/1,https://example.org/audio.m4a,https://example.org/score,https://example.org/1\n")
            outputs = []
            def fake_download(url, dest, *args):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"source")
                outputs.append(dest.relative_to(root).as_posix())
            with patch.object(sys, "argv", ["download", "--root", str(root), "--interval", "0"]), \
                 patch.object(tool, "download", side_effect=fake_download):
                self.assertEqual(tool.main(), 0)
            self.assertEqual(sorted(outputs), ["gangqinpu_audio/1/audio.m4a", "gangqinpu_v1.0.0/1/score.ccmz"])
            sample = GangqinpuDataset(root)[0]
            self.assertEqual(sample.audio_path.name, "audio.m4a")
            self.assertIsNone(sample.musicxml_path)
            sample.file("score.musicxml").write_text("<score/>")
            self.assertEqual(sample.musicxml_path.name, "score.musicxml")

    def test_version_and_export_schema_are_consistent(self):
        from gangqinpu import __version__
        from gangqinpu.dataset import METADATA_CSV
        import step10_export_metadata as export
        tool = load_tool("pack_dataset")
        from gangqinpu.hub import METADATA_COLUMNS
        self.assertEqual(export.COLUMNS, list(METADATA_COLUMNS))
        self.assertEqual(__version__, "1.0.0")
        self.assertEqual(METADATA_CSV, f"gangqinpu_meta_v{__version__}.csv")
        self.assertEqual(tool.MANIFEST, METADATA_CSV)
        self.assertEqual(len(export.COLUMNS), 25)
        self.assertTrue({"audio_path", "score_musicxml_path", "score_ccmz_path"}.isdisjoint(export.COLUMNS))
        self.assertIn(f'version = "{__version__}"', (ROOT / "pyproject.toml").read_text())



if __name__ == "__main__":
    unittest.main()
