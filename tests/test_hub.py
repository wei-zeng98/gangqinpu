"""Round-trip checks for the optional native Hugging Face representation."""
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np
from symusic import Note, Score, Track, Tempo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gangqinpu import GangqinpuDataset, export_original_files
from gangqinpu.hub import FILES, METADATA_COLUMNS, pack_dataset

HAS_HUB = importlib.util.find_spec("datasets") is not None and importlib.util.find_spec("pyarrow") is not None


def fixture(root):
    sample = root / "gangqinpu_v1.0.0/1"
    sample.mkdir(parents=True)
    midi = Score(480)
    track = Track()
    track.notes = [Note(0, 480, 60, 80)]
    midi.tracks.append(track)
    for name in FILES:
        if name.endswith(".mid"):
            midi.dump_midi(str(sample / name))
    np.savez(sample / "performance_align.npz", matched_score_indices=[0], matched_performance_indices=[0], matched_score_onsets=[0.0], matched_score_pitches=[60])
    np.savez(sample / "performance_refined_align.npz", performance_note_indices=[0],
             score_note_source_indices=[0], real_matched_score_indices=[0], real_matched_performance_indices=[0],
             refined_performance_onsets=[0.0], interpolated_mask=[False], alignment_hole_mask=[False],
             rascop_recall=1.0, rascop_precision=1.0, rascop_metrics="{}", rascop_stage_times="{}")
    beat = dict(measure=1, beat_in_measure=1.0, score_onset_quarter=0.0, performance_time_sec=0.0,
                time_sec=0.0, is_downbeat=True, is_pickup=False, time_signature_inferred=False,
                method="matched_anchor", confidence=0.95, label="db")
    (sample / "alignment_metadata.json").write_text(json.dumps({"performance_beat_details": [beat], "midi_score_beats": [0.0]}))
    (sample / "metadata.json").write_text('{"title":"fixture"}')
    for name in ["performance_annotations.txt", "midi_score_annotations.txt"]:
        (sample / name).write_text("0.000000\t0.000000\t4/4\n0.000000\t0.000000\t0\n0.000000\t0.000000\tdb\n")
    row = dict.fromkeys(METADATA_COLUMNS, "")
    row.update(sample_id="1", source="gangqinpu", split="train", group_id="g1", sample_dir="gangqinpu_v1.0.0/1")
    with (root / "gangqinpu_meta_v1.0.0.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    (sample / "score.musicxml").write_text("""<?xml version="1.0"?>
<score-partwise version="3.1"><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1"><measure number="1"><attributes><divisions>4</divisions><key><fifths>0</fifths></key>
<time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves><clef><sign>G</sign><line>2</line></clef></attributes>
<note id="n1"><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>quarter</type><staff>1</staff></note>
<note><rest/><duration>12</duration><voice>1</voice><type>half</type><dot/><staff>1</staff></note>
</measure></part></score-partwise>""")
    return sample


@unittest.skipUnless(HAS_HUB, "install the hub extra")
class HubTests(unittest.TestCase):
    def test_native_and_originals_are_independent_and_lossless(self):
        from datasets import load_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            sample = fixture(source)
            output = root / "hub"
            pack_dataset(source, output, workers=1)
            native = load_dataset("parquet", data_files=str(output / "data/*.parquet"), split="train", cache_dir=str(root / "cache"))
            originals = load_dataset("parquet", data_files=str(output / "originals/*.parquet"), split="train", cache_dir=str(root / "cache"))
            self.assertEqual(len(native.column_names), 31)
            self.assertNotIn("files", native.column_names)
            self.assertEqual(native[0]["sample_id"], "1")
            self.assertEqual(native[0]["performance_notes"][0]["pitch"], 60)
            self.assertEqual(native[0]["score_notes"][0]["onset_quarter"], 0.0)
            self.assertEqual(native[0]["score_notes"][0]["duration_quarter"], 1.0)
            self.assertEqual(native[0]["alignment"]["performance_note_indices"], [0])
            self.assertEqual(native[0]["beats"][0]["label"], "db")
            self.assertEqual(native[0]["beats"][0]["time_signature"], "4/4")
            self.assertEqual(native[0]["beats"][0]["key_signature"]["fifths"], 0)
            self.assertEqual(set(originals[0]["files"]), set(FILES))
            target = root / "restored"
            self.assertEqual(export_original_files(originals, native, target), 1)
            restored = GangqinpuDataset(target)[0]
            self.assertEqual(restored.note_alignment()["performance_index"].tolist(), [0])
            for name in FILES:
                self.assertEqual((sample / name).read_bytes(), restored.file(name).read_bytes())
            with self.assertRaises(FileExistsError):
                export_original_files(originals, native, target)

    def test_score_quarters_use_ticks_across_tempo_changes(self):
        from gangqinpu.hub import native_record, native_features
        from gangqinpu.dataset import read_metadata
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = fixture(root)
            midi = Score(480)
            midi.tempos = [Tempo(0, 60), Tempo(480, 120)]
            track = Track()
            track.notes = [Note(240, 720, 60, 80)]
            midi.tracks.append(track)
            midi.dump_midi(str(sample / "score_refined.mid"))
            record = native_record(root, read_metadata(root)[0])
            note = record["score_notes"][0]
            self.assertAlmostEqual(note["time"], 0.5)
            self.assertAlmostEqual(note["duration"], 1.0)
            self.assertEqual(note["onset_quarter"], 0.5)
            self.assertEqual(note["duration_quarter"], 1.5)
            self.assertNotIn("onset_quarter", record["performance_notes"][0])
            self.assertEqual(len(native_features()), 31)

    def test_beat_signatures_follow_score_time_with_pickup_and_reversed_performance(self):
        from gangqinpu import GangqinpuSample
        from gangqinpu.hub import _beat_records
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = fixture(root)
            annotation = json.loads((path / "alignment_metadata.json").read_text())
            first = annotation["performance_beat_details"][0]
            annotation["midi_score_beats"] = [0.0, 0.5, 1.0, 1.5]
            annotation["performance_beat_details"] = [
                {**first, "measure": measure, "is_pickup": i == 0, "is_downbeat": i != 0,
                 "performance_time_sec": t, "time_sec": t}
                for i, (measure, t) in enumerate([(0, 1.0), (1, 2.0), (2, 1.5), (2, 3.0)])]
            (path / "alignment_metadata.json").write_text(json.dumps(annotation))
            (path / "midi_score_annotations.txt").write_text(
                "0\t0\t4/4\n0\t0\t-2\n1\t1\t6/8\n1\t1\t1\n")
            result = _beat_records(GangqinpuSample(root, {"sample_dir": "gangqinpu_v1.0.0/1"}))
            self.assertEqual([b["time_signature"] for b in result], ["4/4", "4/4", "6/8", "6/8"])
            self.assertEqual([b["beats_per_bar"] for b in result], [4, 4, 2, 2])
            self.assertEqual([b["key_signature"]["fifths"] for b in result], [-2, -2, 1, 1])
            self.assertEqual(result[0]["key_signature"]["major_tonic_pitch_class"], 10)
            self.assertTrue(result[0]["is_pickup"])
            (path / "midi_score_annotations.txt").write_text("1\t1\t4/4\n1\t1\t3/4\n")
            missing = _beat_records(GangqinpuSample(root, {"sample_dir": "gangqinpu_v1.0.0/1"}))
            self.assertIsNone(missing[0]["time_signature"])
            self.assertIsNone(missing[2]["time_signature"])
            self.assertIsNone(missing[2]["key_signature"])

    def test_export_rejects_unsafe_or_inconsistent_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = [{"sample_id": "1", "source": "gangqinpu", "split": "train",
                       "sample_dir": "gangqinpu_v1.0.0/1"}]
            bad = {**native[0], "sample_dir": "../escape", "files": {}}
            with self.assertRaises(ValueError):
                export_original_files([bad], native, root / "unsafe")
            bad = {**native[0], "split": "test", "files": {name: b"x" for name in FILES}}
            with self.assertRaises(ValueError):
                export_original_files([bad], native, root / "mismatch")
            self.assertFalse((root / "mismatch/gangqinpu_v1.0.0").exists())


if __name__ == "__main__":
    unittest.main()
