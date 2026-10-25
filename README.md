# Gangqinpu: A Multi-Genre, Multi-Task Piano Dataset with Aligned Audio, MIDI, and Musical Scores

<p align="center">
  <a href="#paper"><img src="https://img.shields.io/badge/Gangqinpu%20Paper-arXiv%20%7C%20coming%20soon-B31B1B?style=flat-square" alt="Gangqinpu paper: arXiv coming soon"></a>
  <a href="https://huggingface.co/datasets/Wayne42/gangqinpu"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gangqinpu%20Dataset-Hugging%20Face-FFD21E?style=flat-square" alt="Gangqinpu dataset on Hugging Face"></a>
</p>

Official reader and construction code for Gangqinpu v1.0.0: **18,786 piano score–performance pairs**, covering **1,007.1 hours** of source audio, with automatic note alignments and beat/downbeat annotations.

The code is open source; the dataset and classifier weights are released under CC BY-NC-SA 4.0 for non-commercial use.

- <a id="paper"></a>**Paper:** arXiv link coming soon.
- <a id="dataset"></a>**Dataset:** [Hugging Face](https://huggingface.co/datasets/Wayne42/gangqinpu).
- **Code:** [GitHub](https://github.com/wei-zeng98/gangqinpu).

## Read with Hugging Face Datasets

Python ≥ 3.10. Accept the license and confirm non-commercial use on the [dataset page](https://huggingface.co/datasets/Wayne42/gangqinpu), then install and sign in:

```bash
python -m pip install "datasets==4.4.1"
hf auth login
```

```python
from datasets import load_dataset

dataset = load_dataset("Wayne42/gangqinpu", token=True)
sample = dataset["train"][0]
```

The default configuration has 31 columns: 25 metadata fields plus `performance_notes`, `score_notes`, `alignment`, `beats`, `notation` and `performance_pedal`. See the [dataset card](https://huggingface.co/datasets/Wayne42/gangqinpu#dataset-structure) for the full schema and [splits](https://huggingface.co/datasets/Wayne42/gangqinpu#splits-and-groups).

To stream the test split:

```python
test = load_dataset(
    "Wayne42/gangqinpu", split="test", streaming=True, token=True
)
test_sample = next(iter(test))
```

For reproducible Hub downloads, pass the same `revision` commit SHA to each configuration load.

### Read local Parquet

Read the downloaded default shards:

```python
from pathlib import Path
from datasets import load_dataset

root = Path("LOCAL_PARQUET_ROOT")
files = {
    split: sorted(str(p) for p in (root / "data").glob(f"{split}-*.parquet"))
    for split in ("train", "validation", "test")
}
local_data = load_dataset("parquet", data_files=files, streaming=True)
sample = next(iter(local_data["train"]))
```

### Score time in quarter notes

`score_notes` includes `onset_quarter` and `duration_quarter` in quarter-note units; `time` and `duration` remain in score-MIDI seconds. Performance-note times are audio seconds.

Alignment indices refer to the saved note order. See the [alignment example](https://huggingface.co/datasets/Wayne42/gangqinpu#score-time-in-quarter-notes) and [time-origin conversion](https://huggingface.co/datasets/Wayne42/gangqinpu#score-time-origins).

### Meter and key signatures

Each beat carries `time_signature`, `beats_per_bar` and `key_signature`. The key signature's `fifths` value counts sharps or flats. See the [field definitions and example](https://huggingface.co/datasets/Wayne42/gangqinpu#meter-and-key-signatures).

## Install

Install the package for construction, classification or original-file export:

```bash
git clone https://github.com/wei-zeng98/gangqinpu
cd gangqinpu
```

Construction and classification:

```bash
pip install -e ".[pipeline,genre]"
```

Filesystem reader and original-file export:

```bash
pip install -e ".[hub]"
```

For the filesystem reader alone, use `pip install -e .`. The construction pipeline uses Python 3.11 and requires `ffmpeg` and `ffprobe` on PATH. The bundled classifier requires scikit-learn 1.6.1.

### Optional original files and filesystem reader

Load `originals` to restore the 11 files per sample:

```python
from datasets import load_dataset
from gangqinpu import GangqinpuDataset, export_original_files

originals = load_dataset("Wayne42/gangqinpu", name="originals", token=True)
asap = load_dataset("Wayne42/gangqinpu", name="asap_reference", token=True)
# Use a directory without an existing manifest; reference files stay external.
export_original_files(originals, dataset, "gangqinpu", references=asap)
train = GangqinpuDataset("gangqinpu", split="train")
pairs = train[0].note_alignment()  # observed matches; excludes interpolated notes
```

Configurations join by `sample_id`. Export restores the files byte-for-byte under `gangqinpu_v1.0.0/<sample_id>/` and writes `gangqinpu_meta_v1.0.0.csv`. The filesystem reader selects Gangqinpu rows; optional `references=asap` adds the 1,066 external reference rows to the manifest.

```bash
python examples/read_dataset.py --root gangqinpu
```

## Original file layout

The 11 files include source metadata, raw/cleaned/refined MIDI, raw/refined alignment NPZ, and JSON/TSV annotations. See the [file table and format definitions](https://huggingface.co/datasets/Wayne42/gangqinpu#original-files-originalssplitifiles).

Source audio, original CCMZ/MusicXML scores and (n)ASAP files are not bundled.

## Recover source files

Audio and original scores are not redistributed. Download from the recorded source URLs:

```bash
python tools/download_sources.py --root gangqinpu --ids 1000009
```

Omit `--ids` for all samples. After optional original-file export, audio goes to `gangqinpu_audio/<id>/audio.<ext>`, with its extension inferred from the source URL; scores go to `gangqinpu_v1.0.0/<id>/score.ccmz`. Existing files are skipped; failures are logged. Links may expire. Respect the source site's terms and keep requests infrequent.

Convert CCMZ to `score.musicxml` with a tool such as [ccmz-to-midi](https://github.com/bszapp/ccmz-to-midi) (GPL-3.0, not bundled). The original converter is not included; another converter may produce different MusicXML and downstream results.

## Structured notation

`notation.version = 2` identifies the notation schema.

`notation` stores source parts, tie-merged notes, staff/voice labels, grace flags, unfolded measures, and direct performance-note links. Grace duration is zero. `alignment_complete` is true when every performance note, including interpolated notes, has exactly one source-note link; `issues` records extraction diagnostics.

Indices are zero-based. `notation.notes[*].performance_note_index` indexes
`performance_notes`; a missing link is null. `alignment.performance_note_indices[i]` links
`score_notes[i]` to `performance_notes`, with `-1` for an unmatched note. Part and measure indices address `notation.parts` and
`notation.measures`. Staff indices start at 0 within each part; voice integers
are local labels. Unknown labels are null.

Notation and beat quarter positions use the construction-score origin; subtract `notation.midi_origin_offset_quarter` for the saved-MIDI origin. Performance, beat and pedal timestamps are audio seconds. `performance_pedal` stores track-0 CC64 `{time, value}` events from `performance_clean.mid`; value ≥64 means pedal down.

See the [dataset card](https://huggingface.co/datasets/Wayne42/gangqinpu#structured-notation)
for complete field definitions. Example:

```python
notes = sample["notation"]["notes"]
matched = [n for n in notes if n["performance_note_index"] is not None]
for note in matched[:3]:
    performed = sample["performance_notes"][note["performance_note_index"]]
    print(note["pitch"], note["staff"], note["onset_quarter"], performed["time"])
```

### Build structured data

Install `pip install -e ".[hub,notation]"` to extract labels and source hashes from recovered local `score.musicxml`:

```bash
python tools/enrich_parquet.py --input OLD_PARQUET_ROOT \
  --samples LOCAL_SAMPLE_ROOT --output NEW_PARQUET_ROOT --workers 8
```

`LOCAL_SAMPLE_ROOT` contains `<sample_id>/score.musicxml` and the original
symbolic files. `NEW_PARQUET_ROOT` must be new or empty. This command updates
the default shards; `tools/pack_dataset.py` builds all three configurations.

## Construction and classification

### Inputs and execution

```text
ROOT/gangqinpu_v1.0.0/<id>/score.musicxml
ROOT/gangqinpu_v1.0.0/<id>/metadata.json
ROOT/gangqinpu_audio/<id>/audio.<ext>
```

`SOURCE_URLS.csv` must contain `sample_id,audio_url,score_ccmz_url`; the released metadata CSV can be used directly. The source CCMZ-to-MusicXML converter is not included.

```bash
export DASHSCOPE_API_KEY=...  # Qwen-Flash metadata labelling
bash scripts/run_pipeline.sh ROOT/gangqinpu_v1.0.0 ROOT/gangqinpu_audio REPORT_DIR SOURCE_URLS.csv \
  --clap-model models/clap_music
```

The manifest is written to `ROOT/gangqinpu_meta_v1.0.0.csv`; reports go to `REPORT_DIR`. Without `--clap-model`, only the metadata genre stage runs; metadata-unknown samples stay unknown. Per-sample stages accept `--ids`; inspect `--help` for other options. Individual steps 1–6 accept `--workers` and `--force`; the wrapper forwards `--force`.

Successful outputs are reused. `--force` recomputes a stage and can recheck its own rejection and downstream rejections, while preserving earlier filter gates. After changing an upstream input, rerun downstream stages with `--force` as well. The wrapper applies it to all per-sample stages. Per-sample filter evidence is kept in `pipeline_checks.json`; work tables are in `.work/`.

API errors and audio decode errors are failures, not genre labels. Retry failed stages before continuing. Steps 8–10 regenerate their reports.

### Stages

| Script | Paper | What it does | Writes |
|---|---|---|---|
| `step01_filter_sources.py` | Filters 1, 2 | Audio duration ≥ 60 s (ffprobe). Score has ≥ 32 measures and only piano parts sound. | `pipeline_checks.json` |
| `step02_transcribe.py` | Step 1 | Decode compressed audio to mono 44.1 kHz WAV; existing WAV is reused. Transkun v2 (default checkpoint, no pedal-extended offsets). `--device auto/cpu/cuda:N` | `audio_transkun.mid` |
| `step03_prepare_symbolic.py` | Step 1 | Score: Partitura load, ordinary two-pass repeats/first endings unfolded, grace notes and ornaments expanded, duplicates removed (SyMuPe) → score MIDI, score note table, metrical grid. Performance: merged to one track; duplicate and very short notes removed; same-pitch overlaps cut. | `score_MS.mid`, `performance_clean.mid` |
| `step04_align.py` | Steps 2, 3; Filter 3 | DualDTW note matching (Parangonar via SyMuPe), then RAScoP refinement (hole filling, onset cleaning, interpolation of missing notes; no performance synchronisation). Rejects pairs whose raw insertion or deletion rate is > 10 %. | `performance_align.npz`, `performance_refined_align.npz`, `score_refined.mid`, `performance_refined.mid` |
| `step05_annotate.py` | Step 4 | Transfers score beats/downbeats to performance time through the refined alignment, with per-beat confidence. | `performance_annotations.txt`, `midi_score_annotations.txt`, `alignment_metadata.json` |
| `step06_classify_expressive.py` | Filter 4 | Expressive-performance classifier on the raw alignment; rejects p < 0.5. | `alignment_metadata.json` (`expressive_performance_classifier`) |
| `step07_genre.py` | Step 5 | Qwen-Flash on the metadata (`DASHSCOPE_API_KEY`); LAION-CLAP on the audio for samples left `unknown`. | `.work/genre.json` |
| `step08_group_scores.py` | Step 6 | Score-similarity grouping on `score_MS.mid`. Optionally joint with a reference corpus (`--reference-csv source,sample_id,score_midi,work_id`, e.g. ASAP). | `groups.csv`, `pair_evidence.jsonl` |
| `step09_split.py` | Step 6 | Group-disjoint, genre-balanced split (`--heldout-size`, `--protect-test`). Excludes held-out samples whose audio is byte-identical to a training sample. | `splits.csv` |
| `step10_export_metadata.py` | – | Writes the metadata CSV (same columns as the release). | `gangqinpu_meta_v1.0.0.csv` |

#### Method details

- **Beat confidence (step 5).** A beat uses matched notes within ±0.05 quarters of its score onset and takes their median performed onset, with confidence 0.95. If it is supported only by interpolated notes, confidence is 0.40. A mixed set uses both observed and interpolated onsets. Other beats are interpolated between onset anchors (confidence = 0.75 × the lower onset-anchor confidence; onset anchors have their own support-based scores) or extrapolated with the local tempo (confidence 0.25). Beats inside alignment holes are multiplied by 0.35. Beats with confidence < 0.35 are labelled `bR`. Compound x/8 meters use dotted-quarter beats. Initial pickups and overfull measures are handled explicitly.
- **Genre (step 7).** The metadata prompt is defined in `gangqinpu/pipeline/genre.py`. CLAP compares three non-silent 10 s excerpts with 13 piano-specific style prompts. A label is accepted only if the top cosine is ≥ 0.20, the coarse-category margin is ≥ 0.015 and at least 2 of 3 clips agree. Otherwise the sample stays `unknown`. The audio stage needs the official LAION music checkpoint converted to Transformers format: download `music_audioset_epoch_15_esc_90.14.pt` from [lukewys/laion_clap](https://huggingface.co/lukewys/laion_clap), then run `python tools/convert_clap_checkpoint.py --checkpoint music_audioset_epoch_15_esc_90.14.pt --output models/clap_music`.
- **Grouping (step 8).** Scores are normalised to a 1/96-quarter grid. Candidates are retrieved with transposition-invariant 8-interval shingles of the upper and lower voices (TF-IDF cosine ≥ 0.04), then verified by anchored polyphonic note matching. `score_group_id` is a complete-link partition over *highly similar* pairs. `group_id` is the conservative connected component that also includes excerpts, related or uncertain pairs, and shared reference work IDs.

### Official partitions

For comparable experiments, use `split`, `group_id`, and `score_group_id` from the released CSV. The original release used joint grouping with ASAP, external MIDI2ScoreTransformer test-work protection, and subsequent whole-group adjustments to reach ASAP 955/50/61 and Gangqinpu 17,773/499/514. One duplicate-audio validation sample was removed.

For rebuilding known samples, add `--manifest RELEASED_METADATA.csv` to the wrapper to reuse the fixed assignments and retain the ASAP reference rows. Recomputed groups are checked against those assignments; conflicts fail instead of silently moving samples. The wrapper also accepts `--reference-csv` for joint score grouping, and `--heldout-size` / `--protect-test` when generating a new split. A new split is not the official v1.0.0 split.

The URL table and any reference manifest must be kept outside the output manifest path when rebuilding, since the final CSV replaces that path. All primary samples need source audio during duplicate screening.

### Reproducibility

Use the released symbolic files, genre labels and splits as the v1.0.0 reference. `requirements.txt` lists direct dependencies rather than a complete environment lock. Rebuilding may differ with another CCMZ converter, dependency version or Qwen-Flash API revision. Re-extraction can also change the ordering of simultaneous notes and the filling of inherited measure-key labels.

To build all Parquet configurations, install `pip install -e ".[hub,notation]"` and use an exported dataset with recovered `score.musicxml` files:

```bash
python tools/pack_dataset.py --root ROOT --output HF_OUTPUT --train-shards 2
```

`ROOT` contains `gangqinpu_meta_v1.0.0.csv` and `gangqinpu_v1.0.0/<sample_id>/`. `HF_OUTPUT` must be outside `ROOT`, with empty `data/`, `originals/` and `metadata/` subdirectories. The filesystem export restores the 11 original files; recovered XML is added separately for rebuilding notation. Copy the dataset card and the dataset’s CC BY-NC-SA 4.0 `LICENSE` into `HF_OUTPUT` separately.

### Expressive-performance classifier

The bundled classifier predicts expressive timing and dynamics from eight score-informed features of raw note matches.

```bash
python scripts/classify_expressive.py --score score.musicxml --performance performance.mid
```

**Model.** `gangqinpu/classifier/model.joblib` is a scikit-learn pipeline: StandardScaler, then an MLP with hidden layers (32, 16), ReLU, Adam, L2 α = 1e-3 and early stopping. The table follows the input order in `FEATURES`. Explicit grace-note flags are excluded when present; construction expands grace notes and ornaments before building the score table, which does not retain those flags.

| Feature | Definition |
|---|---|
| `tempo_residual_std_ms` | Std. of onset residuals from a local linear score-to-performance time fit (±4 quarters, ≥ 6 onsets) |
| `chord_spread_med_ms` | Median onset range within notated simultaneities |
| `vel_std` | Std. of note velocities |
| `vel_unique` | Number of distinct velocities |
| `voicing_boost` | Mean velocity of upper-staff notes minus that of other staves |
| `pitch_boost` | Mean velocity difference between the highest note of a chord and the rest |
| `tempo_residual_autocorr_lag1` | Lag-1 autocorrelation of those residuals |
| `onset_entropy` | Shannon entropy of residuals in 10 ms bins |

**Training data and accuracy.**

- 779 expressive and 370 non-expressive Gangqinpu performances, labelled by two experts by consensus.
- 982 ASAP performances as additional positives.
- Synthetic score renderings with small onset/velocity noise as negatives, grouped by source score for splitting.
- 75/25 train/held-out split per real source; minority-class oversampling is applied only to training data.

On the 288 expert-labelled held-out samples, accuracy is 98.61 % and macro-F1 is 98.41 % (see the paper).

## License and citation

Code: [MIT](LICENSE), except the Apache-2.0 CLAP converter. Classifier weights: [CC BY-NC-SA 4.0](LICENSES/CC-BY-NC-SA-4.0.txt). See [third-party notices](THIRD_PARTY_NOTICES.md). The separately hosted dataset is also released under CC BY-NC-SA 4.0. Classifier weights are included in this repository; the Hugging Face download confirmations apply only to the dataset. Underlying works and downloaded sources remain subject to their rights holders' terms.

```bibtex
@misc{gangqinpu2026,
  title  = {Gangqinpu: A Multi-Genre, Multi-Task Piano Dataset with Aligned Audio, {MIDI}, and Musical Scores},
  author = {Zeng, Wei and Mathur, Anvay and Wang, Ye},
  year   = {2026},
  note   = {Manuscript}
}
```

The pipeline builds on [Transkun](https://github.com/Yujia-Yan/Transkun), [Partitura](https://github.com/CPJKU/partitura), [Parangonar](https://github.com/sildater/parangonar), [SyMuPe / RAScoP](https://github.com/ilya16/SyMuPe), [symusic](https://github.com/Yikai-Liao/symusic), [LAION-CLAP](https://github.com/LAION-AI/CLAP), and Qwen-Flash. Beat annotation text follows [ASAP](https://github.com/fosfrancesco/asap-dataset).

## Authors and contact

[Wei Zeng](https://wei-zeng98.github.io/), Anvay Mathur, and Ye Wang

Contact: [w.zeng@u.nus.edu](mailto:w.zeng@u.nus.edu)
