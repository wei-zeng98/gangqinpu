#!/usr/bin/env bash
# Build from recovered audio + converted MusicXML. See README.md#construction-and-classification.
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo "Usage: $0 SAMPLES_DIR AUDIO_DIR REPORT_DIR SOURCE_URLS_CSV [--clap-model DIR] [--manifest CSV] [--reference-csv CSV] [--heldout-size N] [--protect-test FILE] [--force]" >&2
  exit 2
fi
SAMPLES=$1; AUDIO=$2; OUT=$3; URLS=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python}
ROOT=$(cd "$SAMPLES/.." && pwd)
COMMON=(--samples-dir "$SAMPLES" --audio-dir "$AUDIO")
GROUPING=(); SPLITTING=(); EXPORT=(); CLAP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clap-model|--manifest|--reference-csv|--heldout-size|--protect-test)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --clap-model) CLAP=$2 ;;
        --manifest) SPLITTING+=(--manifest "$2"); EXPORT+=(--reference-metadata "$2") ;;
        --reference-csv) GROUPING+=(--reference-csv "$2") ;;
        *) SPLITTING+=("$1" "$2") ;;
      esac
      shift 2 ;;
    --force) COMMON+=(--force); shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -f "$URLS" ]] || { echo "Source URL CSV does not exist" >&2; exit 2; }
for step in 01_filter_sources 02_transcribe 03_prepare_symbolic 04_align 05_annotate 06_classify_expressive; do
  "$PYTHON" "$HERE/step${step}.py" "${COMMON[@]}"
done
if [[ -n "$CLAP" ]]; then
  "$PYTHON" "$HERE/step07_genre.py" "${COMMON[@]}" --stage all --clap-model "$CLAP"
else
  "$PYTHON" "$HERE/step07_genre.py" "${COMMON[@]}" --stage metadata
fi
"$PYTHON" "$HERE/step08_group_scores.py" --samples-dir "$SAMPLES" --output "$OUT" "${GROUPING[@]}"
# Step 9 has no --force option; its reports are regenerated.
"$PYTHON" "$HERE/step09_split.py" --samples-dir "$SAMPLES" --audio-dir "$AUDIO" \
  --groups "$OUT/groups.csv" --output "$OUT" "${SPLITTING[@]}"
"$PYTHON" "$HERE/step10_export_metadata.py" --samples-dir "$SAMPLES" --splits "$OUT/splits.csv" \
  --groups "$OUT/groups.csv" --source-urls "$URLS" --output "$ROOT/gangqinpu_meta_v1.0.0.csv" "${EXPORT[@]}"
