#!/usr/bin/env bash
set -e

# ------------------------------------------------------------------
# Sweep all tasks across multiple saturation thresholds with
# baseline, using plain saturation hooks (no LightMax).
# Usage:  ./run_all_saturations.sh  [--dry-run] [--num-samples N]
#         [--csv-name STR] [--in-bw N] [--out-bw N]
# ------------------------------------------------------------------

NUM_SAMPLES=0
SATURATIONS="4 8 12 16 24 32 64"
IN_BW=8
OUT_BW=8
BASE="$(dirname "$0")"
CSV_NAME="_saturations"
DRY=

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=echo; shift ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --in-bw) IN_BW="$2"; shift 2 ;;
        --out-bw) OUT_BW="$2"; shift 2 ;;
        --csv-name) CSV_NAME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== run_all_saturations.sh ==="
echo "Saturations: $SATURATIONS (with baseline)"
echo

# ---- Image ----
$DRY python3 "$BASE/eval_image.py" \
    --saturation $SATURATIONS \
    --in-bw $IN_BW --out-bw $OUT_BW \
    --num-samples $NUM_SAMPLES \
    --csv-name "$CSV_NAME" \
    --output-dir "$BASE/panda"

# ---- Language ----
$DRY python3 "$BASE/eval_language.py" \
    --saturation $SATURATIONS \
    --in-bw $IN_BW --out-bw $OUT_BW \
    --num-samples $NUM_SAMPLES \
    --csv-name "$CSV_NAME" \
    --output-dir "$BASE/panda"

# ---- Speech ----
$DRY python3 "$BASE/eval_speech.py" \
    --saturation $SATURATIONS \
    --in-bw $IN_BW --out-bw $OUT_BW \
    --num-samples $NUM_SAMPLES \
    --csv-name "$CSV_NAME" \
    --output-dir "$BASE/panda"

# ---- Time series ----
$DRY python3 "$BASE/eval_time_series.py" \
    --saturation $SATURATIONS \
    --in-bw $IN_BW --out-bw $OUT_BW \
    --num-samples $NUM_SAMPLES \
    --csv-name "$CSV_NAME" \
    --output-dir "$BASE/panda"

echo
echo "=== All saturation sweeps complete ==="