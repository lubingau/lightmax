#!/usr/bin/env bash
set -e

# ------------------------------------------------------------------
# Run all tasks with LightMax for a single
# representative saturation value.
# Usage:  ./run_lightmax.sh  [--dry-run] [--num-samples N] [--saturation N] [--in-bw N] [--out-bw N]
#         [--k-bits N] [--s-bits N] [--csv-name STR]
# ------------------------------------------------------------------

NUM_SAMPLES=0
SATURATION=24
IN_BW=8
OUT_BW=8
K_BITS=4
S_BITS=6
BASE="$(dirname "$0")"
CSV_NAME="_lightmax_sat_${SATURATION}_bw_${IN_BW}_k${K_BITS}_s${S_BITS}"
DRY=

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=echo; shift ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --saturation) SATURATION="$2"; shift 2 ;;
        --in-bw) IN_BW="$2"; shift 2 ;;
        --out-bw) OUT_BW="$2"; shift 2 ;;
        --k-bits) K_BITS="$2"; shift 2 ;;
        --s-bits) S_BITS="$2"; shift 2 ;;
        --csv-name) CSV_NAME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== run_lightmax.sh ==="
echo "Saturation: $SATURATION, in_bw=$IN_BW, out_bw=$OUT_BW, k_bits=$K_BITS, s_bits=$S_BITS"
echo

# ---- Image ----
$DRY /home/lubin/anaconda3/bin/python "$BASE/eval_image.py" \
    --saturation $SATURATION \
    --use-lightmax --no-baseline \
    --in-bw $IN_BW --out-bw $OUT_BW --k-bits $K_BITS --s-bits $S_BITS \
    --num-samples $NUM_SAMPLES \
    --csv-name "$CSV_NAME" \
    --output-dir "$BASE/panda"

# ---- Language ----
$DRY /home/lubin/anaconda3/bin/python "$BASE/eval_language.py" \
    --saturation $SATURATION \
    --use-lightmax --no-baseline \
    --in-bw $IN_BW --out-bw $OUT_BW --k-bits $K_BITS --s-bits $S_BITS \
    --num-samples $NUM_SAMPLES \
    --csv-name "$CSV_NAME" \
    --output-dir "$BASE/panda"

# ---- Speech ----
$DRY /home/lubin/anaconda3/bin/python "$BASE/eval_speech.py" \
    --saturation $SATURATION \
    --use-lightmax --no-baseline \
    --in-bw $IN_BW --out-bw $OUT_BW --k-bits $K_BITS --s-bits $S_BITS \
    --num-samples $NUM_SAMPLES \
    --csv-name "$CSV_NAME" \
    --output-dir "$BASE/panda"

# ---- Time series ----
$DRY /home/lubin/anaconda3/bin/python "$BASE/eval_time_series.py" \
    --saturation $SATURATION \
    --use-lightmax --no-baseline \
    --in-bw $IN_BW --out-bw $OUT_BW --k-bits $K_BITS --s-bits $S_BITS \
    --num-samples $NUM_SAMPLES \
    --csv-name "$CSV_NAME" \
    --output-dir "$BASE/panda"

echo
echo "=== All LightMax evaluations complete ==="