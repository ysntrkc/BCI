#!/usr/bin/env bash
# One round of Pyramid Pix2Pix on BCI, MIST HER2 and HER2match with the SIU 2026 settings:
# train -> test (seeded 256 crop + full 1024) -> evaluate, one dataset after another.
#
# Usage: scripts/baseline_round.sh [variant] [pattern] [datasets...]
#   defaults: baseline L1_L2_L3 "BCI/combined MIST/HER2/combined HER2match/combined"
#   e.g.      scripts/baseline_round.sh dwt L1_L2_L3_dwt
# Run detached:  setsid nohup scripts/baseline_round.sh > /mnt/d/thesis/runs/_logs/round.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../PyramidPix2pix"

VARIANT=${1:-baseline}
PATTERN=${2:-L1_L2_L3}
shift $(( $# < 2 ? $# : 2 ))
DATASETS=${*:-BCI/combined MIST/HER2/combined HER2match/combined}
NAME=pyramidpix2pix/$VARIANT
RUNS=${RUNS:-/mnt/d/thesis/runs}          # override for smoke tests
EXTRA_TRAIN=${EXTRA_TRAIN:-}                  # e.g. "--max_dataset_size 8 --n_epochs 1 --n_epochs_decay 1"
EXTRA_TEST=${EXTRA_TEST:-}                    # e.g. "--num_test 10"
DATA=/mnt/d/thesis/datasets
LOGS=$RUNS/_logs
mkdir -p "$LOGS"

log() { echo "[$(date '+%F %T')] $*"; }

for DS in $DATASETS; do
    tag=$(date +%Y%m%d_%H%M%S)_${DS//\//_}
    log "train $NAME on $DS"
    if ! uv run python train.py --dataroot "$DATA/$DS" --name "$NAME" --pattern "$PATTERN" --seed 0 \
            --batch_size 20 --num_threads 8 --preprocess crop --crop_size 256 \
            --n_epochs 50 --n_epochs_decay 50 --save_epoch_freq 10 --print_freq 50 --display_id -1 \
            --checkpoints_dir "$RUNS" $EXTRA_TRAIN > "$LOGS/${tag}_train.log" 2>&1; then
        log "FAILED training on $DS, see $LOGS/${tag}_train.log"; continue
    fi
    RUN=$(grep -oP '>>> Run folder: \K\S+' "$LOGS/${tag}_train.log" | head -1)   # <name>/run_<ts>_<dataset>
    mv "$LOGS/${tag}_train.log" "$RUNS/$RUN/training_output.txt"
    log "trained $RUN"

    # test protocols: seeded 256 crop (same regions for every model) and the full 1024 image
    for proto in "crop256s0:--preprocess crop --crop_size 256 --seed 0" "full1024:--preprocess none"; do
        rtag=${proto%%:*}
        # shellcheck disable=SC2086
        if uv run python test.py --dataroot "$DATA/$DS" --name "$NAME" --run_name "$(basename "$RUN")" \
                --eval --num_test -1 --results_tag "$rtag" ${proto#*:} \
                --checkpoints_dir "$RUNS" --results_dir "$RUNS" $EXTRA_TEST > "$RUNS/$RUN/test_${rtag}.log" 2>&1 \
           && uv run python evaluate.py --result_path "$RUNS/$RUN" --images_subdir "test_latest_$rtag" \
                > "$RUNS/$RUN/eval_${rtag}.log" 2>&1; then
            log "tested+evaluated $RUN [$rtag]"
        else
            log "FAILED test/eval $RUN [$rtag], see $RUNS/$RUN/*_${rtag}.log"
        fi
    done
done

python3 ~/workspace/tez/tools/build_runs_index.py "$RUNS"
log "round done"
