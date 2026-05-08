#!/bin/bash
#
# Run KE methods on the unified merged_eval_dataset.json (51 samples × 75 Qs).
#
# Default pipeline (workshop paper): prompt_v2 -> ROME -> FT-M -> GRACE -> WISE.
# Released labeled outputs and post-processing utilities are under results/.
#
# Usage:
#   bash run_merged_eval.sh [methods] [gpu_id] [data_size] [start_index]
#
# Examples:
#   bash run_merged_eval.sh                       # full pipeline on all 51, GPU 0
#   bash run_merged_eval.sh ROME 0                # only ROME on GPU 0
#   bash run_merged_eval.sh "prompt_v2 ROME" 0 3  # sanity: 3 samples
#   bash run_merged_eval.sh FT-M 0 51 21          # FT-M on hb/zs slice only
#

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---- Args ----
METHODS_STR=${1:-"prompt_v2 ROME FT-M GRACE WISE"}
GPU_ID=${2:-0}
DATA_SIZE=${3:-51}
START_INDEX=${4:-0}

read -ra METHODS <<< "$METHODS_STR"

TOPIC=merged_eval_dataset
RESULTS_DIR="${REPO_ROOT}/results/merged_eval"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "========================================"
echo "Merged Eval Dataset Run"
echo "========================================"
echo "Methods    : ${METHODS[@]}"
echo "GPU        : $GPU_ID"
echo "Data size  : $DATA_SIZE  (start_index=$START_INDEX)"
echo "Topic      : $TOPIC"
echo "Output dir : $RESULTS_DIR/<METHOD>/llama3_8b/${TOPIC}_<METHOD>_post.json"
echo "========================================"

# Pin GPU enumeration to PCI bus order so $GPU_ID matches `nvidia-smi` indices.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID

CONDA_ENV=${CONDA_ENV:-easyedit}
CONDA_SH=${CONDA_SH:-"$HOME/miniforge3/etc/profile.d/conda.sh"}

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "Conda is required. Set CONDA_SH=/path/to/conda.sh or initialize conda before running."
    exit 1
fi

conda activate "$CONDA_ENV"

for METHOD in "${METHODS[@]}"; do
    OUT_DIR="${RESULTS_DIR}/${METHOD}"
    mkdir -p "$OUT_DIR"
    PER_METHOD_LOG="${LOG_DIR}/merged_eval_${METHOD}.log"

    echo ""
    echo "----------------------------------------"
    echo "Running $METHOD ($DATA_SIZE samples from idx=$START_INDEX)"
    echo "Per-method log: $PER_METHOD_LOG"
    echo "Started at: $(date)"
    echo "----------------------------------------"

    python code/intervention/run_with_summary_metrics_no_acc.py \
        --edit_method "$METHOD" \
        --model_name llama3-8b \
        --topic_name "$TOPIC" \
        --use_json \
        --dataset_dir "${REPO_ROOT}/data" \
        --hparams_dir "${REPO_ROOT}/code/hparams" \
        --data_size "$DATA_SIZE" \
        --start_index "$START_INDEX" \
        --results_dir "$OUT_DIR" \
        --device_edit 0 \
        --device_eval 0 \
        --max_new_tokens 80 \
        $( [ "$START_INDEX" -gt 0 ] && echo --append ) \
        2>&1 | tee "$PER_METHOD_LOG"

    echo "Finished $METHOD at: $(date)"
done

echo ""
echo "========================================"
echo "All methods complete."
echo "Results saved under: $RESULTS_DIR/<METHOD>/llama3_8b/"
echo "========================================"
