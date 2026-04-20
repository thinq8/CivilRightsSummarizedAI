#!/bin/bash --login
#SBATCH --job-name=qwen7b_lora_run2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --gpus=h200:1
#SBATCH --gres-flags=enforce-binding

module purge
module load Miniforge3
conda activate "${CONDA_ENV:-legal-sum}"

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export HF_HOME="$PROJECT_DIR/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"

mkdir -p "$HF_HUB_CACHE"

cd "$PROJECT_DIR" || { echo "FATAL: Cannot cd to $PROJECT_DIR"; exit 1; }

SCRIPTS="$PROJECT_DIR/scripts"

# Verify scripts exist before doing anything
for f in "$SCRIPTS/prepare_training_data.py" "$SCRIPTS/doc_classifier.py" "$SCRIPTS/train_lora.py"; do
    if [ ! -f "$f" ]; then
        echo "FATAL: Missing $f"
        exit 1
    fi
done

# ── Phase 1: Prepare training data (CPU, ~10-30 min) ──────────────────────
echo "============================================"
echo "Phase 1: Preparing training data"
echo "============================================"

RAW_DATA="${RAW_DATA_DIR:-$PROJECT_DIR/data/training}"
PREP_DATA="${PREP_DATA_DIR:-$PROJECT_DIR/data/training_v2}"
mkdir -p "$PREP_DATA"

for split in train val test; do
    if [ ! -f "$PREP_DATA/${split}.jsonl" ]; then
        echo "Preparing ${split}..."
        python3 "$SCRIPTS/prepare_training_data.py" \
            --input "$RAW_DATA/${split}.jsonl" \
            --output "$PREP_DATA/${split}.jsonl" \
            --strategy extract_first \
            --max-tokens 24000
    else
        echo "${split}.jsonl already exists, skipping preparation"
    fi
done

echo ""
echo "Training data sizes:"
wc -l "$PREP_DATA"/*.jsonl

# ── Phase 2: LoRA fine-tuning (GPU, ~4-6 hrs) ─────────────────────────────
echo ""
echo "============================================"
echo "Phase 2: LoRA fine-tuning"
echo "============================================"

python3 "$SCRIPTS/train_lora.py" \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --train-file "$PREP_DATA/train.jsonl" \
    --val-file "$PREP_DATA/val.jsonl" \
    --output-dir "${RUN_DIR:-$PROJECT_DIR/runs/qwen25_7b_lora_run2}" \
    --max-length 8192 \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-5 \
    --num-train-epochs 3 \
    --warmup-ratio 0.05 \
    --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --logging-steps 10 \
    --save-steps 500 \
    --eval-steps 500

echo ""
echo "============================================"
echo "Training complete!"
echo "============================================"

conda deactivate
