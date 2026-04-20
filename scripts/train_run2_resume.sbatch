#!/bin/bash --login
#SBATCH --job-name=qwen7b_lora_run2_resume
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=a100:1
#SBATCH --exclude=nfh-002
#SBATCH --output=%x-%j.out

module purge
module load Miniforge3
conda activate "${CONDA_ENV:-legal-sum}"

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export HF_HOME="$PROJECT_DIR/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"

cd "$PROJECT_DIR" || { echo "FATAL: Cannot cd to $PROJECT_DIR"; exit 1; }

PREP_DATA="${PREP_DATA_DIR:-$PROJECT_DIR/data/training_v2}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/runs/qwen25_7b_lora_run2}"

# Find the latest checkpoint automatically
LATEST_CKPT=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found in $RUN_DIR"
    exit 1
fi

echo "============================================"
echo "Resuming training from: $LATEST_CKPT"
echo "============================================"

python3 scripts/train_lora.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --train-file "$PREP_DATA/train.jsonl" \
    --val-file "$PREP_DATA/val.jsonl" \
    --output-dir "$RUN_DIR" \
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
    --eval-steps 500 \
    --resume-from-checkpoint "$LATEST_CKPT"

echo ""
echo "============================================"
echo "Training complete!"
echo "============================================"

conda deactivate
