#!/bin/bash --login
#SBATCH --job-name=qwen7b_lora_run2_polish
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=a100:1
#SBATCH --exclude=nfh-002
#SBATCH --output=%x-%j.out

# Final polish run: resumes from the latest checkpoint (expected: checkpoint-3000)
# and finishes the remaining ~690 steps so the LR schedule fully anneals to ~0.
# Previous run hit 8hr wall-time at step 3249/3690 with loss plateauing; 12hr
# here is ~2x the headroom needed for the remaining steps.

module purge
module load Miniforge3
conda activate legal-sum

PROJECT_DIR=/mnt/research/Civil_Rights_Summaries/Fragmentation
export HF_HOME="$PROJECT_DIR/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"

cd "$PROJECT_DIR"

PREP_DATA="$PROJECT_DIR/training_v2"
RUN_DIR="$PROJECT_DIR/runs/qwen25_7b_lora_run2"

# Find the latest checkpoint automatically (expected: checkpoint-3000)
LATEST_CKPT=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found in $RUN_DIR"
    exit 1
fi

echo "============================================"
echo "POLISH RUN — resuming from: $LATEST_CKPT"
echo "Target: complete 3 epochs, fully anneal LR"
echo "Wall time: 12:00:00"
echo "============================================"

python3 Scripts/train_lora.py \
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
    --save-steps 250 \
    --eval-steps 500 \
    --resume-from-checkpoint "$LATEST_CKPT"

echo ""
echo "============================================"
echo "Polish run complete!"
echo "Final checkpoint should be at max_steps (3690)"
echo "============================================"

conda deactivate
