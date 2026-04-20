#!/bin/bash --login
#SBATCH --job-name=eval_ckpt3690
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=a100:1
#SBATCH --exclude=nfh-002
#SBATCH --output=%x-%j.out

# Eval checkpoint-3690 (final polish run):
#   Phase 1: Generate 50 summaries with LoRA (~25 min)
#   Phase 2: ROUGE scoring (~instant)
#   Phase 3: LLM-as-judge using base Qwen2.5 (~25 min)
# Total expected: ~1 hour. 4hr wall time for headroom.

module purge
module load Miniforge3
conda activate "${CONDA_ENV:-legal-sum}"

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export HF_HOME="$PROJECT_DIR/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"

cd "$PROJECT_DIR" || { echo "FATAL: Cannot cd to $PROJECT_DIR"; exit 1; }

pip install rouge-score --quiet 2>/dev/null

echo "============================================"
echo "Eval: checkpoint-3690 (polish final)"
echo "============================================"

python3 scripts/train_lora.py --help > /dev/null 2>&1  # warm up imports

python3 scripts/eval_checkpoint_v2.py \
    --checkpoint-dir "${RUN_DIR:-runs/qwen25_7b_lora_run2/checkpoint-3690}" \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --test-file "${TEST_FILE:-data/training_v2/test.jsonl}" \
    --num-samples 50 \
    --max-new-tokens 2048 \
    --output-file eval_ckpt3690.jsonl

echo ""
echo "============================================"
echo "Eval complete!"
echo "============================================"

conda deactivate
