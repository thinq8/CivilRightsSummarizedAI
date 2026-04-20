#!/bin/bash --login
#SBATCH --job-name=prep_training_data
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out

# ── Phase 1: Prepare training data with document-aware fragmentation ──
# No GPU needed — this is CPU-only (heuristic extraction).
# For Claude extraction, run locally with ANTHROPIC_API_KEY instead.

module purge
module load Miniforge3
conda activate legal-sum

PROJECT_DIR=/mnt/research/Civil_Rights_Summaries/Fragmentation
REPO_DIR="$PROJECT_DIR/CivilRightsSummarizedAI"
RAW_DATA="$PROJECT_DIR/output/First_Train"
OUTPUT_DIR="$REPO_DIR/data/training_v2"

cd "$REPO_DIR"

mkdir -p "$OUTPUT_DIR"

echo "=== Preparing training data with extract_first (heuristic) ==="

# Process all 3 splits
for split in train val test; do
    echo "Processing ${split}..."
    python scripts/prepare_training_data.py \
        --input "$RAW_DATA/${split}.jsonl" \
        --output "$OUTPUT_DIR/${split}.jsonl" \
        --strategy extract_first \
        --max-tokens 24000
done

echo "=== Data preparation complete ==="
ls -lh "$OUTPUT_DIR"/*.jsonl

conda deactivate
