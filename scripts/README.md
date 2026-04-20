# Scripts Guide

This directory contains reproducibility scripts outside the installable `src/clearinghouse/` package.

## Canonical Final-Reproduction Scripts

| Script | Role | Output |
|--------|------|--------|
| `prepare_training_data.py` | Convert raw case JSONL into model-ready prompts with document-aware context control | `data/training_v2/*.jsonl` |
| `train_lora.py` | Fine-tune `Qwen/Qwen2.5-7B-Instruct` with LoRA | `runs/qwen25_7b_lora_run2/` |
| `train_run2.sbatch` | Canonical HPCC training job for the final improved run | LoRA checkpoints |
| `eval_checkpoint_v2.py` | Generate summaries and score LoRA checkpoints | evaluation JSONL/summary files |
| `benchmark_claude.py` | Generate and judge Claude summaries for quality/cost comparison | Claude benchmark JSONL |
| `summary_qa.py` | Run structural QA triage over generated summaries | `qa_report.html`, `qa_report.csv`, `qa_report.jsonl` |

## Supporting or Legacy Scripts

| Script | Status | Purpose |
|--------|--------|---------|
| `doc_classifier.py` | Supporting | Classifies source documents into `PRIMARY`, `SUPPORTING`, `CONTEXTUAL`, or `EXCLUDED` tiers |
| `eval_checkpoint.py` | Quick check | Lightweight checkpoint generation + ROUGE scoring |
| `prepare_data.sbatch` | Supporting HPCC job | CPU-only prepared-data generation |
| `train_run2_resume.sbatch` | Supporting HPCC job | Resume interrupted training from latest checkpoint |
| `train_run2_polish.sbatch` | Historical HPCC job | Finish the run that produced checkpoint 3690 |
| `eval_ckpt3690.sbatch` | Historical HPCC job | Evaluate the final checkpoint for overfit comparison |
| `fetch_document.py` | Utility | Smoke-test live document fetching |
| `hydrate_document_text.py` | Utility | Bulk-hydrate document text from the API |
| `md_to_pdf.py` | Packaging helper | Convert `SHARING_PLAN.md` to PDF |

## Data Preparation

Raw split layout:

```text
data/training/train.jsonl
data/training/val.jsonl
data/training/test.jsonl
```

Prepare one split:

```bash
python scripts/prepare_training_data.py \
  --input data/training/train.jsonl \
  --output data/training_v2/train.jsonl \
  --strategy extract_first \
  --max-tokens 24000 \
  --extraction-backend heuristic
```

Use `--extraction-backend claude` with `ANTHROPIC_API_KEY` for higher-quality extraction.

## Training

Local GPU training uses `train_lora.py` directly. HPCC training uses the canonical SLURM template:

```bash
sbatch scripts/train_run2.sbatch
```

The SLURM templates default to the repository root but accept overrides:

```bash
PROJECT_DIR=/path/to/CivilRightsSummarizedAI \
RAW_DATA_DIR=/path/to/raw_splits \
PREP_DATA_DIR=/path/to/prepared_splits \
RUN_DIR=/path/to/runs/qwen25_7b_lora_run2 \
sbatch scripts/train_run2.sbatch
```

Conda appears in SLURM files because HPCC provides Python/CUDA through cluster modules. Local users should use the virtual environment in `INSTALL.md`.

## Evaluation

Checkpoint comparison:

```bash
python scripts/eval_checkpoint_v2.py \
  --checkpoint-dir runs/qwen25_7b_lora_run2/checkpoint-3000 \
  --test-file data/training_v2/test.jsonl \
  --num-samples 50 \
  --output-file eval_ckpt3000.jsonl
```

Claude benchmark:

```bash
export ANTHROPIC_API_KEY="YOUR_KEY"
python scripts/benchmark_claude.py \
  --model claude-sonnet-4-5 \
  --test-file data/training/test.jsonl \
  --num-samples 50 \
  --output-file eval_claude_sonnet.jsonl
```

## QA Triage

```bash
python scripts/summary_qa.py \
  --input eval_ckpt3000.jsonl \
  --output-dir qa_report_ckpt3000
```

Statuses:

- `PASS`: no critical or warning flags.
- `REVIEW`: warning flags; human review needed.
- `REJECT`: critical flags; regenerate or heavily edit.

The QA tool checks for malformed years, raw document artifacts, chat-template leakage, repetition loops, length collapse, missing legal elements, and suspicious style issues.

## Gitignored Outputs

The following are intentionally not committed:

- `data/training/*.jsonl`
- `data/training_v2/`
- `runs/*/checkpoint-*`
- model adapter weights and tokenizer files under `runs/`
- `eval/results/*.jsonl`
- `eval/results/*.json`
- local SQLite databases such as `data/dev.db`
