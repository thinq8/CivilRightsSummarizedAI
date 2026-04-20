# Scripts Guide

This directory contains the project scripts that sit outside the installable `src/clearinghouse/` package. They are grouped by pipeline stage below.

## Data Preparation

| Script | Purpose | Typical input | Typical output |
|--------|---------|---------------|----------------|
| `doc_classifier.py` | Classify legal documents into priority tiers | Document title/type fields or prompt chunks | Tier labels: `PRIMARY`, `SUPPORTING`, `CONTEXTUAL`, `EXCLUDED` |
| `prepare_training_data.py` | Convert long raw case prompts into model-ready prompts | `data/training/{train,val,test}.jsonl` | `data/training_v2/{train,val,test}.jsonl` |
| `prepare_data.sbatch` | HPCC CPU-only batch job for heuristic training-data prep | Full raw training splits on HPCC | Prepared training splits |

The main strategy used for the final run was `extract_first`: extract structured facts from individual documents, then train the model to synthesize those facts into a narrative summary. This addresses document fragmentation, where a case can contain more text than the model context window can handle.

## Training

| Script | Purpose |
|--------|---------|
| `train_lora.py` | Fine-tune `Qwen/Qwen2.5-7B-Instruct` with LoRA adapters |
| `train_run2.sbatch` | Reproducible HPCC template for the final improved training run |
| `train_run2_resume.sbatch` | Resume a partially completed run from the latest checkpoint |
| `train_run2_polish.sbatch` | Finish the remaining steps after checkpoint 3000; used to produce checkpoint 3690 |

Final run settings:

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Max length | 8192 |
| Learning rate | `2e-5` |
| Epochs | 3 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Gradient accumulation | 8 |
| Gradient clipping | 1.0 |

The important result is that the final checkpoint was not the best checkpoint. Checkpoint 3000 had a better QA profile than checkpoint 3690.

## Evaluation and Benchmarking

| Script | Purpose |
|--------|---------|
| `eval_checkpoint.py` | Quick checkpoint evaluation with generation and ROUGE |
| `eval_checkpoint_v2.py` | Checkpoint evaluation with generation, ROUGE, and Qwen-as-judge option |
| `eval_ckpt3690.sbatch` | HPCC batch job for checkpoint 3690 evaluation |
| `benchmark_claude.py` | Generate and judge summaries with Claude for cost/quality comparison |

Use `eval_checkpoint_v2.py` for LoRA checkpoint comparisons:

```bash
python scripts/eval_checkpoint_v2.py \
  --checkpoint-dir runs/qwen25_7b_lora_run2/checkpoint-3000 \
  --test-file data/training_v2/test.jsonl \
  --num-samples 50 \
  --output-file eval_ckpt3000.jsonl
```

Use `benchmark_claude.py` to estimate the frontier-model quality ceiling and rough API cost:

```bash
export ANTHROPIC_API_KEY="YOUR_KEY"
python scripts/benchmark_claude.py \
  --model claude-sonnet-4-5 \
  --test-file data/training/test.jsonl \
  --num-samples 50 \
  --output-file eval_claude_sonnet.jsonl
```

## QA Triage

| Script | Purpose | Output |
|--------|---------|--------|
| `summary_qa.py` | Batch structural QA over AI-generated or human summaries | `qa_report.html`, `qa_report.csv`, `qa_report.jsonl` |

`summary_qa.py` is reference-free by default. It checks for failure modes observed during model evaluation and required Clearinghouse summary elements:

- garbled years and impossible dates
- raw document artifacts such as `[DOCUMENT]` or chat-template leakage
- repeated phrases or length collapse
- missing filing date, court, statute, remedy, or action type
- style issues such as `judgement` instead of `judgment`

Statuses:

- `PASS`: no critical or warning flags.
- `REVIEW`: warning flags; human review needed.
- `REJECT`: one or more critical flags; do not use without regeneration or major editing.

Example:

```bash
python scripts/summary_qa.py \
  --input eval_ckpt3000.jsonl \
  --output-dir qa_report_ckpt3000
```

## Report and Sharing Helpers

| Script | Purpose |
|--------|---------|
| `md_to_pdf.py` | Convert `SHARING_PLAN.md` to a PDF using ReportLab |

This is a packaging helper only; it is not part of the modeling pipeline.
