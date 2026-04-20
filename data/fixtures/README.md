# Fixture Data

This directory contains small, shareable files used for tests, demos, and figure reproduction. These files are intentionally not the full training corpus.

## Files

| File | Purpose | Provenance |
|------|---------|------------|
| `mock_dataset.json` | Deterministic two-case fixture for mock ingestion tests and the out-of-the-box demo | Hand-built safe fixture matching the subset of the Clearinghouse API shape used by the ingestion pipeline |
| `trainer_state.json` | Source data for the training-loss figure | Extracted from the successful LoRA training run's `trainer_state.json`; contains training/eval loss log history, not model weights |
| `test_chunk_counts.json` | Source data for the document-fragmentation figure | Extracted from metadata fields in the 1,231-record private test split; excludes source document text and reference summaries |
| `eval_summary.json` | Fallback aggregate metrics for evaluation comparison figures | Produced from the 50-case evaluation sample after running generation, scoring, and QA triage |
| `eval_scores.jsonl` | Fallback per-record scores for judge/score distribution figures | Reduced evaluation output used only for plotting; excludes full private source documents |

## Why Fixtures Exist

The full project used large JSONL training and evaluation files derived from Clearinghouse case records. Those files are too large for git and can include partner-controlled material. The fixtures keep the public repository reproducible enough for grading:

- Unit tests run without API credentials.
- The mock ingestion demo creates a real SQLite database.
- Figure reproduction works even when the full model outputs are unavailable.
- Reviewers can inspect the data shape without downloading gigabytes of training data.

## Regenerating Fixture-Like Artifacts

The full training split lives outside git as:

```text
data/training/train.jsonl
data/training/val.jsonl
data/training/test.jsonl
```

Model-ready prepared data is regenerated with:

```bash
python scripts/prepare_training_data.py \
  --input data/training/train.jsonl \
  --output data/training_v2/train.jsonl \
  --strategy extract_first \
  --max-tokens 24000
```

Evaluation outputs are regenerated with:

```bash
cd eval
python generate.py --source claude --sample 50
python evaluate.py results/generations_claude_*.jsonl --judge-sample 25
cd ..
```

See `TUTORIAL.md` for the full local, HPCC, evaluation, and figure workflow.
