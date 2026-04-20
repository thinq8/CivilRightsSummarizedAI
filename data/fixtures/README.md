# Fixture Data

This directory contains small, shareable data artifacts for tests, demos, and figure reproduction. It does not contain the full training corpus, private source text, reference summaries, model checkpoints, or raw model outputs.

| File | Purpose | Provenance | Contains private text? | Contains model outputs? | Used by |
|------|---------|------------|------------------------|-------------------------|---------|
| `mock_dataset.json` | Deterministic two-case ingestion fixture | Hand-built synthetic Clearinghouse-like records matching the API shape used by the ingestion pipeline | No | No | `pytest`, `python -m clearinghouse.cli ingest-mock` |
| `trainer_state.json` | Training dynamics plot input | Reduced Hugging Face Trainer log history from `Polish/checkpoint-3690/trainer_state.json`, the same source used in `project_showcase.ipynb` | No | No | Figure 1, `notebooks/figure_instructions.ipynb` |
| `test_chunk_counts.json` | Document fragmentation metadata | Extracted counts from the 1,231-record test split without document text or summaries | No | No | Legacy checks and context for Figure 2 |
| `eval_summary.json` | Aggregate evaluation fixture from the earlier four-figure notebook | Reduced aggregate metrics from the evaluation pipeline | No | No | Backward-compatible figure fallback |
| `eval_scores.jsonl` | Per-record reduced score fixture from the earlier four-figure notebook | Reduced score rows from evaluation output, without source documents or full summaries | No | No | Backward-compatible figure fallback |
| `final_report_metrics.json` | Final seven-figure report plotting fixture | Reduced aggregate metrics extracted with the original `project_showcase.ipynb` logic from scored CSVs, QA JSONL reports, attribution JSONL reports, and prompt-length metadata | No | No | Figures 2-7, `notebooks/figure_instructions.ipynb` |

## Why Fixtures Exist

The full project used large JSONL files derived from Clearinghouse case records and human-written summaries. Those files are too large for git and may include partner-controlled material. The fixtures keep the public repository useful for review:

- Unit tests run without API credentials.
- The mock ingestion demo creates a real SQLite database.
- The final figure notebook uses the original local project artifacts when they are present next to the repository, and otherwise reproduces all seven final report figures from non-private aggregate data.
- Reviewers can inspect the data shape and pipeline without downloading gigabytes of training data.

## Regenerating Larger Artifacts

Raw training splits belong outside git:

```text
data/training/train.jsonl
data/training/val.jsonl
data/training/test.jsonl
```

Prepared training data is regenerated with:

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

See [TUTORIAL.md](../../TUTORIAL.md) for the full local, HPCC, evaluation, QA, and figure workflow.
