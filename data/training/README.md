# Training Data

This directory holds the training, validation, and test splits for the legal case summarization model.

## Files (not tracked in git — too large)

| File | Size | Records | Description |
|------|------|---------|-------------|
| `train.jsonl` | ~2.8 GB | 9,841 | Training split |
| `val.jsonl` | ~387 MB | 1,231 | Validation split |
| `test.jsonl` | ~340 MB | 1,231 | Test split |

## How to obtain

These files were generated from Civil Rights Litigation Clearinghouse records and human-written reference summaries using the ingestion/export workflow. They are not bundled because they are large and may include partner-controlled material.

Use one of these sources:

1. Rebuild the data from approved Clearinghouse API access using the ingestion pipeline.
2. Copy the course/team-provided private artifact bundle, if you have been granted access.

To place them correctly, copy or symlink them into this directory:

```bash
# Example: copy from external location
cp /path/to/First_Train/*.jsonl data/training/
```

Expected final layout:

```text
data/training/train.jsonl
data/training/val.jsonl
data/training/test.jsonl
```

After these raw split files are present, build model-ready files with:

```bash
mkdir -p data/training_v2

python scripts/prepare_training_data.py \
  --input data/training/train.jsonl \
  --output data/training_v2/train.jsonl \
  --strategy extract_first \
  --max-tokens 24000
```

Repeat for `val.jsonl` and `test.jsonl`.

## Data format

Each JSONL record contains:

```json
{
  "id": "unique_id",
  "case_id": "case_identifier",
  "split": "train|val|test",
  "prompt": "Multi-chunk case documents with summarization instruction",
  "response": "Human-written reference summary",
  "completion": "Same as response",
  "source_chunk_count": 5,
  "used_chunk_count": 5
}
```

The `source_chunk_count` and `used_chunk_count` fields track document fragmentation — how many source documents were combined to produce each case summary.

## Why this is separate from fixtures

`data/fixtures/` contains small, shareable artifacts for tests and figure fallback. Those fixtures are enough to verify the code path and reproduce the bundled figures, but they are not enough to fine-tune or fully evaluate the model.
