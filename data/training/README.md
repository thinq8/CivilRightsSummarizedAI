# Training Data

This directory holds the training, validation, and test splits for the legal case summarization model.

## Files (not tracked in git — too large)

| File | Size | Records | Description |
|------|------|---------|-------------|
| `train.jsonl` | ~2.8 GB | 9,841 | Training split |
| `val.jsonl` | ~387 MB | 1,231 | Validation split |
| `test.jsonl` | ~340 MB | 1,231 | Test split |

## How to obtain

These files were generated from the Civil Rights Litigation Clearinghouse data using the ingestion pipeline. Contact the project team or check the shared Teams directory for access.

To place them correctly, copy or symlink them into this directory:

```bash
# Example: copy from external location
cp /path/to/First_Train/*.jsonl data/training/
```

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
