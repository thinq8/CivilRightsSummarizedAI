# Civil Rights Summarized AI

Ingestion and summarization tooling for Clearinghouse civil rights cases.

## Course Context
- **Institution / Course**: Michigan State University CMSE Capstone, in collaboration with Civil Rights Clearinghouse stakeholders.
- **Team Goal**: Deliver a reproducible pipeline that ingests civil-rights case records and prepares data for high-quality summarization workflows.

This repository now includes a resilient ingestion pipeline designed for two near-term goals:
1. Pull data reliably from the Clearinghouse API.
2. Preserve data lineage so the same corpus can later be used for LLM fine-tuning/evaluation.

## Why This Design

For training and research workflows, normalized tables alone are not enough. Upstream APIs evolve,
and fields that are ignored today may become important training features later. This pipeline stores:
- Normalized entities (`cases`, `dockets`, `documents`) for current app/query use.
- Raw payload snapshots (`raw_api_payloads`) for schema evolution and reproducibility.
- Operational metadata (`ingestion_runs`, `ingestion_checkpoints`) for resume/debug/audit.

That split keeps the project practical for a small student team while preserving options for model
development at the end of the project.

## Repository Layout

- `src/clearinghouse`: Python package (clients, pipeline, storage, CLI)
- `data/fixtures/mock_dataset.json`: local mock dataset for deterministic tests
- `scripts/fetch_document.py`: smoke-test utility for one live document
- `tests/`: unit/integration tests for ingestion behavior
- `archive/`: legacy/reference artifacts (non-runtime materials)

## Step-by-Step Setup (New Machine)

Run these commands from repository root (`CivilRightsSummarizedAI`).

1. Create a local virtual environment:

```bash
python3 -m venv .venv
```

2. Ensure `pip` exists inside `.venv` (some environments create venvs without pip):

```bash
./.venv/bin/python -m ensurepip --upgrade
```

3. Upgrade packaging tools:

```bash
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
```

4. Install project + dev dependencies:

```bash
./.venv/bin/python -m pip install -e ".[dev]"
```

5. Verify the environment with tests:

```bash
./.venv/bin/python -m pytest -q
```

Generated SQLite files (for example `data/dev.db` and `data/live.db`) are local runtime artifacts and are intentionally not part of source control.

## Pipeline Behavior (Current)

### Ingestion flow

For each case returned by API/fixture:
1. Upsert case row.
2. Upsert related docket rows.
3. Upsert related document rows.
4. Build heuristic summary for each document.
5. Archive raw payload snapshots (deduped by SHA256 hash).
6. Advance checkpoint only after case commit succeeds.

### Reliability features

- API retries with exponential backoff + jitter for transient failures.
- `Retry-After` header support on throttling responses.
- Incremental resume from checkpoint (`--resume-from-checkpoint`).
- Per-case error isolation in non-strict mode (`--continue-on-error`).
- Run-level auditing with counts/error fields in `ingestion_runs`.

## Running Ingestion

### 1) Mock run (recommended first)

```bash
./.venv/bin/python -m clearinghouse.cli ingest-mock \
    --db-url sqlite:///data/dev.db \
    --fixture data/fixtures/mock_dataset.json \
    --checkpoint-key mock-default \
    --resume-from-checkpoint \
    --archive-raw-payloads
```

Expected output shape:
`Ingestion complete: run_id=<uuid> cases=<n> dockets=<n> documents=<n> errors=<n>`

### 2) Live run

Set token (raw token or `Token <value>` are both accepted):

```bash
export CLEARINGHOUSE_API_TOKEN="YOUR_TOKEN_HERE"
./.venv/bin/python -m clearinghouse.cli ingest-live \
    --db-url sqlite:///data/live.db \
    --checkpoint-key live-default \
    --resume-from-checkpoint \
    --archive-raw-payloads \
    --continue-on-error \
    --case-limit 25
```

Use `--strict` instead of `--continue-on-error` if you want the run to fail fast on first case error.

## CLI Argument Reference

### Global CLI

```bash
./.venv/bin/python -m clearinghouse.cli --help
```

- `--verbose`, `-v`: enable debug logging.

### `ingest-mock` arguments

```bash
./.venv/bin/python -m clearinghouse.cli ingest-mock --help
```

- `--since TEXT`: ISO timestamp filter (example: `2023-01-01T00:00:00Z`).
- `--case-limit INTEGER`: max number of cases to ingest.
- `--db-url TEXT`: SQLAlchemy DB URL.
- `--fixture PATH`: path to fixture JSON.
- `--checkpoint-key TEXT`: checkpoint namespace key (default: `mock-default`).
- `--resume-from-checkpoint` / `--no-resume-from-checkpoint`: use stored checkpoint timestamp as effective start point.
- `--archive-raw-payloads` / `--no-archive-raw-payloads`: store raw payload snapshots in `raw_api_payloads`.
- `--continue-on-error` / `--strict`: continue on per-case failures vs fail fast.

### `ingest-live` arguments

```bash
./.venv/bin/python -m clearinghouse.cli ingest-live --help
```

- `--since TEXT`: ISO timestamp filter.
- `--case-limit INTEGER`: max number of cases.
- `--db-url TEXT`: SQLAlchemy DB URL.
- `--api-token TEXT`: API token (`Token ` prefix optional).
- `--checkpoint-key TEXT`: checkpoint namespace key (default from `CLEARINGHOUSE_LIVE_CHECKPOINT_KEY`).
- `--resume-from-checkpoint` / `--no-resume-from-checkpoint`: incremental resume behavior.
- `--archive-raw-payloads` / `--no-archive-raw-payloads`: archive raw API payloads.
- `--continue-on-error` / `--strict`: continue or fail fast on case-level errors.

## Operational Verification

After a run, inspect key counts:

```bash
sqlite3 data/dev.db "SELECT 'cases', count(*) FROM cases UNION ALL SELECT 'dockets', count(*) FROM dockets UNION ALL SELECT 'documents', count(*) FROM documents;"
```

Inspect run metadata:

```bash
sqlite3 data/dev.db "SELECT id, source, status, started_at, finished_at, cases_ingested, dockets_ingested, documents_ingested, errors FROM ingestion_runs ORDER BY started_at DESC LIMIT 5;"
```

Inspect checkpoint state:

```bash
sqlite3 data/dev.db "SELECT key, source, last_case_id, last_case_last_checked, last_run_id FROM ingestion_checkpoints;"
```

Inspect raw payload archive volume:

```bash
sqlite3 data/dev.db "SELECT resource_type, count(*) FROM raw_api_payloads GROUP BY resource_type ORDER BY resource_type;"
```

## Environment Variables

All settings use `CLEARINGHOUSE_` prefix (loaded from shell or `.env`):

- `CLEARINGHOUSE_DATABASE_URL` (default `sqlite:///data/dev.db`)
- `CLEARINGHOUSE_API_TOKEN`
- `CLEARINGHOUSE_API_BASE_URL` (default `https://clearinghouse.net/api/v2p1`)
- `CLEARINGHOUSE_API_TIMEOUT` (default `30.0`)
- `CLEARINGHOUSE_API_MAX_RETRIES` (default `4`)
- `CLEARINGHOUSE_API_BACKOFF_SECONDS` (default `0.5`)
- `CLEARINGHOUSE_API_MAX_BACKOFF_SECONDS` (default `8.0`)
- `CLEARINGHOUSE_LIVE_CHECKPOINT_KEY` (default `live-default`)
- `CLEARINGHOUSE_LIVE_RESUME_FROM_CHECKPOINT` (default `true`)
- `CLEARINGHOUSE_ARCHIVE_RAW_PAYLOADS` (default `true`)
- `CLEARINGHOUSE_CONTINUE_ON_ERROR` (default `true`)

## Fetch One Document (Smoke Test)

Quickly inspect one live document and optional text:

```bash
./.venv/bin/python scripts/fetch_document.py <case_id> <document_id> \
    --api-token "YOUR_TOKEN_HERE" \
    --download-text
```

## Development Notes

- `tests/test_mock_ingestion.py` validates basic end-to-end mock ingestion counts.
- `tests/test_ingestion_resilience.py` validates checkpoint resume, raw payload dedupe, and token normalization.
- The project currently uses SQLAlchemy `create_all` (no Alembic migrations yet).
- For production growth, add migration tooling and background workers for PDF/OCR text extraction.

## References

- Clearinghouse API docs: <https://api.clearinghouse.net>
