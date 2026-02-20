# Clearinghouse Ingestion & Summarization Plan

## 1. Objectives
- Pull docket-level metadata and associated documents from the Clearinghouse API (union cases, individual rights, etc.).
- Persist raw responses plus normalized relational structures so we can track provenance, reprocess documents, and build datasets for summarization/fine-tuning.
- Generate structured summaries of important filings (complaints, orders, settlements) using metadata + text, suitable for legal professionals and LLM training.
- Design for incremental updates, reproducibility, and transparent logging.

## 2. Assumptions & Constraints
- API credentials/endpoints may not yet be available; mock responses or static snapshots may be used for early development.
- Network is restricted in some environments; plan should support "offline" development via recorded fixtures.
- Clearinghouse data model roughly follows: cases → dockets → documents, with document metadata (type, date, parties) and text/PDF links.
- Target environment: Python 3.11+, Postgres (or SQLite for local dev) for structured data, object storage/local fs for document blobs.

## 3. High-Level Architecture
1. **Ingestion Orchestrator** (CLI / scheduled job)
   - Loads configuration (API keys, pagination limits, start dates).
   - Coordinates individual extract tasks (cases, dockets, documents).
2. **Extractor Layer**
   - API client with retries/backoff, pagination helpers, and schema validation.
   - Source-specific mappers to transform JSON into internal dataclasses (Case, Docket, Document, DocumentText).
3. **Storage Layer**
   - Raw JSON bucket (timestamped) for reproducibility.
   - Normalized database (SQLAlchemy models) for queryable metadata and relationships.
   - Document file store (local `data/raw_docs/<case>/<doc>.pdf` or similar) + optional text extraction cache.
4. **Processing Layer**
   - Text extraction (PDF → text) and metadata enrichment (NER for parties, judge names, topics).
   - Summarization jobs leveraging existing LLM (baseline: GPT-4/claude via API; future: fine-tuned local model).
5. **Dataset Builder**
   - Creates curated datasets (JSONL/Parquet) of `metadata + text + summary` pairs for evaluation or fine-tuning.
6. **Observability**
   - Structured logging, run metadata, and lightweight dashboard/reporting (CLI summary for now).

## 4. Incremental Milestones
1. **Scaffolding (Week 0-1)**
   - Repo structure (`src`, `scripts`, `data`, `tests`).
   - Config management (pydantic `.env`).
   - Basic CLI entrypoint.
2. **Mocked Ingestion (Week 1)**
   - Stub Clearinghouse API client reading fixture JSON.
   - Dataclasses + persistence to SQLite.
   - Unit tests for transformations.
3. **Real API Integration (Week 2)**
   - Replace stubs with real HTTP calls; handle auth and pagination.
   - Document download pipeline (PDF/text) with checksum dedupe.
4. **Processing & Summaries (Week 3)**
   - Text extraction service + metadata enrichers.
   - Baseline summarization (prompt template + call out to LLM provider or local summarizer placeholder).
5. **Dataset Export & Automation (Week 4+)**
   - JSONL/Parquet exporters for `metadata + summary`.
   - Airflow/Prefect or cron-style scheduling.
   - Monitoring + alerts for failures.

## 5. Detailed Work Breakdown

### 5.1 Repo Structure & Tooling
- `src/clearinghouse/` package with submodules: `config`, `clients`, `models`, `storage`, `ingest`, `processing`.
- `scripts/` for CLI wrappers (e.g., `scripts/ingest_dockets.py`).
- `tests/` covering clients, mappers, processors.
- `data/` (gitignored) for fixtures, raw dumps, documents.
- Poetry or Hatch for packaging; for now, use `uv`/`pip-tools`? (Decide soon; default to Poetry.)
- Pre-commit hooks (black, ruff, mypy) to enforce quality.

### 5.2 Configuration
- Use `pydantic-settings` to load `.env` files.
- Config keys: `CLEARINGHOUSE_API_BASE`, `API_KEY`, pagination size, rate limits, storage paths, DB URL.
- Provide sample `.env.example` with placeholders.

### 5.3 API Client & Fixtures
- Define interface `ClearinghouseClient` with methods:
  - `list_cases(updated_after)`
  - `list_dockets(case_id)`
  - `list_documents(docket_id)`
  - `get_document(document_id)` (metadata + download URL).
- For offline dev, create `MockClearinghouseClient` reading fixture JSON from `data/fixtures/` with deterministic behaviour.
- Add HTTP client implementation using `httpx` with retry/backoff middleware and structured logging.

### 5.4 Data Modeling & Persistence
- Dataclasses/TypedDicts for case/docket/document metadata.
- SQLAlchemy models mirroring relationships: `cases`, `dockets`, `documents`, `document_blobs`, `runs`.
- Provide migration tool (SQLModel or Alembic). For MVP, create `metadata.create_all` script for SQLite.
- Document files go to `data/raw_documents/<case>/<doc_id>.pdf`; store metadata linking file path + checksum.

### 5.5 Ingestion Orchestration
- CLI command `python -m clearinghouse.ingest --since 2020-01-01 --limit 1000`.
- Steps per case/docket:
  1. Fetch case metadata; persist.
  2. Fetch linked dockets; persist.
  3. Fetch documents; store metadata & raw JSON.
  4. Queue document downloads/extractions (for now synchronous).
- Idempotency: use `updated_at` + `external_id` unique constraints.
- Logging + run summary (counts, errors).

### 5.6 Processing & Summaries
- Text extraction via `pypdf`/`pdfminer` or `textract`; fallback to OCR placeholder.
- Summarization pipeline accepts `DocumentRecord` + heuristics to categorize document importance (docket roles, file type, keywords).
- Prompt template referencing metadata (case name, court, stage) to produce structured summary sections (issues, ruling, outcome, citations).
- Store summaries in DB + dataset export.

### 5.7 Dataset Export
- Exporter modules to produce JSONL/Parquet `[{metadata, text, summary}]` for fine-tuning.
- Include provenance fields (document_id, docket_id, run_id, timestamp, prompt parameters).

### 5.8 Testing & Validation
- Unit tests for config loader, client fixtures, mappers, persistence.
- Integration test using fixture dataset covering full ingestion run into SQLite.
- CLI smoke test script verifying ingestion + summary generation.

## 6. Execution Next Steps (This Sprint)
1. Stand up package scaffolding w/ Poetry, `.env.example`, base config + data dirs.
2. Implement mock API client + fixture loader (read sample JSON describing 1-2 cases/dockets/documents).
3. Create ingestion orchestrator that walks mock data and writes to SQLite w/ SQLAlchemy models.
4. Add CLI entry + tests for ingestion + simple summary stub (e.g., rule-based bullet summary) to ensure pipeline end-to-end.
5. Document usage in `README.md`.

## 7. Risks & Mitigations
- **API changes**: use versioned client + schema validation; store raw payloads.
- **Document size / rate limits**: implement pagination/backoff + checkpointing.
- **OCR/Text extraction errors**: add fallback pipeline and track extraction quality metrics.
- **LLM cost/latency**: start with offline summarizer baseline; add async queue for heavy summarization later.
- **Security/PII**: ensure `.env` not committed; consider encryption for stored documents.

## 8. Open Questions
- Confirm actual Clearinghouse endpoints + auth flows once access granted.
- Decide final storage backend (managed Postgres vs. local) and summarization target model.
- Determine criteria for "important" documents (maybe by docket role, manual labels, or heuristics from Clearinghouse metadata).

