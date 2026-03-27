# Civil Rights Summarized AI

Ingestion, fine-tuning, and evaluation pipeline for summarizing civil rights litigation cases from the [Civil Rights Litigation Clearinghouse](https://clearinghouse.net).

## Install and Demo Instructions

- See [INSTALL.md](INSTALL.md) for the reproducible setup, test, and demo workflow.

## Figure Reproduction

- See [`notebooks/figure_instructions.ipynb`](notebooks/figure_instructions.ipynb) for complete instructions to reproduce all project figures.
- Exported figures are saved to the [`figures/`](figures/) directory.
- **Figures 1 & 2** can be reproduced immediately — all required data is bundled in `data/fixtures/` (no large downloads needed).
- **Figures 3 & 4** require running the evaluation pipeline first (see notebook for steps).
- Pre-generated PNGs are included in `figures/` for reference if you don't have the full training data or model weights.

## Course Context

- **Institution / Course**: Michigan State University CMSE Capstone, in collaboration with Civil Rights Clearinghouse stakeholders.
- **Team Goal**: Deliver a reproducible pipeline that ingests civil-rights case records, fine-tunes a summarization model, and evaluates output quality.

## Project Components

### 1. Data Ingestion Pipeline

A resilient pipeline that pulls case data from the Clearinghouse API:
- Normalized entities (`cases`, `dockets`, `documents`) for querying.
- Raw payload snapshots (`raw_api_payloads`) for schema evolution and reproducibility.
- Operational metadata (`ingestion_runs`, `ingestion_checkpoints`) for resume/debug/audit.

### 2. LoRA Fine-Tuning

Fine-tuning of Qwen2.5-7B-Instruct using LoRA adapters on 9,841 case summarization examples. The training data addresses the **document fragmentation** challenge: legal cases span multiple documents (complaints, motions, orders) that must be synthesized into coherent summaries.

### 3. Evaluation Pipeline

Three-tier evaluation of generated summaries:
- **ROUGE** (1/2/L) — lexical overlap with reference summaries
- **BERTScore** — semantic similarity
- **LLM-as-Judge** — Claude scores on 5 dimensions (factual accuracy, completeness, conciseness, legal reasoning, overall quality)

## Repository Layout

```
├── src/clearinghouse/          # Python package (clients, pipeline, storage, CLI)
├── eval/                       # Evaluation pipeline (generate + score summaries)
│   ├── config.py               # Paths, model config, judge prompts
│   ├── generate.py             # Summary generation (local model or Claude API)
│   ├── evaluate.py             # ROUGE, BERTScore, LLM-as-Judge scoring
│   └── results/                # Output scores and summaries (gitignored)
├── notebooks/                  # Figure reproduction instructions
│   └── figure_instructions.ipynb
├── figures/                    # Exported figures
├── scripts/                    # Utility scripts
│   ├── fetch_document.py       # Smoke test for single live document
│   └── hydrate_document_text.py # Bulk text hydration from API
├── data/
│   ├── fixtures/               # Mock dataset for deterministic tests
│   └── training/               # Training data (gitignored, see README)
├── runs/                       # Model checkpoints (gitignored, see README)
├── tests/                      # Unit and integration tests
├── .env.example                # Environment variable template
├── INSTALL.md                  # Setup and demo instructions
├── pyproject.toml              # Package config and dependencies
└── LICENSE                     # MIT License
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/thinq8/CivilRightsSummarizedAI.git
cd CivilRightsSummarizedAI
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest -q

# Run mock ingestion demo
python -m clearinghouse.cli ingest-mock \
    --db-url sqlite:///data/dev.db \
    --fixture data/fixtures/mock_dataset.json \
    --checkpoint-key mock-default \
    --resume-from-checkpoint \
    --archive-raw-payloads
```

For ML/evaluation work, install the ML extras:

```bash
pip install -e ".[ml]"
```

## Running Ingestion

### Mock run (recommended first)

```bash
python -m clearinghouse.cli ingest-mock \
    --db-url sqlite:///data/dev.db \
    --fixture data/fixtures/mock_dataset.json \
    --checkpoint-key mock-default \
    --resume-from-checkpoint \
    --archive-raw-payloads
```

### Live run (API token required)

```bash
export CLEARINGHOUSE_API_TOKEN="YOUR_TOKEN_HERE"
python -m clearinghouse.cli ingest-live \
    --db-url sqlite:///data/live.db \
    --checkpoint-key live-default \
    --resume-from-checkpoint \
    --archive-raw-payloads \
    --continue-on-error \
    --case-limit 25
```

## Running Evaluation

```bash
cd eval

# Generate summaries (choose source)
python generate.py --source local --sample 50
python generate.py --source claude --sample 50

# Score summaries
python evaluate.py results/generations_local_*.jsonl --judge-sample 25
```

## Environment Variables

All ingestion settings use `CLEARINGHOUSE_` prefix (see `.env.example`):

- `CLEARINGHOUSE_API_TOKEN` — API authentication
- `CLEARINGHOUSE_DATABASE_URL` — SQLAlchemy DB URL (default: `sqlite:///data/dev.db`)
- `CLEARINGHOUSE_API_BASE_URL` — API base (default: `https://clearinghouse.net/api/v2p1`)

For evaluation, set `ANTHROPIC_API_KEY` for Claude API access.

## References

- Clearinghouse API docs: <https://api.clearinghouse.net>
