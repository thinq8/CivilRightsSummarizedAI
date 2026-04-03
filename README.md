# Civil Rights Summarized AI

End-to-end pipeline for multi-document summarization of civil rights litigation cases from the [Civil Rights Litigation Clearinghouse](https://clearinghouse.net), including data ingestion, LoRA fine-tuning, and automated evaluation.

## Install and Demo Instructions

- See [INSTALL.md](INSTALL.md) for the reproducible setup, test, and demo workflow.

## Usage Overview

The pipeline supports three main workflows:

- Ingestion: Retrieve and normalize civil rights case data from the Clearinghouse API
- Fine-tuning: Train LoRA adapters for multi-document case summarization
- Evaluation: Score generated summaries using ROUGE-L, BERTScore, and an LLM-as-Judge rubric

See INSTALL.md for step-by-step execution instructions.

## Pipeline Overview

### 1. Data Ingestion Pipeline

A resilient pipeline that pulls case data from the Clearinghouse API:
- Normalized entities (`cases`, `dockets`, `documents`) for querying.
- Raw payload snapshots (`raw_api_payloads`) for schema evolution and reproducibility.
- Operational metadata (`ingestion_runs`, `ingestion_checkpoints`) for resume/debug/audit.

### 2. LoRA Fine-Tuning

Fine-tuning of Qwen2.5-7B-Instruct using LoRA adapters on 9,841 case summarization examples. The training data addresses the **document fragmentation** challenge: legal cases span multiple documents (complaints, motions, orders) that must be synthesized into coherent summaries.

### 3. Evaluation Pipeline

Three-tier evaluation of generated summaries:
- **ROUGE-L** — measures lexical overlap with reference summaries
- **BERTScore** — measures semantic similarity using contextual embeddings
- **LLM-as-Judge** — Claude scores on 5 dimensions (factual accuracy, completeness, conciseness, legal reasoning, overall quality)

Combining lexical, semantic, and rubric-based metrics provides a more robust assessment than any single metric, particularly for multi-document legal case summaries where wording may differ while meaning remains consistent.

## Repository Structure

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
└── LICENSE.md                     # MIT License
```

## References

- Clearinghouse API docs: <https://api.clearinghouse.net>

## License

This project is released under the MIT License. See LICENSE.md for details.
