# INSTALL.md

Setup, test, and demo instructions for the Civil Rights Summarized AI project.

## 1) Software Requirements

- Git
- Python 3.11+
- Terminal (macOS/Linux shell or Windows command prompt)
- SQLite (usually included with Python)
- (Optional) CUDA-compatible GPU for local model inference

Verify Python version:
```bash
python3 --version
```

## 2) Clone the Repository

```bash
git clone https://github.com/thinq8/CivilRightsSummarizedAI.git
cd CivilRightsSummarizedAI
```

## 3) Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

python -m pip install --upgrade pip setuptools wheel
```

## 4) Install Dependencies

### Base install (ingestion pipeline + tests)

```bash
pip install -e ".[dev]"
```

### ML install (fine-tuning + evaluation + figures)

```bash
pip install -e ".[ml]"
```

This adds PyTorch, Transformers, PEFT, evaluation metrics (ROUGE, BERTScore), Anthropic SDK, matplotlib, and Jupyter.

## 5) Test the Installation
Make sure the virtual environment is activated.

Run from the repository root directory:

```bash
pytest -q
```

Expected result: `4 passed` (or higher if new tests are added later).

## 6) Run the Demo (Out-of-the-Box)
Run the end-to-end mock ingestion demo:

```bash
python -m clearinghouse.cli ingest-mock \
  --db-url sqlite:///data/dev.db \
  --fixture data/fixtures/mock_dataset.json \
  --checkpoint-key mock-default \
  --resume-from-checkpoint \
  --archive-raw-payloads
```

This creates a SQLite database at: data/dev.db

Expected output pattern: `Ingestion complete: run_id=... cases=... dockets=... documents=... errors=...`

## 7) Environment Variables (optional)

The mock demo does not require any environment variables.

The following variables are only needed for advanced usage:

- CLEARINGHOUSE_API_TOKEN — required for live API ingestion
- ANTHROPIC_API_KEY — required for Claude-based generation and LLM-as-Judge evaluation

Example (macOS/Linux):

export CLEARINGHOUSE_API_TOKEN="YOUR_TOKEN"
export ANTHROPIC_API_KEY="YOUR_KEY"

## 8) Data Instructions

### Included example data (default for grading/demo)

- Use `data/fixtures/mock_dataset.json` — safe to share, used for install testing and demo runs.

### Training data (for ML pipeline)

- Training/validation/test JSONL files are too large for git.
- See `data/training/README.md` for download instructions.
- Place files in `data/training/` (train.jsonl, val.jsonl, test.jsonl).

### Model weights (for local inference)

- LoRA adapter weights are too large for git.
- See `runs/README.md` for reproduction or download instructions.

### Community partner/private data

- Private partner data is **not** required for the reproducible demo.
- Do not commit private data to this repository.
- If private files are used locally, store them under ignored paths such as `data/raw_documents/` or `data/tmp/`.

## 9) Reproduce Figures

Figure reproduction instructions are provided in `notebooks/figure_instructions.ipynb`.

Notes:
- Figures 1 & 2 can be reproduced immediately using bundled fixture data
- Figures 3 & 4 require running the evaluation pipeline first
- Exported figures are saved to the figures/ directory
- Pre-generated PNGs are included for reference if full training data is unavailable

```bash
cd notebooks
jupyter notebook figure_instructions.ipynb
```

Run cells sequentially. See the notebook for per-figure data requirements and time estimates.

## 10) Run Evaluation Pipeline

```bash
cd eval

# Generate summaries (choose one)
python generate.py --source local --sample 50   # requires GPU + model weights
python generate.py --source claude --sample 50   # requires ANTHROPIC_API_KEY

# Score summaries
python evaluate.py results/generations_*.jsonl --judge-sample 25
```

## 11) Optional Live API Demo (Token Required)

If you have approved Clearinghouse API credentials:

```bash
export CLEARINGHOUSE_API_TOKEN="YOUR_TOKEN_HERE"
python -m clearinghouse.cli ingest-live --db-url sqlite:///data/live.db --case-limit 25
```

This step is optional and not required for course install verification.

## 12) Remove Environment (Optional Cleanup)

```bash
deactivate
rm -rf .venv
```
