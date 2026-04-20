# Civil Rights Summarized AI Tutorial

This is the linear walkthrough for reproducing the project from a fresh clone. It starts with the local demo that works without private data, then shows where live API access, HPCC training, evaluation, and figure reproduction fit.

Use this page as the working tutorial. `INSTALL.md` is the shorter setup checklist.

## What Runs Where

| Stage | Runs locally? | Requires private data or keys? | Main files |
|-------|---------------|--------------------------------|------------|
| Install and tests | Yes | No | `INSTALL.md`, `pyproject.toml`, `tests/` |
| Mock ingestion demo | Yes | No | `data/fixtures/mock_dataset.json`, `src/clearinghouse/cli.py` |
| Live Clearinghouse ingestion | Yes | `CLEARINGHOUSE_API_TOKEN` | `src/clearinghouse/clients/http.py`, `src/clearinghouse/ingest/` |
| Training data preparation | Yes or HPCC | Full training JSONL files | `scripts/prepare_training_data.py` |
| LoRA fine-tuning | HPCC/GPU machine | Full training JSONL files | `scripts/train_lora.py`, `scripts/train_run2.sbatch` |
| Evaluation | GPU for local model, API key for Claude | Model checkpoint or `ANTHROPIC_API_KEY` | `eval/`, `scripts/summary_qa.py` |
| Figures | Yes | No for bundled fixture figures | `notebooks/figure_instructions.ipynb`, `data/fixtures/` |

## Environment Decision

Use a plain Python virtual environment for local install, testing, mock ingestion, and figure reproduction:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip "setuptools<82" wheel
python -m pip install -e ".[dev]"
```

Use conda only on HPCC or another managed cluster where Python/CUDA modules are provided through conda. The SLURM script activates the cluster environment because that is how the training run was launched, not because local users need both environment systems.

The `pip install -e ".[dev]"` command means "install this repository in editable mode, plus the dependencies listed in the `dev` optional dependency group in `pyproject.toml`."

## Cell 1: Confirm the Repository and Python Environment

Run this from the repository root after activating `.venv`.

```bash
pwd
python --version
python -m pip show civil-rights-summarizer
```

Expected: the path ends in `CivilRightsSummarizedAI`, Python is 3.11 or newer, and pip shows the local editable package.

## Cell 2: Run the Unit Tests

```bash
pytest -q
```

Expected: all tests pass. These tests use the bundled mock data and do not call the live Clearinghouse API.

## Cell 3: Run the Simplest Mock Ingestion Demo

The CLI has defaults for the database path, fixture path, checkpoint key, raw-payload archiving, and error handling. Start with the shortest command:

```bash
python -m clearinghouse.cli ingest-mock
```

Expected output pattern:

```text
Ingestion complete: run_id=... cases=2 dockets=2 documents=4 errors=0
```

This creates `data/dev.db`, which is ignored by git.

To see the available options:

```bash
python -m clearinghouse.cli --help
python -m clearinghouse.cli ingest-mock --help
```

Only use the longer command when you want to override defaults:

```bash
python -m clearinghouse.cli ingest-mock \
  --db-url sqlite:///data/dev.db \
  --fixture data/fixtures/mock_dataset.json \
  --checkpoint-key mock-default \
  --resume-from-checkpoint \
  --archive-raw-payloads
```

## Cell 4: Inspect the Demo Database

```bash
python - <<'PY'
import sqlite3
from pathlib import Path

db = Path("data/dev.db")
print(f"database exists: {db.exists()} ({db})")

with sqlite3.connect(db) as conn:
    for table in ["cases", "dockets", "documents", "raw_api_payloads"]:
        count = conn.execute(f"select count(*) from {table}").fetchone()[0]
        print(f"{table}: {count}")
PY
```

Expected counts after one mock run:

```text
cases: 2
dockets: 2
documents: 4
raw_api_payloads: 8
```

## Cell 5: Get API Keys for Optional Live Runs

Live ingestion is optional for grading the local demo. To request access:

- Create a Clearinghouse account: <https://clearinghouse.net/registration>
- Request a Clearinghouse API key: <https://www.clearinghouse.net/api-request>
- Read the API quick start: <https://api.clearinghouse.net/quick-start>

After approval, set the token in your shell:

```bash
export CLEARINGHOUSE_API_TOKEN="Token YOUR_TOKEN_HERE"
```

The code also accepts the raw token without the `Token ` prefix.

For Claude generation or Claude judging:

```bash
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY_HERE"
```

Do not commit `.env` files or keys. `.env` is ignored by git.

## Cell 6: Run a Small Live API Smoke Test

Use a small `--case-limit` first.

```bash
python -m clearinghouse.cli ingest-live --case-limit 5
```

Expected output pattern:

```text
Live ingestion complete: run_id=... cases=... dockets=... documents=... errors=...
```

The live database also defaults to `data/dev.db` unless `CLEARINGHOUSE_DATABASE_URL` or `--db-url` overrides it.

## Cell 7: Understand the Fixture Data

The repository includes small, shareable fixture files so reviewers can run tests and reproduce figures without downloading the full training set.

| Fixture | What it is | How it was produced |
|---------|------------|---------------------|
| `data/fixtures/mock_dataset.json` | Two synthetic Clearinghouse-like cases for deterministic ingestion tests | Hand-built safe fixture matching the API shape used by the pipeline |
| `data/fixtures/trainer_state.json` | Training logs used for the loss curve | Extracted from the successful LoRA run's `trainer_state.json` |
| `data/fixtures/test_chunk_counts.json` | Document-count metadata for the 1,231-case test split | Extracted from the private test split metadata, without document text or summaries |
| `data/fixtures/eval_summary.json` | Aggregated evaluation metrics for figure fallback | Produced by the evaluation scripts from the 50-case evaluation sample |
| `data/fixtures/eval_scores.jsonl` | Per-record judge/metric scores for figure fallback | Produced by `eval/evaluate.py` and reduced to shareable scores |

These fixtures are not the training corpus. They are small artifacts that make the repository reproducible without redistributing large or partner-controlled files.

## Cell 8: Prepare Full Training Data

The full `train.jsonl`, `val.jsonl`, and `test.jsonl` files are not tracked because they are large. Put them in `data/training/` when available.

Expected raw split paths:

```text
data/training/train.jsonl
data/training/val.jsonl
data/training/test.jsonl
```

The preparation script converts raw long-document examples into shorter model-ready examples. The run used for the final project used the `extract_first` strategy:

```bash
mkdir -p data/training_v2

python scripts/prepare_training_data.py \
  --input data/training/train.jsonl \
  --output data/training_v2/train.jsonl \
  --strategy extract_first \
  --max-tokens 24000 \
  --extraction-backend heuristic
```

For the strongest extraction quality, use the Claude backend with `ANTHROPIC_API_KEY` set:

```bash
python scripts/prepare_training_data.py \
  --input data/training/train.jsonl \
  --output data/training_v2/train.jsonl \
  --strategy extract_first \
  --max-tokens 24000 \
  --extraction-backend claude
```

Repeat for `val.jsonl` and `test.jsonl`.

## Cell 9: Reproduce the HPCC Training Run

Training was done on MSU ICER HPCC with SLURM, a Miniforge/conda environment, and one H200 GPU. The local repo contains the same Python training entry point and a configurable SLURM template.

First copy the repository and full training data to the cluster. Then, from the repo root on HPCC:

```bash
module purge
module load Miniforge3
conda create -n legal-sum python=3.11 -y
conda activate legal-sum
python -m pip install --upgrade pip "setuptools<82" wheel
python -m pip install -e ".[ml]"
```

Submit the training job:

```bash
sbatch scripts/train_run2.sbatch
```

The script does two phases:

1. Builds `data/training_v2/{train,val,test}.jsonl` with `scripts/prepare_training_data.py`
2. Fine-tunes `Qwen/Qwen2.5-7B-Instruct` with LoRA using `scripts/train_lora.py`

The main training settings were:

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Adapter | LoRA, rank 16, alpha 32, dropout 0.05 |
| Max length | 8192 |
| Learning rate | `2e-5` |
| Epochs | 3 |
| Gradient accumulation | 8 |
| Gradient clipping | 1.0 |
| Checkpoint cadence | every 500 steps |
| Best observed checkpoint | step 3000 by training-loss EMA |
| Final checkpoint | step 3690 |

The important methodological decision was to evaluate both step 3000 and the final checkpoint. The final checkpoint looked like the natural endpoint, but evaluation showed it overfit and produced more structural failures.

## Cell 10: Run Local or Claude Evaluation

Evaluation needs the full test split. Local model evaluation also needs a LoRA adapter under `runs/`.

Claude generation:

```bash
cd eval
python generate.py --source claude --sample 50
python evaluate.py results/generations_claude_*.jsonl --judge-sample 25
cd ..
```

Local LoRA generation:

```bash
cd eval
python generate.py --source local --sample 50
python evaluate.py results/generations_local_*.jsonl --judge-sample 25
cd ..
```

Structural QA triage:

```bash
python scripts/summary_qa.py \
  --input eval/results/generations_local_YYYYMMDD_HHMMSS.jsonl \
  --output-dir eval/results/qa_report_local
```

How to interpret QA:

| Status | Meaning | Action |
|--------|---------|--------|
| `PASS` | No critical or warning flags | Candidate for normal editorial review |
| `REVIEW` | Warning flags present | Human editor should inspect before use |
| `REJECT` | Critical flags present | Regenerate or heavily edit before use |

The QA tool is important because it catches failures that ROUGE can hide. In this project, checkpoint 3690 had similar-looking reference metrics to checkpoint 3000, but many more structural failures, especially malformed dates. That is why the report recommends checkpoint comparison plus QA routing instead of simply shipping the last checkpoint.

## Cell 11: Use the Partner-Facing Tools

The final submission includes standalone tools in `tools/`. They are prototypes for the Clearinghouse workflow, not a production deployment.

Offline single-summary QA:

```bash
open tools/summary_qa_standalone.html
```

Generator/evaluator with optional live API loading:

```bash
python tools/clearinghouse_api_proxy.py
```

Then open <http://127.0.0.1:8765/>.

Recommended tool flow:

1. Generate a metadata-only draft for a thin/simple case, or a metadata-plus-documents draft for a complex case.
2. Export the generated package.
3. Import it into the evaluator.
4. Run local QA and source review.
5. Export reviewer feedback JSON.

See `tools/README.md` for details.

## Cell 12: Reproduce Figures

For the course reproducibility check, use the notebook:

```bash
jupyter notebook notebooks/figure_instructions.ipynb
```

Run the cells top to bottom. Figures 1 and 2 use bundled fixture files. Figures 3 and 4 use evaluation results if present; otherwise they fall back to the bundled aggregate fixtures in `data/fixtures/`.

## Cell 13: Find the Final Submission Materials

Use `FINAL_SUBMISSION.md` as the final map. It lists every final deliverable and includes a data-flow diagram showing where ingestion, training data preparation, HPCC training, evaluation, QA, figures, and browser tools fit.

Key final files:

- `REPORT.md`
- `SHARING_PLAN.md`
- `UofM_SHARING_PLAN.pdf`
- `20260410_Final_Presentation_Plan_CivilRightsSummarizedAI.md`
- `20260410_Final_Presentation_Plan_CivilRightsSummarizedAI.pptx`
- `FINAL_SUBMISSION.md`
- `scripts/README.md`
- `tools/README.md`

## Cell 14: What to Commit

Safe to commit:

- Source code in `src/`, `scripts/`, `eval/`
- Tests in `tests/`
- Small fixture metadata in `data/fixtures/`
- Documentation and notebooks
- Figure PNGs intended for the report

Do not commit:

- `.env` or API keys
- `data/dev.db`
- Full `data/training/*.jsonl`
- `data/training_v2/`
- Model checkpoints and adapter weights under `runs/`
- Generated evaluation result JSONL files unless a small fixture was intentionally curated
