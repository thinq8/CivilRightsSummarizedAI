# Civil Rights Summarized AI Tutorial

This is the full walkthrough. It separates the no-private-data grader path from optional live API work and full ML/HPCC reproduction.

Use [INSTALL.md](INSTALL.md) for the short setup checklist. Use [FINAL_SUBMISSION.md](FINAL_SUBMISSION.md) for the final deliverable map.

## Track A: Grader Path Without Private Data

This track verifies the repository from a fresh clone without API keys, private data, model weights, or GPU access.

### A1. Install

```bash
git clone https://github.com/thinq8/CivilRightsSummarizedAI.git
cd CivilRightsSummarizedAI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip "setuptools<82" wheel
python -m pip install -e ".[dev]"
```

Expected: `python -m pip show civil-rights-summarizer` shows the package installed from this repository.

### A2. Run Tests

```bash
pytest -q
```

Expected: `4 passed` or more, with zero failures.

### A3. Run the Mock Ingestion Demo

```bash
python -m clearinghouse.cli ingest-mock
```

Expected:

```text
Ingestion complete: run_id=... cases=2 dockets=2 documents=4 errors=0
```

This writes `data/dev.db`. That file is local output and is ignored by git.

### A4. Inspect the Demo Database

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

Expected after one mock run:

```text
cases: 2
dockets: 2
documents: 4
raw_api_payloads: 8
```

### A5. Reproduce the Seven Final Figures

```bash
python -m pip install -e ".[ml]"
jupyter notebook notebooks/figure_instructions.ipynb
```

Run all cells. The notebook uses bundled non-private fixtures and saves:

- `figures/figure1_training_dynamics.png`
- `figures/figure2_prompt_length_distribution.png`
- `figures/figure3_checkpoint_comparison.png`
- `figures/figure4_qa_triage_3systems.png`
- `figures/figure5_source_attribution.png`
- `figures/figure6_cost_quality.png`
- `figures/figure7_flag_frequency.png`

### A6. Read the Final Artifacts

- [REPORT.md](REPORT.md) - final technical report.
- [FINAL_SUBMISSION.md](FINAL_SUBMISSION.md) - deliverable index and data-flow diagram.
- [scripts/README.md](scripts/README.md) - what the training/evaluation/QA scripts do.
- [tools/README.md](tools/README.md) - how to open the browser tools.
- [data/fixtures/README.md](data/fixtures/README.md) - metadata for all bundled fixtures.

## Track B: Live API and Partner Tool Path

This track is optional. It is for reviewers or partner users with approved Clearinghouse API access.

### B1. Get API Credentials

- Clearinghouse API quick start: <https://api.clearinghouse.net/quick-start>
- API access request: <https://www.clearinghouse.net/api-request>

Set the token:

```bash
export CLEARINGHOUSE_API_TOKEN="Token YOUR_TOKEN_HERE"
```

The code also accepts the raw token without the `Token ` prefix.

### B2. Run a Small Live Ingestion Smoke Test

```bash
python -m clearinghouse.cli ingest-live --case-limit 5
```

Expected output pattern:

```text
Live ingestion complete: run_id=... cases=... dockets=... documents=... errors=...
```

Use a small `--case-limit` first. Larger live runs may take longer and may create a larger local SQLite database.

### B3. Open the Browser Generator and Evaluator

Start the local proxy:

```bash
python tools/clearinghouse_api_proxy.py
```

Open <http://127.0.0.1:8765/>.

Recommended workflow:

1. Generate a metadata-only draft for a thin/simple case, or a metadata-plus-documents draft for a complex case.
2. Export the generator package.
3. Import the package into the evaluator.
4. Run local QA and optional source-grounding review.
5. Export reviewer feedback JSON.

For a fully offline QA check, open:

```bash
open tools/summary_qa_standalone.html
```

## Track C: ML and HPCC Reproduction Path

This track requires large private artifacts or approved credentials. It is not required for the basic grader path.

### C1. Place Raw Training Splits

Expected local layout:

```text
data/training/train.jsonl
data/training/val.jsonl
data/training/test.jsonl
```

These files are not tracked in git. See [data/training/README.md](data/training/README.md).

### C2. Prepare Model-Ready Training Data

Heuristic extraction, no API key:

```bash
mkdir -p data/training_v2
python scripts/prepare_training_data.py \
  --input data/training/train.jsonl \
  --output data/training_v2/train.jsonl \
  --strategy extract_first \
  --max-tokens 24000 \
  --extraction-backend heuristic
```

Claude extraction, higher quality:

```bash
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY_HERE"
python scripts/prepare_training_data.py \
  --input data/training/train.jsonl \
  --output data/training_v2/train.jsonl \
  --strategy extract_first \
  --max-tokens 24000 \
  --extraction-backend claude
```

Repeat for `val.jsonl` and `test.jsonl`.

### C3. Train on HPCC

Conda is used here because MSU ICER HPCC exposes Python/CUDA through cluster modules. Local users should stay with the virtual environment from Track A.

```bash
module purge
module load Miniforge3
conda create -n legal-sum python=3.11 -y
conda activate legal-sum
python -m pip install --upgrade pip "setuptools<82" wheel
python -m pip install -e ".[ml]"
sbatch scripts/train_run2.sbatch
```

The canonical training job:

1. Reads raw data from `data/training/` unless `RAW_DATA_DIR` overrides it.
2. Writes prepared data to `data/training_v2/` unless `PREP_DATA_DIR` overrides it.
3. Trains `Qwen/Qwen2.5-7B-Instruct` with LoRA.
4. Saves checkpoints under `runs/qwen25_7b_lora_run2/`.

Final run settings:

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Max length | 8192 |
| Learning rate | `2e-5` |
| Epochs | 3 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Gradient accumulation | 8 |
| Gradient clipping | 1.0 |

### C4. Evaluate Checkpoints

Checkpoint comparison:

```bash
python scripts/eval_checkpoint_v2.py \
  --checkpoint-dir runs/qwen25_7b_lora_run2/checkpoint-3000 \
  --test-file data/training_v2/test.jsonl \
  --num-samples 50 \
  --output-file eval_ckpt3000.jsonl
```

Claude benchmark:

```bash
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY_HERE"
python scripts/benchmark_claude.py \
  --model claude-sonnet-4-5 \
  --test-file data/training/test.jsonl \
  --num-samples 50 \
  --output-file eval_claude_sonnet.jsonl
```

### C5. Run Structural QA

```bash
python scripts/summary_qa.py \
  --input eval_ckpt3000.jsonl \
  --output-dir qa_report_ckpt3000
```

QA statuses:

| Status | Meaning | Action |
|--------|---------|--------|
| `PASS` | No critical or warning flags | Candidate for normal editorial review |
| `REVIEW` | Warning flags present | Human editor should inspect before use |
| `REJECT` | Critical flags present | Regenerate or heavily edit before use |

The QA layer is central to the final result. Checkpoint 3690 had similar-looking reference metrics to checkpoint 3000, but many more structural failures, especially malformed dates. That is why the final report recommends checkpoint comparison plus QA routing instead of shipping the last checkpoint automatically.

## What Is Safe To Commit

Safe:

- Source code in `src/`, `scripts/`, `eval/`, and `tools/`
- Tests in `tests/`
- Small non-private fixtures in `data/fixtures/`
- Documentation, notebooks, and final figure PNGs

Do not commit:

- `.env` or API keys
- `data/dev.db`
- full `data/training/*.jsonl`
- `data/training_v2/`
- model checkpoints or adapter weights under `runs/`
- raw generated evaluation result JSONL files unless intentionally reduced into a fixture
