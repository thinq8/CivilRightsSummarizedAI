# Install and Smoke-Test Guide

This is the shortest setup path for any handoff recipient. It uses a local Python virtual environment and bundled fixture data only. It does not require conda, Clearinghouse credentials, large training data files, model weights, a GPU, or Claude/Anthropic credentials.

For the full workflow after this smoke test, use [TUTORIAL.md](TUTORIAL.md).

## Requirements

- Git
- Python 3.11 or newer
- Terminal, PowerShell, or another shell

## 1. Clone and Create a Local Environment

```bash
git clone https://github.com/thinq8/CivilRightsSummarizedAI.git
cd CivilRightsSummarizedAI

python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell/CMD

python -m pip install --upgrade pip "setuptools<82" wheel
python -m pip install -e ".[dev]"
```

## 2. Run Tests

```bash
pytest -q
```

Expected output:

```text
4 passed
```

The exact number may be higher if more tests are added later, but failures should be zero.

## 3. Run the No-Private-Data Demo

```bash
python -m clearinghouse.cli ingest-mock
```

Expected output pattern:

```text
Ingestion complete: run_id=... cases=2 dockets=2 documents=4 errors=0
```

This creates `data/dev.db`, which is ignored by git. The demo uses `data/fixtures/mock_dataset.json`, a synthetic two-case fixture documented in [data/fixtures/README.md](data/fixtures/README.md).

## 4. Optional Next Steps

| Goal | Where to go |
|------|-------------|
| Full walkthrough | [TUTORIAL.md](TUTORIAL.md) |
| Final deliverable map | [FINAL_SUBMISSION.md](FINAL_SUBMISSION.md) |
| Reproduce the seven final figures | [notebooks/figure_instructions.ipynb](notebooks/figure_instructions.ipynb) |
| Understand scripts and QA | [scripts/README.md](scripts/README.md) |
| Open browser review tools | [tools/README.md](tools/README.md) |

## Optional API Keys

Keys are not needed for the smoke test.

- Clearinghouse live ingestion requires `CLEARINGHOUSE_API_TOKEN`.
  - API quick start: <https://api.clearinghouse.net/quick-start>
  - Access request: <https://www.clearinghouse.net/api-request>
- Claude generation or judging requires `ANTHROPIC_API_KEY`.

Create a local `.env` file only if you are running live/API-backed workflows:

```bash
cat > .env <<'EOF'
CLEARINGHOUSE_API_TOKEN=your_token_here
ANTHROPIC_API_KEY=your_key_here
EOF
```

Never commit `.env` or API keys.

## Optional Handoff Package (If The Recipient Should Not Re-Run Expensive Steps)

The repository already includes the main outputs needed for a normal review:

- `REPORT.md`
- `figures/figure1_training_dynamics.png`
- `figures/figure2_prompt_length_distribution.png`
- `figures/figure3_checkpoint_comparison.png`
- `figures/figure4_qa_triage_3systems.png`
- `figures/figure5_source_attribution.png`
- `figures/figure6_cost_quality.png`
- `figures/figure7_flag_frequency.png`
- `data/fixtures/`
- `notebooks/figure_instructions.ipynb`

If you are handing off the larger project artifacts outside git, keep the same repo-relative paths when you place those files:

```text
data/training/train.jsonl
data/training/val.jsonl
data/training/test.jsonl
data/training_v2/train.jsonl
data/training_v2/val.jsonl
data/training_v2/test.jsonl
runs/qwen25_7b_lora_run2/checkpoint-3000/
runs/qwen25_7b_lora_run2/checkpoint-3690/
eval/results/
qa_report_ckpt3000/
qa_report_ckpt3690/
eval_ckpt3000.jsonl
eval_ckpt3690.jsonl
eval_claude_sonnet.jsonl
REPORT.pdf
```

Use this split when preparing the handoff:

- Keep in the public git repo:
  - source code, docs, tests, tools, notebooks, final figures, and `data/fixtures/`
- Share in the handoff folder, Teams drive, or partner drive:
  - `data/training/*.jsonl`
  - `data/training_v2/`
  - `runs/*/checkpoint-*`
  - `eval/results/`
  - raw evaluation JSONL outputs
  - generated QA report directories
  - `REPORT.pdf`

These files are shared outside git because they are large handoff artifacts, not because different recipients need different instructions.

The final video can stay in the Teams drive or handoff folder. Reference that location in the final report, README, or submission email rather than trying to commit the video itself.

## Optional Cleanup

```bash
deactivate
rm -rf .venv data/dev.db
```
