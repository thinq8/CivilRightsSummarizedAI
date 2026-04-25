# Sharing Plan -- Civil Rights Summarized AI

This guide is for anyone receiving the project handoff: grader, instructor, or community partner. It explains how to install the repo, run the lightweight demo, and locate the larger handoff artifacts without using different instructions for different audiences.

For the shortest setup path, start with [INSTALL.md](INSTALL.md). For the complete technical walkthrough, use [TUTORIAL.md](TUTORIAL.md). For the final deliverable index and data-flow diagram, use [FINAL_SUBMISSION.md](FINAL_SUBMISSION.md).

## What You Need

1. Git and Python 3.11 or newer.
2. A terminal or PowerShell window.
3. About 500 MB of free disk space for the basic demo.
4. Optional API credentials only if you want live Clearinghouse ingestion or Claude-backed evaluation.

## Step 1: Download and Install

```bash
git clone https://github.com/thinq8/CivilRightsSummarizedAI.git
cd CivilRightsSummarizedAI

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip "setuptools<82" wheel
python -m pip install -e ".[dev]"
```

Windows activation command:

```powershell
.venv\Scripts\activate
```

## Step 2: Run Tests

```bash
pytest -q
```

Expected output:

```text
4 passed
```

The number may be higher if more tests are added later. The important part is zero failures.

## Step 3: Run the Mock Demo

```bash
python -m clearinghouse.cli ingest-mock
```

Expected output:

```text
Ingestion complete: run_id=... cases=2 dockets=2 documents=4 errors=0
```

This demo uses synthetic fixture data. It does not call the live Clearinghouse API.

## Step 4: Reproduce the Seven Final Figures

```bash
python -m pip install -e ".[ml]"
jupyter notebook notebooks/figure_instructions.ipynb
```

Run all notebook cells. The notebook uses bundled non-private aggregate fixtures and writes seven figure files:

- `figure1_training_dynamics.png`
- `figure2_prompt_length_distribution.png`
- `figure3_checkpoint_comparison.png`
- `figure4_qa_triage_3systems.png`
- `figure5_source_attribution.png`
- `figure6_cost_quality.png`
- `figure7_flag_frequency.png`

## Step 5: Review the Final Submission

| Item | Location | What it is |
|------|----------|------------|
| Final submission map | `FINAL_SUBMISSION.md` | Deliverable index, data-flow diagram, and QA interpretation |
| Full tutorial | `TUTORIAL.md` | Basic repo path, live API path, and ML/HPCC path |
| Final report | `REPORT.md` | Main technical report with results and figures |
| Figure notebook | `notebooks/figure_instructions.ipynb` | Reproduces the seven report figures |
| Scripts guide | `scripts/README.md` | Data prep, training, evaluation, benchmark, and QA script guide |
| Tools guide | `tools/README.md` | Browser generator, evaluator, QA checker, and local proxy |
| Sample data | `data/fixtures/` | Small non-private fixtures for tests, demos, and figures |

## Shared Handoff Files

The git repository is the main code handoff. Larger reproducibility artifacts may be shared alongside it in the handoff folder or Teams drive. Keep these repo-relative paths if you include them:

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

These files are not committed because they are large project artifacts. The same handoff package applies to graders and community partners.

## Optional: Evaluate Summary Quality

Full local model evaluation requires two artifacts that are usually shared through the handoff folder rather than git:

- `data/training/test.jsonl`
- a LoRA checkpoint such as `runs/qwen25_7b_lora_run2/checkpoint-3000`

With those files present:

```bash
python scripts/eval_checkpoint_v2.py \
  --checkpoint-dir runs/qwen25_7b_lora_run2/checkpoint-3000 \
  --test-file data/training_v2/test.jsonl \
  --num-samples 50 \
  --output-file eval_ckpt3000.jsonl
```

Run QA triage on generated summaries:

```bash
python scripts/summary_qa.py \
  --input eval_ckpt3000.jsonl \
  --output-dir qa_report_ckpt3000
```

QA output includes:

- `qa_report.html` for human review.
- `qa_report.csv` for spreadsheet review.
- `qa_report.jsonl` for structured downstream use.

## What Is Not in the Repo

| Item | Why not |
|------|---------|
| Full training data | Too large for the git repository |
| Prepared `data/training_v2/` files | Regenerable from the raw training splits |
| Model weights and checkpoints | Large model artifacts |
| Raw generated evaluation JSONL files | Large handoff artifacts; reduced fixtures are enough for the basic repo walkthrough |
| API keys | Security |

The included fixtures are enough for installation checks, mock ingestion, and final figure reproduction.
