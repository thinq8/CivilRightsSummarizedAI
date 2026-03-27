"""Shared configuration, paths, rubric, and I/O helpers for the eval pipeline."""

import json
import os
import random
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "training" / "test.jsonl"
LORA_ADAPTER_PATH = ROOT / "runs" / "qwen25_7b_lora_run1"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── Model config ───────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0  # greedy for reproducibility

# ── Claude config ──────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_CONCURRENT = 10

# ── Summarization system message (used for Claude generation) ──────────────────
SUMMARIZATION_SYSTEM = (
    "You are a legal case summarization assistant for the Civil Rights "
    "Litigation Clearinghouse. Produce a concise case summary in past tense "
    "covering: the dispute and stakes, filing date and court, whether class or "
    "individual action, type of counsel, legal claims and statutory basis, key "
    "procedural events, and outcome or current status."
)

# ── LLM-as-Judge prompts ──────────────────────────────────────────────────────
JUDGE_SYSTEM = """\
You are an expert legal editor for the Civil Rights Litigation Clearinghouse \
evaluating AI-generated case summaries. You will receive a human-written \
reference summary and an AI-generated summary of the same case. Rate the \
AI summary on each dimension below using a 1-5 scale.

## Dimensions

### 1. Factual Accuracy (1-5)
5 = All facts (dates, party descriptions, court names, rulings, statutory \
citations) match the reference exactly.
4 = Minor discrepancies that don't change meaning (e.g., imprecise date).
3 = One or two meaningful factual errors.
2 = Several errors or one critical error (wrong ruling, wrong party, wrong court).
1 = Pervasive errors or hallucinated facts not in the reference.

### 2. Completeness (1-5)
Check for these required elements from the reference:
- Opening sentence conveying the stakes of the case
- Filing date and full formal court name
- Whether class or individual action
- Type of counsel (identified by name for legal services orgs)
- Causes of action / legal claims with statutory or constitutional basis
- Remedies sought
- Key procedural events and outcome
5 = All required elements present. 4 = Misses one minor element. \
3 = Misses one significant element. 2 = Misses multiple significant elements. \
1 = Covers only a fraction of the case.

### 3. Conciseness & Style (1-5)
5 = Tight, focused, similar length to reference. Clear prose with shorter \
sentences, one idea per sentence. Correct legal terminology ("judgment" not \
"judgement"). Past tense. Acronyms spelled out on first use.
4 = Slightly wordy but no irrelevant content.
3 = Noticeably padded or includes tangential information.
2 = Significantly longer than needed with filler.
1 = Rambling or disorganized.

### 4. Legal Reasoning (1-5)
5 = Correctly identifies legal standards, doctrines, and statutory basis. \
Distinguishes allegations from findings. Captures how courts addressed \
opposing arguments.
4 = Identifies legal framework with minor imprecision.
3 = Mentions legal concepts but doesn't connect them to the outcome.
2 = Mischaracterizes legal reasoning or confuses standards.
1 = No meaningful engagement with legal reasoning.

### 5. Overall Quality (1-5)
Holistic judgment: would a Clearinghouse editor find this summary publishable \
with minimal edits?

Respond with ONLY valid JSON, no other text."""

JUDGE_USER_TEMPLATE = """\
Reference summary:
{reference}

AI-generated summary:
{generated}

Rate the AI-generated summary. Respond with ONLY this JSON format:
{{"factual_accuracy": N, "completeness": N, "conciseness_style": N, "legal_reasoning": N, "overall": N, "brief_rationale": "One sentence explaining the overall score."}}"""


# ── I/O helpers ────────────────────────────────────────────────────────────────
def load_test_data(path=None, sample_n=None, seed=42):
    """Load test JSONL, optionally sampling n records."""
    path = path or DATA_PATH
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if sample_n and sample_n < len(records):
        random.seed(seed)
        records = random.sample(records, sample_n)
    return records


def load_generations(path):
    """Load a generations JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records, path):
    """Write list of dicts to JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}")
