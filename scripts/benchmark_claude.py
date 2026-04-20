#!/usr/bin/env python3
"""Benchmark Claude API on the same 50 test cases used for Qwen eval.

Generates summaries with a Claude model, then runs Claude-as-judge on each
result using the same 5-dimension rubric as eval_checkpoint_v2.py. Output
format is drop-in compatible with the existing eval JSONL / summary.json
so it plugs straight into eval_analysis.ipynb.

Handles:
- Prompts that exceed the 200K-token context window (char-budget truncation)
- Prompt caching on the judge system prompt (cuts judge cost ~90%)
- Resume: re-running with the same --output-file skips already-done indices
- Running cost tracking with live printout

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/benchmark_claude.py \\
        --model claude-sonnet-4-5 \\
        --test-file /Users/liamsandy/Documents/Legal/First_Train/test.jsonl \\
        --num-samples 50 \\
        --output-file /Users/liamsandy/Documents/Legal/Polish/eval2/eval_claude_sonnet.jsonl

    # Cheap baseline:
    python scripts/benchmark_claude.py --model claude-haiku-4-5 ...

    # Skip judge (summaries only):
    python scripts/benchmark_claude.py --skip-judge ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    sys.exit("ERROR: pip install anthropic")


# ── Pricing (USD per 1M tokens) ───────────────────────────────────────────────
# Keep in sync with https://www.anthropic.com/pricing
PRICING = {
    "claude-opus-4-5":   {"input": 15.00, "output": 75.00, "cache_read": 1.50},
    "claude-sonnet-4-5": {"input":  3.00, "output": 15.00, "cache_read": 0.30},
    "claude-haiku-4-5":  {"input":  1.00, "output":  5.00, "cache_read": 0.10},
    # Older aliases you might pass
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
    "claude-3-5-haiku-latest":  {"input": 0.80, "output":  4.00, "cache_read": 0.08},
}


# ── Rubric (same as eval_checkpoint_v2.py) ───────────────────────────────────

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

JUDGE_DIMS = [
    "factual_accuracy",
    "completeness",
    "conciseness_style",
    "legal_reasoning",
    "overall",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

# Claude's context is 200K tokens. Reserve ~8K for system + generation budget.
# At ~4 chars/token that's 768K chars of input. Be conservative: 700K.
MAX_PROMPT_CHARS = 700_000


def truncate_prompt(prompt: str) -> tuple[str, bool]:
    """If prompt exceeds budget, keep the head and tail so we don't lose the
    caption or the disposition. Returns (prompt, was_truncated)."""
    if len(prompt) <= MAX_PROMPT_CHARS:
        return prompt, False
    keep_head = MAX_PROMPT_CHARS // 2
    keep_tail = MAX_PROMPT_CHARS - keep_head
    head = prompt[:keep_head]
    tail = prompt[-keep_tail:]
    dropped = len(prompt) - MAX_PROMPT_CHARS
    notice = f"\n\n[... {dropped:,} characters elided to fit context window ...]\n\n"
    return head + notice + tail, True


def parse_judge_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def load_test_records(path: str, num_samples: int, seed: int) -> list[dict]:
    random.seed(seed)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if num_samples > 0 and num_samples < len(records):
        records = random.sample(records, num_samples)
    return records


def load_existing(path: Path) -> dict[int, dict]:
    """Load already-written records (by index) for resume."""
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["index"]] = rec
    return out


def price_for(model: str) -> dict:
    if model not in PRICING:
        raise KeyError(
            f"No pricing for {model}. Known: {list(PRICING)}. "
            f"Add it to PRICING at the top of this script."
        )
    return PRICING[model]


def usage_cost(usage, model: str) -> float:
    p = price_for(model)
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    # cache_creation is billed at 1.25x input; cache_read at ~10% of input
    return (
        inp * p["input"] / 1e6
        + cache_read * p["cache_read"] / 1e6
        + cache_create * p["input"] * 1.25 / 1e6
        + out * p["output"] / 1e6
    )


# ── API calls ─────────────────────────────────────────────────────────────────

SUMMARIZE_SYSTEM = (
    "You are a legal editor for the Civil Rights Litigation Clearinghouse. "
    "Produce a concise case summary of the provided civil rights case materials. "
    "Match the style of the Clearinghouse: factual, past tense, one idea per "
    "sentence, acronyms spelled out on first use, covering the opening stakes, "
    "filing date and court, party and counsel type, causes of action with "
    "statutory basis, remedies sought, key procedural events, and outcome."
)


def call_summarize(client, model: str, prompt: str, max_tokens: int) -> tuple[str, object]:
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        system=SUMMARIZE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return text.strip(), resp.usage


def call_judge(client, model: str, reference: str, generated: str) -> tuple[dict | None, str, object]:
    user_msg = JUDGE_USER_TEMPLATE.format(reference=reference, generated=generated)
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0,
        system=[
            {
                "type": "text",
                "text": JUDGE_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return parse_judge_json(text), text, resp.usage


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Claude model id (e.g. claude-sonnet-4-5)")
    ap.add_argument("--judge-model", default=None,
                    help="Model to use as judge. Defaults to --model.")
    ap.add_argument("--test-file", required=True, help="Path to test.jsonl")
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42,
                    help="Must match the seed used for the Qwen eval to hit the same 50 cases")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load data and print cost estimate, don't call API")
    return ap.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    judge_model = args.judge_model or args.model

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: set ANTHROPIC_API_KEY before running.")

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load test records with matching seed so we hit the same 50 cases
    records = load_test_records(args.test_file, args.num_samples, args.seed)
    existing = load_existing(out_path)
    if existing:
        print(f"Resume: found {len(existing)} existing records in {out_path}, "
              f"will skip those.")

    # Dry-run cost estimate
    if args.dry_run:
        total_in = sum(min(len(r["prompt"]), MAX_PROMPT_CHARS) for r in records) // 4
        total_out = 50 * 1000  # rough
        p = price_for(args.model)
        est = total_in / 1e6 * p["input"] + total_out / 1e6 * p["output"]
        print(f"Dry run: ~{total_in:,} in tokens + ~{total_out:,} out tokens")
        print(f"Summarize cost (est): ${est:.2f}")
        if not args.skip_judge:
            jp = price_for(judge_model)
            # Judge system prompt ~800 tokens, cached after first call
            # Per call: ~1500 in (pred+ref+template) + ~200 out
            judge_in = 50 * 1500
            judge_out = 50 * 200
            judge_cost = judge_in / 1e6 * jp["input"] + judge_out / 1e6 * jp["output"]
            print(f"Judge cost (est, with cache): ${judge_cost:.2f}")
            print(f"TOTAL (est): ${est + judge_cost:.2f}")
        return

    client = anthropic.Anthropic()

    results = [existing.get(i) for i in range(len(records))]
    total_cost = 0.0
    truncated_count = 0

    # Append-mode writes so a crash mid-run doesn't lose completed work
    # (We rewrite the whole file at the end to keep it sorted.)

    print(f"Model: {args.model}  |  Judge: {judge_model}")
    print(f"Samples: {len(records)}  |  Output: {out_path}")
    print("=" * 70)

    for i, rec in enumerate(records):
        if results[i] is not None:
            print(f"[{i+1:2d}/{len(records)}] skip (already in output)")
            continue

        raw_prompt = rec["prompt"]
        reference = rec.get("completion", rec.get("response", ""))
        case_id = rec.get("case_id", "")

        prompt, was_truncated = truncate_prompt(raw_prompt)
        if was_truncated:
            truncated_count += 1

        # ── Summarize ──
        t0 = time.time()
        try:
            prediction, sum_usage = call_summarize(
                client, args.model, prompt, args.max_new_tokens
            )
        except anthropic.BadRequestError as e:
            print(f"[{i+1:2d}/{len(records)}] ERROR: {e}")
            continue
        sum_cost = usage_cost(sum_usage, args.model)
        elapsed = time.time() - t0

        result = {
            "index": i,
            "case_id": case_id,
            "model": args.model,
            "truncated": was_truncated,
            "raw_prompt_chars": len(raw_prompt),
            "sent_prompt_chars": len(prompt),
            "reference_len": len(reference.split()),
            "prediction_len": len(prediction.split()),
            "elapsed_sec": round(elapsed, 2),
            "summarize_input_tokens": sum_usage.input_tokens,
            "summarize_output_tokens": sum_usage.output_tokens,
            "summarize_cost_usd": round(sum_cost, 4),
            "prediction": prediction,
            "reference": reference,
        }

        # ── Judge ──
        judge_cost = 0.0
        if not args.skip_judge:
            try:
                judge_json, judge_raw, judge_usage = call_judge(
                    client, judge_model, reference, prediction
                )
                judge_cost = usage_cost(judge_usage, judge_model)
                if judge_json:
                    for dim in JUDGE_DIMS:
                        result[f"judge_{dim}"] = judge_json.get(dim)
                    result["judge_rationale"] = judge_json.get("brief_rationale", "")
                else:
                    result["judge_parse_failed"] = True
                    result["judge_raw"] = judge_raw
                result["judge_model"] = judge_model
                result["judge_cost_usd"] = round(judge_cost, 4)
                result["judge_input_tokens"] = judge_usage.input_tokens
                result["judge_output_tokens"] = judge_usage.output_tokens
                result["judge_cache_read_tokens"] = getattr(
                    judge_usage, "cache_read_input_tokens", 0) or 0
            except anthropic.APIError as e:
                print(f"   judge error: {e}")
                result["judge_error"] = str(e)

        results[i] = result
        total_cost += sum_cost + judge_cost

        overall = result.get("judge_overall", "-")
        trunc_flag = " TRUNC" if was_truncated else ""
        print(
            f"[{i+1:2d}/{len(records)}] {elapsed:5.1f}s "
            f"pred={result['prediction_len']:4d}w "
            f"ref={result['reference_len']:4d}w "
            f"score={overall} "
            f"cost=${sum_cost+judge_cost:.3f} "
            f"running=${total_cost:.2f}{trunc_flag}"
        )

        # Checkpoint: write the file after every sample (cheap, 50 records)
        with open(out_path, "w") as f:
            for r in results:
                if r is not None:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Summary ──
    done = [r for r in results if r is not None]
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  Records written:    {len(done)}/{len(records)}")
    print(f"  Truncated prompts:  {truncated_count}")
    print(f"  Total cost:         ${total_cost:.2f}")

    if done:
        avg_pred = sum(r["prediction_len"] for r in done) / len(done)
        avg_ref = sum(r["reference_len"] for r in done) / len(done)
        print(f"  Avg prediction len: {avg_pred:.0f} words")
        print(f"  Avg reference len:  {avg_ref:.0f} words")

    # Judge aggregate
    if not args.skip_judge:
        print()
        for dim in JUDGE_DIMS:
            key = f"judge_{dim}"
            vals = [r[key] for r in done if r.get(key) is not None]
            if vals:
                mean_v = sum(vals) / len(vals)
                print(f"  judge_{dim}: {mean_v:.2f} (n={len(vals)})")

    # Write summary.json
    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "model": args.model,
        "judge_model": judge_model if not args.skip_judge else None,
        "num_samples": len(done),
        "truncated_count": truncated_count,
        "total_cost_usd": round(total_cost, 4),
        "avg_prediction_len_words": round(
            sum(r["prediction_len"] for r in done) / max(len(done), 1), 1),
        "avg_reference_len_words": round(
            sum(r["reference_len"] for r in done) / max(len(done), 1), 1),
    }
    if not args.skip_judge:
        summary["judge"] = {}
        for dim in JUDGE_DIMS:
            key = f"judge_{dim}"
            vals = [r[key] for r in done if r.get(key) is not None]
            if vals:
                summary["judge"][dim] = {
                    "mean": round(sum(vals) / len(vals), 2),
                    "min": min(vals),
                    "max": max(vals),
                    "n": len(vals),
                }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {out_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
