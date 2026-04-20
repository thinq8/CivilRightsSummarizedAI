#!/usr/bin/env python3
"""Evaluate a LoRA checkpoint: generate summaries, compute ROUGE, and run
LLM-as-judge using the base Qwen2.5 model itself.

Generates summaries for N test records, computes ROUGE-1/2/L/Lsum,
then uses the BASE model (without LoRA) as a judge following the
Civil Rights Clearinghouse rubric (5 dimensions, 1-5 scale).

Usage:
    python scripts/eval_checkpoint_v2.py \
        --checkpoint-dir runs/qwen25_7b_lora_run2/checkpoint-3000 \
        --test-file data/training_v2/test.jsonl \
        --num-samples 50 \
        --output-file eval_ckpt3000.jsonl

    # Skip judge (generation + ROUGE only):
    python scripts/eval_checkpoint_v2.py \
        --checkpoint-dir runs/qwen25_7b_lora_run2/checkpoint-3690 \
        --test-file data/training_v2/test.jsonl \
        --num-samples 50 \
        --output-file eval_ckpt3690.jsonl \
        --skip-judge
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── LLM-as-Judge Rubric (from eval/config.py) ────────────────────────────────

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model name")
    parser.add_argument("--test-file", required=True,
                        help="Path to test.jsonl (prompt + completion columns)")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of test samples to evaluate (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max tokens to generate per sample")
    parser.add_argument("--output-file", default=None,
                        help="Path to save per-record results as JSONL")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM-as-judge (generation + ROUGE only)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_records(path: str, num_samples: int, seed: int) -> list[dict]:
    import random
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


# ── Generation ────────────────────────────────────────────────────────────────

def generate_summary(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── ROUGE ─────────────────────────────────────────────────────────────────────

def compute_rouge(predictions: list[str], references: list[str]) -> tuple[dict, list[dict]]:
    """Returns (aggregate_dict, per_record_list)."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
        )
        keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        per_record = []
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            per_record.append({k: round(s[k].fmeasure, 4) for k in keys})
        agg = {k: round(sum(r[k] for r in per_record) / len(per_record), 4) for k in keys}
        return agg, per_record
    except ImportError:
        print("WARNING: rouge_score not installed, skipping ROUGE")
        return {}, [{} for _ in predictions]


# ── LLM-as-Judge (local Qwen) ────────────────────────────────────────────────

def parse_judge_json(text: str) -> dict | None:
    """Parse JSON from judge response, tolerating markdown fences and noise."""
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting first {...} block
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def judge_single(model, tokenizer, reference: str, generated: str) -> dict | None:
    """Use the base model (no LoRA) as a judge for one sample."""
    user_msg = JUDGE_USER_TEMPLATE.format(reference=reference, generated=generated)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(response, skip_special_tokens=True).strip()
    return parse_judge_json(response_text)


def run_judge(base_model, lora_model, tokenizer, results: list[dict]) -> list[dict]:
    """Unload LoRA, run judge on each sample with the base model, reload LoRA.

    The base model is accessed by disabling the LoRA adapter via
    model.disable_adapter_layers(), then re-enabling after judging.
    """
    print(f"\n{'='*60}")
    print("LLM-AS-JUDGE (using base Qwen2.5 without LoRA)")
    print(f"{'='*60}")

    # Disable LoRA so we're using the raw base model as judge
    lora_model.disable_adapter_layers()
    print("LoRA adapters disabled — using base model as judge")

    judged = 0
    failed = 0
    for i, rec in enumerate(results):
        t0 = time.time()
        scores = judge_single(lora_model, tokenizer, rec["reference"], rec["prediction"])
        elapsed = time.time() - t0

        if scores:
            for dim in JUDGE_DIMS:
                rec[f"judge_{dim}"] = scores.get(dim)
            rec["judge_rationale"] = scores.get("brief_rationale", "")
            judged += 1
            dim_str = " ".join(f"{d[:4]}={scores.get(d, '?')}" for d in JUDGE_DIMS)
            print(f"  [{i+1}/{len(results)}] {elapsed:.1f}s | {dim_str}")
        else:
            failed += 1
            print(f"  [{i+1}/{len(results)}] {elapsed:.1f}s | PARSE FAILED")

    # Re-enable LoRA
    lora_model.enable_adapter_layers()
    print(f"\nJudged {judged}/{len(results)} records ({failed} parse failures)")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    ckpt_path = Path(args.checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"{'='*60}")
    print(f"Evaluating checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    # Load base model + LoRA adapter
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    model.eval()

    # Load test data
    print(f"Loading test data from {args.test_file}...")
    records = load_test_records(args.test_file, args.num_samples, args.seed)
    print(f"Evaluating {len(records)} samples\n")

    # ── Phase 1: Generate summaries with LoRA ──
    predictions = []
    references = []
    results = []

    for i, rec in enumerate(records):
        prompt = rec["prompt"]
        reference = rec.get("completion", rec.get("response", ""))

        t0 = time.time()
        prediction = generate_summary(model, tokenizer, prompt, args.max_new_tokens)
        elapsed = time.time() - t0

        predictions.append(prediction)
        references.append(reference)

        result = {
            "index": i,
            "reference_len": len(reference.split()),
            "prediction_len": len(prediction.split()),
            "elapsed_sec": round(elapsed, 2),
            "prediction": prediction,
            "reference": reference,
        }
        results.append(result)

        pred_preview = prediction[:120].replace("\n", " ")
        print(f"[{i+1}/{len(records)}] {elapsed:.1f}s | "
              f"pred={len(prediction.split())}w ref={len(reference.split())}w | "
              f"{pred_preview}...")

    # ── Phase 2: ROUGE ──
    print(f"\n{'='*60}")
    print("ROUGE SCORES")
    print(f"{'='*60}")

    rouge_agg, rouge_per = compute_rouge(predictions, references)
    for i, rscores in enumerate(rouge_per):
        results[i].update(rscores)

    if rouge_agg:
        for k, v in rouge_agg.items():
            print(f"  {k}: {v:.4f}")

    # ── Phase 3: LLM-as-Judge (base Qwen, no LoRA) ──
    if not args.skip_judge:
        results = run_judge(base_model, model, tokenizer, results)

    # ── Aggregate ──
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")

    avg_pred_len = sum(len(p.split()) for p in predictions) / len(predictions)
    avg_ref_len = sum(len(r.split()) for r in references) / len(references)
    print(f"  Avg prediction length: {avg_pred_len:.0f} words")
    print(f"  Avg reference length:  {avg_ref_len:.0f} words")

    if rouge_agg:
        for k, v in rouge_agg.items():
            print(f"  {k}: {v:.4f}")

    # Judge aggregates
    if not args.skip_judge:
        print()
        for dim in JUDGE_DIMS:
            key = f"judge_{dim}"
            vals = [r[key] for r in results if key in r and r[key] is not None]
            if vals:
                mean_v = sum(vals) / len(vals)
                print(f"  judge_{dim}: {mean_v:.2f} (n={len(vals)})")

    # ── Save ──
    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary_path = out_path.with_suffix(".summary.json")
        summary = {
            "checkpoint": str(ckpt_path),
            "num_samples": len(records),
            "rouge": rouge_agg,
            "avg_prediction_len_words": round(avg_pred_len, 1),
            "avg_reference_len_words": round(avg_ref_len, 1),
        }
        if not args.skip_judge:
            summary["judge"] = {}
            for dim in JUDGE_DIMS:
                key = f"judge_{dim}"
                vals = [r[key] for r in results if key in r and r[key] is not None]
                if vals:
                    summary["judge"][dim] = {
                        "mean": round(sum(vals) / len(vals), 2),
                        "min": min(vals),
                        "max": max(vals),
                        "n": len(vals),
                    }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {out_path}")
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
