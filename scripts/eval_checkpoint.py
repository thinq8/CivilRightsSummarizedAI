#!/usr/bin/env python3
"""Quick evaluation of a LoRA checkpoint on the test set.

Generates summaries for N test records and computes ROUGE scores
against the expert reference summaries.

Usage:
    python scripts/eval_checkpoint.py \
        --checkpoint-dir runs/qwen25_7b_lora_run2/checkpoint-1500 \
        --test-file data/training_v2/test.jsonl \
        --num-samples 50 \
        --output-file eval_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_test_records(path: str, num_samples: int, seed: int) -> list[dict]:
    """Load test records, optionally sampling a subset."""
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


def generate_summary(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Generate a summary given a prompt."""
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
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE scores. Falls back gracefully if rouge_score not installed."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            for k in scores:
                scores[k].append(s[k].fmeasure)
        return {k: sum(v) / len(v) for k, v in scores.items()}
    except ImportError:
        print("WARNING: rouge_score not installed, skipping ROUGE computation")
        print("  Install with: pip install rouge-score")
        return {}


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

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(ckpt_path))
    model.eval()

    # Load test data
    print(f"Loading test data from {args.test_file}...")
    records = load_test_records(args.test_file, args.num_samples, args.seed)
    print(f"Evaluating {len(records)} samples\n")

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

        # Print progress
        pred_preview = prediction[:120].replace("\n", " ")
        print(f"[{i+1}/{len(records)}] {elapsed:.1f}s | "
              f"pred={len(prediction.split())}w ref={len(reference.split())}w | "
              f"{pred_preview}...")

    # Compute aggregate metrics
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")

    rouge_scores = compute_rouge(predictions, references)
    if rouge_scores:
        for k, v in rouge_scores.items():
            print(f"  {k}: {v:.4f}")

    avg_pred_len = sum(len(p.split()) for p in predictions) / len(predictions)
    avg_ref_len = sum(len(r.split()) for r in references) / len(references)
    print(f"  Avg prediction length: {avg_pred_len:.0f} words")
    print(f"  Avg reference length:  {avg_ref_len:.0f} words")

    # Save per-record results
    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        # Also save summary
        summary_path = out_path.with_suffix(".summary.json")
        summary = {
            "checkpoint": str(ckpt_path),
            "num_samples": len(records),
            "rouge": rouge_scores,
            "avg_prediction_len_words": round(avg_pred_len, 1),
            "avg_reference_len_words": round(avg_ref_len, 1),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {out_path}")
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
