"""Score generated legal summaries with ROUGE, BERTScore, and LLM-as-judge.

Usage:
    python eval/evaluate.py results/generations_local_*.jsonl
    python eval/evaluate.py results/generations_local_*.jsonl --skip-judge
    python eval/evaluate.py results/generations_local_*.jsonl --judge-sample 50
"""

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from config import (
    CLAUDE_MAX_CONCURRENT,
    CLAUDE_MODEL,
    JUDGE_SYSTEM,
    JUDGE_USER_TEMPLATE,
    RESULTS_DIR,
    load_generations,
    save_jsonl,
)


# ── ROUGE ──────────────────────────────────────────────────────────────────────
def compute_rouge(records):
    """Compute ROUGE-1/2/L F1 for each record."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    for rec in tqdm(records, desc="ROUGE"):
        scores = scorer.score(rec["reference"], rec["generated"])
        rec["rouge1_f"] = round(scores["rouge1"].fmeasure, 4)
        rec["rouge2_f"] = round(scores["rouge2"].fmeasure, 4)
        rec["rougeL_f"] = round(scores["rougeL"].fmeasure, 4)
    return records


# ── BERTScore ──────────────────────────────────────────────────────────────────
def compute_bertscore(records):
    """Compute BERTScore P/R/F1 for each record."""
    from bert_score import score as bert_score

    refs = [r["reference"] for r in records]
    gens = [r["generated"] for r in records]

    print("Computing BERTScore (this may take a minute)...")
    P, R, F1 = bert_score(gens, refs, lang="en", verbose=True)

    for i, rec in enumerate(records):
        rec["bertscore_p"] = round(P[i].item(), 4)
        rec["bertscore_r"] = round(R[i].item(), 4)
        rec["bertscore_f1"] = round(F1[i].item(), 4)
    return records


# ── LLM Judge ─────────────────────────────────────────────────────────────────
def compute_judge_scores(records, sample_n=None, seed=42):
    """Use Claude to judge summary quality on 5 dimensions."""
    import random

    import anthropic

    if sample_n and sample_n < len(records):
        random.seed(seed)
        judge_indices = set(random.sample(range(len(records)), sample_n))
    else:
        judge_indices = set(range(len(records)))

    to_judge = [(i, records[i]) for i in judge_indices]
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CLAUDE_MAX_CONCURRENT)

    async def _judge(idx, rec):
        async with semaphore:
            user_msg = JUDGE_USER_TEMPLATE.format(
                reference=rec["reference"], generated=rec["generated"]
            )
            try:
                resp = await client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=256,
                    system=JUDGE_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                text = resp.content[0].text.strip()
                # Parse JSON, tolerating markdown code fences
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                scores = json.loads(text)
                return idx, scores
            except (json.JSONDecodeError, Exception) as e:
                print(f"  Judge error on {rec['id']}: {e}")
                return idx, None

    async def _run_all():
        tasks = [_judge(idx, rec) for idx, rec in to_judge]
        results = []
        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="LLM Judge"
        ):
            results.append(await coro)
        return results

    judge_results = asyncio.run(_run_all())

    judge_dims = [
        "factual_accuracy",
        "completeness",
        "conciseness_style",
        "legal_reasoning",
        "overall",
    ]
    for idx, scores in judge_results:
        if scores:
            for dim in judge_dims:
                records[idx][f"judge_{dim}"] = scores.get(dim)
            records[idx]["judge_rationale"] = scores.get("brief_rationale", "")

    judged_count = sum(1 for _, s in judge_results if s is not None)
    failed_count = len(judge_results) - judged_count
    print(f"Judged {judged_count}/{len(to_judge)} records ({failed_count} failures)")
    return records


# ── Aggregation ────────────────────────────────────────────────────────────────
def aggregate(records):
    """Compute aggregate stats for all numeric score columns."""
    summary = {"n_records": len(records)}

    metric_keys = [
        "rouge1_f", "rouge2_f", "rougeL_f",
        "bertscore_p", "bertscore_r", "bertscore_f1",
        "judge_factual_accuracy", "judge_completeness",
        "judge_conciseness_style", "judge_legal_reasoning", "judge_overall",
    ]

    for key in metric_keys:
        vals = [r[key] for r in records if key in r and r[key] is not None]
        if vals:
            summary[key] = {
                "mean": round(statistics.mean(vals), 4),
                "median": round(statistics.median(vals), 4),
                "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "n": len(vals),
            }
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate generated legal summaries")
    parser.add_argument("input", help="Path to generations JSONL file")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM judge")
    parser.add_argument("--judge-sample", type=int, default=None, help="Judge N records")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for judge sampling")
    args = parser.parse_args()

    records = load_generations(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    # Filter out any error records
    valid = [r for r in records if not r["generated"].startswith("ERROR:")]
    if len(valid) < len(records):
        print(f"Skipping {len(records) - len(valid)} error records")
    records = valid

    t0 = time.time()

    # Tier 1: ROUGE (always)
    records = compute_rouge(records)

    # Tier 2: BERTScore (always)
    records = compute_bertscore(records)

    # Tier 3: LLM Judge (optional)
    if not args.skip_judge:
        records = compute_judge_scores(records, sample_n=args.judge_sample, seed=args.seed)

    elapsed = time.time() - t0

    # Determine source from the input filename
    input_name = Path(args.input).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save per-record scores
    scores_path = RESULTS_DIR / f"scores_{input_name}_{timestamp}.jsonl"
    save_jsonl(records, scores_path)

    # Save aggregate summary
    summary = aggregate(records)
    summary["source_file"] = args.input
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary_path = RESULTS_DIR / f"summary_{input_name}_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Print summary to console
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY ({len(records)} records, {elapsed:.1f}s)")
    print(f"{'='*60}")
    for key, stats in summary.items():
        if isinstance(stats, dict) and "mean" in stats:
            print(f"  {key:30s}  mean={stats['mean']:.4f}  med={stats['median']:.4f}  std={stats['std']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
