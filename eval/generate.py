"""Generate legal case summaries from a local fine-tuned model or Claude API.

Usage:
    python eval/generate.py --source local [--sample 100] [--seed 42]
    python eval/generate.py --source claude [--sample 100] [--seed 42]
"""

import argparse
import asyncio
import json
import time
from datetime import datetime

from tqdm import tqdm

from config import (
    BASE_MODEL,
    CLAUDE_MAX_CONCURRENT,
    CLAUDE_MODEL,
    LORA_ADAPTER_PATH,
    MAX_NEW_TOKENS,
    RESULTS_DIR,
    SUMMARIZATION_SYSTEM,
    TEMPERATURE,
    load_test_data,
    save_jsonl,
)


def generate_local(records):
    """Generate summaries using the local Qwen + LoRA model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from {LORA_ADAPTER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH)

    print(f"Loading base model {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )

    print(f"Loading LoRA adapter from {LORA_ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, str(LORA_ADAPTER_PATH))
    model.eval()

    max_ctx = getattr(model.config, "max_position_embeddings", 32768)
    results = []
    skipped = 0

    for rec in tqdm(records, desc="Generating (local)"):
        messages = [
            {"role": "system", "content": SUMMARIZATION_SYSTEM},
            {"role": "user", "content": rec["prompt"]},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        prompt_len = input_ids.shape[1]

        if prompt_len + MAX_NEW_TOKENS > max_ctx:
            print(f"  Skipping {rec['id']}: {prompt_len} tokens exceeds context")
            skipped += 1
            continue

        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                do_sample=TEMPERATURE > 0,
            )

        generated_ids = output_ids[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        results.append(
            {
                "id": rec["id"],
                "case_id": rec["case_id"],
                "reference": rec["response"],
                "generated": generated_text.strip(),
                "source": "local",
            }
        )

    if skipped:
        print(f"Skipped {skipped}/{len(records)} records (exceeded context window)")
    return results


def generate_claude(records):
    """Generate summaries using the Claude API with async concurrency."""
    import anthropic

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CLAUDE_MAX_CONCURRENT)

    async def _summarize(rec):
        async with semaphore:
            try:
                resp = await client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=MAX_NEW_TOKENS,
                    system=SUMMARIZATION_SYSTEM,
                    messages=[{"role": "user", "content": rec["prompt"]}],
                )
                text = resp.content[0].text
            except Exception as e:
                print(f"  Error on {rec['id']}: {e}")
                text = f"ERROR: {e}"
            return {
                "id": rec["id"],
                "case_id": rec["case_id"],
                "reference": rec["response"],
                "generated": text.strip(),
                "source": "claude",
            }

    async def _run_all():
        tasks = [_summarize(rec) for rec in records]
        results = []
        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Generating (claude)"
        ):
            results.append(await coro)
        return results

    return asyncio.run(_run_all())


def main():
    parser = argparse.ArgumentParser(description="Generate legal case summaries")
    parser.add_argument(
        "--source", required=True, choices=["local", "claude"], help="Model source"
    )
    parser.add_argument("--sample", type=int, default=None, help="Sample N records")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    records = load_test_data(sample_n=args.sample, seed=args.seed)
    print(f"Loaded {len(records)} records")

    t0 = time.time()
    if args.source == "local":
        results = generate_local(records)
    else:
        results = generate_claude(records)
    elapsed = time.time() - t0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"generations_{args.source}_{timestamp}.jsonl"
    save_jsonl(results, out_path)
    print(f"Done in {elapsed:.1f}s ({len(results)} summaries)")


if __name__ == "__main__":
    main()
