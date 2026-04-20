#!/usr/bin/env python3
"""LoRA fine-tuning of Qwen2.5-7B-Instruct for legal case summarization.

Based on the original train_lora.py with critical fixes:
  - max-length raised from 4096 → 8192 (raw prompts were exceeding context)
  - learning-rate lowered from 1e-4 → 2e-5 (was causing loss spikes to 5000+)
  - max-grad-norm added at 1.0 (prevents gradient explosions)
  - num-train-epochs raised from 1 → 3 (more passes over the smaller, better data)
  - warmup-ratio raised from 0.03 → 0.05

Usage:
    python scripts/train_lora.py \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --train-file data/training_v2/train.jsonl \\
        --val-file data/training_v2/val.jsonl \\
        --output-dir runs/qwen25_7b_lora_run2 \\
        --max-length 8192 \\
        --learning-rate 2e-5 \\
        --num-train-epochs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Path to a checkpoint directory to resume training from")
    return parser.parse_args()


def load_json_dataset(path: str):
    return load_dataset("json", data_files=str(Path(path).resolve()), split="train")


def ensure_prompt_completion_columns(dataset):
    column_names = set(dataset.column_names)
    if "prompt" not in column_names:
        raise ValueError("Dataset must contain a 'prompt' column.")
    if "completion" in column_names:
        return dataset
    if "response" in column_names:
        return dataset.map(lambda row: {"completion": row["response"]})
    raise ValueError("Dataset must contain either 'completion' or 'response'.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print(f"{'='*60}")
    print(f"LoRA Fine-Tuning: {args.model_name}")
    print(f"{'='*60}")
    print(f"Train: {args.train_file}")
    print(f"Val:   {args.val_file}")
    print(f"Output: {args.output_dir}")
    print(f"Max length: {args.max_length}")
    print(f"LR: {args.learning_rate}, Epochs: {args.num_train_epochs}")
    print(f"Grad clip: {args.max_grad_norm}, Weight decay: {args.weight_decay}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print()

    train_dataset = ensure_prompt_completion_columns(load_json_dataset(args.train_file))
    eval_dataset = None
    if args.val_file:
        eval_dataset = ensure_prompt_completion_columns(load_json_dataset(args.val_file))

    print(f"Train records: {len(train_dataset)}")
    if eval_dataset:
        print(f"Val records: {len(eval_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=3,
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
