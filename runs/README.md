# Model Runs

This directory holds LoRA fine-tuning runs. Model weights are not tracked in git due to size.

## Runs

### `qwen25_7b_lora_run2` (Final evaluated run)

- **Base model**: `Qwen/Qwen2.5-7B-Instruct`
- **Method**: LoRA (rank=16, alpha=32, dropout=0.05)
- **Target modules**: all linear layers through PEFT (`target_modules="all-linear"`)
- **Training**: 3 epochs on prepared `extract_first` data
- **Max length**: 8192
- **Learning rate**: `2e-5`
- **Gradient accumulation**: 8
- **Gradient clipping**: 1.0
- **Checkpoints evaluated**: checkpoint-3000 and checkpoint-3690
- **Result**: checkpoint-3000 was better by QA triage and loss EMA; checkpoint-3690 overfit.

### `qwen25_7b_lora_run1` (earlier baseline)

- **Base model**: `Qwen/Qwen2.5-7B-Instruct`
- **Method**: LoRA (rank=16, alpha=32, dropout=0.05)
- **Training**: 1 epoch, 1,231 steps
- **Checkpoints**: checkpoint-1200, checkpoint-1231

### `preflight_run`

- Baseline/test run with same config, single checkpoint for validation.

## How to reproduce

1. Install ML dependencies: `pip install -e ".[ml]"`
2. Place raw training data in `data/training/` (see `data/training/README.md`)
3. Prepare model-ready data into `data/training_v2/` with `scripts/prepare_training_data.py`
4. Train locally on a suitable GPU or submit the HPCC template:

```bash
sbatch scripts/train_run2.sbatch
```

The training entry point is `scripts/train_lora.py`. The HPCC template documents the SLURM resources and exact command used for the final run.

## How to use for evaluation

The eval pipeline loads the LoRA adapter from this directory:

```bash
cd eval
python generate.py --source local --sample 50
```

This requires the adapter files (adapter_model.safetensors, adapter_config.json, tokenizer files) to be present in the run directory.

For checkpoint-specific evaluation, use `scripts/eval_checkpoint_v2.py`; see `TUTORIAL.md` for the full evaluation flow.
