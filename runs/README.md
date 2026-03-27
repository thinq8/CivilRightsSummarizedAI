# Model Runs

This directory holds LoRA fine-tuning runs. Model weights are not tracked in git due to size.

## Runs

### `qwen25_7b_lora_run1` (Primary)

- **Base model**: `Qwen/Qwen2.5-7B-Instruct`
- **Method**: LoRA (rank=16, alpha=32, dropout=0.05)
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training**: 1 epoch, 1,231 steps
- **Checkpoints**: checkpoint-1200, checkpoint-1231

### `preflight_run`

- Baseline/test run with same config, single checkpoint for validation.

## How to reproduce

1. Install ML dependencies: `pip install -e ".[ml]"`
2. Place training data in `data/training/` (see `data/training/README.md`)
3. Training was done using TRL's SFTTrainer. Contact the project team for the training script or check shared resources.

## How to use for evaluation

The eval pipeline loads the LoRA adapter from this directory:

```bash
cd eval
python generate.py --source local --sample 50
```

This requires the adapter files (adapter_model.safetensors, adapter_config.json, tokenizer files) to be present in the run directory.
