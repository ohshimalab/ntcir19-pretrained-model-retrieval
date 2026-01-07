## Deterministic CUDA runs

- Export before launching: `CUBLAS_WORKSPACE_CONFIG=:16:8` (or `:4096:8`), `PYTORCH_DETERMINISTIC=1`; optionally add `CUDA_LAUNCH_BLOCKING=1` for debugging.
- The entrypoint disables TF32 and forces deterministic kernels when CUDA is present; leave AMP/bfloat16 off if you need reproducibility.
- Keep `dataloader_num_workers=0` (default here) to avoid multi-worker nondeterminism.
- Pin Torch/Transformers/driver versions across runs to prevent kernel-level drift.
- Pin model/tokenizer revisions in the config to avoid pulling newer checkpoint versions across machines.
- Use a consistent dataset revision (config `download.revision`) and shared caches (`HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`).
- Seeding is applied before model creation to stabilize classifier head initialization; reuse the same seed on every machine.
