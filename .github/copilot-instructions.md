Purpose

This file orients AI coding agents to the repository structure, runtime patterns, and gotchas so you can be productive immediately.

**Big Picture**

- **What**: This repo processes many text classification datasets, fine-tunes transformer models, and collects results (download -> preprocess -> finetune -> aggregate).
- **Components**: CLI entrypoint in [main.py](main.py); dataset processing in [ntcir19_pretrained_model_retrieval/cli.py](ntcir19_pretrained_model_retrieval/cli.py); core training logic in [ntcir19_pretrained_model_retrieval/trainer.py](ntcir19_pretrained_model_retrieval/trainer.py); job-slicing helper in [run_assigned.py](run_assigned.py).
- **Data flow**: Tasks Excel -> processed dataset directories (under [bert-data](bert-data)) each containing `train.jsonl`, `val.jsonl`, `test.jsonl` -> experiments run produce `experiment_results/<dataset>_<model>/` containing `test_results.json` and `training_history.csv`.

**Key runtime conventions**

- **Data format**: JSONL files with at least `text` and `labels` columns. See dataset writer in [ntcir19_pretrained_model_retrieval/cli.py](ntcir19_pretrained_model_retrieval/cli.py) where `df_train.to_json(... "train.jsonl")` is used.
- **Label mapping**: `trainer.build_label_mapping` supports boolean (False/True -> 0/1), numeric (sorted unique -> 0..N-1) and text/categorical (stable string-sorted ordering). Unknown labels in val/test are dropped and logged.
- **Experiment naming**: Run names are `{basename(data_dir)}_{model_id.replace('/', '-')}` and outputs are written under the `output_root` configured in [config.toml](config.toml).
- **Skip behavior**: If `test_results.json` exists in an experiment output dir, `run_experiment` will skip that job (idempotent retries supported).

**Training specifics worth preserving**

- **Tokenization**: Uses `AutoTokenizer.from_pretrained(model_id)` and caps `model_max_length` at 512 if the tokenizer reports an absurdly large value.
- **Model loading**: `AutoModelForSequenceClassification.from_pretrained(..., ignore_mismatched_sizes=True)` — agents should avoid changing this unless necessary for model/card compatibility.
- **Training recipe**: `TrainingArguments` set: `num_train_epochs=1000`, `eval_strategy='epoch'`, early stopping via `EarlyStoppingCallback(early_stopping_patience=10)`, `save_total_limit=2`, `load_best_model_at_end=True`, `full_determinism=True`. Be careful: epochs are large but early stopping controls runtime.

**Important files & configs**

- **Config**: [config.toml](config.toml) drives most defaults: dataset roots (`[finetune].data_dir_root`), `model_list_excel`, `output_root`, `batch_size`, log files.
- **Model list**: Excel of model IDs read by [run_assigned.py](run_assigned.py) and [ntcir19_pretrained_model_retrieval/cli.py](ntcir19_pretrained_model_retrieval/cli.py). Default column name is `model_name` (see `model_list_column` in [config.toml](config.toml)).
- **Output layout**: [config.toml](config.toml) default `output_root` is `experiment_results`; each run writes `test_results.json` and `training_history.csv`.
- **Dependencies**: See [pyproject.toml](pyproject.toml) for required packages (`transformers`, `datasets`, `torch`, `pandas`, `scikit-learn`, etc.).

**Common developer workflows / commands**

- **Process/download datasets**: `python main.py download-datasets --config config.toml` (reads task Excel and writes per-dataset `train.jsonl`/`val.jsonl`/`test.jsonl` under `bert-data`).
- **Run all finetunes locally**: `python main.py finetune-all --config config.toml` (reads all dirs under `data_dir_root` and all models in `model_list_excel`).
- **Assign jobs across machines**: Example: `python run_assigned.py --machine-id 0 --num-machines 5 --config config.toml --dry-run` then remove `--dry-run` to execute. This implements modulo slicing: `linear_index = dataset_idx * num_models + model_idx`.
- **Quick single run**: call `run_experiment(data_dir, model_id, output_root, seed, batch_size)` from Python or run via CLI loops above.

**Project-specific patterns / gotchas**

- **Typer command names**: functions are defined with snake_case (`finetune_all`) but invoked on CLI as kebab-case (`finetune-all`). Use `python main.py <command>`.
- **Excel-driven orchestration**: Datasets and models are enumerated from Excel files. Validate Excel columns: tasks must include `dataset_name,text_col,label_col,train_split,val_split,test_split` (see [ntcir19_pretrained_model_retrieval/cli.py](ntcir19_pretrained_model_retrieval/cli.py)).
- **Idempotency**: The trainer skips completed experiments to support interrupted runs and retries — preserves compute.
- **Determinism**: `full_determinism=True` in training arguments makes runs deterministic but may restrict some accelerations; be mindful when changing.

**When changing code**

- Prefer small, focused edits. If modifying training hyperparameters, update [ntcir19_pretrained_model_retrieval/trainer.py](ntcir19_pretrained_model_retrieval/trainer.py) and document the rationale in a PR.
- Preserve the `ignore_mismatched_sizes=True` and label-mapping behavior unless you confirm model/dataset mismatches.

If anything here is unclear or you want more examples (e.g., a walkthrough for adding a new dataset row in the task Excel or adding a new column to the model list Excel), tell me which part to expand and I will iterate.
