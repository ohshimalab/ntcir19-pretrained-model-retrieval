import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from .logger_setup import get_logger

# Training hyperparameters - centralized for easy adjustment
TRAINING_CONFIG = {
    "learning_rate": 2e-5,
    "lr_scheduler_type": "constant",
    "weight_decay": 0.01,
    "optimizer": "adamw_torch",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "num_train_epochs": 1000,
    "early_stopping_patience": 10,
    "metric_for_best_model": "loss",
    "max_token_length": 512,
}


def compute_metrics(pred):
    """Compute classification metrics for model evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def build_label_mapping(series: pd.Series) -> Tuple[dict, dict]:
    """
    Build label-to-id mapping from a pandas Series.

    Supports boolean, numeric, and categorical/text labels with stable ordering.

    Args:
        series: Pandas Series containing label values

    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    s = series.dropna()

    # Booleans: map False->0, True->1 (stable ordering)
    if pd.api.types.is_bool_dtype(s):
        label2id = {False: 0, True: 1}
        id2label = {0: False, 1: True}
        return label2id, id2label

    # Numeric types: preserve unique values but map them to 0..N-1
    if pd.api.types.is_numeric_dtype(s):
        unique = sorted(s.unique().tolist())
        label2id = {v: i for i, v in enumerate(unique)}
        id2label = {i: v for i, v in enumerate(unique)}
        return label2id, id2label

    # Fallback: categorical/text labels. Use stable ordering by string
    unique = sorted(s.unique().tolist(), key=lambda x: str(x))
    label2id = {v: i for i, v in enumerate(unique)}
    id2label = {i: v for i, v in enumerate(unique)}
    return label2id, id2label


def apply_label_mapping(df: pd.DataFrame, mapping: dict, split_name: str, run_name: str) -> pd.DataFrame:
    """
    Apply label mapping to dataframe and handle unknown labels.

    Args:
        df: DataFrame with 'labels' column
        mapping: Dictionary mapping labels to integer ids
        split_name: Name of split (for logging)
        run_name: Experiment name (for logging)

    Returns:
        DataFrame with labels mapped to integers
    """
    logger = get_logger()
    df = df.copy()
    df["labels"] = df["labels"].map(mapping)

    if df["labels"].isnull().any():
        dropped_count = int(df["labels"].isnull().sum())
        logger.warning(f"LABEL WARNING: {run_name} - Dropping {dropped_count} rows in {split_name}.")
        df = df.dropna(subset=["labels"])

    return df.astype({"labels": "int"})


def load_datasets(data_dir: Path, run_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train/val/test JSONL files from data directory.

    Args:
        data_dir: Path to directory containing train.jsonl, val.jsonl, test.jsonl
        run_name: Experiment name for logging

    Returns:
        Tuple of (train_df, val_df, test_df)

    Raises:
        Exception: If any file cannot be loaded
    """
    logger = get_logger()
    try:
        train_df = pd.read_json(Path(data_dir) / "train.jsonl", lines=True)
        val_df = pd.read_json(Path(data_dir) / "val.jsonl", lines=True)
        test_df = pd.read_json(Path(data_dir) / "test.jsonl", lines=True)
        return train_df, val_df, test_df
    except Exception as e:
        logger.error(f"DATA LOAD ERROR: {run_name} - {e}")
        raise


def prepare_tokenizer(model_id: str, run_name: str) -> Tuple[PreTrainedTokenizer, int]:
    """
    Load tokenizer and determine appropriate max length.

    Args:
        model_id: HuggingFace model identifier
        run_name: Experiment name for logging

    Returns:
        Tuple of (tokenizer, max_length)
    """
    logger = get_logger()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    max_len = tokenizer.model_max_length

    if max_len > 100_000:
        logger.warning(
            f"MAX LEN WARNING: {model_id} reports max_length={max_len}. Capping at {TRAINING_CONFIG['max_token_length']}."
        )
        max_len = TRAINING_CONFIG["max_token_length"]

    logger.info(f"TOKENIZATION: {run_name} - Using max_length={max_len}")
    return tokenizer, max_len


def create_training_history(trainer_state) -> pd.DataFrame:
    """
    Extract and organize training history from trainer state.

    Args:
        trainer_state: Trainer.state containing log_history

    Returns:
        DataFrame with epoch-level training metrics
    """
    df_logs = pd.DataFrame(trainer_state.log_history)
    epoch_stats = []

    if not df_logs.empty and "epoch" in df_logs.columns:
        df_logs = df_logs.dropna(subset=["epoch"])
        for epoch, group in df_logs.groupby("epoch"):
            record = {"epoch": epoch}
            for _, row in group.iterrows():
                record.update(row.dropna().to_dict())
            if "eval_loss" in record:
                epoch_stats.append(record)

    return pd.DataFrame(epoch_stats)


def run_experiment(data_dir, model_id, output_root, seed: int, batch_size: int):
    """
    Run a complete fine-tuning experiment for one dataset-model pair.

    Args:
        data_dir: Directory containing train/val/test JSONL files
        model_id: HuggingFace model identifier
        output_root: Root directory for experiment outputs
        seed: Random seed for reproducibility
        batch_size: Training and evaluation batch size
    """
    logger = get_logger()
    run_name = f"{os.path.basename(data_dir)}_{model_id.replace('/', '-')}"
    output_dir = Path(output_root) / run_name

    final_result_file = output_dir / "test_results.json"
    if final_result_file.exists():
        logger.info(f"SKIP: {run_name} - Found existing test_results.json")
        return

    logger.info(f"STARTING: {run_name} | Dir: {data_dir} | Model: {model_id}")

    # Load datasets
    try:
        train_df, val_df, test_df = load_datasets(data_dir, run_name)
    except Exception:
        return

    # Build label mappings from training data
    label2id, id2label = build_label_mapping(train_df["labels"])
    num_labels = len(label2id)

    train_df = apply_label_mapping(train_df, label2id, "train", run_name)
    val_df = apply_label_mapping(val_df, label2id, "validation", run_name)
    test_df = apply_label_mapping(test_df, label2id, "test", run_name)

    # Create HuggingFace datasets
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df),
        }
    )

    # Prepare tokenizer and tokenize datasets
    tokenizer, max_len = prepare_tokenizer(model_id, run_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Configure training
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        optim=TRAINING_CONFIG["optimizer"],
        adam_beta1=TRAINING_CONFIG["adam_beta1"],
        adam_beta2=TRAINING_CONFIG["adam_beta2"],
        adam_epsilon=TRAINING_CONFIG["adam_epsilon"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=TRAINING_CONFIG["metric_for_best_model"],
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        seed=seed,
        data_seed=seed,
        full_determinism=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"])],
    )

    # Train
    logger.info(f"TRAINING START: {run_name} - From scratch")
    trainer.train()

    # Evaluate on test set
    logger.info(f"EVALUATING: {run_name} - On test set")
    test_results = trainer.evaluate(tokenized_datasets["test"])

    # Save training history
    df_final = create_training_history(trainer.state)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_dir / "training_history.csv", index=False)

    # Save test results
    pd.Series(test_results).to_json(final_result_file)
    logger.info(f"COMPLETED: {run_name} - Results saved.")
