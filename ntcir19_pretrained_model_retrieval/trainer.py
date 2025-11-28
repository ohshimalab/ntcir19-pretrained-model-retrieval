import os

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .logger_setup import get_logger


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def run_experiment(data_dir, model_id, output_root, seed: int, batch_size: int):
    logger = get_logger()
    run_name = f"{os.path.basename(data_dir)}_{model_id.replace('/', '-') }"
    output_dir = os.path.join(str(output_root), run_name)

    final_result_file = os.path.join(output_dir, "test_results.json")
    if os.path.exists(final_result_file):
        logger.info(f"SKIP: {run_name} - Found existing test_results.json")
        return

    logger.info(f"STARTING: {run_name} | Dir: {data_dir} | Model: {model_id}")

    try:
        train_df = pd.read_json(os.path.join(data_dir, "train.jsonl"), lines=True)
        val_df = pd.read_json(os.path.join(data_dir, "val.jsonl"), lines=True)
        test_df = pd.read_json(os.path.join(data_dir, "test.jsonl"), lines=True)
    except Exception as e:
        logger.error(f"DATA LOAD ERROR: {run_name} - {e}")
        return

    if not pd.api.types.is_numeric_dtype(train_df["labels"]):
        logger.info(f"LABEL MAPPING: {run_name} - Text labels detected.")
        unique_labels = sorted(train_df["labels"].unique().tolist())
        num_labels = len(unique_labels)

        label2id = {label: i for i, label in enumerate(unique_labels)}
        id2label = {i: label for i, label in enumerate(unique_labels)}

        def map_labels(df, split_name):
            df["labels"] = df["labels"].map(label2id)
            if df["labels"].isnull().any():
                dropped_count = df["labels"].isnull().sum()
                logger.warning(f"LABEL WARNING: {run_name} - Dropping {dropped_count} rows in {split_name}.")
                df = df.dropna(subset=["labels"])
            return df.astype({"labels": "int"})

        train_df = map_labels(train_df, "train")
        val_df = map_labels(val_df, "validation")
        test_df = map_labels(test_df, "test")
    else:
        unique_labels = sorted(train_df["labels"].unique().tolist())
        num_labels = len(unique_labels)
        label2id = {l: l for l in unique_labels}
        id2label = {l: l for l in unique_labels}

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    max_len = tokenizer.model_max_length
    if max_len > 100_000:
        logger.warning(f"MAX LEN WARNING: {model_id} reports max_length={max_len}. Capping at 512.")
        max_len = 512

    logger.info(f"TOKENIZATION: {run_name} - Using max_length={max_len}")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels, label2id=label2id, id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1000,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    logger.info(f"TRAINING START: {run_name} - From scratch")
    train_result = trainer.train()

    logger.info(f"EVALUATING: {run_name} - On Test set")
    test_results = trainer.evaluate(tokenized_datasets["test"])

    history = trainer.state.log_history
    df_logs = pd.DataFrame(history)

    epoch_stats = []
    if not df_logs.empty and "epoch" in df_logs.columns:
        df_logs = df_logs.dropna(subset=["epoch"])
        for epoch, group in df_logs.groupby("epoch"):
            record = {"epoch": epoch}
            for _, row in group.iterrows():
                record.update(row.dropna().to_dict())
            if "eval_loss" in record:
                epoch_stats.append(record)

    df_final = pd.DataFrame(epoch_stats)
    df_final.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    pd.Series(test_results).to_json(final_result_file)
    logger.info(f"COMPLETED: {run_name} - Results saved.")
