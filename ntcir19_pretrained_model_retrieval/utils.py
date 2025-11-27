from pathlib import Path

import pandas as pd
from datasets import load_dataset

from .logger_setup import get_logger


def limit_rows(df: pd.DataFrame, max_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=seed)
    return df


def one_hot_label_to_id(df: pd.DataFrame, cols: list[str]) -> list[int]:
    ids = []
    for _, _row in df.iterrows():
        found_id = -1
        for i, col_name in enumerate(cols):
            if _row[col_name] == 1:
                found_id = i
        ids.append(found_id)
    return ids


def load_dataset_safe(row: pd.Series, revision: str):
    logger = get_logger()
    dataset_name = row["dataset_name"]
    subset = row["subset"]
    try:
        if pd.isna(subset):
            ds = load_dataset(dataset_name, revision=revision)
            subset_name = "default"
        else:
            ds = load_dataset(dataset_name, subset)
            subset_name = subset
        return ds, subset_name
    except Exception as e:
        logger.error(f"Error loading {dataset_name}: {e}")
        return None, None


def get_splits(ds, row: pd.Series, seed: int = 42):
    train_split = row["train_split"]
    val_split = row["val_split"]
    test_split = row["test_split"]

    if pd.isna(val_split) and pd.isna(test_split):
        ds_split_1 = ds[train_split].train_test_split(test_size=0.2, seed=seed)
        ds_train = ds_split_1["train"]

        ds_split_2 = ds_split_1["test"].train_test_split(test_size=0.5, seed=seed)
        ds_val = ds_split_2["train"]
        ds_test = ds_split_2["test"]

    elif pd.isna(test_split):
        ds_test = ds[val_split]
        ds_train_val = ds[train_split].train_test_split(test_size=0.1, seed=seed)
        ds_train = ds_train_val["train"]
        ds_val = ds_train_val["test"]

    elif pd.isna(val_split):
        ds_test = ds[test_split]
        ds_train_val = ds[train_split].train_test_split(test_size=0.1, seed=seed)
        ds_train = ds_train_val["train"]
        ds_val = ds_train_val["test"]

    else:
        ds_train = ds[train_split]
        ds_val = ds[val_split]
        ds_test = ds[test_split]

    return ds_train, ds_val, ds_test


def process_split(ds_split, text_col: str, label_col: str, seed: int = 42, max_rows: int = 5000):
    if ds_split is None:
        return pd.DataFrame()

    df = pd.DataFrame(ds_split)

    if "," in text_col:
        text_cols = [c.strip() for c in text_col.split(",")]
        df[text_col] = df[text_cols].agg(". ".join, axis=1)

    if "," in label_col:
        label_cols = [c.strip() for c in label_col.split(",")]
        df[label_col] = one_hot_label_to_id(df, label_cols)

    if text_col in df.columns and label_col in df.columns:
        df = df[[text_col, label_col]].copy()
        df.columns = ["text", "labels"]
    else:
        return pd.DataFrame()

    return limit_rows(df, max_rows=max_rows, seed=seed)
