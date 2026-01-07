from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from .logger_setup import get_logger

# Dataset split ratio constants
VAL_TEST_FROM_TRAIN_RATIO = 0.2  # When both val and test are missing
VAL_TEST_SPLIT_RATIO = 0.5  # Split the 20% into val (10%) and test (10%)
VAL_FROM_TRAIN_RATIO = 0.1  # When only val is missing


def limit_rows(df: pd.DataFrame, max_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Limit DataFrame to maximum number of rows via random sampling.

    Args:
        df: DataFrame to limit
        max_rows: Maximum number of rows to keep
        seed: Random seed for reproducible sampling

    Returns:
        DataFrame with at most max_rows rows
    """
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=seed)
    return df


def one_hot_label_to_id(df: pd.DataFrame, cols: list[str]) -> list[int]:
    """Convert one-hot encoded columns to label ids using vectorized operations."""
    one_hot_array = df[cols].values
    # Find the index of the maximum value (should be 1) in each row
    ids = one_hot_array.argmax(axis=1)
    # Replace with -1 where the max value is not 1 (no valid label found)
    ids = [int(idx) if one_hot_array[i, idx] == 1 else -1 for i, idx in enumerate(ids)]
    return ids


def load_dataset_safe(row: pd.Series, revision: str) -> Tuple[Optional[DatasetDict], Optional[str]]:
    """
    Safely load a HuggingFace dataset with error handling.

    Args:
        row: Pandas Series containing 'dataset_name' and 'subset' fields
        revision: Dataset revision to load (e.g., 'refs/convert/parquet')

    Returns:
        Tuple of (dataset, subset_name) or (None, None) on error
    """
    logger = get_logger()
    dataset_name = row["dataset_name"]
    subset = row["subset"]
    try:
        if pd.isna(subset):
            ds = load_dataset(dataset_name, revision=revision)
            subset_name = "default"
        else:
            ds = load_dataset(dataset_name, subset, revision=revision)
            subset_name = subset
        return ds, subset_name
    except Exception as e:
        logger.error(f"Error loading {dataset_name}: {e}")
        return None, None


def get_splits(ds: DatasetDict, row: pd.Series, seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Extract or create train/validation/test splits from a dataset.

    Handles four cases:
    1. All splits specified: use them directly
    2. Missing val & test: split train into 80/10/10
    3. Missing val only: split train into 90/10, use existing test
    4. Missing test only: split train into 90/10, use existing val as test

    Args:
        ds: HuggingFace DatasetDict containing dataset splits
        row: Pandas Series with 'train_split', 'val_split', 'test_split' fields
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_split = row["train_split"]
    val_split = row["val_split"]
    test_split = row["test_split"]

    if pd.isna(val_split) and pd.isna(test_split):
        # Split train into 80/20, then 20 into 10/10 (val/test)
        ds_split_1 = ds[train_split].train_test_split(test_size=VAL_TEST_FROM_TRAIN_RATIO, seed=seed)
        ds_train = ds_split_1["train"]

        ds_split_2 = ds_split_1["test"].train_test_split(test_size=VAL_TEST_SPLIT_RATIO, seed=seed)
        ds_val = ds_split_2["train"]
        ds_test = ds_split_2["test"]

    elif pd.isna(test_split):
        # Use existing val as test, create new val from train
        ds_test = ds[val_split]
        ds_train_val = ds[train_split].train_test_split(test_size=VAL_FROM_TRAIN_RATIO, seed=seed)
        ds_train = ds_train_val["train"]
        ds_val = ds_train_val["test"]

    elif pd.isna(val_split):
        # Use existing test, create val from train
        ds_test = ds[test_split]
        ds_train_val = ds[train_split].train_test_split(test_size=VAL_FROM_TRAIN_RATIO, seed=seed)
        ds_train = ds_train_val["train"]
        ds_val = ds_train_val["test"]

    else:
        # All splits specified
        ds_train = ds[train_split]
        ds_val = ds[val_split]
        ds_test = ds[test_split]

    return ds_train, ds_val, ds_test


def process_split(
    ds_split: Optional[Dataset], text_col: str, label_col: str, seed: int = 42, max_rows: int = 5000
) -> pd.DataFrame:
    """
    Process a dataset split into standardized text/labels format.

    Handles:
    - Multiple text columns (comma-separated): joins with ". "
    - Multiple label columns (comma-separated): converts from one-hot to single id
    - Missing values: drops rows with null text or labels
    - Row limit: samples down to max_rows if needed

    Args:
        ds_split: HuggingFace Dataset split (or None)
        text_col: Column name(s) for text. Comma-separated for multiple columns.
        label_col: Column name(s) for labels. Comma-separated for one-hot encoding.
        seed: Random seed for sampling
        max_rows: Maximum rows to keep per split

    Returns:
        DataFrame with 'text' and 'labels' columns, limited to max_rows
    """
    if ds_split is None:
        return pd.DataFrame()

    df = pd.DataFrame(ds_split)

    # Handle multiple text columns
    if "," in text_col:
        text_cols = [c.strip() for c in text_col.split(",")]
        df[text_col] = df[text_cols].agg(". ".join, axis=1)

    # Handle one-hot encoded labels
    if "," in label_col:
        label_cols = [c.strip() for c in label_col.split(",")]
        df[label_col] = one_hot_label_to_id(df, label_cols)

    # Standardize column names
    if text_col in df.columns and label_col in df.columns:
        df = df[[text_col, label_col]].copy()
        df.columns = ["text", "labels"]
    else:
        return pd.DataFrame()

    # Drop missing values
    na_row_count = df["text"].isnull().sum() + df["labels"].isnull().sum()
    if na_row_count > 0:
        logger = get_logger()
        logger.warning(f"DATA WARNING: Dropping {na_row_count} rows with missing text or labels.")
    df = df.dropna(subset=["text", "labels"])
    df = df.reset_index(drop=True)

    return limit_rows(df, max_rows=max_rows, seed=seed)
