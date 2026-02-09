from typing import Optional, Tuple

import pandas as pd
from datasets import Dataset


def preserve_features_from_df(df: pd.DataFrame, template_ds: Dataset) -> Dataset:
    """Create a Dataset from pandas DataFrame and attempt to preserve template features.

    If casting fails, return the Dataset built from pandas as-is.
    """
    ds = Dataset.from_pandas(df.reset_index(drop=True))
    try:
        ds = ds.cast(template_ds.features)
    except Exception:
        # Best-effort: if cast fails (schema mismatch), return raw ds
        pass
    return ds


def safe_stratified_resplit(
    df_full: pd.DataFrame, label_col: str, seed: int, t1: float, t2: float, logger
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Attempt sklearn stratified 2-stage split. Return None on failure.

    t1: first-stage test_size (e.g., VAL_TEST_FROM_TRAIN_RATIO)
    t2: second-stage test_size (e.g., VAL_TEST_SPLIT_RATIO)
    """
    try:
        from sklearn.model_selection import train_test_split

        df_train_val, df_test = train_test_split(df_full, test_size=t1, random_state=seed, stratify=df_full[label_col])
        df_train, df_val = train_test_split(
            df_train_val, test_size=t2, random_state=seed, stratify=df_train_val[label_col]
        )
        return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)
    except Exception as e:
        # Could be ValueError from stratify due to too-few-samples per class
        logger.warning("Stratified resplit failed (%s). Falling back to non-stratified split.", e)
        return None
