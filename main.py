import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import typer
from datasets import load_dataset

SEED = 0


@dataclass
class Config:
    output_dir: Path = Path("bert-data")
    max_rows: int = 5000
    default_revision: str = "refs/convert/parquet"


def load_config(path: Path) -> Config:
    """Load configuration from a TOML or JSON file into a Config.

    Supported formats: .toml, .json
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() == ".toml":
        with path.open("rb") as f:
            data = tomllib.load(f)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported config format: use .toml or .json")

    cfg = Config()
    if "output_dir" in data:
        cfg.output_dir = Path(data["output_dir"])
    if "max_rows" in data:
        cfg.max_rows = int(data["max_rows"])
    if "default_revision" in data:
        cfg.default_revision = data["default_revision"]
    return cfg


app = typer.Typer(help="CLI for dataset processing")


def limit_rows(df: pd.DataFrame, max_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    If dataframe is larger than max_rows, sample it down.
    Otherwise, return original dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.
        max_rows (int, optional): Maximum number of rows to keep. Defaults to 5000.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: The sampled DataFrame if larger than max_rows, else original DataFrame.
    """
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=seed)
    return df


def one_hot_label_to_id(df: pd.DataFrame, cols: list[str]) -> list[int]:
    """
    Convert one-hot encoded columns to a single label ID column.

    Args:
        df (pd.DataFrame): The input DataFrame with one-hot columns.
        cols (list[str]): List of column names representing one-hot labels.

    Returns:
        list[int]: List of label IDs corresponding to the one-hot encoding.
    """
    ids = []
    for _, _row in df.iterrows():
        found_id = -1
        for i, col_name in enumerate(cols):
            if _row[col_name] == 1:
                found_id = i
                # break
        ids.append(found_id)
    return ids


def load_dataset_safe(row: pd.Series, revision: str) -> tuple:
    """
    Attempts to load the dataset based on row configuration.

    Args:
        row (pd.Series): Row containing dataset configuration.

    Returns:
        tuple: (dataset object or None, subset name or None)
    """
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
        print(f"Error loading {dataset_name}: {e}")
        return None, None


def get_splits(ds, row: pd.Series, seed: int = 42) -> tuple:
    """
    Determines and generates train, val, and test splits.

    Args:
        ds: Loaded dataset object.
        row (pd.Series): Row containing split configuration.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (train split, val split, test split)
    """
    train_split = row["train_split"]
    val_split = row["val_split"]
    test_split = row["test_split"]

    # Case A: No Validation AND No Test defined
    if pd.isna(val_split) and pd.isna(test_split):
        ds_split_1 = ds[train_split].train_test_split(test_size=0.2, seed=seed)
        ds_train = ds_split_1["train"]

        ds_split_2 = ds_split_1["test"].train_test_split(test_size=0.5, seed=seed)
        ds_val = ds_split_2["train"]
        ds_test = ds_split_2["test"]

    # Case B: Test is missing (Use Val as Test, Split Val from Train)
    elif pd.isna(test_split):
        ds_test = ds[val_split]
        ds_train_val = ds[train_split].train_test_split(test_size=0.1, seed=seed)
        ds_train = ds_train_val["train"]
        ds_val = ds_train_val["test"]

    # Case C: Val is missing (We have Train and Test)
    elif pd.isna(val_split):
        ds_test = ds[test_split]
        ds_train_val = ds[train_split].train_test_split(test_size=0.1, seed=seed)
        ds_train = ds_train_val["train"]
        ds_val = ds_train_val["test"]

    # Case D: All splits exist
    else:
        ds_train = ds[train_split]
        ds_val = ds[val_split]
        ds_test = ds[test_split]

    return ds_train, ds_val, ds_test


def process_split(ds_split, text_col: str, label_col: str, seed: int = 42, max_rows: int = 5000) -> pd.DataFrame:
    """
    Converts dataset split to DataFrame, handles column logic, renames columns, and limits row count.

    Args:
        ds_split: Dataset split object or None.
        text_col (str): Name of the text column(s), comma separated if multiple.
        label_col (str): Name of the label column(s), comma separated if multiple.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: Processed DataFrame with 'text' and 'label' columns, sampled to max_rows.
    """
    if ds_split is None:
        return pd.DataFrame()

    df = pd.DataFrame(ds_split)

    # Handle multiple text columns (comma separated)
    if "," in text_col:
        text_cols = [c.strip() for c in text_col.split(",")]
        # Combine text columns with a separator
        df[text_col] = df[text_cols].agg(". ".join, axis=1)

    # Handle multiple label columns (one-hot encoding)
    if "," in label_col:
        label_cols = [c.strip() for c in label_col.split(",")]
        df[label_col] = one_hot_label_to_id(df, label_cols)

    # Select and Rename
    # Use .copy() to avoid SettingWithCopy warnings
    if text_col in df.columns and label_col in df.columns:
        df = df[[text_col, label_col]].copy()
        df.columns = ["text", "labels"]
    else:
        # Fallback if columns are missing (though logic usually prevents this)
        return pd.DataFrame()

    # Apply Sampling
    return limit_rows(df, max_rows=max_rows, seed=seed)


@app.command()
def download_datasets(
    task_excel: Path = typer.Argument(..., help="Path to Excel file listing dataset tasks"),
    seed: int = typer.Option(SEED, help="Random seed for sampling"),
    config: Path | None = typer.Option(None, "--config", help="Path to TOML/JSON config file"),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Output directory for processed datasets"),
    max_rows: int | None = typer.Option(None, "--max-rows", help="Max rows to keep per split"),
    revision: str | None = typer.Option(
        None, "--revision", help="Revision to use when loading datasets without subset"
    ),
) -> None:
    # --- Main Processing Loop ---

    # Iterate over the rows of your dataframe
    # Assuming 'df' is defined in the environment calling this script

    # Build effective config from defaults, optional config file, and explicit CLI overrides
    effective_cfg = Config()
    if config is not None:
        file_cfg = load_config(config)
        effective_cfg = file_cfg
    if output_dir is not None:
        effective_cfg.output_dir = output_dir
    if max_rows is not None:
        effective_cfg.max_rows = max_rows
    if revision is not None:
        effective_cfg.default_revision = revision

    # Accept Path or str; pandas can read Path directly
    # Validate task_excel exists and looks like an Excel file
    if not task_excel.exists():
        typer.secho(f"Error: task Excel file not found: {task_excel}", fg="red")
        raise typer.Exit(code=2)
    if not task_excel.is_file():
        typer.secho(f"Error: task Excel path is not a file: {task_excel}", fg="red")
        raise typer.Exit(code=2)
    # Optionally validate extension
    if task_excel.suffix.lower() not in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
        typer.secho(
            f"Warning: task file does not have a typical Excel extension: {task_excel.suffix}",
            fg="yellow",
        )

    df = pd.read_excel(task_excel)
    for index, row in df.iterrows():

        # 1. Load Data
        ds, subset_name = load_dataset_safe(row, revision=effective_cfg.default_revision)
        if ds is None:
            continue

        # 2. Get Splits
        ds_train, ds_val, ds_test = get_splits(ds, row, seed)

        # 3. Process each split (Convert -> Format Columns -> Sample)
        text_col = row["text_col"]
        label_col = row["label_col"]

        df_train = process_split(ds_train, text_col, label_col, seed, max_rows=effective_cfg.max_rows)
        df_val = process_split(ds_val, text_col, label_col, seed, max_rows=effective_cfg.max_rows)
        df_test = process_split(ds_test, text_col, label_col, seed, max_rows=effective_cfg.max_rows)

        # 4. Save to Disk
        df_train["split"] = "train"
        df_val["split"] = "val"
        df_test["split"] = "test"

        dataset_name = row["dataset_name"]
        safe_dataset_name = dataset_name.replace("/", "_")
        safe_subset_name = str(subset_name).replace("/", "_")

        # Use configured output directory
        if safe_subset_name != "default":
            dataset_dir = effective_cfg.output_dir / f"{safe_dataset_name}@{safe_subset_name}"
        else:
            dataset_dir = effective_cfg.output_dir / f"{safe_dataset_name}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # save into jsonl
        df_train.to_json(dataset_dir / "train.jsonl", orient="records", lines=True)
        df_val.to_json(dataset_dir / "val.jsonl", orient="records", lines=True)
        df_test.to_json(dataset_dir / "test.jsonl", orient="records", lines=True)

        print(
            f"Processed {dataset_name} - {subset_name} | "
            f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
        )


def main():
    app()


if __name__ == "__main__":
    main()
