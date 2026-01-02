from pathlib import Path
from typing import Tuple

import pandas as pd
import typer

from .config import Config, FinetuneConfig, load_config
from .logger_setup import get_logger
from .trainer import run_experiment
from .utils import get_splits, load_dataset_safe, process_split

app = typer.Typer(help="CLI for dataset processing")

# Required columns in task Excel file
REQUIRED_TASK_COLUMNS = {
    "dataset_name",
    "subset",
    "train_split",
    "val_split",
    "test_split",
    "text_col",
    "label_col",
}


def _validate_task_excel(task_excel: Path) -> None:
    """
    Validate that task Excel file exists and has proper format.

    Args:
        task_excel: Path to task Excel file

    Raises:
        typer.Exit: If validation fails
    """
    if not task_excel.exists() or not task_excel.is_file():
        typer.secho(f"Error: task Excel file not found: {task_excel}", fg="red")
        raise typer.Exit(code=2)

    if task_excel.suffix.lower() not in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
        typer.secho(
            f"Warning: task file does not have a typical Excel extension: {task_excel.suffix}",
            fg="yellow",
        )


def _validate_task_columns(df: pd.DataFrame) -> None:
    """
    Validate that task DataFrame has all required columns.

    Args:
        df: Task DataFrame to validate

    Raises:
        typer.Exit: If required columns are missing
    """
    missing = REQUIRED_TASK_COLUMNS.difference(set(df.columns))
    if missing:
        typer.secho(f"Error: task Excel is missing required columns: {', '.join(sorted(missing))}", fg="red")
        raise typer.Exit(code=2)


def _create_output_directory(dataset_name: str, subset_name: str, output_dir: Path, logger) -> Path:
    """
    Create and return output directory for a dataset.

    Args:
        dataset_name: Name of the dataset
        subset_name: Subset name (or 'default')
        output_dir: Base output directory
        logger: Logger instance

    Returns:
        Path to created dataset directory, or None on error
    """
    safe_dataset_name = dataset_name.replace("/", "_")
    safe_subset_name = str(subset_name).replace("/", "_")

    if safe_subset_name != "default":
        dataset_dir = output_dir / f"{safe_dataset_name}@{safe_subset_name}"
    else:
        dataset_dir = output_dir / f"{safe_dataset_name}"

    try:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir
    except Exception as e:
        logger.error(f"IO ERROR: Could not create dataset dir {dataset_dir}: {e}")
        return None


def _save_dataset_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    dataset_dir: Path,
    logger,
) -> bool:
    """
    Save train/val/test DataFrames as JSONL files.

    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame
        df_test: Test DataFrame
        dataset_dir: Directory to save files to
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    try:
        df_train.to_json(dataset_dir / "train.jsonl", orient="records", lines=True)
        df_val.to_json(dataset_dir / "val.jsonl", orient="records", lines=True)
        df_test.to_json(dataset_dir / "test.jsonl", orient="records", lines=True)
        return True
    except Exception as e:
        logger.error(f"IO ERROR: Failed to write jsonl for {dataset_dir}: {e}")
        return False


def _process_all_splits(ds_train, ds_val, ds_test, text_col: str, label_col: str, seed: int, max_rows: int):
    """
    Process train/val/test splits with common parameters.

    Args:
        ds_train: Training dataset split
        ds_val: Validation dataset split
        ds_test: Test dataset split
        text_col: Text column name(s)
        label_col: Label column name(s)
        seed: Random seed for sampling
        max_rows: Maximum rows per split

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df_train = process_split(ds_train, text_col, label_col, seed, max_rows=max_rows)
    df_val = process_split(ds_val, text_col, label_col, seed, max_rows=max_rows)
    df_test = process_split(ds_test, text_col, label_col, seed, max_rows=max_rows)
    return df_train, df_val, df_test


def _process_single_task(row: pd.Series, cfg_download, logger) -> None:
    """
    Process a single dataset task: load, split, and save.

    Args:
        row: Row from task Excel containing dataset configuration
        cfg_download: Download configuration object
        logger: Logger instance
    """
    ds, subset_name = load_dataset_safe(row, revision=cfg_download.revision)
    if ds is None:
        return

    ds_train, ds_val, ds_test = get_splits(ds, row, cfg_download.seed)

    df_train, df_val, df_test = _process_all_splits(
        ds_train,
        ds_val,
        ds_test,
        row["text_col"],
        row["label_col"],
        cfg_download.seed,
        cfg_download.max_rows,
    )

    dataset_name = row["dataset_name"]
    dataset_dir = _create_output_directory(dataset_name, subset_name, cfg_download.output_dir, logger)
    if dataset_dir is None:
        return

    if _save_dataset_splits(df_train, df_val, df_test, dataset_dir, logger):
        logger.info(
            f"Processed {dataset_name} - {subset_name} | "
            f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
        )


def _load_datasets(data_dir_root: Path, logger) -> list[Path]:
    """
    Load and sort dataset directories.

    Args:
        data_dir_root: Root directory containing dataset subdirectories
        logger: Logger instance

    Returns:
        Sorted list of dataset directories

    Raises:
        typer.Exit: If loading fails
    """
    try:
        data_dirs = sorted(p for p in data_dir_root.iterdir() if p.is_dir())
        if not data_dirs:
            logger.error(f"No dataset directories found in {data_dir_root}")
            raise typer.Exit(code=2)
        return data_dirs
    except Exception as e:
        logger.error(f"Failed to list datasets in {data_dir_root}: {e}")
        raise typer.Exit(code=2)


def _load_models(model_list_excel: Path, model_column: str, logger) -> list[str]:
    """
    Load model IDs from Excel file.

    Args:
        model_list_excel: Path to Excel file containing model IDs
        model_column: Column name containing model IDs
        logger: Logger instance

    Returns:
        List of model ID strings

    Raises:
        typer.Exit: If loading fails
    """
    try:
        df_models = pd.read_excel(model_list_excel)
        if model_column not in df_models.columns:
            available = df_models.columns.tolist()
            logger.error(
                f"Column '{model_column}' not found in {model_list_excel}. " f"Available columns: {available}"
            )
            raise typer.Exit(code=2)
        model_ids = df_models[model_column].dropna().tolist()
        if not model_ids:
            logger.error(f"No models found in column '{model_column}' of {model_list_excel}")
            raise typer.Exit(code=2)
        return model_ids
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Failed to load models from {model_list_excel}: {e}")
        raise typer.Exit(code=2)


def _order_jobs(jobs: list[tuple], order: str) -> list[tuple]:
    """Return jobs ordered ascending (default) or descending."""
    if order == "desc":
        return list(reversed(jobs))
    return jobs


def _slice_jobs(jobs: list[tuple], portion_index: int, portion_total: int) -> list[tuple]:
    """Return only the jobs in the requested portion (1-indexed)."""
    if portion_total <= 1:
        return jobs

    chunk = (len(jobs) + portion_total - 1) // portion_total
    start = (portion_index - 1) * chunk
    end = start + chunk
    return jobs[start:end]


@app.command()
def download_datasets(
    config: Path = typer.Option(Path("config.toml"), "--config", help="Path to TOML/JSON config file"),
):
    """Download and process datasets from task Excel file."""
    cfg = load_config(config)
    dl = cfg.download
    logger = get_logger()

    # Log the loaded config for this command
    try:
        logger.info("Loaded config for download_datasets: %s", cfg.model_dump())
    except Exception:
        logger.info("Loaded config for download_datasets")

    if dl.task_excel is None:
        typer.secho("Error: `task_excel` must be set in the [download] section of the config file.", fg="red")
        raise typer.Exit(code=2)

    _validate_task_excel(dl.task_excel)
    df = pd.read_excel(dl.task_excel)
    _validate_task_columns(df)

    for index, row in df.iterrows():
        _process_single_task(row, dl, logger)


def _finetune_distributed(
    data_dirs: list[Path],
    model_ids: list[str],
    machine_id: int,
    num_machines: int,
    total_jobs: int,
    ft: FinetuneConfig,
    dry_run: bool,
    order: str,
    portion_index: int,
    portion_total: int,
) -> None:
    """Run assigned jobs on this machine (distributed mode).

    Uses modulo slicing to distribute jobs: a job at linear_index is assigned
    to machine M if (linear_index % num_machines == M).

    Args:
        data_dirs: List of dataset directories
        model_ids: List of model IDs
        machine_id: This machine's ID (0..num_machines-1)
        num_machines: Total number of machines
        total_jobs: Total job count (for logging)
        ft: Finetune config
        dry_run: If True, preview jobs without running
    """
    logger = get_logger()
    num_models = len(model_ids)

    # Validate machine_id
    if machine_id < 0 or machine_id >= num_machines:
        typer.secho(f"Error: --machine-id must be in range [0, {num_machines-1}]", fg="red")
        raise typer.Exit(code=2)

    # Assign jobs via modulo slicing
    assigned = []
    for di, data_dir in enumerate(data_dirs):
        for mi, model_id in enumerate(model_ids):
            linear_index = di * num_models + mi
            if linear_index % num_machines == machine_id:
                assigned.append((linear_index, data_dir, model_id))

    logger.info(f"Distributed mode: Machine {machine_id}/{num_machines} assigned {len(assigned)}/{total_jobs} jobs")

    assigned = _order_jobs(assigned, order)
    assigned = _slice_jobs(assigned, portion_index, portion_total)

    logger.info("Portioning: index %d of %d -> %d jobs (post-ordering)", portion_index, portion_total, len(assigned))

    if dry_run:
        typer.secho(f"\n[DRY-RUN] Machine {machine_id} would run {len(assigned)} jobs:", fg="blue")
        for idx, data_dir, model_id in assigned:
            typer.echo(f"  [{idx}] {data_dir.name:50s} + {model_id}")
        return

    # Run assigned jobs
    for idx, data_dir, model_id in assigned:
        try:
            logger.info(f"RUN [{idx}]: {data_dir.name} / {model_id}")
            run_experiment(data_dir, model_id, ft.output_root, ft.seed, ft.batch_size)
        except Exception:
            logger.exception(f"FAILED [{idx}]: {data_dir} / {model_id}")


def _finetune_single(
    data_dirs: list[Path],
    model_ids: list[str],
    total_jobs: int,
    ft: FinetuneConfig,
    dry_run: bool,
    order: str,
    portion_index: int,
    portion_total: int,
) -> None:
    """Run all jobs on this machine (single-machine mode).

    Args:
        data_dirs: List of dataset directories
        model_ids: List of model IDs
        total_jobs: Total job count (for logging)
        ft: Finetune config
        dry_run: If True, preview jobs without running
    """
    logger = get_logger()
    num_models = len(model_ids)

    logger.info("Single-machine mode: running all jobs")

    jobs: list[tuple[int, Path, str]] = []
    for di, data_dir in enumerate(data_dirs):
        for mi, model_id in enumerate(model_ids):
            idx = di * num_models + mi
            jobs.append((idx, data_dir, model_id))

    jobs = _order_jobs(jobs, order)
    jobs = _slice_jobs(jobs, portion_index, portion_total)

    logger.info("Portioning: index %d of %d -> %d jobs (post-ordering)", portion_index, portion_total, len(jobs))

    if dry_run:
        typer.secho(f"\n[DRY-RUN] Would run {total_jobs} jobs:", fg="blue")
        for idx, data_dir, model_id in jobs:
            typer.echo(f"  [{idx}] {data_dir.name:50s} + {model_id}")
        return

    # Run all jobs
    for idx, data_dir, model_id in jobs:
        try:
            logger.info(f"RUN [{idx}]: {data_dir.name} / {model_id}")
            run_experiment(data_dir, model_id, ft.output_root, ft.seed, ft.batch_size)
        except Exception:
            logger.exception(f"FAILED: {data_dir} / {model_id}")


@app.command()
def finetune_all(
    config: Path = typer.Option(Path("config.toml"), "--config", help="Path to TOML/JSON config file"),
    machine_id: int | None = typer.Option(
        None, "--machine-id", help="Machine ID for distributed mode (0..num-machines-1)"
    ),
    num_machines: int | None = typer.Option(
        None, "--num-machines", help="Total number of machines in distributed mode"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview assigned jobs without running"),
    order: str = typer.Option("asc", "--order", help="Job order: asc (0..N) or desc (N..0)"),
    portion_index: int = typer.Option(1, "--portion-index", help="Portion number to run (1-indexed)"),
    portion_total: int = typer.Option(1, "--portion-total", help="Total portions to split assigned jobs"),
):
    """Fine-tune all (or assigned) models on all (or assigned) datasets.

    Can run in two modes:

    1. Single-machine (default): Runs all model-dataset pairs locally
       python main.py finetune-all --config config.toml

    2. Distributed: Uses modulo slicing to assign jobs across machines
       python main.py finetune-all --config config.toml --machine-id 0 --num-machines 5
       python main.py finetune-all --config config.toml --machine-id 0 --num-machines 5 --dry-run
    """
    cfg = load_config(config)
    ft = cfg.finetune
    logger = get_logger()

    logger.info("Loaded config for finetune_all")

    # Validate required config
    if ft.data_dir_root is None:
        typer.secho("Error: `data_dir_root` must be set in the [finetune] section of the config file.", fg="red")
        raise typer.Exit(code=2)
    if ft.model_list_excel is None:
        typer.secho("Error: `model_list_excel` must be set in the [finetune] section of the config file.", fg="red")
        raise typer.Exit(code=2)

    # Load datasets and models
    data_dirs = _load_datasets(ft.data_dir_root, logger)
    model_ids = _load_models(ft.model_list_excel, ft.model_list_column, logger)

    num_models = len(model_ids)
    total_jobs = len(data_dirs) * num_models
    logger.info(f"Inventory: {len(data_dirs)} datasets × {num_models} models = {total_jobs} total jobs")

    order = order.lower()
    if order not in {"asc", "desc"}:
        typer.secho("Error: --order must be 'asc' or 'desc'", fg="red")
        raise typer.Exit(code=2)

    if portion_total < 1:
        typer.secho("Error: --portion-total must be >= 1", fg="red")
        raise typer.Exit(code=2)
    if portion_index < 1 or portion_index > portion_total:
        typer.secho("Error: --portion-index must be between 1 and --portion-total", fg="red")
        raise typer.Exit(code=2)

    # Mode dispatch: distributed or single-machine
    if machine_id is not None and num_machines is not None:
        _finetune_distributed(
            data_dirs,
            model_ids,
            machine_id,
            num_machines,
            total_jobs,
            ft,
            dry_run,
            order,
            portion_index,
            portion_total,
        )
    elif machine_id is None and num_machines is None:
        _finetune_single(data_dirs, model_ids, total_jobs, ft, dry_run, order, portion_index, portion_total)
    else:
        typer.secho(
            "Error: Both --machine-id and --num-machines must be provided together for distributed mode.",
            fg="red",
        )
        raise typer.Exit(code=2)
