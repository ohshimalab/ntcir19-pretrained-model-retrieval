from pathlib import Path

import pandas as pd
import typer

from .config import load_config
from .logger_setup import get_logger
from .trainer import run_experiment
from .utils import get_splits, load_dataset_safe, process_split

app = typer.Typer(help="CLI for dataset processing")


@app.command()
def download_datasets(
    config: Path = typer.Option(Path("config.toml"), "--config", help="Path to TOML/JSON config file"),
):
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

    task_excel = dl.task_excel
    if not task_excel.exists() or not task_excel.is_file():
        typer.secho(f"Error: task Excel file not found: {task_excel}", fg="red")
        raise typer.Exit(code=2)

    if task_excel.suffix.lower() not in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
        typer.secho(
            f"Warning: task file does not have a typical Excel extension: {task_excel.suffix}",
            fg="yellow",
        )

    df = pd.read_excel(task_excel)

    required_columns = {
        "dataset_name",
        "subset",
        "train_split",
        "val_split",
        "test_split",
        "text_col",
        "label_col",
    }
    missing = required_columns.difference(set(df.columns))
    if missing:
        typer.secho(f"Error: task Excel is missing required columns: {', '.join(sorted(missing))}", fg="red")
        raise typer.Exit(code=2)

    for index, row in df.iterrows():
        ds, subset_name = load_dataset_safe(row, revision=dl.revision)
        if ds is None:
            continue

        ds_train, ds_val, ds_test = get_splits(ds, row, dl.seed)

        text_col = row["text_col"]
        label_col = row["label_col"]

        df_train = process_split(ds_train, text_col, label_col, dl.seed, max_rows=dl.max_rows)
        df_val = process_split(ds_val, text_col, label_col, dl.seed, max_rows=dl.max_rows)
        df_test = process_split(ds_test, text_col, label_col, dl.seed, max_rows=dl.max_rows)

        df_train["split"] = "train"
        df_val["split"] = "val"
        df_test["split"] = "test"

        dataset_name = row["dataset_name"]
        safe_dataset_name = dataset_name.replace("/", "_")
        safe_subset_name = str(subset_name).replace("/", "_")

        if safe_subset_name != "default":
            dataset_dir = dl.output_dir / f"{safe_dataset_name}@{safe_subset_name}"
        else:
            dataset_dir = dl.output_dir / f"{safe_dataset_name}"

        try:
            dataset_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger = get_logger()
            logger.error(f"IO ERROR: Could not create dataset dir {dataset_dir}: {e}")
            continue

        try:
            df_train.to_json(dataset_dir / "train.jsonl", orient="records", lines=True)
            df_val.to_json(dataset_dir / "val.jsonl", orient="records", lines=True)
            df_test.to_json(dataset_dir / "test.jsonl", orient="records", lines=True)
        except Exception as e:
            logger = get_logger()
            logger.error(f"IO ERROR: Failed to write jsonl for {dataset_dir}: {e}")
            continue

        logger.info(
            f"Processed {dataset_name} - {subset_name} | "
            f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
        )


@app.command()
def finetune_all(
    config: Path = typer.Option(Path("config.toml"), "--config", help="Path to TOML/JSON config file"),
):
    cfg = load_config(config)
    ft = cfg.finetune
    logger = get_logger()

    # Log the loaded config for this command
    try:
        logger.info("Loaded config for finetune_all: %s", cfg.model_dump())
    except Exception:
        logger.info("Loaded config for finetune_all")

    if ft.data_dir_root is None:
        typer.secho("Error: `data_dir_root` must be set in the [finetune] section of the config file.", fg="red")
        raise typer.Exit(code=2)
    if ft.model_list_excel is None:
        typer.secho("Error: `model_list_excel` must be set in the [finetune] section of the config file.", fg="red")
        raise typer.Exit(code=2)

    data_dir_root = ft.data_dir_root
    model_list_excel = ft.model_list_excel

    data_dirs = [p for p in data_dir_root.iterdir() if p.is_dir()]
    data_dirs = list(sorted(data_dirs))
    df_models = pd.read_excel(model_list_excel)
    model_ids = df_models.get(ft.model_list_column, []).tolist()
    for data_dir in data_dirs:
        for model_id in model_ids:
            try:
                run_experiment(data_dir, model_id, ft.output_root, ft.seed, ft.batch_size)
            except Exception:
                logger.error(f"CRITICAL FAILURE: {data_dir} / {model_id}", exc_info=True)
