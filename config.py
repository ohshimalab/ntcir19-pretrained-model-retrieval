import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DownloadConfig:
    task_excel: Path | None = None
    seed: int = 0
    output_dir: Path = Path("bert-data")
    max_rows: int = 5000
    revision: str = "refs/convert/parquet"
    log_file: str | None = None


@dataclass
class FinetuneConfig:
    data_dir_root: Path | None = None
    model_list_excel: Path | None = None
    model_list_column: str = "model_name"
    seed: int = 0
    output_root: Path = Path("./experiment_results")
    log_file: str | None = None
    batch_size: int = 32


@dataclass
class Config:
    download: DownloadConfig = field(default_factory=DownloadConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)


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

    download_section = {}
    finetune_section = {}

    if isinstance(data, dict):
        if "download" in data:
            download_section = data["download"] or {}
        elif "download_datasets" in data:
            download_section = data["download_datasets"] or {}
        else:
            for k in ("task_excel", "output_dir", "max_rows", "default_revision", "revision", "seed"):
                if k in data:
                    download_section[k] = data[k]

        if "finetune" in data:
            finetune_section = data["finetune"] or {}
        elif "finetune_all" in data:
            finetune_section = data["finetune_all"] or {}
        else:
            for k in ("data_dir_root", "model_list_excel", "model_list_column"):
                if k in data:
                    finetune_section[k] = data[k]

    # Populate download config
    if "task_excel" in download_section:
        cfg.download.task_excel = Path(download_section["task_excel"]) if download_section["task_excel"] else None
    if "seed" in download_section:
        cfg.download.seed = int(download_section["seed"])
    if "output_dir" in download_section:
        cfg.download.output_dir = Path(download_section["output_dir"])
    if "max_rows" in download_section:
        cfg.download.max_rows = int(download_section["max_rows"])
    if "revision" in download_section:
        cfg.download.revision = download_section["revision"]
    if "default_revision" in download_section:
        cfg.download.revision = download_section["default_revision"]

    # Populate finetune config
    if "data_dir_root" in finetune_section:
        cfg.finetune.data_dir_root = (
            Path(finetune_section["data_dir_root"]) if finetune_section["data_dir_root"] else None
        )
    if "model_list_excel" in finetune_section:
        cfg.finetune.model_list_excel = (
            Path(finetune_section["model_list_excel"]) if finetune_section["model_list_excel"] else None
        )
    if "model_list_column" in finetune_section:
        cfg.finetune.model_list_column = finetune_section["model_list_column"]

    # Globals mapping for backward compatibility
    globals_section = {}
    if isinstance(data, dict):
        if "globals" in data:
            globals_section = data["globals"] or {}
        else:
            for k in ("seed", "output_root", "log_file", "data_root_dir", "batch_size"):
                if k in data:
                    globals_section[k] = data[k]

    if "seed" in globals_section:
        gseed = int(globals_section["seed"])
        if "seed" not in download_section:
            cfg.download.seed = gseed
        if "seed" not in finetune_section:
            cfg.finetune.seed = gseed

    if "output_root" in globals_section:
        gout = globals_section["output_root"]
        if "output_root" not in finetune_section:
            cfg.finetune.output_root = Path(gout)

    if "log_file" in globals_section:
        glog = globals_section["log_file"]
        if "log_file" not in download_section:
            cfg.download.log_file = glog
        if "log_file" not in finetune_section:
            cfg.finetune.log_file = glog

    if "data_root_dir" in globals_section:
        gdata = globals_section["data_root_dir"]
        if "output_dir" not in download_section and "data_root_dir" not in download_section:
            cfg.download.output_dir = Path(gdata)

    if "batch_size" in globals_section:
        gb = int(globals_section["batch_size"])
        if "batch_size" not in finetune_section:
            cfg.finetune.batch_size = gb

    return cfg
