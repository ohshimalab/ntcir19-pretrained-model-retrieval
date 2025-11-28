import json
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DownloadConfig(BaseModel):
    task_excel: Path | None = Field(default=None)
    seed: int = Field(default=0)
    output_dir: Path = Field(default=Path("bert-data"))
    max_rows: int = Field(default=5000)
    revision: str = Field(default="refs/convert/parquet")
    log_file: str | None = Field(default=None)


class FinetuneConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    data_dir_root: Path | None = Field(default=None)
    model_list_excel: Path | None = Field(default=None)
    model_list_column: str = Field(default="model_name")
    seed: int = Field(default=0)
    output_root: Path = Field(default=Path("./experiment_results"))
    log_file: str | None = Field(default=None)
    batch_size: int = Field(default=32)


class Config(BaseModel):
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)


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

    # Globals mapping for backward compatibility
    globals_section = {}
    if isinstance(data, dict):
        if "globals" in data:
            globals_section = data["globals"] or {}
        else:
            for k in ("seed", "output_root", "log_file", "data_root_dir", "batch_size"):
                if k in data:
                    globals_section[k] = data[k]

    # Apply globals to sections if not already set
    if "seed" in globals_section and "seed" not in download_section:
        download_section["seed"] = globals_section["seed"]
    if "seed" in globals_section and "seed" not in finetune_section:
        finetune_section["seed"] = globals_section["seed"]

    if "output_root" in globals_section and "output_root" not in finetune_section:
        finetune_section["output_root"] = globals_section["output_root"]

    if "log_file" in globals_section and "log_file" not in download_section:
        download_section["log_file"] = globals_section["log_file"]
    if "log_file" in globals_section and "log_file" not in finetune_section:
        finetune_section["log_file"] = globals_section["log_file"]

    if "data_root_dir" in globals_section and "output_dir" not in download_section:
        download_section["output_dir"] = globals_section["data_root_dir"]

    if "batch_size" in globals_section and "batch_size" not in finetune_section:
        finetune_section["batch_size"] = globals_section["batch_size"]

    # Build the config dict
    config_dict = {
        "download": download_section,
        "finetune": finetune_section,
    }

    return Config.model_validate(config_dict)
