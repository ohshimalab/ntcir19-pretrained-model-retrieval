import json
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseConfig(BaseModel):
    """Base configuration with common fields."""

    seed: int = Field(default=0, description="Random seed for reproducibility")
    log_file: Path | None = Field(default=None, description="Path to log file")


class DownloadConfig(BaseConfig):
    """Configuration for dataset download and preprocessing."""

    task_excel: Path | None = Field(default=None, description="Path to Excel file with dataset task definitions")
    output_dir: Path = Field(default=Path("bert-data"), description="Directory to save processed datasets")
    max_rows: int = Field(default=5000, description="Maximum rows to keep per split")
    revision: str = Field(default="refs/convert/parquet", description="Dataset revision to load from HuggingFace")

    @field_validator("max_rows")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_rows must be positive")
        return v


class FinetuneConfig(BaseConfig):
    """Configuration for model fine-tuning experiments."""

    model_config = ConfigDict(protected_namespaces=())

    data_dir_root: Path | None = Field(default=None, description="Root directory containing dataset subdirectories")
    model_list_excel: Path | None = Field(default=None, description="Path to Excel file listing models to fine-tune")
    model_list_column: str = Field(default="model_name", description="Column name in model list Excel")
    model_revision: str | None = Field(default=None, description="Model checkpoint revision to pin")
    tokenizer_revision: str | None = Field(default=None, description="Tokenizer revision to pin")
    output_root: Path = Field(default=Path("experiment_results"), description="Root directory for experiment outputs")
    batch_size: int = Field(default=32, description="Training and evaluation batch size")

    @field_validator("batch_size")
    @classmethod
    def validate_positive_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v


class Config(BaseModel):
    """Top-level configuration containing download and finetune sections."""

    download: DownloadConfig = Field(default_factory=DownloadConfig)
    finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)


def _apply_globals(section: dict, globals_section: dict, keys: list[str]) -> dict:
    """
    Apply global config values to section if not already set.

    Args:
        section: Configuration section dict
        globals_section: Global configuration dict
        keys: Keys to apply from globals

    Returns:
        New dict with globals applied (does not mutate input)
    """
    result = section.copy()
    for key in keys:
        if key in globals_section and key not in result:
            result[key] = globals_section[key]
    return result


def load_config(path: Path) -> Config:
    """
    Load configuration from TOML or JSON file.

    Supports optional [globals] section for shared settings across download/finetune sections.

    Args:
        path: Path to config file (.toml or .json)

    Returns:
        Validated Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is not .toml or .json
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Load raw data
    if path.suffix.lower() == ".toml":
        with path.open("rb") as f:
            data = tomllib.load(f)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported config format: use .toml or .json")

    # Extract sections with defaults
    download_section = data.get("download", {})
    finetune_section = data.get("finetune", {})
    globals_section = data.get("globals", {})

    # Apply globals (pure - returns new dicts)
    download_section = _apply_globals(download_section, globals_section, ["seed", "log_file"])
    finetune_section = _apply_globals(
        finetune_section, globals_section, ["seed", "log_file", "output_root", "batch_size"]
    )

    return Config.model_validate({"download": download_section, "finetune": finetune_section})
