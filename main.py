"""Minimal CLI entrypoint with logger initialization.

Loads config from config.toml (if present) and initializes logging before running CLI.
"""

import os
from pathlib import Path

from ntcir19_pretrained_model_retrieval import cli
from ntcir19_pretrained_model_retrieval.config import Config, load_config
from ntcir19_pretrained_model_retrieval.logger_setup import get_logger, setup_logger


def _initialize_logger() -> None:
    """Load config and set up logger with file and console handlers."""
    default_config_path = Path("config.toml")

    logger_path = None
    if default_config_path.exists():
        try:
            default_cfg = load_config(default_config_path)
            logger_path = default_cfg.download.log_file or default_cfg.finetune.log_file
        except Exception as e:
            get_logger().warning(f"Failed to load config: {e}. Using defaults.")

    # Use explicit fallback if no path configured
    if not logger_path:
        logger_path = Path("process_status.log")

    setup_logger(logger_path)


def _enable_cuda_determinism() -> None:
    """Force deterministic CUDA behavior when available."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("PYTORCH_DETERMINISTIC", "1")

    try:
        import torch

        if torch.cuda.is_available():
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            get_logger().info("CUDA determinism enabled (deterministic kernels, TF32 off).")
    except Exception as e:
        get_logger().warning(f"Could not enforce CUDA determinism: {e}")


def main() -> None:
    """Initialize and run the CLI application."""
    _initialize_logger()
    _enable_cuda_determinism()
    cli.app()


if __name__ == "__main__":
    main()
"""Minimal entrypoint: initialize logger (from config) and run CLI app.

This file intentionally keeps very little logic. The original large `main.py`
was split across modules for clarity.
"""

from pathlib import Path

# Determine log file from default config (if present) and initialize logger
default_config_path = Path("config.toml")
default_cfg = Config()
try:
    if default_config_path.exists():
        default_cfg = load_config(default_config_path)
except Exception:
    default_cfg = Config()

chosen_log = default_cfg.download.log_file or default_cfg.finetune.log_file or "process_status.log"
setup_logger(chosen_log)


def main():
    _enable_cuda_determinism()
    cli.app()


if __name__ == "__main__":
    main()
