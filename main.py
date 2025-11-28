"""Minimal entrypoint: initialize logger (from config) and run CLI app.

This file intentionally keeps very little logic. The original large `main.py`
was split across modules for clarity.
"""

from pathlib import Path

from ntcir19_pretrained_model_retrieval.config import Config, load_config
from ntcir19_pretrained_model_retrieval.logger_setup import setup_logger, get_logger

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
logger = get_logger()
try:
    # Log loaded default config for diagnostics
    logger.info("Loaded default config: %s", default_cfg.model_dump())
except Exception:
    logger.info("Loaded default config")

from ntcir19_pretrained_model_retrieval import cli


def main():
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
    cli.app()


if __name__ == "__main__":
    main()
