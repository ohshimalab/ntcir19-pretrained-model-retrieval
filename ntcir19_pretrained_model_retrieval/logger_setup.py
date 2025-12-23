import logging
from pathlib import Path

_logger = None


def setup_logger(log_file: Path | str | None) -> logging.Logger:
    """
    Create and configure logger with file and console handlers.

    Args:
        log_file: Optional path to log file. If provided but inaccessible,
                  logs to console only and warns about the failure.

    Returns:
        Configured logger instance
    """
    global _logger
    logger = logging.getLogger("ExperimentRunner")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Add file handler if log_file provided
    if log_file:
        try:
            file_path = Path(log_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(file_path)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            logger.warning(f"Could not create file handler for {log_file}: {e}")

    # Always add console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """
    Get the configured logger instance.

    Returns the global logger if setup_logger() has been called,
    otherwise returns a default logger instance.

    Returns:
        Logger instance configured by setup_logger() or default ExperimentRunner logger
    """
    global _logger
    return _logger or logging.getLogger("ExperimentRunner")
