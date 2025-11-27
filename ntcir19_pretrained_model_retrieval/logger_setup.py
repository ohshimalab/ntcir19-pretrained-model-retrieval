import logging

_logger = None


def setup_logger(log_file: str | None):
    """Create and configure the ExperimentRunner logger and store it.

    Returns the configured logger.
    """
    global _logger
    logger = logging.getLogger("ExperimentRunner")
    logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicates when re-running
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler (best-effort)
    if log_file:
        try:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception:
            # Continue with console-only
            pass

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    _logger = logger
    return logger


def get_logger():
    global _logger
    if _logger is not None:
        return _logger
    return logging.getLogger("ExperimentRunner")
