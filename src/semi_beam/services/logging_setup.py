# path: src/semi_beam/services/logging_setup.py
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union


def get_default_log_dir() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "Lambert" / "Calculeitor" / "logs"
    return Path.home() / ".calculeitor" / "logs"


def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    log_name: str = "app.log",
) -> logging.Logger:
    if log_dir is None:
        log_dir = get_default_log_dir()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    logger = logging.getLogger("semi_beam")
    logger.setLevel(logging.INFO)

    # Evitar duplicar handlers si se llama más de una vez
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    fh = RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Logging inicializado. Archivo: %s", log_path)
    return logger
