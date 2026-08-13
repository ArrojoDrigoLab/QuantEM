"""
Centralized configuration module for directory paths.

Everything QuantEM writes -- the database, downloaded model weights, image
renditions, overlays, caches and logs -- lives under a single data directory.
That directory is chosen by :mod:`quantem.cli` and published to the process as
``QUANTEM_DATA_DIR`` before Django is configured.

By owner ruling (2026-08-09) the default location lives **with the
installation**: ``<install>\\data`` beside the frozen executable, or
``<sys.prefix>/quantem-data`` for a pip install (the environment is the
install). ``QUANTEM_DATA_DIR`` is the explicit override, and an unwritable
location is a hard error naming that override -- never a silent fallback to a
per-user directory. See :func:`quantem.cli.default_data_dir`.
"""

import logging
import os
from pathlib import Path

from quantem.cli import default_data_dir

#: Environment variable naming the user data directory. ``quantem.cli`` sets it
#: in ``_prepare_env()`` before ``django.setup()`` runs; the fallback below only
#: applies when Django is started some other way (tests, ``django-admin``).
DATA_DIR_ENV_VAR = "QUANTEM_DATA_DIR"


def _resolve_storage_dir() -> Path:
    raw = os.environ.get(DATA_DIR_ENV_VAR, "").strip()
    if not raw:
        return default_data_dir().expanduser().resolve(strict=False)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise RuntimeError(
            f"{DATA_DIR_ENV_VAR} must be an absolute path (got {raw!r}). "
            "Storage must live in the user data directory, never relative to "
            "the installation or the current working directory."
        )
    return candidate.resolve(strict=False)


# Root of everything this application writes.
STORAGE_DIR = _resolve_storage_dir()

#: The one SQLite file. See :mod:`quantem.core.db_config`.
DB_PATH = STORAGE_DIR / "quantem.sqlite3"

#: Per-install Django secret key, generated on first run. Never committed.
SECRET_KEY_PATH = STORAGE_DIR / "secret_key"

# Directory paths - all relative to STORAGE_DIR
LOGS_DIR = STORAGE_DIR / "logs"

#: The server's rotating log file. Written when the launcher enables file
#: logging (``quantem serve`` and the frozen build set ``QUANTEM_LOG_TO_FILE=1``;
#: see the LOGGING block in :mod:`quantem.core.settings`). Named here so the
#: CLI can announce the same path the settings module writes to.
SERVER_LOG_PATH = LOGS_DIR / "quantem-server.log"


def file_logging_enabled() -> bool:
    """Whether this process will actually write :data:`SERVER_LOG_PATH`.

    One process, one writer: a spawned job worker inherits the server's whole
    environment, flag included, but rotation renames the file and on Windows
    that rename fails while another process holds it open. Shared with the CLI
    so that ``quantem serve`` cannot promise a log file it will never write.
    """
    flag = os.environ.get("QUANTEM_LOG_TO_FILE")
    if flag is None or flag.strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    return os.environ.get("QUANTEM_JOB_WORKER") != "1"


CACHE_DIR = STORAGE_DIR / "cache"
MODELS_DIR = STORAGE_DIR / "models"
DATA_DIR = STORAGE_DIR / "data"

# DATA_DIR specific paths
TMP_DIR = DATA_DIR / "tmp"
NGFF_TMP_DIR = TMP_DIR / "ngff"
SEGMENTATION_OVERLAYS_TMP_DIR = TMP_DIR / "segmentation_overlays"
UPLOADS_DIR = TMP_DIR / "uploads"  # Temporary upload storage
IMAGES_DIR = DATA_DIR / "images"  # Final PNG storage
ROIS_DIR = TMP_DIR / "rois"  # Temporary ROI PNG storage
PROB_MAPS_DIR = DATA_DIR / "prob_maps"
GLOBAL_MASKS_DIR = DATA_DIR / "global_masks"


def ensure_directories():
    """
    Create all required directories if they don't exist.

    This should be called at Django startup to ensure directories are available.
    """
    logger = logging.getLogger(__name__)
    directories = [
        STORAGE_DIR,
        LOGS_DIR,
        CACHE_DIR,
        MODELS_DIR,
        DATA_DIR,
        TMP_DIR,
        NGFF_TMP_DIR,
        SEGMENTATION_OVERLAYS_TMP_DIR,
        UPLOADS_DIR,
        IMAGES_DIR,
        ROIS_DIR,
        PROB_MAPS_DIR,
        GLOBAL_MASKS_DIR,
    ]

    created_dirs = []
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if directory.exists():
                created_dirs.append(str(directory))
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}", exc_info=True)
            # Never fall back to another location (owner ruling): storage that
            # silently relocates is storage nobody can find. Name the path and
            # the override instead.
            raise RuntimeError(
                f"QuantEM cannot create its storage directory {directory} "
                f"({e}). Storage never falls back to another location; set "
                f"{DATA_DIR_ENV_VAR} (or pass --data-dir) to a writable "
                "directory."
            ) from e

    if created_dirs:
        logger.info(f"Ensured directories exist: {len(created_dirs)} directories checked")

    return directories
