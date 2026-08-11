"""
Django settings for the QuantEM local server.

QuantEM is a single-user offline desktop application. The server binds to
loopback only, is started by :mod:`quantem.cli` with a per-launch token, and has
no user accounts, no sessions and no admin site. The settings below are written
for that deployment and no other -- there is no "production" variant.

For the full list of settings and their values, see
https://docs.djangoproject.com/en/6.0/ref/settings/
"""

import contextlib
import os
import secrets
from pathlib import Path

from quantem.core.env_files import load_backend_env_files

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env files, for every setting read below.
#
# This is NOT where QUANTEM_DATA_DIR gets its chance: importing this module
# imports the `quantem.core` package first, and that package resolves the data
# directory in its own body. The .env load that QUANTEM_DATA_DIR depends on is
# therefore in `quantem/core/__init__.py`, which runs earlier. Both calls are
# safe -- `read_env` never overwrites a variable that is already set.
load_backend_env_files(BASE_DIR)

# Deliberately after load_backend_env_files(): importing this module resolves
# STORAGE_DIR, and moving it above would fix the data directory before a .env
# could name it.
from quantem.core.config import (  # noqa: E402
    SECRET_KEY_PATH,
    SERVER_LOG_PATH,
    TMP_DIR,
    ensure_directories,
)

# The data directory has to exist before the secret key can be persisted in it.
ensure_directories()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_or_create_secret_key() -> str:
    """Return this install's secret key, generating it on first run.

    The key is generated per install and persisted in the user data directory.
    It is never committed and never shipped: a value baked into the source would
    be identical on every machine that installed QuantEM.
    """
    from_env = os.environ.get("DJANGO_SECRET_KEY", "").strip()
    if from_env:
        return from_env

    try:
        existing = SECRET_KEY_PATH.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass

    candidate = secrets.token_urlsafe(64)
    try:
        # Exclusive create so two processes racing on first launch cannot end up
        # with different keys.
        with open(SECRET_KEY_PATH, "x", encoding="utf-8") as handle:
            handle.write(candidate)
    except FileExistsError:
        return SECRET_KEY_PATH.read_text(encoding="utf-8").strip()
    # Best-effort: a filesystem that cannot express the mode (a Windows share,
    # some FUSE mounts) is not a reason to refuse to start.
    with contextlib.suppress(OSError):
        os.chmod(SECRET_KEY_PATH, 0o600)
    return candidate


SECRET_KEY = _load_or_create_secret_key()

# Off unless explicitly asked for. quantem.cli sets DJANGO_DEBUG=0.
DEBUG = _env_flag("DJANGO_DEBUG", False)

# The server binds to loopback (see quantem.cli). Pinning the accepted Host
# values as well is what stops a DNS-rebinding page from reaching the API.
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "[::1]"]

# Application definition

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "rest_framework",  # DRF
    "django_filters",  # Django filter for DRF
    "corsheaders",
    "quantem.assets",  # Image/asset management app
    "quantem.library",  # Experiments and datasets over the image library
    "quantem.jobs",  # DB-backed job queue
    "quantem.segmentation",  # Segmentations, segments, ROIs, overlays
    "quantem.seg_core",  # Segmenter registry and shared inference plumbing
    "quantem.analysis",  # Quantitative analysis runs and their export bundles
    "quantem.finetune",  # Guided fine-tuning adapters
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "quantem.core.middleware.LocalOnlyMiddleware",
    # Above the rest so its process_exception runs last (Django unwinds
    # process_exception from the bottom up) and it is the final chance to turn
    # a request refused during parsing into JSON instead of Django's own 400
    # page. See ApiErrorShapeMiddleware.
    "quantem.core.middleware.ApiErrorShapeMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# CORS. In a packaged build the UI is loaded from the same loopback origin as
# the API, so CORS only matters for the separately-served dev SPA. Never
# CORS_ALLOW_ALL_ORIGINS: that let any page on the machine call the API.
CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOW_CREDENTIALS = False

_ui_origins = [
    origin.strip()
    for origin in os.environ.get("QUANTEM_UI_ORIGIN", "").split(",")
    if origin.strip()
]
if _ui_origins:
    CORS_ALLOWED_ORIGINS = _ui_origins
    CORS_ALLOWED_ORIGIN_REGEXES = []
else:
    # Both the server port (quantem.cli.free_port) and the dev SPA port are
    # chosen at runtime, so the default allowlist is "any loopback origin".
    # Set QUANTEM_UI_ORIGIN to pin it to one exact origin.
    CORS_ALLOWED_ORIGINS = []
    CORS_ALLOWED_ORIGIN_REGEXES = [
        r"^http://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$",
    ]

# Kept beside the CORS block it extends rather than hoisted away from it.
from corsheaders.defaults import default_headers  # noqa: E402

CORS_ALLOW_HEADERS = list(default_headers)

# Custom response headers the SPA reads cross-origin (the dev SPA runs on its
# own port). Without exposing these, the browser hides them from JS and the
# overlay LUT parser reads maxLabel/lut_revision/bundle_version as 0 -- which
# blanks the ID-map overlay (maxLabel=0 => every label is skipped on colorize).
CORS_EXPOSE_HEADERS = [
    "X-Overlay-Lut-Revision",
    "X-Overlay-Bundle-Version",
    "X-Overlay-Max-Label",
]

# Modest hardening; none of it costs anything on loopback HTTP.
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"

ROOT_URLCONF = "quantem.core.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "quantem.core.wsgi.application"


# Database
# https://docs.djangoproject.com/en/6.0/ref/settings/#databases
from quantem.core.db_config import get_database_config  # noqa: E402

DATABASES = {"default": get_database_config()}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# Internationalization
# https://docs.djangoproject.com/en/6.0/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/6.0/howto/static-files/

STATIC_URL = "static/"


# Django REST Framework configuration.
#
# There are no user accounts: django.contrib.auth is not installed, so DRF must
# not fall back to its Session/Basic defaults (they dereference request.user).
# There is no authentication: QuantEM is single-user and loopback-only.
# LocalOnlyMiddleware rejects a non-loopback request before any view runs,
# so every view is reached already authorized.
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "UNAUTHENTICATED_USER": None,
    "UNAUTHENTICATED_TOKEN": None,
    # JSON only: the browsable API renders templates that expect contrib.auth.
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
        "rest_framework.filters.SearchFilter",
        "rest_framework.filters.OrderingFilter",
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 100,
}

# Uploads.
#
# Where a multipart upload's bytes are staged before the view moves them into
# place. Django's default is ``tempfile.gettempdir()``, which on Windows is the
# signed-in user's own temporary folder under AppData, on the system drive, and
# that is wrong here twice over. It writes the whole image to the system drive
# -- forbidden by this project, and by
# the owner's ruling that storage lives with the installation -- and it puts
# the staged file on a *different volume* from the directory it is about to be
# moved into, which turns what should be a rename into a second full copy of a
# gigabyte. Pointing it at the data directory's own ``tmp`` makes the staged
# file a sibling of its destination.
#
# TMP_DIR is created by ensure_directories() at the top of this module, so it
# exists before the first request can arrive.
FILE_UPLOAD_TEMP_DIR = str(TMP_DIR)

# The largest request body the local server will accept, in bytes.
#
# waitress's default is 1 GiB, enforced against Content-Length before a single
# body byte is read: fourteen of the forty TIFFs over 400 MB in this
# laboratory's own collection are larger than that, so the shipped application
# simply could not import them. It answered 413 and closed the socket while the
# browser was still uploading, which the browser reports as a network error,
# and nothing in the UI ever mentioned that a limit existed.
#
# 64 GiB, because:
#
# * the largest image this laboratory has produced is 2 074 034 677 B (1.93
#   GiB), so this clears real work by more than thirty times, and clears the
#   4 GiB ceiling of a classic (non-Big) TIFF by sixteen;
# * it is still finite. There is no adversary on a loopback single-user server,
#   but there is a malformed Content-Length, and an unlimited server would
#   spool it onto the data volume until the disk filled. The real ceiling on an
#   import remains free disk space, and that failure reports itself honestly
#   from the write that hits it.
#
# quantem.cli passes this to waitress and reports the refusal in words; note
# waitress compares with ``>=``, so the largest accepted body is one byte less.
QUANTEM_MAX_UPLOAD_BYTES = 64 * 1024 * 1024 * 1024

# Logging. Console always; plus a rotating file under the data directory when
# the launcher asks for it. ``quantem serve`` and the frozen desktop build set
# ``QUANTEM_LOG_TO_FILE=1`` (paper-cut: the packaged server wrote no log file
# at all, so a session that crashed left nothing to attach to a bug report).
# A dev ``runserver`` and the test suite never set the flag and are unchanged.
LOG_LEVEL = os.environ.get("DJANGO_LOG_LEVEL", "WARNING").upper()
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(levelname)s:%(name)s:%(message)s"},
        # Timestamps matter in a file read days later; on a console they are noise.
        "file": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"},
    },
    "handlers": {
        # The console handler carries its own level so that lowering the root
        # (which the file handler needs) never changes what the console shows.
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": LOG_LEVEL,
        },
    },
    "root": {
        "handlers": ["console"],
        "level": LOG_LEVEL,
    },
    "loggers": {},
}


def _file_logging_enabled() -> bool:
    # Shared with quantem.cli, which announces the path this decides to write.
    from quantem.core.config import file_logging_enabled

    return file_logging_enabled()


if _file_logging_enabled():
    import logging as _logging

    FILE_LOG_LEVEL = os.environ.get("QUANTEM_FILE_LOG_LEVEL", "INFO").upper()
    if FILE_LOG_LEVEL not in _logging.getLevelNamesMapping():
        FILE_LOG_LEVEL = "INFO"
    LOGGING["handlers"]["file"] = {
        "class": "logging.handlers.RotatingFileHandler",
        # ensure_directories() above created logs/; delay=True keeps the file
        # unopened until the first record, so probing settings costs nothing.
        "filename": str(SERVER_LOG_PATH),
        "maxBytes": 5 * 1024 * 1024,
        "backupCount": 3,
        "encoding": "utf-8",
        "delay": True,
        "formatter": "file",
        "level": FILE_LOG_LEVEL,
    }
    LOGGING["root"]["handlers"].append("file")
    # The root must pass records down to the most verbose handler; each
    # handler's own level then does the filtering.
    _levels = _logging.getLevelNamesMapping()
    LOGGING["root"]["level"] = _logging.getLevelName(
        min(
            _levels.get(LOG_LEVEL, _logging.WARNING),
            _levels.get(FILE_LOG_LEVEL, _logging.INFO),
        )
    )
