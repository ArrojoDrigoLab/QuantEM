"""Optional ``.env`` loading for developers running from a source checkout.

A packaged install has no ``.env``; everything it needs comes from
:mod:`quantem.cli`. ``read_env`` never overwrites a variable that is already
set, so anything the launcher exported wins.
"""

from pathlib import Path

import environ

BACKEND_ENV_FILES = (
    ".env",
    "local.env",
)


def load_backend_env_files(base_dir: Path) -> None:
    for env_file in BACKEND_ENV_FILES:
        environ.Env.read_env(str(base_dir / env_file))
