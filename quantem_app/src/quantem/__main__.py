"""``python -m quantem`` — the same CLI as the ``quantem`` console script.

The console script (``[project.scripts]`` in pyproject.toml) only exists for
an *installed* copy; a checkout driven with ``PYTHONPATH=src`` has no scripts
directory, and ``python -m quantem`` is the standard spelling that works in
both. Delegates to :func:`quantem.cli.main` with no behaviour of its own.
"""

from __future__ import annotations

from quantem.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
