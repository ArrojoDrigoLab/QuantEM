# Building QuantEM release artifacts

Three channels, one package. The pip wheel, the conda package and the desktop
installer all deliver the same `quantem-app` Python distribution — the desktop shell
merely wraps it. Model weights ship in **none** of them; they are downloaded at
runtime (see the README's *Models* section).

## Wheel and sdist

The built frontend rides inside the wheel as `quantem/_frontend/` (mapped from
`frontend/dist` by `[tool.hatch.build.targets.wheel.force-include]` in
pyproject.toml), and `quantem.core.spa` resolves it through
`importlib.resources` when the server is not running from a checkout. So the
frontend must be built first, and the order matters:

```bash
cd frontend && npm run build && cd ..     # typechecks, then vite build
python -m build                            # sdist + wheel into dist/
```

The sdist also carries `frontend/dist` (not the frontend sources), so a wheel
rebuilt from the sdist — which is what the conda recipe does — is identical and
needs no node.

**Version:** single source is `[project].version` in pyproject.toml.
`quantem.__version__` reads it back via `importlib.metadata` (installed) or from
pyproject.toml itself (uninstalled checkout). Bump it in one place only.

**Deliberately not in the wheel:** tests (`**/tests/**`; they assume a dev
checkout and pytest-django — the sdist keeps them), in-package READMEs
(`**/README.md`; development documents, the sdist keeps them too), model
weights, and any build scratch.

**Sdist include patterns are rooted** (`/src/quantem`, `/README.md`, …).
Hatchling treats an unanchored pattern as a recursive glob — unrooted
`"README.md"` once shipped an sdist carrying 800+ READMEs and LICENSEs from
`frontend/node_modules`. The gate below now fails any artifact with an
unexpected top-level path, so that cannot recur silently.

## Release gate

Every artifact that leaves this machine must pass:

```bash
python packaging/check_wheel.py dist/quantem_app-<version>-py3-none-any.whl
python packaging/check_wheel.py dist/quantem_app-<version>.tar.gz
```

It runs `quantem.registry.release.find_local_paths` — the same scanner that
sanitises model release bundles — over every text file, and fails on model
weights, `__pycache__`, `node_modules`, any unexpected top-level path, or any
machine path that is not **pinned**. Four kinds of scanner match are filtered
as non-findings: URL routes the app serves (`/api/...`, `/roi/...`),
data-directory fragments (`<QUANTEM_DATA_DIR>/cache/hf`), a platform's own
documented storage root quoted home-relative in documentation
(`~/Library/Application Support/QuantEM`), and regex/wasm noise in minified
vendor bundles.

Extensionless members are scanned by **name** (`METADATA`, `PKG-INFO`,
`RECORD`, `WHEEL`, `LICENSE`, `NOTICE`). A suffix test cannot see a
file with no suffix, and `METADATA` embeds the whole README as the long
description — which is the PyPI project page. Everything in `README.md`
therefore reached the public through a file the gate could not read.

**Inside tests** the scan is narrowed rather than skipped. Tests ship only in
the sdist and are genuinely full of path-shaped strings — the scrubber's own
fixtures have to *be* leaks — but skipping the whole file left 1.8 MB of the
sdist unread, and four test files carried a live laboratory share, two of them
as module-level defaults resolved on import. So only one shape is reported
there: an absolute path that names a specific machine's storage (a drive letter
or UNC host, then two or more directory-shaped segments). A fixture that needs
a machine-shaped path writes one with a visibly-example word in a segment —
`D:\example\legacy\head.pt`, `\\EXAMPLEHOST\share\weights`,
`C:\Users\someone\AppData\Local\Temp` — and real data lives behind an
environment variable with no default, so the test skips where the data is
absent.

What survives filtering must appear in the script's `PINNED` table — exact,
per-file documentary examples, chiefly the docstrings of the path-sanitising
modules themselves. The table **only shrinks** (owner ruling D8): a category
that deserves to ship becomes a rule, not a new line in it. Pinned hits are
printed on every pass, so nothing ships silently; any hit not pinned fails.

Then prove the wheel like a stranger would use it: fresh venv, install the
wheel, `quantem-app serve`, and check the UI loads, `/api/models/` answers, and an
image imports and preprocesses against a clean data directory.

## Conda package

The recipe is `packaging/conda/meta.yaml`. It builds `noarch: python` from the
published sdist; the renames from PyPI are `torch` → `pytorch`,
`opencv-python-headless` → `py-opencv` and `huggingface-hub` →
`huggingface_hub`.

```bash
python packaging/lint_conda.py                                  # no conda-build needed
conda build packaging/conda -c conda-forge --output-folder <dir>
```

The lint renders the recipe and cross-checks it against pyproject — version,
entry point, python bounds, and every run requirement under the renames — so
the recipe cannot drift from the package without a failure here.

At release time set `source.url` to the published sdist and fill in `sha256`;
for a local build use the `path: ../..` alternative noted in the recipe (after
`python -m build --sdist`). conda-build is not part of the application
environment — build in a dedicated environment or CI.

## Desktop installer

Out of scope here: the shell wraps this same package, frozen by PyInstaller into
a `quantem-server` binary, and spawns `quantem-server serve --port 0` — it does
not go through the console script. See the repository's desktop packaging notes.
