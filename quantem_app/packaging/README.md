# Building QuantEM release artifacts

Three channels, one package. The pip wheel, the conda package and the desktop
installer all deliver the same `quantem` Python distribution — the desktop shell
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
weights, anything from `.scratch`.

**Sdist include patterns are rooted** (`/src/quantem`, `/README.md`, …).
Hatchling treats an unanchored pattern as a recursive glob — unrooted
`"README.md"` once shipped an sdist carrying 800+ READMEs and LICENSEs from
`frontend/node_modules`. The gate below now fails any artifact with an
unexpected top-level path, so that cannot recur silently.

## Release gate

Every artifact that leaves this machine must pass:

```bash
python packaging/check_wheel.py dist/quantem-<version>-py3-none-any.whl
python packaging/check_wheel.py dist/quantem-<version>.tar.gz
```

It runs `quantem.registry.release.find_local_paths` — the same scanner that
sanitises model release bundles — over every text file, and fails on model
weights, `__pycache__`, `node_modules`, any unexpected top-level path, or any
machine path that is not **pinned**. Three kinds of scanner match are filtered
as non-findings: URL routes the app serves (`/api/...`, `/roi/...`),
data-directory fragments (`<QUANTEM_DATA_DIR>/cache/hf`), and regex/wasm noise
in minified vendor bundles. Test files are not path-scanned: they ship only in
the sdist, and the scrubber's own tests hold leak-shaped fixtures by
construction.

What survives filtering must appear in the script's `PINNED` table — exact,
per-file documentary examples (docstrings of the path-sanitising modules
themselves, and the Models screen's `e.g. D:\quantem-models-0.1.0`
placeholder). Pinned hits are printed on every pass, so nothing ships silently;
any hit not pinned fails.

Then prove the wheel like a stranger would use it: fresh venv, install the
wheel, `quantem serve`, and check the UI loads, `/api/models/` answers, and an
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

Out of scope here: the shell wraps this same package and spawns
`quantem serve --port 0`. See the repository's desktop packaging notes.
