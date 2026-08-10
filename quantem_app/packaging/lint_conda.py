"""Lint packaging/conda/meta.yaml without conda-build.

Usage::

    python packaging/lint_conda.py

conda-build is deliberately not part of the application environment, so this
performs the checks that matter for keeping the recipe truthful:

1. the jinja in the recipe renders (the tiny subset it actually uses) and the
   result is valid YAML with the sections conda-build requires;
2. the ``run`` requirements are exactly pyproject's ``[project.dependencies]``
   plus ``python`` — same version specs — under the PyPI→conda renames
   (``torch``→``pytorch``, ``opencv-python-headless``→``py-opencv``,
   dashes→underscores where conda uses them);
3. version, entry point and python bounds match pyproject.

So the recipe cannot drift from the package it builds without this failing.
Exit status 0 iff every check passes.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RECIPE = HERE / "conda" / "meta.yaml"
PYPROJECT = HERE.parent / "pyproject.toml"

#: PyPI name (normalised to lower, ``-`` kept) -> conda-forge name.
CONDA_NAMES = {
    "torch": "pytorch",
    "opencv-python-headless": "py-opencv",
    "huggingface-hub": "huggingface_hub",
}


def render_jinja(text: str) -> str:
    """Render the jinja subset the recipe uses: {% set %}, {{ var }}, filters."""
    variables: dict[str, str] = {"PYTHON": "python"}
    for name, value in re.findall(r'{%\s*set\s+(\w+)\s*=\s*"([^"]*)"\s*%}', text):
        variables[name] = value
    text = re.sub(r"{%.*?%}\n?", "", text)

    def substitute(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        m = re.fullmatch(r"(\w+)(\[0\])?(?:\s*\|\s*(\w+))?", expr)
        if not m:
            raise SystemExit(f"jinja expression not supported by this lint: {{{{ {expr} }}}}")
        name, index, filt = m.groups()
        if name not in variables:
            raise SystemExit(f"undefined jinja variable: {name}")
        value = variables[name]
        if index:
            value = value[0]
        if filt == "lower":
            value = value.lower()
        elif filt:
            raise SystemExit(f"jinja filter not supported by this lint: {filt}")
        return value

    return re.sub(r"{{(.*?)}}", substitute, text)


def split_requirement(req: str) -> tuple[str, str]:
    """``"Django>=5.2,<5.3"`` -> ``("django", ">=5.2,<5.3")``, spaces dropped."""
    m = re.match(r"\s*([A-Za-z0-9_.\-]+)\s*(.*)$", req)
    assert m, req
    return m.group(1).lower(), m.group(2).replace(" ", "")


def main() -> int:
    problems: list[str] = []
    recipe = yaml.safe_load(render_jinja(RECIPE.read_text(encoding="utf-8")))
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]

    for section in ("package", "source", "build", "requirements", "test", "about"):
        if section not in recipe:
            problems.append(f"recipe is missing the `{section}` section")

    if recipe["package"]["version"] != project["version"]:
        problems.append(
            f"version drift: recipe {recipe['package']['version']} vs "
            f"pyproject {project['version']}"
        )
    if recipe["build"].get("noarch") != "python":
        problems.append("recipe must be `noarch: python`")

    entry_points = recipe["build"].get("entry_points", [])
    expected_ep = [f"{k} = {v}" for k, v in project["scripts"].items()]
    if entry_points != expected_ep:
        problems.append(f"entry points {entry_points} != pyproject scripts {expected_ep}")

    python_bound = project["requires-python"].replace(" ", "")
    expected = {"python": python_bound, "pip": "", "hatchling": ""}
    host = dict(split_requirement(r) for r in recipe["requirements"]["host"])
    if host != expected:
        problems.append(f"host requirements {host} != {expected}")

    run = dict(split_requirement(r) for r in recipe["requirements"]["run"])
    wanted = {"python": python_bound}
    for dep in project["dependencies"]:
        name, spec = split_requirement(dep)
        wanted[CONDA_NAMES.get(name, name)] = spec
    # conda uses the PyPI name for everything not in CONDA_NAMES; case-folding
    # is enough (Django -> django, PyYAML -> pyyaml, Pillow -> pillow).
    for name, spec in wanted.items():
        if name not in run:
            problems.append(f"run requirements missing `{name} {spec}`")
        elif run[name] != spec:
            problems.append(f"run spec drift for `{name}`: recipe `{run[name]}` vs pyproject `{spec}`")
    for name in run:
        if name not in wanted:
            problems.append(f"run requirement `{name}` is not in pyproject dependencies")

    if recipe["about"].get("license") != "BSD-3-Clause":
        problems.append("about.license must be BSD-3-Clause")
    if "sha256" not in recipe["source"] and "path" not in recipe["source"]:
        problems.append("source must carry url+sha256 (release) or path (local build)")

    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(" ", p)
        return 1
    n_run = len(recipe["requirements"]["run"])
    print(f"meta.yaml renders, parses, and matches pyproject ({n_run} run requirements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
