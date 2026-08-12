"""Release-build contracts that ordinary source tests cannot exercise.

The Windows installer embeds a PyInstaller sidecar. Its module graph is a
second dependency boundary: an import can work in the build environment and
still be absent from the installed application. These checks pin the two
non-obvious decisions that have caused release-only failures.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SPEC = PROJECT_ROOT / "packaging" / "pyinstaller" / "quantem-server.spec"
BUILD_SCRIPT = PROJECT_ROOT / "packaging" / "pyinstaller" / "build.ps1"


def _analysis_excludes(source: str) -> set[str]:
    tree = ast.parse(source, filename=str(SPEC))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "Analysis":
            continue
        for keyword in node.keywords:
            if keyword.arg == "excludes":
                values = ast.literal_eval(keyword.value)
                return {str(value) for value in values}
    raise AssertionError("the PyInstaller Analysis() call has no excludes list")


def test_frozen_bundle_keeps_torch_runtime_support_modules():
    """Production Torch 2.13 imports must not be mistaken for test-only code."""

    source = SPEC.read_text(encoding="utf-8")
    assert "torch.testing._internal" in source
    assert "torch.testing._internal" not in _analysis_excludes(source)


def test_normal_pyinstaller_build_is_quiet_without_hiding_real_warnings():
    source = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert 'if ($VerboseBuild) { "INFO" } else { "WARN" }' in source
    assert "$env:PYI_LOG_LEVEL = $logLevel" in source
    assert "Assuming this is not an Anaconda environment" in source
    assert "--log-level $logLevel" in source
