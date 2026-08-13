"""Release-build contracts that ordinary source tests cannot exercise.

The Windows and macOS installers embed the same PyInstaller sidecar. Its module
graph is a second dependency boundary: an import can work in the build
environment and still be absent from the installed application. These checks
pin the non-obvious decisions that have caused release-only failures.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SPEC = PROJECT_ROOT / "packaging" / "pyinstaller" / "quantem-server.spec"
BUILD_SCRIPT = PROJECT_ROOT / "packaging" / "pyinstaller" / "build.ps1"
ENTRY_POINT = PROJECT_ROOT / "packaging" / "pyinstaller" / "quantem_server_entry.py"
CI_WORKFLOW = PROJECT_ROOT.parent / ".github" / "workflows" / "quantem-app.yml"
RELEASE_WORKFLOW = PROJECT_ROOT.parent / ".github" / "workflows" / "quantem-app-desktop-release.yml"


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


def test_frozen_bundle_keeps_sympy_for_eager_model_execution():
    """OmniEM reaches torch's lazy symbolic-shape imports during inference."""

    source = SPEC.read_text(encoding="utf-8")
    assert 'hiddenimports += ["sympy"]' in source
    assert "sympy" not in _analysis_excludes(source)

    entry_source = ENTRY_POINT.read_text(encoding="utf-8")
    assert 'os.environ.get("QUANTEM_FROZEN_SELFTEST") == "1"' in entry_source
    assert "import sympy" in entry_source
    assert "import torch.fx.experimental.symbolic_shapes" in entry_source


def test_platform_builds_execute_the_frozen_model_runtime_selftest():
    """A server-start check alone cannot catch lazy model-runtime imports."""

    assert "QUANTEM_FROZEN_SELFTEST=1" in CI_WORKFLOW.read_text(encoding="utf-8")
    release_source = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert release_source.count("QUANTEM_FROZEN_SELFTEST") >= 2


def test_normal_pyinstaller_build_is_quiet_without_hiding_real_warnings():
    source = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert 'if ($VerboseBuild) { "INFO" } else { "WARN" }' in source
    assert "$env:PYI_LOG_LEVEL = $logLevel" in source
    assert "Assuming this is not an Anaconda environment" in source
    assert "--log-level $logLevel" in source
