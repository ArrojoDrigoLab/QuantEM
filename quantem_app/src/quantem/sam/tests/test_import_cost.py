"""Having this app installed must cost nothing until a box is drawn.

The scientific stack's import budget exists because of the 8 GB laptop floor in
owner ruling R3, and it is measured at *import* time. This app's whole reason
for being lazy is that most launches never prompt a box: the URLconf and the
views are imported on every start, the 375 MB of weights and the torch graph
that loads them are imported on the first prompt and not before.

These tests run in a subprocess because ``torch`` is almost certainly already
in ``sys.modules`` by the time any other test has run -- asserting on the
current process would pass for the wrong reason.
"""

from __future__ import annotations

import json
import subprocess
import sys

from django.test import SimpleTestCase

#: Modules that must not be pulled in merely by importing this app.
FORBIDDEN = ("torch", "segment_anything", "micro_sam", "skimage")


def _import_probe(*statements: str) -> set[str]:
    """Which forbidden modules ``statements`` drag in, in a clean interpreter."""
    script = "\n".join(
        [
            "import json, sys",
            *statements,
            f"print(json.dumps([n for n in {FORBIDDEN!r} if n in sys.modules]))",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise AssertionError(f"probe failed:\n{completed.stdout}\n{completed.stderr}")
    return set(json.loads(completed.stdout.strip().splitlines()[-1]))


class ImportCostTests(SimpleTestCase):
    def test_the_config_module_is_free(self):
        self.assertEqual(_import_probe("import quantem.sam.config"), set())

    def test_the_geometry_module_does_not_import_torch(self):
        pulled = _import_probe("import quantem.sam.geometry")
        self.assertNotIn("torch", pulled)
        self.assertNotIn("segment_anything", pulled)

    def test_the_checkpoint_module_is_stdlib_only(self):
        self.assertEqual(_import_probe("import quantem.sam.checkpoint"), set())

    def test_choosing_a_backend_module_does_not_build_one(self):
        pulled = _import_probe("import quantem.sam.backends")
        self.assertNotIn("torch", pulled)
        self.assertNotIn("segment_anything", pulled)

    def test_the_whole_django_app_imports_without_torch(self):
        """The real check: what a launch that never draws a box pays."""
        pulled = _import_probe(
            "import os, django",
            "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'quantem.core.settings')",
            "django.setup()",
            "import quantem.sam.urls",
        )
        self.assertNotIn(
            "segment_anything",
            pulled,
            "importing the SAM routes pulled in the SAM runtime",
        )
