"""The vendored Segment Anything must stay diffable against upstream.

Vendoring only keeps its value if the copy stays a copy. These digests are the
files exactly as they came from
``facebookresearch/segment-anything@dca509fe793f601edb92606367a655c15ac00fdf``;
a failure here means someone edited third-party source in place, which is the
one thing a vendored directory must not quietly allow. If a change is genuinely
wanted, mark it with a ``QUANTEM:`` comment, note it in the package docstring
and update the digest in the same commit.

The digests use upstream's LF line endings. Git for Windows may materialize a
CRLF worktree even though the indexed source is unchanged, so the comparison
normalizes that checkout-only difference before hashing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from django.test import SimpleTestCase

VENDOR = Path(__file__).resolve().parent.parent / "_vendor" / "segment_anything"

#: Upstream files, copied byte for byte. ``__init__.py`` and ``predictor.py``
#: are deliberately absent -- both carry a marked QuantEM change.
VERBATIM = {
    "build_sam.py": "77bc4faf728118625ddd184dc4e6375bd1b5c636fc66fb429f9fc5276f331cc5",
    "modeling/__init__.py": "47f6363b86c0bfaa7a6abf723975136d17fae4a5f002d2814a5cce8834d18980",
    "modeling/common.py": "59a53bfa5d0a4d446df7f2bb57d2e95808b130601592b934600fb663d256ce17",
    "modeling/image_encoder.py": "1665ada2902722c9a8cd566c840e324d623ef2b12f7d9ae9f38b24aed0e39886",
    "modeling/mask_decoder.py": "6a2455e990515c67a061187c92c236b879b189f21ef6ebfce1627cae9a6b2f93",
    "modeling/prompt_encoder.py": "bd8f2f5903f0647f55c21c608bcbb334eec2901ebe76aa41061b1c513df4a06f",
    "modeling/sam.py": "2d501173b7345f6829f745a2ade240c98026f6b3237bf6cf8e9473405e86c54e",
    "modeling/transformer.py": "566a8f6db8b608398de939548bd0f485b21752fda2bdda36900e6444ef938d11",
    "utils/__init__.py": "34bd8069c54764e7b8d73a78905dbe6467140a2f73170875128f6ca4d8cdd0aa",
    "utils/transforms.py": "26a649f2bca8a4a24b4d56617487a9a54c4279e559b2b813c9de38f58a60107c",
}

#: Left out on purpose -- nothing on the box-prompt path reaches them.
OMITTED = ("automatic_mask_generator.py", "utils/amg.py", "utils/onnx.py")


def _upstream_digest(source: bytes) -> str:
    source = source.replace(b"\r\n", b"\n")
    return hashlib.sha256(source).hexdigest()


class VendoredSourceTests(SimpleTestCase):
    def test_upstream_files_are_unmodified(self):
        for name, expected in VERBATIM.items():
            path = VENDOR / name
            self.assertTrue(path.is_file(), f"{name} is missing from the vendored copy")
            actual = _upstream_digest(path.read_bytes())
            self.assertEqual(actual, expected, f"{name} no longer matches upstream")

    def test_windows_checkout_newlines_do_not_look_like_a_vendor_edit(self):
        source = b"Copyright (c) Meta Platforms, Inc.\nsource line\n"
        self.assertEqual(_upstream_digest(source), _upstream_digest(source.replace(b"\n", b"\r\n")))

    def test_the_omitted_modules_stayed_out(self):
        for name in OMITTED:
            self.assertFalse(
                (VENDOR / name).exists(),
                f"{name} was vendored; it is not on the box-prompt path",
            )

    def test_every_upstream_file_keeps_its_copyright_header(self):
        for name in VERBATIM:
            head = (VENDOR / name).read_text(encoding="utf-8")[:400]
            self.assertIn("Copyright (c) Meta Platforms", head, f"{name} lost its header")

    def test_the_two_changed_files_say_so(self):
        for name in ("__init__.py", "predictor.py"):
            text = (VENDOR / name).read_text(encoding="utf-8")
            self.assertIn("QUANTEM:", text, f"{name} differs from upstream without saying where")

    def test_no_vendored_file_imports_a_top_level_segment_anything(self):
        """The one functional edit, checked across the whole vendored tree.

        Line-by-line rather than a substring search: the ``QUANTEM:`` comment on
        the changed line quotes the original import, so a naive ``assertNotIn``
        matches its own explanation and fails on correct code.
        """
        for path in sorted(VENDOR.rglob("*.py")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                statement = line.split("#", 1)[0].strip()
                self.assertFalse(
                    statement.startswith(("import segment_anything", "from segment_anything")),
                    f"{path.name}:{number} imports a top-level segment_anything, "
                    "which does not exist here",
                )

        predictor = (VENDOR / "predictor.py").read_text(encoding="utf-8")
        self.assertIn("from .modeling import Sam", predictor)


class VendoredImportTests(SimpleTestCase):
    """It has to actually work, not merely be present."""

    def test_the_registry_builds_a_vit_b(self):
        from quantem.sam._vendor.segment_anything import sam_model_registry

        self.assertIn("vit_b", sam_model_registry)
        model = sam_model_registry["vit_b"]()
        width = model.state_dict()["image_encoder.patch_embed.proj.weight"].shape[0]
        self.assertEqual(width, 768, "vit_b did not build at the expected width")

    def test_the_predictor_imports(self):
        from quantem.sam._vendor.segment_anything.predictor import SamPredictor

        self.assertTrue(callable(SamPredictor))

    def test_it_needs_torchvision(self):
        """Stated rather than assumed: the transforms are on the prompt path."""
        text = (VENDOR / "utils" / "transforms.py").read_text(encoding="utf-8")
        self.assertIn("torchvision", text)
        import torchvision  # noqa: F401  -- already a QuantEM dependency
