"""Nothing new may open an image file for pixels, anywhere in the tree.

The saturating decode has now been found in **four different functions** --
``ngff.read_source_plane``'s Pillow arm and its pyvips arm (which clamps where
the canonical path scales, and was invisible to three verification rounds
because libvips is not installed on the box they ran on), and two of the
``task_utils`` fallbacks. Three rounds of human review added a guard at the
occurrence that was tested and missed the next one. This test is the mechanism
that stops there being a fifth.

**Why it was rewritten.** The first version matched on the *spelling* of a
call: an ``ast.Attribute`` whose owner was literally named ``Image`` or
``tifffile``. An adversarial pass planted ten realistic decodes and **seven
walked straight through**, including ``import tifffile as tf`` followed by
``tf.imread(...)`` inside a module this file asserts holds zero decodes, and
``from PIL.Image import open as _open``. A guard that a two-word alias defeats
is not a guard.

So the scanner now resolves names before it judges them. For each module it
builds a symbol table from the imports -- ``import tifffile as tf`` binds
``tf`` to ``tifffile``; ``from PIL import Image as _I`` binds ``_I`` to
``PIL.Image``; ``from PIL.Image import open as _open`` binds ``_open`` to
``PIL.Image.open``; a plain assignment ``_reader = tifffile.imread`` binds
``_reader`` too -- and then resolves every attribute chain and every bare name
against it. The alias is gone by the time the comparison happens, so aliasing
cannot help.

Four independent layers, because each one has a shape of attack it cannot see:

1. **Resolved references to a decode entry point.** ``PIL.Image.open``,
   ``tifffile.imread``, ``cv2.imread``, ``imageio.v3.imread`` and the rest of
   :data:`BANNED_ENTRY_POINTS`, under any alias, whether called or merely
   mentioned. Mentions count because ``def read(open=Image.open)`` is a decode
   the call-only version could not see.
2. **Imports of a library whose only use is decoding.** ``pyvips``,
   ``imageio``, ``skimage.io``, ``matplotlib.image`` and the microscopy-format
   readers in :data:`DECODE_ONLY_LIBRARIES` have no other reason to appear, so
   the import itself is the finding -- and an import is the one thing an alias
   cannot avoid. ``PIL``, ``tifffile`` and ``cv2`` are deliberately **not** on
   this list: PIL and tifffile also *write* the canonical PNG and the
   thumbnails, and cv2 is the morphology and contour library. Those three are
   caught at their entry points by layer 1 instead.
3. **Dynamic import and dynamic attribute lookup with a constant name.**
   ``importlib.import_module("tifffile")``, ``__import__("PIL")`` and
   ``getattr(Image, "open")``.
4. **A hard zero for the pyramid and reader modules.** Those six may not so
   much as *import* an image library, which is the layer that would have caught
   the aliased ``tifffile`` planted in ``task_utils``.

Plus the original string-method check, now also matching the keyword form:
``.convert("L")`` **and** ``.convert(mode="L")`` -- the second was another of
the seven that got through.

Two lists, and the difference between them matters:

* :data:`ALLOWED_MODULES` -- the modules that are *allowed* to decode.
  ``canonical_decode`` is the one decoder; ``volume_readers`` reads geometry
  and per-page planes for the 3-D path under its own documented contract.
* :data:`FROZEN_LEGACY` -- modules that still contain occurrences and are not
  in this change's boundary. Their counts are frozen: a new occurrence in any
  of them fails, and so does a new *file* joining the list. A count going
  **down** is reported as a warning rather than a failure, because
  ``segmentation/**`` and ``seg_core/**`` belong to other workflows and one of
  them removing a decode -- the correct thing to do -- must not turn this gate
  red. The warning names the number to lower.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]

#: May open an image file for pixel data.
ALLOWED_MODULES = {
    "quantem/assets/canonical_decode.py",
    "quantem/assets/volume_readers.py",
}

#: Modules outside this change's boundary that still decode, with the exact
#: number of banned references each holds today. See the module docstring.
FROZEN_LEGACY = {
    # The pre-authority import helpers. No longer on the pyramid or the reader
    # path -- ``tasks`` and ``task_utils`` both route through
    # ``canonical_decode`` now -- but still reached by ``convert_tiff_to_png``
    # and the ROI PNG writer. Handoff: fold into ``canonical_decode``.
    # 21 under the old spelling-based scanner; 22 under this one, which also
    # sees the ``import pyvips`` on line 24.
    "quantem/assets/utils.py": 22,
    # Probability maps are a different pixel domain with their own dtype rules,
    # and ``seg_core/**`` / ``segmentation/**`` belong to another workflow.
    # Handoff: ``canonical_decode.decode_probability_map``.
    "quantem/seg_core/db/prob_maps.py": 3,
    "quantem/segmentation/prob_maps/features.py": 2,
    "quantem/segmentation/prob_maps/io.py": 3,
    "quantem/segmentation/services/adapt/extract_crops.py": 3,
    "quantem/segmentation/utils.py": 3,
}

#: Test modules and ``quantem/testing.py`` are out of the production scan on
#: purpose: a fixture has to *write* the files under test, and the independent
#: reference decoder the source matrix compares against has to decode by hand
#: -- an oracle that called the code under test would prove nothing. Shipped
#: code importing a test module would put a decoder back on the pixel path, so
#: :func:`test_shipped_code_cannot_borrow_a_decoder_from_an_unscanned_corner`
#: closes that door, and the vendored ``_fig3`` one beside it.
_TEST_PATH_MARKERS = ("/tests/", "quantem/testing.py")

#: Fully-qualified functions that turn a file into pixels. Matched after alias
#: resolution, so the spelling in the source is irrelevant.
BANNED_ENTRY_POINTS = {
    "PIL.Image.open": "PIL.Image.open",
    "PIL.Image.frombuffer": "PIL.Image.frombuffer",
    "PIL.Image.frombytes": "PIL.Image.frombytes",
    "PIL.ImageFile.ImageFile": "PIL.ImageFile.ImageFile",
    "tifffile.imread": "tifffile.imread",
    "tifffile.TiffFile": "tifffile.TiffFile",
    "tifffile.TiffSequence": "tifffile.TiffSequence",
    "tifffile.TiffReader": "tifffile.TiffReader",
    "tifffile.imopen": "tifffile.imopen",
    "tifffile.memmap": "tifffile.memmap",
    "tifffile.ZarrTiffStore": "tifffile.ZarrTiffStore",
    "cv2.imread": "cv2.imread",
    "cv2.imreadmulti": "cv2.imreadmulti",
    "cv2.imdecode": "cv2.imdecode",
    "cv2.VideoCapture": "cv2.VideoCapture",
    "pyvips.Image.new_from_file": "pyvips.Image.new_from_file",
    "pyvips.Image.new_from_buffer": "pyvips.Image.new_from_buffer",
    "pyvips.Image.tiffload": "pyvips.Image.tiffload",
    "imageio.imread": "imageio.imread",
    "imageio.v2.imread": "imageio.v2.imread",
    "imageio.v3.imread": "imageio.v3.imread",
    "imageio.mimread": "imageio.mimread",
    "imageio.volread": "imageio.volread",
    "skimage.io.imread": "skimage.io.imread",
    "skimage.io.imread_collection": "skimage.io.imread_collection",
    "matplotlib.image.imread": "matplotlib.image.imread",
    "matplotlib.pyplot.imread": "matplotlib.pyplot.imread",
}

#: Libraries with no use in this tree except reading images. Importing one at
#: all, in any spelling, is the finding. ``PIL``, ``tifffile`` and ``cv2`` are
#: excluded on purpose -- see the module docstring.
DECODE_ONLY_LIBRARIES = {
    "pyvips",
    "imageio",
    "skimage.io",
    "matplotlib.image",
    "matplotlib.pyplot",
    "rasterio",
    "openslide",
    "nd2",
    "nd2reader",
    "czifile",
    "aicsimageio",
    "bioformats",
    "javabridge",
    "slideio",
    "pims",
    "mrcfile",
    "ncempy",
    "hyperspy",
    "dm3_lib",
    "pydicom",
    "nibabel",
    "SimpleITK",
    "itk",
    "OpenEXR",
    "rawpy",
    "pillow_heif",
    "imagecodecs",
}

#: Every module that any banned entry point lives in, for layer 4's hard zero
#: and for the dynamic-import check.
_IMAGE_LIBRARY_ROOTS = {"PIL", "tifffile", "cv2"} | DECODE_ONLY_LIBRARIES

#: The pyramid and reader path. These may not import an image library at all.
PIXEL_PATH_MODULES = (
    "quantem/assets/ngff.py",
    "quantem/assets/task_utils.py",
    "quantem/assets/tasks.py",
    "quantem/assets/views.py",
    "quantem/assets/pyramid_authority.py",
    "quantem/assets/asset_openable.py",
)

#: Two of the six build images rather than read them -- ``ngff`` writes the
#: preview thumbnail and ``task_utils`` builds and crops the ROI PNG -- and
#: both need Pillow to do it. The exception is named here rather than waved
#: through, and :data:`ENCODE_ONLY_ATTRIBUTES` bounds what they may touch, so
#: "it is allowed to import PIL" cannot quietly become "it is allowed to open
#: files with PIL".
PIXEL_PATH_ENCODE_ONLY = {
    "quantem/assets/ngff.py": {"PIL"},
    "quantem/assets/task_utils.py": {"PIL"},
}

#: The only members an encode-only module may reach on its allowed library.
#: ``open``, ``frombytes`` and the rest are absent on purpose.
ENCODE_ONLY_ATTRIBUTES = {
    "PIL.Image.fromarray",
    "PIL.Image.MAX_IMAGE_PIXELS",
    "PIL.Image.new",
    "PIL.Image.merge",
    "PIL.Image.Image",
    "PIL.Image.Resampling",
    "PIL.Image.NEAREST",
    "PIL.Image.BILINEAR",
    "PIL.Image.BICUBIC",
    "PIL.Image.LANCZOS",
}

_BANNED_STRING_METHODS = {
    ("convert", "L"): '.convert("L")',
    ("cast", "uchar"): '.cast("uchar")',
}

_REPLACEMENT = (
    "call quantem.assets.canonical_decode.decode_canonical_plane(path) instead -- "
    "it is the only function in the tree allowed to turn a file into pixels, and "
    "it is why the pyramid no longer depends on which of four decoders ran"
)


def _dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for an attribute chain rooted in a plain name, else ``None``."""

    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


class _Symbols:
    """Local name -> fully-qualified name, built from a module's own imports."""

    def __init__(self) -> None:
        self.bindings: dict[str, str] = {}
        self.imported: list[tuple[int, str]] = []

    def resolve(self, dotted: str | None) -> str | None:
        if not dotted:
            return None
        head, _, tail = dotted.partition(".")
        base = self.bindings.get(head)
        if base is None:
            return dotted
        return f"{base}.{tail}" if tail else base

    def visit_import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            target = alias.name if alias.asname else alias.name.split(".")[0]
            self.bindings[local] = target
            self.imported.append((node.lineno, alias.name))

    def visit_import_from(self, node: ast.ImportFrom) -> None:
        if node.level:  # a relative import cannot reach a third-party library
            return
        module = node.module or ""
        for alias in node.names:
            local = alias.asname or alias.name
            self.bindings[local] = f"{module}.{alias.name}" if module else alias.name
            self.imported.append((node.lineno, f"{module}.{alias.name}"))

    def visit_assign(self, node: ast.Assign) -> None:
        """``_reader = tifffile.imread`` is an alias too."""

        resolved = self.resolve(_dotted(node.value))
        if resolved is None:
            return
        if resolved not in BANNED_ENTRY_POINTS and resolved.split(".")[0] not in (
            _IMAGE_LIBRARY_ROOTS
        ):
            return
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.bindings[target.id] = resolved


def _build_symbols(tree: ast.AST) -> _Symbols:
    symbols = _Symbols()
    # Imports first, so an assignment that aliases an imported name resolves.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            symbols.visit_import(node)
        elif isinstance(node, ast.ImportFrom):
            symbols.visit_import_from(node)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            symbols.visit_assign(node)
    return symbols


def _library_of(dotted: str) -> str | None:
    """The decode-only library ``dotted`` names, if any (longest match)."""

    for library in sorted(DECODE_ONLY_LIBRARIES, key=len, reverse=True):
        if dotted == library or dotted.startswith(library + "."):
            return library
    return None


def banned_references(tree: ast.AST) -> list[tuple[int, str]]:
    """Every decode reference in one parsed module, after alias resolution."""

    symbols = _build_symbols(tree)
    found: list[tuple[int, str]] = []

    # Layer 2 -- importing a library that exists only to decode.
    for lineno, imported in symbols.imported:
        library = _library_of(imported)
        if library is not None:
            found.append((lineno, f"imports {library}"))

    for node in ast.walk(tree):
        # Layer 1 -- a resolved reference to a decode entry point, called or not.
        if isinstance(node, (ast.Attribute, ast.Name)) and isinstance(node.ctx, ast.Load):
            resolved = symbols.resolve(_dotted(node))
            if resolved in BANNED_ENTRY_POINTS:
                found.append((node.lineno, BANNED_ENTRY_POINTS[resolved]))
                continue

        if not isinstance(node, ast.Call):
            continue
        func = node.func

        # Layer 3 -- dynamic import and dynamic attribute lookup.
        name = symbols.resolve(_dotted(func))
        if name in {"importlib.import_module", "__import__"}:
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                root = first.value.split(".")[0]
                if root in _IMAGE_LIBRARY_ROOTS or _library_of(first.value):
                    found.append((node.lineno, f"imports {first.value} dynamically"))
        if name == "getattr" and len(node.args) >= 2:
            owner = symbols.resolve(_dotted(node.args[0]))
            attribute = node.args[1]
            if (
                owner
                and isinstance(attribute, ast.Constant)
                and isinstance(attribute.value, str)
                and f"{owner}.{attribute.value}" in BANNED_ENTRY_POINTS
            ):
                found.append((node.lineno, f"{owner}.{attribute.value} via getattr"))

        # The original string-method check, positional and keyword.
        if isinstance(func, ast.Attribute):
            candidates = []
            if node.args and isinstance(node.args[0], ast.Constant):
                candidates.append(node.args[0].value)
            for keyword in node.keywords:
                if keyword.arg in {"mode", "format"} and isinstance(
                    keyword.value, ast.Constant
                ):
                    candidates.append(keyword.value.value)
            for value in candidates:
                if not isinstance(value, str):
                    continue
                label = _BANNED_STRING_METHODS.get((func.attr, value))
                if label is not None:
                    found.append((node.lineno, label))

    # One reference can trip two layers (an aliased import *and* its call);
    # de-duplicate on (line, label) so the frozen counts stay meaningful.
    return sorted(set(found))


def library_members_used(tree: ast.AST, library: str) -> set[str]:
    """Every fully-qualified member of ``library`` a module reaches for.

    Resolved, so ``from PIL import Image as _I`` then ``_I.fromarray`` reports
    ``PIL.Image.fromarray``. Bare references to the module itself (``PIL`` or
    ``PIL.Image`` alone, as in ``Image.MAX_IMAGE_PIXELS = None``'s target) are
    reported as the dotted name they resolve to.
    """

    symbols = _build_symbols(tree)
    members: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Attribute, ast.Name)):
            continue
        resolved = symbols.resolve(_dotted(node))
        if not resolved:
            continue
        if resolved == library or resolved.startswith(library + "."):
            members.add(resolved)
    # Keep only the longest chain of each: ``PIL.Image.fromarray`` implies
    # ``PIL.Image``, and reporting both would be noise.
    return {
        member
        for member in members
        if not any(other != member and other.startswith(member + ".") for other in members)
    }


def imported_image_libraries(tree: ast.AST) -> set[str]:
    """Which image libraries a module imports, by root name."""

    symbols = _build_symbols(tree)
    roots = set()
    for _lineno, imported in symbols.imported:
        root = imported.split(".")[0]
        if root in _IMAGE_LIBRARY_ROOTS:
            roots.add(root)
        library = _library_of(imported)
        if library:
            roots.add(library.split(".")[0])
    return roots


def _iter_sources(*, tests: bool = False):
    for path in sorted((SRC / "quantem").rglob("*.py")):
        if "__pycache__" in path.parts or "_fig3" in path.parts:
            continue
        relative = path.relative_to(SRC).as_posix()
        is_test = any(marker in relative for marker in _TEST_PATH_MARKERS)
        if is_test is not tests:
            continue
        yield path, relative


def test_no_module_outside_the_allowlist_opens_an_image_for_pixels():
    offences: list[str] = []
    counts: dict[str, int] = {}
    for path, relative in _iter_sources():
        if relative in ALLOWED_MODULES:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover - a broken file is its own failure
            offences.append(f"{relative}: could not be parsed ({exc})")
            continue
        calls = banned_references(tree)
        if not calls:
            continue
        counts[relative] = len(calls)
        if relative not in FROZEN_LEGACY:
            for line, label in calls:
                offences.append(f"{relative}:{line} reaches {label}; {_REPLACEMENT}")

    for relative, frozen in sorted(FROZEN_LEGACY.items()):
        actual = counts.get(relative, 0)
        if actual > frozen:
            offences.append(
                f"{relative} now holds {actual} image-decoding references, up from the "
                f"frozen {frozen}. A new decode outside canonical_decode.py is exactly "
                "the defect this test exists to stop; " + _REPLACEMENT
            )
        elif actual < frozen:
            warnings.warn(
                f"{relative} holds {actual} image-decoding references, fewer than the "
                f"frozen {frozen}. Good -- lower the number in FROZEN_LEGACY so the "
                "next one cannot creep back under the old ceiling.",
                stacklevel=1,
            )

    assert offences == [], "\n".join(offences)


def test_the_pyramid_and_reader_modules_hold_no_decode_at_all():
    """The modules this change owns are clean, not merely frozen."""

    for relative in PIXEL_PATH_MODULES:
        tree = ast.parse((SRC / relative).read_text(encoding="utf-8"))
        assert banned_references(tree) == [], (
            f"{relative} decodes an image itself; {_REPLACEMENT}"
        )


def test_the_pyramid_and_reader_modules_do_not_even_import_an_image_library():
    """The layer an alias cannot dodge.

    ``import tifffile as tf`` inside ``task_utils`` followed by ``tf.imread``
    walked through every earlier version of this file. There is no legitimate
    reason for any of these six to hold a decoder at all, so the import is the
    line, and the two-word alias has nothing left to hide behind.
    """

    offences: list[str] = []
    for relative in PIXEL_PATH_MODULES:
        tree = ast.parse((SRC / relative).read_text(encoding="utf-8"))
        allowed = PIXEL_PATH_ENCODE_ONLY.get(relative, set())
        for library in sorted(imported_image_libraries(tree) - allowed):
            offences.append(
                f"{relative} imports {library}. Nothing on the pyramid or reader path "
                f"may import an image library, whatever it calls it; {_REPLACEMENT}"
            )
        # And the encode-only exception stays encode-only.
        for library in sorted(allowed):
            for member in sorted(library_members_used(tree, library)):
                if member not in ENCODE_ONLY_ATTRIBUTES:
                    offences.append(
                        f"{relative} reaches {member}. That module may use {library} to "
                        "build an image, not to read one; add it to "
                        "ENCODE_ONLY_ATTRIBUTES only if it cannot open a file"
                    )
    assert offences == [], "\n".join(offences)


def test_shipped_code_cannot_borrow_a_decoder_from_an_unscanned_corner():
    """The scan has two exemptions; neither may be used as a back door.

    Test modules are exempt because a fixture has to write the files under test
    and the reference decoder has to decode by hand. ``inference/_fig3`` is
    exempt because it is vendored third-party code this project does not edit.
    Either one becomes a hole the moment shipped code imports from it, so:
    nothing shipped imports a test module, and nothing in the asset package --
    the whole of the pixel path -- imports the vendored stack.
    """

    offences: list[str] = []
    for path, relative in _iter_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            for name in names:
                if name == "quantem.testing" or ".tests" in f".{name}.":
                    offences.append(
                        f"{relative}:{node.lineno} imports the test module {name}, which "
                        "the decode scan deliberately does not cover"
                    )
                if relative.startswith("quantem/assets/") and "_fig3" in name.split("."):
                    offences.append(
                        f"{relative}:{node.lineno} imports {name} from the vendored stack, "
                        "which the decode scan deliberately does not cover"
                    )
    assert offences == [], "\n".join(offences)


def test_the_builder_cannot_be_handed_a_path():
    """``build_pyramid`` takes a ticket and a plane, never a source path.

    This is the type-level lock -- the one that holds when someone edits the
    linters. Round 3's ``regenerate_ngff_for_image(image)`` took a *path*,
    decided from its suffix that a staged 16-bit PNG upload was the canonical
    PNG, and published an all-white pyramid over a FAILED asset.

    It locks the *builder*. It does not lock the readers: ``load_image_array``
    and its siblings in ``task_utils`` still return bare ``ndarray``, so a
    decode that slipped past every layer above could still reach segmentation
    through them. That gap is recorded in ``canonical_decode``'s docstring and
    is a change in a module this package does not own.
    """

    import inspect

    from quantem.assets import ngff

    parameters = list(inspect.signature(ngff.build_pyramid).parameters)
    assert parameters[:3] == ["ticket", "image", "plane"], parameters
    for gone in ("read_source_plane", "ensure_ngff_for_image", "regenerate_ngff_for_image"):
        assert not hasattr(ngff, gone), (
            f"ngff.{gone} is back. Each of these took a path and decided for itself "
            "what to do with it; that is the defect class this change removed."
        )


def test_the_scanner_catches_every_shape_of_decode_that_defeated_it():
    """The guard is tested against the attacks that beat its predecessor.

    Seven of these ten passed the spelling-based scanner. They are kept here as
    executable source strings rather than as planted files, so the mechanism is
    checked on every run instead of only when someone re-runs an adversarial
    harness.
    """

    attacks = {
        "literal": "from PIL import Image\ndef read(p):\n    return Image.open(p).convert('L')\n",
        "aliased_pil": (
            "from PIL import Image as _Img\nimport numpy as np\n"
            "def read(p):\n    return np.asarray(_Img.open(p).convert('L'))\n"
        ),
        "from_import_open": (
            "from PIL.Image import open as _open\nimport numpy as np\n"
            "def read(p):\n    return np.asarray(_open(p))\n"
        ),
        "convert_keyword": (
            "import PIL.Image as pil\nimport numpy as np\n"
            "def read(p):\n    h = pil.open(p)\n    return np.asarray(h.convert(mode='L'))\n"
        ),
        "aliased_tifffile": (
            "import tifffile as tf\nimport numpy as np\n"
            "def read(p):\n    return np.asarray(tf.imread(str(p)), dtype=np.uint8)\n"
        ),
        "imageio": (
            "import imageio.v3 as iio\nimport numpy as np\n"
            "def read(p):\n    return np.asarray(iio.imread(p)).astype(np.uint8)\n"
        ),
        "cv2": "import cv2\ndef read(p):\n    return cv2.imread(str(p), 0)\n",
        "matplotlib": (
            "import matplotlib.image as mpimg\ndef read(p):\n    return mpimg.imread(str(p))\n"
        ),
        "deep_alias": (
            "import tifffile\n_reader = tifffile.imread\n"
            "def read(p):\n    return _reader(str(p))\n"
        ),
        "default_argument": (
            "from PIL import Image\n"
            "def read(p, opener=Image.open):\n    return opener(p)\n"
        ),
        "dynamic_import": (
            "import importlib\n"
            "def read(p):\n    return importlib.import_module('tifffile').imread(str(p))\n"
        ),
        "getattr_lookup": (
            "from PIL import Image\n"
            "def read(p):\n    return getattr(Image, 'open')(p)\n"
        ),
        "dunder_import": (
            "def read(p):\n    return __import__('imageio').imread(p)\n"
        ),
        "aliased_pyvips": (
            "import pyvips as v\n"
            "def read(p):\n    return v.Image.new_from_file(str(p)).cast('uchar')\n"
        ),
    }
    missed = [
        name for name, source in attacks.items() if not banned_references(ast.parse(source))
    ]
    assert missed == [], (
        "the chokepoint scanner did not see these decodes: " + ", ".join(missed)
    )


def test_the_scanner_does_not_fire_on_the_things_that_are_not_decodes():
    """A guard that cries wolf gets deleted, so its false positives are pinned.

    Writing a PNG, resampling a mask with cv2, measuring regions with skimage
    and opening a zarr store are all legitimate and common in this tree.
    """

    innocent = {
        "writes_a_png": (
            "from PIL import Image\n"
            "def write(a, p):\n    Image.fromarray(a).save(p)\n"
        ),
        "cv2_morphology": (
            "import cv2\n"
            "def close(mask, k):\n    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)\n"
        ),
        "cv2_resize": "import cv2\ndef up(a, s):\n    return cv2.resize(a, s)\n",
        "skimage_measure": (
            "from skimage.measure import regionprops\ndef m(a):\n    return regionprops(a)\n"
        ),
        "zarr_store": "import zarr\ndef open_store(p):\n    return zarr.open(p, mode='r')\n",
        "convert_rgb": (
            "from PIL import Image\n"
            "def rgb(h):\n    return h.convert('RGB')\n"
        ),
        "tifffile_write": (
            "import tifffile\ndef save(a, p):\n    tifffile.imwrite(str(p), a)\n"
        ),
    }
    fired = {
        name: banned_references(ast.parse(source))
        for name, source in innocent.items()
        if banned_references(ast.parse(source))
    }
    assert fired == {}, f"the chokepoint scanner fired on legitimate code: {fired}"
