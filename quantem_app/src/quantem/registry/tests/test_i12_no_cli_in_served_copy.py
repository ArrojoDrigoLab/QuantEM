"""Invariant I-12, gated where the copy actually lives: the serialised API.

**Why this file exists.** I-12 ("no user-facing string may contain a shell
command") was accepted by a grep over the built JavaScript bundle. On
2026-08-10 that grep returned zero hits while a verifier had the forbidden
string on screen three times in one five-minute session -- the create-run
dialog, the labeling header after a failed run, and the viewer's overlay card::

    quantem:er cannot run on this machine. Not installed yet. Install it from
    the Models screen or with `quantem models install <pack id, e.g.
    quantem:mito>` -- QuantEM downloads and verifies it from Hugging Face.
    Offline, download a QuantEM model release instead, unzip it, and install it
    with `quantem models install <the directory you unzipped the release into>`.

The bundle grep could not see it because the frontend does not own that
sentence. The backend does: it is ``quantem.registry.cache.INSTALL_HINT``,
reaching the screen through ``reason`` on ``GET /api/models/`` and through a
segmentation's ``status_error``. **A gate that only reads the bundle is blind to
every string the server composes**, which is where all the dangerous copy is.

So this gate enumerates the *server side* of every surface that shows install
advice, serialises it exactly as the API would, and runs
:mod:`quantem.registry.tests.copy_gate` over every string that comes out.
``test_the_gate_sees_the_string_that_shipped`` pins the original sentence as a
literal and asserts the detector reports all three of its defects, so weakening
the detector fails here rather than silently.

Scope note: the surfaces below are the registry's and the model-error
translator's, which is where F2 lived and what this package owns. The detector
is written to be reused; other apps' serialisers should be added to it as their
owners reach them.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from django.test import SimpleTestCase, TestCase, override_settings
from rest_framework.test import APIClient

from quantem.inference.specs import MODEL_SPECS
from quantem.registry import cache, catalogue, hf
from quantem.registry.tests.copy_gate import find_violations, walk_strings
from quantem.seg_core.model_errors import (
    MODEL_UNAVAILABLE_CLASS_NAMES,
    translate_model_error,
)

TEST_URLCONF = "quantem.registry.tests.urls"
PACK_ID = "quantem:mito"

#: Verbatim, from ``w0_verify_report.md`` F2: what a user saw on 2026-08-10.
#: Three defects in one sentence, and this file exists because none of them was
#: catchable by the gate that was supposed to catch them.
SHIPPED_STRING = (
    "quantem:er cannot run on this machine. Not installed yet. Install it from "
    "the Models screen or with `quantem models install <pack id, e.g. "
    "quantem:mito>` -- QuantEM downloads and verifies it from Hugging Face. "
    "Offline, download a QuantEM model release instead, unzip it, and install "
    "it with `quantem models install <the directory you unzipped the release "
    "into>`."
)

#: Also seen on a card in the same session, truncated by the layout.
SHIPPED_MODULE_STRING = (
    "(or, if the console script is not on your PATH: python -m quantem.registry.instal"
)


def _assert_clean(pairs, surface: str, *, user_supplied=()) -> None:
    """Fail with every offending string, not just the first.

    ``user_supplied`` lists values the caller handed the application in the
    request being checked -- the folder picked in "Install from a local
    folder". Quoting one back is the app answering the question it was asked;
    see :func:`copy_gate.find_violations`.
    """
    violations = [
        v
        for where, text in pairs
        for v in find_violations(text, where, user_supplied=user_supplied)
    ]
    if violations:
        report = "\n".join(f"  {v}" for v in violations)
        raise AssertionError(f"I-12: {len(violations)} shell/CLI defect(s) in {surface}:\n{report}")


def _assert_no_terminal_copy(pairs, surface: str) -> None:
    """The terminal register must never cross into a served string."""
    for where, text in pairs:
        for terminal in cache.TERMINAL_ONLY_COPY:
            assert terminal not in text, (
                f"I-12: {surface} serialises terminal-only copy at {where}: "
                f"{terminal!r} appears in {text!r}"
            )


# --- The detector itself -----------------------------------------------------


class DetectorTests(SimpleTestCase):
    """The predicate has to catch the real thing and leave English alone."""

    def test_the_gate_sees_the_string_that_shipped(self):
        kinds = {v.kind for v in find_violations(SHIPPED_STRING)}

        # The three defects called out in F2, each independently detected.
        assert "shell-command" in kinds, kinds
        assert "double-hyphen" in kinds, kinds
        assert "placeholder" in kinds, kinds

    def test_the_gate_sees_the_module_path_variant(self):
        kinds = {v.kind for v in find_violations(SHIPPED_MODULE_STRING)}

        assert "shell-command" in kinds, kinds
        assert "module-path" in kinds, kinds

    def test_the_gate_passes_the_copy_that_replaced_it(self):
        assert find_violations(cache.INSTALL_HINT) == []

    def test_the_gate_still_flags_the_terminal_copy(self):
        # Not a bug -- INSTALL_INSTRUCTIONS is *meant* to name commands. The
        # point is that the detector would catch it the moment it is served.
        assert find_violations(cache.INSTALL_INSTRUCTIONS)

    def test_it_does_not_flag_ordinary_product_prose(self):
        innocent = [
            "QuantEM downloads it and checks every file before anything runs.",
            "Mitochondria 62% \u00b7 531 of 858 tiles \u00b7 about 4 min",
            "quantem:mito cannot run on this machine.",
            "QuantEM \u2014 Mitochondria",
            "Encoder and head trained by the Arrojo e Drigo Lab on the QuantEM EM corpus.",
            "This run found 0 objects at include level 0.50.",
            "Could not reach the QuantEM model repository "
            "(https://huggingface.co/ArrojoeDrigoLab/quantem).",
        ]
        for text in innocent:
            assert find_violations(text) == [], text

    def test_walk_strings_ignores_keys_and_finds_nested_values(self):
        body = {"packs": [{"reason": "bad -- copy", "download_bytes": 1}]}
        found = list(walk_strings(body))

        assert found == [("$.packs[0].reason", "bad -- copy")]


# --- The catalogue: every branch of "why can this pack not run" --------------


class _FakeCache:
    """A models root under a tmp dir, with packs built file by file.

    Mirrors ``test_catalogue._FakeCache``: the developer box has all eight packs
    installed and exported, so the only way to reach the interesting ``reason``
    strings is to build broken packs by hand.
    """

    def __init__(self, root: Path):
        self.root = root

    def install(self, pack_id: str, *, index: dict | None = None, exported: bool = False):
        pack = self.root / "packs" / pack_id.replace(":", "__")
        pack.mkdir(parents=True, exist_ok=True)
        (pack / cache.HEAD_NAME).write_bytes(b"head")
        (pack / cache.CONFIG_NAME).write_text("{}", encoding="utf-8")
        (pack / cache.RECORD_NAME).write_text(
            json.dumps({"pack_id": pack_id, "head": {"sha256": "x"}}), encoding="utf-8"
        )
        if index is not None:
            (pack / cache.INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")
        if exported:
            (pack / cache.EXPORTED_ENCODER_NAME).write_bytes(b"ts")
        return pack


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    fake = _FakeCache(tmp_path / "models")
    monkeypatch.setattr(cache, "models_root", lambda: fake.root)
    return fake


def _reasons(fake_cache, monkeypatch) -> list[tuple[str, str]]:
    """Every ``reason`` ``probe_runnable`` can produce, labelled by branch."""
    out: list[tuple[str, str]] = []

    def reason(label: str, pack_id: str = PACK_ID) -> None:
        text = catalogue.probe_runnable(pack_id).reason
        assert text, f"{label} produced no reason"
        out.append((f"probe_runnable[{label}]", text))

    # 1. no torch at all.
    monkeypatch.setattr(catalogue, "_module_available", lambda name: name != "torch")
    reason("no-torch")
    monkeypatch.undo()
    monkeypatch.setattr(cache, "models_root", lambda: fake_cache.root)

    # 2. not installed -- the F2 branch, reached by every fresh machine.
    reason("not-installed")

    # 3. unknown pack id.
    out.append(("probe_runnable[unknown]", catalogue.probe_runnable("nope:nope").reason or ""))

    # 4. installed, no export, no checkpoint index.
    fake_cache.install(PACK_ID)
    reason("no-index")

    # 5. installed, index naming a framework QuantEM cannot build.
    fake_cache.install(PACK_ID, index={"encoder": {"framework": "jax_magic"}})
    reason("unknown-framework")

    # 6. QuantEM family, no export, and neither builder present. A DINOv3
    #    index alone is no longer a blocker -- the engine renames the tensors
    #    and builds through timm -- so both have to be absent to reach this.
    fake_cache.install(PACK_ID, index={"encoder": {"framework": "dinov3"}})
    monkeypatch.setattr(catalogue, "_dinov3_available", lambda: False)
    monkeypatch.setattr(catalogue, "_module_available", lambda name: name == "torch")
    reason("no-encoder-builder")

    # 7. OmniEM family, no export, timm absent.
    fake_cache.install("omniem:mito", index={"encoder": {"framework": "timm_vit"}})
    reason("timm-missing", "omniem:mito")

    return out


def test_no_probe_reason_tells_a_clicker_to_open_a_terminal(fake_cache, monkeypatch):
    pairs = _reasons(fake_cache, monkeypatch)

    assert len(pairs) == 7, [p[0] for p in pairs]
    _assert_clean(pairs, "catalogue.probe_runnable")
    _assert_no_terminal_copy(pairs, "catalogue.probe_runnable")


@pytest.mark.django_db
def test_the_whole_models_body_is_clean_on_a_machine_with_nothing_installed(
    fake_cache, monkeypatch
):
    """The exact machine state F2 was found on: no packs, all eight listed."""
    body = catalogue.catalogue()
    pairs = list(walk_strings(body, "$"))

    assert [p["id"] for p in body["packs"]] == sorted(MODEL_SPECS)
    assert all(p["reason"] for p in body["packs"]), "a blocked pack with no reason"
    _assert_clean(pairs, "GET /api/models/ (nothing installed)")
    _assert_no_terminal_copy(pairs, "GET /api/models/ (nothing installed)")


# --- The model-error translator: what lands in status_error ------------------


class _Fake(Exception):
    """Stands in for a model-layer exception, by class name."""


def _unavailable(name: str) -> Exception:
    maintainer_text = (
        "Set QUANTEM_DINOV3_PATH to a checkout of "
        "github.com/facebookresearch/dinov3 and run "
        "python -m quantem.inference.export quantem:mito --all"
    )
    return type(name, (_Fake,), {})(maintainer_text)


def test_status_error_never_carries_the_maintainers_command():
    """The labeling header renders this verbatim. It used to be terminal copy."""
    pairs = [
        (
            f"translate_model_error[{name}]",
            translate_model_error(_unavailable(name), pack_id=PACK_ID),
        )
        for name in sorted(MODEL_UNAVAILABLE_CLASS_NAMES)
    ]

    assert len(pairs) == 5, pairs
    for _, message in pairs:
        assert PACK_ID in message
    _assert_clean(pairs, "seg_core.translate_model_error")
    _assert_no_terminal_copy(pairs, "seg_core.translate_model_error")


def test_a_run_that_failed_for_another_reason_keeps_its_own_words():
    """Not I-12's business: only availability errors are rewritten."""
    message = translate_model_error(ValueError("tile 3 is empty"), pack_id=PACK_ID)

    assert message == "tile 3 is empty"


def test_the_pack_not_installed_exception_is_app_copy(tmp_path, monkeypatch):
    """``resolve_pack`` raises straight into ``status_error``."""
    monkeypatch.setattr(cache, "models_root", lambda: tmp_path / "models")

    with pytest.raises(cache.PackNotInstalled) as excinfo:
        cache.resolve_pack(PACK_ID)

    _assert_clean([("resolve_pack", str(excinfo.value))], "cache.resolve_pack")
    _assert_no_terminal_copy([("resolve_pack", str(excinfo.value))], "cache.resolve_pack")


def test_the_offline_download_error_is_app_copy():
    """Shown verbatim on the Models screen when a download cannot start."""
    message = str(hf._offline_error("the model card", OSError("no route to host")))

    _assert_clean([("hf._offline_error", message)], "registry.hf")
    _assert_no_terminal_copy([("hf._offline_error", message)], "registry.hf")
    assert "Models screen" in message


# --- The endpoints, serialised for real --------------------------------------


@override_settings(ROOT_URLCONF=TEST_URLCONF)
class ServedResponseTests(TestCase):
    """Real requests, real response bodies, every string in them checked."""

    def setUp(self):
        self.client = APIClient()

    def _check(self, body, surface: str, *, user_supplied=()) -> None:
        pairs = list(walk_strings(body, "$"))
        assert pairs, f"{surface} serialised no strings at all"
        _assert_clean(pairs, surface, user_supplied=user_supplied)
        _assert_no_terminal_copy(pairs, surface)

    def test_model_list(self):
        self._check(self.client.get("/api/models/").json(), "GET /api/models/")

    def test_unknown_pack(self):
        response = self.client.post("/api/models/nope:nope/install/", {}, format="json")

        assert response.status_code == 404
        self._check(response.json(), "POST install (unknown pack)")

    def test_source_path_is_not_a_directory(self):
        response = self.client.post(
            f"/api/models/{PACK_ID}/install/",
            {"source_path": "Z:/not/a/real/place"},
            format="json",
        )

        assert response.status_code == 400
        self._check(
            response.json(),
            "POST install (bad source_path)",
            user_supplied=["Z:/not/a/real/place"],
        )

    def test_a_directory_that_is_neither_a_release_nor_a_model(self):
        import tempfile

        with tempfile.TemporaryDirectory() as empty:
            response = self.client.post(
                f"/api/models/{PACK_ID}/install/",
                {"source_path": empty},
                format="json",
            )

        assert response.status_code == 400
        self._check(
            response.json(),
            "POST install (empty directory)",
            user_supplied=[empty],
        )

    def test_an_install_already_in_flight(self):
        """The 409 body, without writing a Job row.

        The job row is faked rather than created: this gate is about copy, and
        it must not go red because someone else's unreleased migration is
        halfway through the tree.
        """

        class _Job:
            id = "6f1d4c2e-0000-4000-8000-000000000001"
            status = "RUNNING"
            progress_current_bytes = 1024
            progress_total_bytes = 4096

        with patch.object(catalogue, "active_install_job", lambda pack_id: _Job()):
            response = self.client.post(f"/api/models/{PACK_ID}/install/", {}, format="json")

        assert response.status_code == 409
        self._check(response.json(), "POST install (409 conflict)")


# --- Installing from a folder: the offline route the app copy now points at ---


def test_the_offline_install_route_fails_in_app_copy(tmp_path, monkeypatch):
    """INSTALL_HINT sends people to "Install from a local folder"; its refusals
    have to be app copy too, or the fix just moves the shell command one click
    further in. Every message here is shown verbatim on the Models screen.
    """
    from quantem.registry import install, release

    monkeypatch.setattr(cache, "models_root", lambda: tmp_path / "models")
    pairs: list[tuple[str, str]] = []

    # Pointed at a folder that is not a bundle, at the zip, and at packs/.
    empty = tmp_path / "Downloads"
    empty.mkdir()
    with pytest.raises(release.BundleError) as exc:
        release.read_bundle(empty)
    pairs.append(("read_bundle[not-a-bundle]", str(exc.value)))

    zipped = tmp_path / "quantem-models-0.1.0.zip"
    zipped.write_bytes(b"PK")
    with pytest.raises(release.BundleError) as exc:
        release.read_bundle(zipped)
    pairs.append(("read_bundle[still-zipped]", str(exc.value)))

    bundle = tmp_path / "quantem-models-0.1.0"
    (bundle / release.PACKS_DIRNAME).mkdir(parents=True)
    (bundle / release.MANIFEST_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(release.BundleError) as exc:
        release.read_bundle(bundle / release.PACKS_DIRNAME)
    pairs.append(("read_bundle[packs-subdir]", str(exc.value)))

    with pytest.raises(release.BundleError) as exc:
        release.read_bundle(bundle)  # kind/schema not declared
    pairs.append(("read_bundle[bad-manifest]", str(exc.value)))

    with pytest.raises(install.InstallError) as exc:
        install.install_pack_from_bundle(PACK_ID, empty)
    pairs.append(("install_pack_from_bundle[not-a-bundle]", str(exc.value)))

    with pytest.raises(install.InstallError) as exc:
        install.install_pack_from_bundle("nope:nope", bundle)
    pairs.append(("install_pack_from_bundle[unknown-pack]", str(exc.value)))

    # Every path in these messages is the folder the caller pointed at, which
    # is the folder the user picked; quoting it back is the whole point of the
    # message. Nothing else absolute is allowed through.
    chosen = [str(empty), str(zipped), str(bundle), str(bundle / release.PACKS_DIRNAME)]
    _assert_clean(pairs, "the local-folder install route", user_supplied=chosen)
    _assert_no_terminal_copy(pairs, "the local-folder install route")


# --- The other half of the old acceptance, kept honest ------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_the_frontend_does_not_hardcode_terminal_copy():
    """The bundle grep's half of I-12, done on source so it needs no build.

    Checking source rather than ``dist/`` is not a weakening: a string that is
    not in the source cannot be in the bundle, and this runs on every developer
    machine instead of only after ``vite build``.
    """
    src = _repo_root() / "frontend" / "src"
    assert src.is_dir(), src

    offenders: list[str] = []
    for path in src.rglob("*.ts*"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for terminal in cache.TERMINAL_ONLY_COPY:
            if terminal in text:
                offenders.append(f"{path.relative_to(src)}: {terminal!r}")

    assert not offenders, "I-12: terminal copy hardcoded in the frontend:\n" + "\n".join(offenders)
