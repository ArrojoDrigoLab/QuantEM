"""The release-bundle format, its installer, and the CLI that drives them.

These tests are about **packaging**, so they use a synthetic bundle rather than
the real one: the properties under test -- that a tampered file is refused, that
an incomplete bundle is refused, that a stranger's invocation works -- must hold
for bytes of any kind, and asserting them against 7 GB of real weights would
make them a test that this machine has the weights.

The one thing they cannot show is that a real export loads and segments. That is
:mod:`quantem.inference.tests.test_real_models`, and it is marked
``requires_weights`` for the same reason.

Background: the packaging these tests pin down replaces one where the only
documented way to obtain the models was
``python -m quantem.registry.install local --all``, whose ``--heads-root``
default was a path on one developer's computer -- so a bundle assembled from it
installed on someone else's machine and then could not run, because the exported
encoder was never in it. Hence :func:`test_a_bundle_without_an_export_is_refused`
and :func:`test_installed_bundle_pack_is_runnable`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from quantem.cli import _resolve_data_dir, build_parser, default_data_dir
from quantem.registry import cache, catalogue, install, release

PACK_ID = "quantem:mito"
OTHER_PACK_ID = "omniem:er"

#: Enough of a checkpoint index for `catalogue.probe_runnable` to read the
#: framework off it. Never parsed as a real manifest here.
FAKE_INDEX = {"encoder": {"framework": "dinov3"}}

#: The encoder run the synthetic packs claim, echoed in their descriptors.
RUN_DIR = "foundation_weights/m1_dinov3_vitb"
RUN_ID = "m1_dinov3_vitb"
STEP = 674999

#: A ``checkpoint_index.json`` exactly as the research tree writes one: every
#: checkpoint addressed by an absolute path on the machine that trained it, one
#: of them through a **UNC share, which names a host**. Transcribed from the
#: real quantem index rather than invented, because this is the file that was
#: shipped verbatim to Hugging Face in eight copies.
DIRTY_INDEX = {
    "schema_version": 1,
    "checkpoints": [
        {
            "kind": "teacher",
            "path": r"\\EXAMPLEHOST\share\checkpoints\m1_teacher_24999.pth",
            "sha256": None,
            "step": 24999,
        },
        {
            "kind": "teacher",
            "path": "/mnt/d/example/checkpoints/m1_teacher_674999.pth",
            "sha256": None,
            "step": STEP,
        },
    ],
    "encoder": {
        "arch": "vit_base",
        "depth": 12,
        "embedding_dim": 768,
        "patch_size": 16,
        "framework": "dinov3",
        "image_mean": [0.583175],
        "image_std": [0.244468],
        "input_channels": 1,
        "run_id": "M1_dinov3_vitb_512",
        "notes": "Final = m1_teacher_674999. Index synthesised from the V: mirror.",
    },
}

#: And the config beside it: the training data root on a lab drive, and the
#: experiment YAML's path inside the training container.
DIRTY_CONFIG = f"""\
name: F4v2_qem_cem
encoder:
  run_dir: /root/dino/{RUN_DIR}
  checkpoint_step: {STEP}
  adapt: last_n
neck:
  type: naive_1x1
decoder:
  type: affinity_mws
data:
  data_root: V:\\example\\seg_data
  num_classes: 2
config_path: /root/dino/fig3/configs/experiments/FIG4_MITO_V2/F4v2_qem_cem.yaml
"""


# --- Fixtures ---------------------------------------------------------------


def _write_bundle(
    root: Path,
    *,
    pack_ids: tuple[str, ...] = (PACK_ID,),
    with_export: bool = True,
    release_name: str = "9.9.9",
    config: str = "neck: naive_1x1\n",
    encoder_in_descriptor: bool = True,
) -> Path:
    """Hand-write a bundle whose files are tiny but whose manifest is real.

    Written by hand rather than by :func:`quantem.registry.release.build_bundle`
    because the builder needs torch and real weights, and because a reader that
    only ever parses its own writer's output is not tested against the format --
    it is tested against itself.
    """
    root.mkdir(parents=True, exist_ok=True)
    files: list[dict] = []
    packs: list[dict] = []

    for pack_id in pack_ids:
        dirname = cache.pack_dirname(pack_id)
        pack_dir = root / release.PACKS_DIRNAME / dirname
        pack_dir.mkdir(parents=True, exist_ok=True)

        contents: dict[str, bytes] = {
            cache.HEAD_NAME: f"head of {pack_id}".encode(),
            cache.CONFIG_NAME: config.encode(),
            cache.INDEX_NAME: json.dumps(FAKE_INDEX).encode(),
        }
        roles = {"head": cache.HEAD_NAME, "config": cache.CONFIG_NAME,
                 "index": cache.INDEX_NAME}
        if with_export:
            contents[cache.EXPORTED_ENCODER_NAME] = f"torchscript of {pack_id}".encode()
            roles["export"] = cache.EXPORTED_ENCODER_NAME

        descriptor = {
            "schema_version": release.BUNDLE_SCHEMA_VERSION,
            "kind": release.BUNDLE_KIND,
            "pack_id": pack_id,
            "release": release_name,
            "licence": "see NOTICE",
        }
        if encoder_in_descriptor:
            descriptor["encoder"] = {
                "run_dir": RUN_DIR,
                "run_id": RUN_ID,
                "checkpoint_step": STEP,
            }
        contents[release.PACK_DESCRIPTOR_NAME] = (
            json.dumps(descriptor, indent=2) + "\n"
        ).encode()

        for name, blob in contents.items():
            (pack_dir / name).write_bytes(blob)
            files.append(
                {
                    "path": f"{release.PACKS_DIRNAME}/{dirname}/{name}",
                    "sha256": cache.sha256_file(pack_dir / name),
                    "size_bytes": len(blob),
                }
            )
        packs.append(
            {
                "pack_id": pack_id,
                "dir": f"{release.PACKS_DIRNAME}/{dirname}",
                "files": roles,
                "architecture": {"tile": 512},
            }
        )

    manifest = {
        "schema_version": release.BUNDLE_SCHEMA_VERSION,
        "kind": release.BUNDLE_KIND,
        "release": release_name,
        "generated_at": "2026-01-01T00:00:00+0000",
        "generated_by": {"python": "3.13.0"},
        "packs": packs,
        "files": sorted(files, key=lambda f: f["path"]),
        "total_bytes": sum(f["size_bytes"] for f in files),
    }
    (root / release.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return root


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    return _write_bundle(tmp_path / "quantem-models-9.9.9")


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh, empty model cache, as a machine that has never installed one."""
    models = tmp_path / "data" / "models"
    monkeypatch.setattr(cache, "models_root", lambda: models)
    return models


# --- What may not ship ------------------------------------------------------
#
# The defect these pin down: `release.py` promised, in the README it writes into
# every bundle, that a release "does not refer to the machine that built it",
# and then copied `checkpoint_index.json` and `resolved_config.yaml` in
# verbatim. Eight copies of a lab file server's UNC name, a WSL mount and two
# drive letters went to Hugging Face and Zenodo under the author's real name.


def test_the_shipped_index_keeps_the_step_and_loses_the_machine() -> None:
    """What identifies the encoder stays; what identifies the build box goes."""
    shipped = release.sanitise_checkpoint_index(json.dumps(DIRTY_INDEX))

    assert release.find_local_paths(shipped) == []
    for leak in ("EXAMPLEHOST", "share", "/mnt/", "example", "V:"):
        assert leak not in shipped, leak

    parsed = json.loads(shipped)
    # The two facts a user or a rebuild actually needs.
    assert parsed["encoder"]["run_id"] == "M1_dinov3_vitb_512"
    assert [c["step"] for c in parsed["checkpoints"]] == [24999, STEP]
    # The checkpoint is still named -- by the filename, which carries the step.
    assert [c["path"] for c in parsed["checkpoints"]] == [
        "m1_teacher_24999.pth",
        "m1_teacher_674999.pth",
    ]
    # And the architecture, which is why the index is shipped at all.
    assert parsed["encoder"]["framework"] == "dinov3"


def test_the_shipped_config_keeps_the_encoder_run_and_loses_the_data_root() -> None:
    shipped = release.sanitise_resolved_config(DIRTY_CONFIG)

    assert release.find_local_paths(shipped) == []
    assert "seg_data" not in shipped
    assert "/root/dino" not in shipped

    parsed = yaml.safe_load(shipped)
    assert parsed["data"]["num_classes"] == 2
    assert "data_root" not in parsed["data"]
    assert "config_path" not in parsed
    # An absolute run dir is reduced to the run's name, not blanked: the run is
    # the one thing the field is for.
    assert parsed["encoder"]["run_dir"] == RUN_ID
    assert parsed["encoder"]["checkpoint_step"] == STEP


def test_a_relative_run_dir_ships_unchanged() -> None:
    """Half the released configs name the run relatively; that is not a path."""
    config = DIRTY_CONFIG.replace(f"/root/dino/{RUN_DIR}", RUN_DIR)
    parsed = yaml.safe_load(release.sanitise_resolved_config(config))
    assert parsed["encoder"]["run_dir"] == RUN_DIR


def test_the_config_is_otherwise_read_exactly_as_before() -> None:
    """Sanitising must not change what the head is rebuilt into."""
    from quantem.inference._fig3.schema import HeadConfig

    before = HeadConfig.from_dict(yaml.safe_load(DIRTY_CONFIG))
    after = HeadConfig.from_dict(yaml.safe_load(release.sanitise_resolved_config(DIRTY_CONFIG)))
    before.encoder.run_dir = after.encoder.run_dir  # the one intended difference
    assert before == after


@pytest.mark.parametrize("role", release.SANITISED_ROLES)
def test_sanitising_twice_changes_nothing(role: str) -> None:
    """``--skip-existing`` re-sanitises a pack directory an earlier run left.

    A resumed build has to produce the same bytes as a fresh one, so the
    rewriting has to be a fixed point.
    """
    source = json.dumps(DIRTY_INDEX) if role == "index" else DIRTY_CONFIG
    once = release.sanitise_pack_file(role, source)
    assert release.sanitise_pack_file(role, once) == once


def test_redacting_removes_exactly_what_scanning_looks_for() -> None:
    """The property the build gate rests on, over the shapes that turned up."""
    for text in (
        r"\\EXAMPLEHOST\share\checkpoints\m1_teacher_674999.pth",
        "/mnt/d/example/training/foundation_weights/backbone.pt",
        r"V:\example\seg_data",
        "/root/dino/fig3/configs/experiments/FIG4/F4_omni_ld.yaml",
        "D:/example/legacy/storage/models",
    ):
        assert release.find_local_paths(text), text
        assert release.find_local_paths(release.redact_local_paths(text)) == []


def test_what_is_not_a_local_path_survives() -> None:
    """Over-redaction would quietly destroy provenance, so it is pinned too."""
    for text in (
        "foundation_weights/m1_dinov3_vitb",   # a run dir, relative
        "m1_teacher_674999.pth",               # a checkpoint, by name
        "packs/quantem__mito/head.pt",         # a path inside the bundle
        "bioRxiv 10.1101/2025.04.13.648639",   # a DOI
        "https://doi.org/10.1101/2025.04.13.648639",
        "quantem:mito",
    ):
        assert release.redact_local_paths(text) == text, text


def test_scan_passes_on_a_clean_bundle(bundle: Path) -> None:
    assert release.scan_bundle_for_local_paths(bundle) == {}


def test_scan_finds_a_pack_file_that_was_shipped_verbatim(bundle: Path) -> None:
    """The gate that would have caught the original defect."""
    pack = bundle / release.PACKS_DIRNAME / "quantem__mito"
    (pack / cache.INDEX_NAME).write_text(json.dumps(DIRTY_INDEX), encoding="utf-8")
    (pack / cache.CONFIG_NAME).write_text(DIRTY_CONFIG, encoding="utf-8")

    offenders = release.scan_bundle_for_local_paths(bundle)
    assert set(offenders) == {
        f"packs/quantem__mito/{cache.INDEX_NAME}",
        f"packs/quantem__mito/{cache.CONFIG_NAME}",
    }
    hits = " ".join(offenders[f"packs/quantem__mito/{cache.INDEX_NAME}"])
    assert "EXAMPLEHOST" in hits
    assert "/mnt/d/example" in hits


def test_scan_ignores_path_shaped_float_noise_in_a_weight_file(bundle: Path) -> None:
    """A weight file is mostly float noise, and noise spells things.

    All five of these came out of a real 341 MB ``encoder_ts.pt`` on the first
    run of this gate. A check that fails on every clean bundle is a check
    somebody switches off, so the binary rules require every segment to look
    like a directory name and at least two of them.
    """
    head = bundle / release.PACKS_DIRNAME / "quantem__mito" / cache.HEAD_NAME
    noise = b"H:/(v=)J> L:\\z{=%0 n:/F$=@Z V:/aw:!ez /C/>`BO= b:\\a}<{%d<a"
    head.write_bytes(b"\x00\x91" + noise + b"\x00\x02")
    assert release.scan_bundle_for_local_paths(bundle) == {}

    # A real path in the same file is still found: what separates them is the
    # shape of the match, not the file's extension.
    head.write_bytes(b"\x00\x91" + rb"D:\example\legacy\head.pt" + b"\x00\x02")
    assert release.scan_bundle_for_local_paths(bundle)


def test_scan_ignores_the_archives_own_zip_members(bundle: Path) -> None:
    """A TorchScript file is a zip, and its member names look like paths.

    ``.../constants/255PK\\x01\\x02...\\x99/encoder_ts/constants/256PK`` is the
    central directory of a real ``encoder_ts.pt``: relative member names, one of
    them preceded by a byte that happens to be ``/``. Four of the eight packs
    tripped the gate on this before the binary rule required an absolute path to
    start where a filesystem starts.
    """
    export = bundle / release.PACKS_DIRNAME / "quantem__mito" / cache.EXPORTED_ENCODER_NAME
    export.write_bytes(
        b"\x10\x04\x99/encoder_ts/constants/256PK\x01\x02\x00\x00"
        b"\x90\x44\x99/encoder_ts/code/__torch__/quantem.pyPK\x01\x02"
    )
    assert release.scan_bundle_for_local_paths(bundle) == {}

    export.write_bytes(b"\x99/mnt/d/example/legacy/m1_teacher_674999.pth\x00")
    assert release.scan_bundle_for_local_paths(bundle)


def test_scan_reads_the_cli_and_reports_an_exit_code(bundle: Path) -> None:
    assert release.main(["scan", str(bundle)]) == 0
    pack = bundle / release.PACKS_DIRNAME / "quantem__mito"
    (pack / cache.CONFIG_NAME).write_text(DIRTY_CONFIG, encoding="utf-8")
    assert release.main(["scan", str(bundle)]) == 1


# --- Reading a bundle -------------------------------------------------------


def test_read_bundle_reports_its_packs_and_files(bundle: Path) -> None:
    parsed = release.read_bundle(bundle)
    assert parsed.pack_ids == [PACK_ID]
    assert parsed.release == "9.9.9"
    assert {f.path for f in parsed.files} == {
        f"packs/quantem__mito/{name}"
        for name in (
            cache.HEAD_NAME,
            cache.CONFIG_NAME,
            cache.INDEX_NAME,
            cache.EXPORTED_ENCODER_NAME,
            release.PACK_DESCRIPTOR_NAME,
        )
    }


def test_a_directory_that_is_not_a_bundle_says_so(tmp_path: Path) -> None:
    with pytest.raises(release.BundleError, match=release.MANIFEST_NAME):
        release.read_bundle(tmp_path)


def test_pointing_at_the_packs_subdirectory_is_redirected(bundle: Path) -> None:
    """The commonest wrong directory gets the right one named back."""
    with pytest.raises(release.BundleError, match=r"one level up"):
        release.read_bundle(bundle / release.PACKS_DIRNAME)


def test_pointing_at_the_parent_of_a_bundle_is_redirected(bundle: Path) -> None:
    with pytest.raises(release.BundleError, match=r"Did you mean"):
        release.read_bundle(bundle.parent)


def test_a_future_schema_version_is_refused(bundle: Path) -> None:
    manifest_path = bundle / release.MANIFEST_NAME
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["schema_version"] = release.BUNDLE_SCHEMA_VERSION + 1
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(release.BundleError, match="Upgrade QuantEM"):
        release.read_bundle(bundle)


def test_verify_bundle_passes_on_an_untouched_bundle(bundle: Path) -> None:
    results = release.verify_bundle(bundle)
    assert results and all(results.values())


def test_verify_bundle_catches_a_changed_byte(bundle: Path) -> None:
    head = bundle / release.PACKS_DIRNAME / "quantem__mito" / cache.HEAD_NAME
    head.write_bytes(head.read_bytes() + b"!")
    results = release.verify_bundle(bundle)
    assert results[f"packs/quantem__mito/{cache.HEAD_NAME}"] is False
    assert results[f"packs/quantem__mito/{cache.EXPORTED_ENCODER_NAME}"] is True


# --- Installing from a bundle -----------------------------------------------


def test_install_copies_every_file_and_records_the_release(
    bundle: Path, data_dir: Path
) -> None:
    installed = install.install_pack_from_bundle(PACK_ID, bundle)

    pack_root = cache.pack_dir(PACK_ID)
    assert installed.root == pack_root
    for name in (cache.HEAD_NAME, cache.CONFIG_NAME, cache.INDEX_NAME,
                 cache.EXPORTED_ENCODER_NAME, cache.RECORD_NAME):
        assert (pack_root / name).is_file(), name

    record = cache.read_record(PACK_ID)
    assert record is not None
    assert record["source"] == "release-bundle"
    assert record["release"] == "9.9.9"
    assert "verified-against-release-manifest" in record["digest_origin"]
    # The publisher's own descriptor is carried through rather than re-derived.
    assert record["release_descriptor"]["licence"] == "see NOTICE"


def test_install_needs_no_encoder_blob(bundle: Path, data_dir: Path) -> None:
    """A bundle ships no ``encoder.pth`` and the install must not want one.

    The exported encoder makes the raw foundation weights dead weight at run
    time, so recording an encoder that is not there would make every pack in a
    bundle report itself broken.
    """
    install.install_pack_from_bundle(PACK_ID, bundle)
    record = cache.read_record(PACK_ID)
    assert record is not None
    assert not (cache.pack_dir(PACK_ID) / cache.ENCODER_NAME).exists()
    assert record.get("encoder") is None
    assert cache.installed(PACK_ID)
    assert cache.resolve_pack(PACK_ID).encoder_path is None


def test_install_records_which_encoder_run_and_step_the_head_came_from(
    bundle: Path, data_dir: Path
) -> None:
    """``quantem.analysis.provenance`` reads these two keys off the record.

    A local-path install has the training config and writes them from it. A
    bundle install cannot read them off the shipped ``checkpoint_index.json``
    any more -- that file no longer names a path -- so the publisher states them
    in the pack descriptor and they are carried through from there.
    """
    install.install_pack_from_bundle(PACK_ID, bundle)
    record = cache.read_record(PACK_ID)
    assert record is not None
    assert record["encoder_run_dir"] == RUN_DIR
    assert record["checkpoint_step"] == STEP


def test_encoder_provenance_falls_back_to_the_installed_config(
    tmp_path: Path, data_dir: Path
) -> None:
    """A bundle whose descriptor predates the ``encoder`` block still reports."""
    bundle = _write_bundle(tmp_path / "older", config=DIRTY_CONFIG, encoder_in_descriptor=False)
    install.install_pack_from_bundle(PACK_ID, bundle)
    record = cache.read_record(PACK_ID)
    assert record is not None
    assert record["encoder_run_dir"] == RUN_ID
    assert record["checkpoint_step"] == STEP


def test_installed_bundle_pack_is_runnable(bundle: Path, data_dir: Path) -> None:
    """The whole point: installed *and* runnable, with no dinov3 anywhere."""
    install.install_pack_from_bundle(PACK_ID, bundle)
    entry = catalogue.pack_entry(PACK_ID)
    assert entry["installed"] is True
    assert entry["runnable"] is True, entry["reason"]
    assert entry["encoder_tier"] == "exported"


def test_a_tampered_file_aborts_the_install(bundle: Path, data_dir: Path) -> None:
    export = bundle / release.PACKS_DIRNAME / "quantem__mito" / cache.EXPORTED_ENCODER_NAME
    export.write_bytes(b"not what the publisher hashed")

    with pytest.raises(install.InstallError, match="does not match the release manifest"):
        install.install_pack_from_bundle(PACK_ID, bundle)
    # No install record, so nothing thinks this pack is usable.
    assert not cache.installed(PACK_ID)


def test_a_missing_file_aborts_the_install(bundle: Path, data_dir: Path) -> None:
    (bundle / release.PACKS_DIRNAME / "quantem__mito" / cache.HEAD_NAME).unlink()
    with pytest.raises(install.InstallError, match="download is incomplete"):
        install.install_pack_from_bundle(PACK_ID, bundle)
    assert not cache.installed(PACK_ID)


def test_a_bundle_without_an_export_is_refused(tmp_path: Path, data_dir: Path) -> None:
    """Installing a pack that cannot then run is the bug this format exists for."""
    bundle = _write_bundle(tmp_path / "no-export", with_export=False)
    with pytest.raises(install.InstallError, match=r"\['export'\]"):
        install.install_pack_from_bundle(PACK_ID, bundle)
    assert not cache.installed(PACK_ID)


def test_install_all_reports_a_pack_the_bundle_does_not_have(
    bundle: Path, data_dir: Path
) -> None:
    with pytest.raises(install.InstallError, match=OTHER_PACK_ID):
        install.install_all_from_bundle(bundle, pack_ids=[OTHER_PACK_ID])


def test_install_all_takes_every_pack_in_the_bundle(
    tmp_path: Path, data_dir: Path
) -> None:
    bundle = _write_bundle(tmp_path / "two", pack_ids=(PACK_ID, OTHER_PACK_ID))
    results = install.install_all_from_bundle(bundle)
    assert [r.pack_id for r in results] == [PACK_ID, OTHER_PACK_ID]
    assert cache.installed_packs() == sorted([PACK_ID, OTHER_PACK_ID])


def test_reinstall_is_a_no_op_without_force(bundle: Path, data_dir: Path) -> None:
    first = install.install_pack_from_bundle(PACK_ID, bundle)
    again = install.install_pack_from_bundle(PACK_ID, bundle)
    assert first.bytes_written > 0
    assert again.bytes_written == 0


def test_verify_pack_passes_after_a_bundle_install(bundle: Path, data_dir: Path) -> None:
    install.install_pack_from_bundle(PACK_ID, bundle)
    results = cache.verify_pack(PACK_ID)
    assert results and all(results.values()), results


# --- Installing from a directory of files -----------------------------------


def _loose_pack(root: Path, *, with_export: bool, with_index: bool = False) -> Path:
    """A directory of files, shaped like an unpacked pack or a training output."""
    root.mkdir(parents=True, exist_ok=True)
    (root / cache.HEAD_NAME).write_bytes(b"head")
    (root / cache.CONFIG_NAME).write_text(DIRTY_CONFIG, encoding="utf-8")
    if with_export:
        (root / cache.EXPORTED_ENCODER_NAME).write_bytes(b"torchscript")
    if with_index:
        (root / cache.INDEX_NAME).write_text(json.dumps(DIRTY_INDEX), encoding="utf-8")
    return root


def test_a_directory_with_an_exported_encoder_needs_nothing_else(
    tmp_path: Path, data_dir: Path
) -> None:
    """``encoder_ts.pt`` is the whole encoder; asking for more refuses a pack
    that can already run.

    This is the shape of a pack directory inside a release, and it used to fail
    with "encoder checkpoint for step=674999 not found ... Pass --search-dir" --
    sending the user after a research checkpoint they have never seen, past a
    file sitting in the same directory that made it unnecessary.
    """
    source = _loose_pack(tmp_path / "quantem__mito", with_export=True)
    installed = install.install_pack_from_path(PACK_ID, source)

    assert installed.encoder_sha256 is None
    assert (cache.pack_dir(PACK_ID) / cache.EXPORTED_ENCODER_NAME).is_file()
    assert not (cache.pack_dir(PACK_ID) / cache.ENCODER_NAME).exists()
    assert cache.installed(PACK_ID)
    assert catalogue.pack_entry(PACK_ID)["runnable"] is True

    record = cache.read_record(PACK_ID)
    assert record is not None
    assert record["encoder_run_dir"] == RUN_ID
    assert record["checkpoint_step"] == STEP


def test_an_index_pointing_at_a_missing_checkpoint_is_survivable_with_an_export(
    tmp_path: Path, data_dir: Path
) -> None:
    """A pack directory out of a bundle carries both; the index is provenance."""
    source = _loose_pack(tmp_path / "quantem__mito", with_export=True, with_index=True)
    install.install_pack_from_path(PACK_ID, source)
    assert (cache.pack_dir(PACK_ID) / cache.INDEX_NAME).is_file()
    assert cache.resolve_pack(PACK_ID).encoder_path is None


def test_a_directory_with_neither_names_the_shapes_that_would_work(
    tmp_path: Path, data_dir: Path
) -> None:
    source = _loose_pack(tmp_path / "mito_quantem", with_export=False)
    with pytest.raises(install.InstallError) as excinfo:
        install.install_pack_from_path(PACK_ID, source)

    message = str(excinfo.value)
    # Every path shape it names is one the user can actually look at.
    assert cache.EXPORTED_ENCODER_NAME in message
    assert release.MANIFEST_NAME in message
    assert f"{release.PACKS_DIRNAME}/quantem__mito/" in message
    # And nothing it names is a flag that does not exist where the user is.
    assert "--search-dir" not in message


# --- The advice users are given ---------------------------------------------


def test_no_user_facing_command_names_the_old_developer_only_one() -> None:
    """The install advice must be runnable on a machine that is not the lab's.

    ``install local`` needs ``--heads-root``/``--weights-root``, which used to
    default to one developer's drive. Any string that still sends a user there
    is the original defect.
    """
    strings = [
        cache.INSTALL_HINT,
        cache.INSTALL_INSTRUCTIONS,
        cache.INSTALL_COMMAND,
        cache.INSTALL_COMMAND_MODULE,
        catalogue.probe_runnable(PACK_ID).reason or "",
    ]
    for text in strings:
        assert "install local" not in text, text
        assert "heads-root" not in text, text
    assert "models install" in cache.INSTALL_COMMAND
    assert "install bundle" in cache.INSTALL_COMMAND_MODULE


def test_not_installed_error_names_a_way_out_that_works(data_dir: Path) -> None:
    """And it is the app's way out, not the terminal's.

    ``resolve_pack``'s message is written into a segmentation's
    ``status_error``, which the labeling header renders verbatim, so it carries
    :data:`cache.INSTALL_HINT` and never the terminal copy (I-12).
    """
    with pytest.raises(cache.PackNotInstalled) as excinfo:
        cache.resolve_pack(PACK_ID)
    message = str(excinfo.value)
    assert cache.INSTALL_HINT in message
    for terminal in cache.TERMINAL_ONLY_COPY:
        assert terminal not in message


def test_install_local_has_no_built_in_roots() -> None:
    """The defaults that made the old command machine-specific are gone."""
    assert not hasattr(install, "DEV_HEADS_ROOT")
    assert not hasattr(install, "DEV_WEIGHTS_ROOT")
    with pytest.raises(TypeError):
        install.install_all_from_paths()  # type: ignore[call-arg]
    assert release.default_heads_root() is None or "QUANTEM_HEADS_ROOT" in (
        release.HEADS_ROOT_ENV_VAR
    )


# --- The command line -------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["--data-dir", "{d}", "serve"],
        ["serve", "--data-dir", "{d}"],
        ["--data-dir", "{d}", "run"],
        ["run", "--data-dir", "{d}"],
        ["--data-dir", "{d}", "models", "install", "B"],
        ["models", "--data-dir", "{d}", "install", "B"],
        ["models", "install", "--data-dir", "{d}", "B"],
        ["--data-dir", "{d}", "models", "list"],
        ["models", "list", "--data-dir", "{d}"],
    ],
)
def test_data_dir_is_accepted_on_either_side_of_the_subcommand(
    argv: list[str], tmp_path: Path
) -> None:
    """``quantem --data-dir X serve`` and ``quantem serve --data-dir X`` agree.

    ``--data-dir`` used to be top-level only, so one of these two spellings
    failed outright and ``serve --help`` documented neither.
    """
    wanted = tmp_path / "elsewhere"
    args = build_parser().parse_args([a.format(d=str(wanted)) for a in argv])
    assert _resolve_data_dir(args) == wanted


def test_omitting_data_dir_falls_back_to_the_default() -> None:
    args = build_parser().parse_args(["serve"])
    assert _resolve_data_dir(args) == default_data_dir()


def test_serve_help_documents_data_dir(capsys: pytest.CaptureFixture[str]) -> None:
    """The flag exists *and* the place a user would look says so.

    ``quantem serve --help`` used not to mention ``--data-dir`` at all, which is
    why "it must go before the subcommand" was undiscoverable rather than merely
    inconvenient.
    """
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--help"])
    assert "--data-dir" in capsys.readouterr().out


def test_data_dir_env_var_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTEM_DATA_DIR", str(Path("/tmp/qem").resolve()))
    assert default_data_dir() == Path("/tmp/qem").resolve()


def test_a_relative_data_dir_env_var_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Storage relative to the current directory is never what was meant."""
    monkeypatch.setenv("QUANTEM_DATA_DIR", "some/relative/path")
    assert default_data_dir().is_absolute()
