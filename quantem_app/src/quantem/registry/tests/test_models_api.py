"""``GET /api/models/`` and ``POST /api/models/<pack_id>/install/``.

Against the API contract's Models section. The urlconf is overridden to
:mod:`quantem.registry.tests.urls` so these do not wait on ``core/urls.py`` --
owned elsewhere -- mounting the routes; when it does, the override is a no-op
and these keep passing unchanged.

No pack is ever really copied here: installing one means hashing and writing
660 MB, which is :mod:`quantem.registry.install`'s job and is covered by
actually running it from the CLI. What is under test is the *endpoint* --
which source it picks, what it refuses, and what it records.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from quantem.inference.specs import MODEL_SPECS
from quantem.jobs.models import Job
from quantem.registry import cache
from quantem.registry.install import InstalledPack
from quantem.registry.views import INSTALL_JOB_TYPE

TEST_URLCONF = "quantem.registry.tests.urls"


class _Installed:
    """``cache.installed`` stand-in whose answer flips once a pack is written."""

    def __init__(self, installed: bool = False):
        self.installed = installed

    def __call__(self, pack_id: str) -> bool:
        return self.installed


@override_settings(ROOT_URLCONF=TEST_URLCONF)
class ModelListTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_the_body_has_the_contract_keys(self):
        # Re-pinned 2026-08-08: "registry" joined the body when the download
        # landed -- the pinned repo/revision a not-installed pack fetches from.
        # Re-pinned 2026-08-10: "storage" joined it for owner ruling D8. The
        # install-from-a-folder field used to be placeheld with this build
        # machine's drive letter; the example is now composed from the running
        # install's own resolved models directory, which only the server knows.
        body = self.client.get("/api/models/").json()
        assert set(body) == {"packs", "adapted", "device", "registry", "storage"}
        assert body["registry"]["repo_id"] == "ArrojoeDrigoLab/quantem"
        assert body["registry"]["revision"]

    def test_the_storage_block_is_built_from_this_machine(self):
        """D8: the example the UI shows has to come from here, not from a literal."""
        from quantem.core.config import MODELS_DIR

        storage = self.client.get("/api/models/").json()["storage"]

        assert storage["models_dir"] == str(MODELS_DIR)
        # A sibling of the data directory, with the platform's own separator,
        # so a Mac never sees a drive letter and Windows never sees a slash.
        example = storage["local_source_example"]
        assert example.startswith(str(Path(MODELS_DIR).parent.parent))
        assert example.endswith("quantem-models")

    def test_all_eight_released_packs_are_listed(self):
        body = self.client.get("/api/models/").json()
        assert [p["id"] for p in body["packs"]] == sorted(MODEL_SPECS)

    def test_each_pack_says_whether_it_can_actually_run(self):
        body = self.client.get("/api/models/").json()
        for pack in body["packs"]:
            assert isinstance(pack["runnable"], bool)
            if not pack["runnable"]:
                # Never a bare False: the UI has to be able to say why it greyed
                # the pack out.
                assert pack["reason"]

    def test_adapted_lists_the_users_successful_adapters(self):
        from django.utils import timezone

        from quantem.finetune.models import STATUS_SUCCESS, Adapter

        Adapter.objects.create(
            base_model="quantem:mito",
            name="mito @ my-liver-set",
            status=STATUS_SUCCESS,
            mode="threshold_only",
            calibrated_threshold=0.35,
            split_mode="image-disjoint",
            sweep={"heldout_dice_at_calibrated": 0.87},
            applied_at=timezone.now(),
        )
        Adapter.objects.create(base_model="quantem:er", status="FAILED")

        body = self.client.get("/api/models/").json()
        assert len(body["adapted"]) == 1
        entry = body["adapted"][0]
        assert entry["base"] == "quantem:mito"
        assert entry["name"] == "mito @ my-liver-set"
        assert entry["calibrated_threshold"] == 0.35
        assert entry["heldout_dice"] == 0.87
        # A held-out score never travels without the split it was measured on.
        assert entry["split_mode"] == "image-disjoint"
        assert entry["id"].startswith("adapted:")

    def test_the_device_block_is_present_and_concrete(self):
        device = self.client.get("/api/models/").json()["device"]
        assert device["kind"] in {"cpu", "cuda", "mps"}
        assert isinstance(device["cuda"], bool)
        assert isinstance(device["mps"], bool)


@override_settings(ROOT_URLCONF=TEST_URLCONF)
class ActiveInstallTests(TestCase):
    """UAT round 13, paper-cut 1: the Models screen was blind to in-flight
    installs, and its Download button queued a real duplicate 1.2 GB download
    while the installer-requested job for the same pack was RUNNING.

    Two halves, pinned to `the API contract` §Models: every pack entry carries
    ``active_install`` (null, or the live job with byte progress), and the
    install POST refuses a duplicate with a 409 naming the existing job --
    the exact guard ``pending_installs._queue_pack`` already had.
    """

    def setUp(self):
        self.client = APIClient()

    def _live_job(self, pack_id="omniem:mito", job_status="RUNNING", **fields):
        job = Job.enqueue(
            job_type=INSTALL_JOB_TYPE,
            payload={"pack_id": pack_id, "source": "huggingface"},
        )
        Job.objects.filter(id=job.id).update(status=job_status, **fields)
        job.refresh_from_db()
        return job

    def _entry(self, pack_id="omniem:mito"):
        body = self.client.get("/api/models/").json()
        return next(p for p in body["packs"] if p["id"] == pack_id)

    def test_every_pack_entry_carries_active_install_null_when_idle(self):
        body = self.client.get("/api/models/").json()
        for pack in body["packs"]:
            assert "active_install" in pack, pack["id"]
            assert pack["active_install"] is None, pack["id"]

    def test_a_running_download_shows_on_its_pack_with_byte_progress(self):
        job = self._live_job(
            job_status="RUNNING",
            progress_current_bytes=214_000_000,
            progress_total_bytes=1_243_000_000,
        )

        entry = self._entry()
        assert entry["active_install"] == {
            "job_id": str(job.id),
            "status": "RUNNING",
            "progress_current_bytes": 214_000_000,
            "progress_total_bytes": 1_243_000_000,
        }
        # Only that pack: the sibling packs stay null.
        body = self.client.get("/api/models/").json()
        for pack in body["packs"]:
            if pack["id"] != "omniem:mito":
                assert pack["active_install"] is None, pack["id"]

    def test_a_waiting_download_reads_queued_with_null_bytes(self):
        # PENDING and RETRY both mean "it will run on its own"; the screen's
        # distinction is queued-vs-downloading, not the queue's bookkeeping.
        for raw in ("PENDING", "RETRY"):
            Job.objects.all().delete()
            job = self._live_job(job_status=raw)
            active = self._entry()["active_install"]
            assert active["status"] == "QUEUED", raw
            assert active["job_id"] == str(job.id)
            assert active["progress_current_bytes"] is None
            assert active["progress_total_bytes"] is None

    def test_a_finished_job_is_not_an_active_install(self):
        for raw in ("SUCCESS", "FAILED", "CANCELLED"):
            Job.objects.all().delete()
            self._live_job(job_status=raw)
            assert self._entry()["active_install"] is None, raw

    def test_a_second_install_of_an_in_flight_pack_is_a_409_naming_the_job(self):
        job = self._live_job(job_status="RUNNING")

        with patch.object(cache, "installed", return_value=False):
            response = self.client.post("/api/models/omniem:mito/install/", {}, format="json")

        assert response.status_code == 409
        body = response.json()
        assert body["job_id"] == str(job.id)
        assert body["status"] == "RUNNING"
        # The id is a field, not a sentence: it is how the client finds the row
        # to watch. Putting it in ``error`` broke I-12's raw-uuid rule, and no
        # screen shows a job id for a reader to match it against.
        assert str(job.id) not in body["error"]
        assert "Tasks & Queues" in body["error"]
        assert body["active_install"]["status"] == "RUNNING"
        # The whole point: no second download job exists.
        assert Job.objects.filter(type=INSTALL_JOB_TYPE).count() == 1

    def test_a_queued_install_blocks_a_duplicate_too(self):
        job = self._live_job(job_status="PENDING")

        with patch.object(cache, "installed", return_value=False):
            response = self.client.post("/api/models/omniem:mito/install/", {}, format="json")

        assert response.status_code == 409
        assert response.json()["job_id"] == str(job.id)
        assert Job.objects.filter(type=INSTALL_JOB_TYPE).count() == 1

    def test_a_local_path_install_is_also_refused_while_one_is_in_flight(self):
        """Copying files into the pack dir under a running download races it
        exactly as a second download would."""
        self._live_job(job_status="RUNNING")

        with patch.object(cache, "installed", return_value=False):
            response = self.client.post(
                "/api/models/omniem:mito/install/",
                {"source_path": "D:/anywhere"},
                format="json",
            )

        assert response.status_code == 409
        assert Job.objects.filter(type=INSTALL_JOB_TYPE).count() == 1

    def test_a_different_packs_install_does_not_block_this_one(self):
        self._live_job(pack_id="omniem:er", job_status="RUNNING")

        with patch.object(cache, "installed", return_value=False):
            response = self.client.post("/api/models/omniem:mito/install/", {}, format="json")

        assert response.status_code == 202
        assert Job.objects.filter(type=INSTALL_JOB_TYPE).count() == 2

    def test_a_terminal_job_does_not_block_a_reinstall(self):
        self._live_job(job_status="FAILED")

        with patch.object(cache, "installed", return_value=False):
            response = self.client.post("/api/models/omniem:mito/install/", {}, format="json")

        assert response.status_code == 202


@override_settings(ROOT_URLCONF=TEST_URLCONF)
class ModelInstallTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def _install(self, pack_id="quantem:mito", **body):
        return self.client.post(f"/api/models/{pack_id}/install/", body, format="json")

    def test_a_downloaded_pack_can_be_removed(self):
        with patch.object(cache, "remove_pack", return_value=True) as remove:
            response = self.client.delete("/api/models/quantem:mito/")

        assert response.status_code == 204
        remove.assert_called_once_with("quantem:mito")

    def test_removing_a_pack_that_is_not_downloaded_is_a_404(self):
        with patch.object(cache, "remove_pack", return_value=False):
            response = self.client.delete("/api/models/quantem:mito/")

        assert response.status_code == 404
        assert response.json()["error"] == "This model is not downloaded."

    def test_removal_waits_for_a_different_pack_download_to_finish(self):
        with (
            patch(
                "quantem.registry.views.catalogue.active_install_job",
                side_effect=lambda pack_id: object() if pack_id == "omniem:er" else None,
            ),
            patch.object(cache, "remove_pack") as remove,
        ):
            response = self.client.delete("/api/models/quantem:mito/")

        assert response.status_code == 409
        assert "still downloading" in response.json()["error"]
        remove.assert_not_called()

    def test_removal_waits_for_an_active_inference_task_using_the_pack(self):
        Job.enqueue(
            job_type="run_segmentation_full",
            payload={"source_model": "quantem:mito"},
            tags=["source_model:quantem:mito"],
        )
        with patch.object(cache, "remove_pack") as remove:
            response = self.client.delete("/api/models/quantem:mito/")

        assert response.status_code == 409
        assert "active task" in response.json()["error"]
        remove.assert_not_called()

    def test_an_unknown_pack_is_a_404_that_names_the_known_ones(self):
        response = self._install("quantem:golgi")
        assert response.status_code == 404
        assert "quantem:mito" in response.json()["error"]

    def test_an_installed_pack_reports_itself_installed_and_does_no_work(self):
        with patch.object(cache, "installed", return_value=True):
            response = self._install()
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "SUCCESS"
        assert body["detail"] == "Already installed."
        assert body["pack"]["id"] == "quantem:mito"
        assert not Job.objects.exists()

    def test_no_source_queues_a_real_download_job(self):
        # Re-pinned 2026-08-08: the remote registry now exists. No source means
        # "download from the QuantEM Hugging Face repository", and the promise
        # is kept by a real PENDING job with a registered handler -- not a 501,
        # and not a terminal row pretending to have run.
        with patch.object(cache, "installed", return_value=False):
            response = self._install()
        assert response.status_code == 202
        body = response.json()
        assert body["source"] == "huggingface"
        assert body["status"] == "PENDING"
        assert body["download_bytes"] > 0
        assert body["repo_id"] and body["revision"]

        job = Job.objects.get(id=body["job_id"])
        assert job.type == INSTALL_JOB_TYPE
        assert job.status == "PENDING"
        assert job.payload_json["pack_id"] == "quantem:mito"
        assert job.payload_json["source"] == "huggingface"
        # The handler is registered, so the scheduler can actually dispatch it.
        from quantem.jobs.registry import get_handler

        assert get_handler(INSTALL_JOB_TYPE)

    def test_a_source_path_that_is_not_a_directory_is_refused(self):
        with patch.object(cache, "installed", return_value=False):
            response = self._install(source_path="D:/definitely/not/here")
        assert response.status_code == 400
        assert "no folder at" in response.json()["error"]

    def test_a_source_path_on_an_unreachable_drive_is_refused_not_crashed(self):
        """A disconnected mapped drive raises from ``is_dir()`` on Windows.

        Uncaught, that is a 500 and a traceback page where a sentence belongs.
        """
        with patch.object(cache, "installed", return_value=False):
            with patch.object(
                Path, "is_dir", side_effect=OSError(1326, "The user name or password is incorrect")
            ):
                response = self._install(source_path="Z:/somewhere/on/a/dead/share")

        assert response.status_code == 400
        assert "no folder at" in response.json()["error"]

    def test_a_directory_with_no_head_names_what_is_missing(self):
        with patch.object(cache, "installed", return_value=False):
            response = self._install(source_path=str(Path(__file__).parent))
        assert response.status_code == 400
        error = response.json()["error"]
        assert "head.pt" in error
        assert "mito_quantem" in error  # the released directory naming

    def test_a_local_install_runs_and_is_recorded_as_a_job(self):
        source = Path(self.tmp_source())
        installed = InstalledPack(
            pack_id="quantem:mito",
            root=source,
            head_sha256="a" * 64,
            encoder_sha256="b" * 64,
            bytes_written=123,
            reused_blobs=1,
        )
        with (
            patch.object(cache, "installed", _Installed(False)),
            patch(
                "quantem.registry.install.install_pack_from_path",
                return_value=installed,
            ) as install,
        ):
            response = self._install(source_path=str(source))

        assert response.status_code == 202
        body = response.json()
        assert body["source"] == "local-path"
        assert body["bytes_written"] == 123
        assert body["reused_blobs"] == 1

        # The contract's polling flow works: the job id resolves to a real row.
        job = Job.objects.get(id=body["job_id"])
        assert job.type == INSTALL_JOB_TYPE
        assert job.status == "SUCCESS"
        assert job.payload_json["pack_id"] == "quantem:mito"
        assert job.result_json["source"] == "local-path"

        # The encoder and index beside the head were found and passed through.
        kwargs = install.call_args.kwargs
        assert kwargs["encoder_index"] == source / cache.INDEX_NAME
        assert kwargs["encoder_file"] == source / cache.ENCODER_NAME

    def test_a_failed_install_is_an_error_and_a_failed_job(self):
        source = Path(self.tmp_source())
        with (
            patch.object(cache, "installed", return_value=False),
            patch(
                "quantem.registry.install.install_pack_from_path",
                side_effect=RuntimeError("encoder checkpoint not found"),
            ),
        ):
            response = self._install(source_path=str(source))

        assert response.status_code == 400
        assert "encoder checkpoint not found" in response.json()["error"]
        job = Job.objects.get()
        assert job.status == "FAILED"

    def test_a_directory_with_no_head_names_only_shapes_that_exist(self):
        """Every path it suggests must be one the user can go and look at.

        The old message named ``mito_quantem/head.pt`` as the release shape.
        Releases ship ``packs/quantem__mito/``; ``mito_quantem/`` is the
        training-output naming, which is a different thing and has to say so.
        """
        with patch.object(cache, "installed", return_value=False):
            response = self._install(source_path=str(Path(__file__).parent))
        error = response.json()["error"]
        assert "MANIFEST.json" in error
        assert "packs/quantem__mito/" in error
        assert "training outputs, named mito_quantem/" in error

    def test_installing_from_one_pack_directory_uses_the_release_above_it(self):
        """The shape the on-screen instruction actually describes.

        "The folder holding head.pt and resolved_config.yaml" inside an unzipped
        release is ``packs/quantem__mito``. That used to miss the bundle branch,
        fall through to the raw-training-outputs path, and fail asking for a
        research checkpoint.
        """
        bundle = self.tmp_bundle()
        pack_dir = bundle / "packs" / "quantem__mito"
        with (
            patch.object(cache, "installed", return_value=False),
            patch(
                "quantem.registry.install.install_pack_from_bundle",
                return_value=InstalledPack("quantem:mito", pack_dir, "a", None, 7, 0),
            ) as install,
        ):
            response = self._install(source_path=str(pack_dir))

        assert response.status_code == 202
        body = response.json()
        assert body["source"] == "release-bundle"
        # Reported back as the release it resolved to, not as what was typed.
        assert body["source_path"] == str(bundle)
        assert install.call_args.args[1] == bundle

    def test_installing_from_the_packs_directory_uses_the_release_above_it(self):
        bundle = self.tmp_bundle()
        with (
            patch.object(cache, "installed", return_value=False),
            patch(
                "quantem.registry.install.install_pack_from_bundle",
                return_value=InstalledPack("quantem:mito", bundle, "a", None, 7, 0),
            ) as install,
        ):
            response = self._install(source_path=str(bundle / "packs"))

        assert response.status_code == 202
        assert response.json()["source"] == "release-bundle"
        assert install.call_args.args[1] == bundle

    def test_installing_from_the_release_root_still_works(self):
        bundle = self.tmp_bundle()
        with (
            patch.object(cache, "installed", return_value=False),
            patch(
                "quantem.registry.install.install_pack_from_bundle",
                return_value=InstalledPack("quantem:mito", bundle, "a", None, 7, 0),
            ) as install,
        ):
            response = self._install(source_path=str(bundle))

        assert response.status_code == 202
        assert install.call_args.args[1] == bundle

    def test_a_directory_of_packs_resolves_the_released_subdirectory(self):
        import tempfile

        root = Path(tempfile.mkdtemp(dir=self.tmp_root()))
        (root / "mito_quantem").mkdir()
        (root / "mito_quantem" / "head.pt").write_bytes(b"h")
        with (
            patch.object(cache, "installed", return_value=False),
            patch(
                "quantem.registry.install.install_pack_from_path",
                return_value=InstalledPack("quantem:mito", root, "a", None, 0, 0),
            ) as install,
        ):
            response = self._install(source_path=str(root))
        assert response.status_code == 202
        assert install.call_args.args[1] == root / "mito_quantem"

    # --- helpers -----------------------------------------------------------

    def tmp_root(self) -> Path:
        import tempfile

        if not hasattr(self, "_tmp_root"):
            self._tmp_root = Path(tempfile.mkdtemp())
            self.addCleanup(self._cleanup)
        return self._tmp_root

    def _cleanup(self) -> None:
        import shutil

        shutil.rmtree(self._tmp_root, ignore_errors=True)

    def tmp_source(self) -> Path:
        """A directory of loose model files, as raw training outputs are."""
        source = self.tmp_root() / "training-outputs"
        source.mkdir(exist_ok=True)
        for name in (
            "head.pt",
            cache.CONFIG_NAME,
            cache.INDEX_NAME,
            cache.ENCODER_NAME,
        ):
            (source / name).write_bytes(b"x")
        return source

    def tmp_bundle(self) -> Path:
        """An unzipped release: MANIFEST.json at the top, packs/<pack> below."""
        from quantem.registry import release

        bundle = self.tmp_root() / "quantem-models-9.9.9"
        pack_dir = bundle / release.PACKS_DIRNAME / cache.pack_dirname("quantem:mito")
        pack_dir.mkdir(parents=True, exist_ok=True)
        (bundle / release.MANIFEST_NAME).write_text("{}", encoding="utf-8")
        for name in (cache.HEAD_NAME, cache.CONFIG_NAME, cache.EXPORTED_ENCODER_NAME):
            (pack_dir / name).write_bytes(b"x")
        return bundle
