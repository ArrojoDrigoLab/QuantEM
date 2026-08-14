import os
from datetime import timedelta
from unittest.mock import patch
from uuid import uuid4

from django.test import TestCase
from django.utils import timezone

from quantem.jobs.constants import (
    JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
    JOB_TYPE_REFRESH_SEGMENT_FEATURES,
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
)
from quantem.jobs.models import Job
from quantem.jobs.runner import (
    JobRunner,
    RunningJob,
    _detect_accelerator_devices,
    run_job_in_subprocess,
)

# TODO(quantem): the assertions that a failed segmentation job also flips
# ImageSegmentation.status_stage to FAILED need a real asset fixture.
# Restore them once one exists.


class _DeadProcess:
    def __init__(self, exitcode: int):
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        return None


class _AliveProcess:
    def is_alive(self) -> bool:
        return True


class _PersistentWorkerStub:
    def __init__(
        self,
        _ctx=None,
        *,
        pool_key: str = "gpu",
        device_name: str | None = None,
    ):
        self.pool_key = pool_key
        self.device_name = device_name
        self.active_job_id: str | None = None
        self.exitcode: int | None = None
        self.completed = False
        self.terminated = False
        self.closed = False

    def assign(self, job_id: str) -> None:
        self.active_job_id = str(job_id)
        self.completed = False

    def is_alive(self) -> bool:
        return not self.terminated

    def join(self, timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True
        self.exitcode = -15

    def try_consume_completion(self) -> bool:
        if not self.completed or self.active_job_id is None:
            return False
        self.active_job_id = None
        self.completed = False
        return True

    def close(self) -> None:
        self.closed = True


class JobRunnerFailureHandlingTests(TestCase):
    def test_full_segmentation_failure_is_failed_without_retry(self):
        job = Job.enqueue(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            payload={"segmentation_id": str(uuid4())},
            max_attempts=3,
        )

        with patch("quantem.jobs.registry.get_handler") as get_handler:

            def failing_handler(payload, reporter, cancel):
                raise RuntimeError("simulated full-run failure")

            get_handler.return_value = failing_handler
            run_job_in_subprocess(str(job.id))

        job.refresh_from_db()

        self.assertEqual(job.status, "FAILED")
        # The exception's sentence, and not the name of its class: this string
        # is rendered verbatim in the Tasks drawer (invariant I-12).
        self.assertEqual(job.message, "simulated full-run failure")
        self.assertNotIn("RuntimeError", job.message)
        self.assertIn("RuntimeError", job.error_traceback)

    def test_adapter_training_failure_is_failed_without_retry(self):
        job = Job.enqueue(
            job_type=JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
            payload={"segmentation_id": str(uuid4()), "base_model": "quantem:mito"},
            max_attempts=3,
        )

        with patch("quantem.jobs.registry.get_handler") as get_handler:

            def failing_handler(payload, reporter, cancel):
                raise RuntimeError("simulated adaptation failure")

            get_handler.return_value = failing_handler
            run_job_in_subprocess(str(job.id))

        job.refresh_from_db()
        self.assertEqual(job.status, "FAILED")

    def test_retryable_job_still_transitions_to_retry(self):
        job = Job.enqueue(
            job_type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            payload={"asset_id": str(uuid4())},
            max_attempts=3,
        )
        Job.objects.filter(id=job.id).update(
            attempts=1,
            started_at=timezone.now() - timedelta(seconds=5),
            status="RUNNING",
        )

        with patch("quantem.jobs.registry.get_handler") as get_handler:

            def failing_handler(payload, reporter, cancel):
                raise RuntimeError("simulated upload failure")

            get_handler.return_value = failing_handler
            run_job_in_subprocess(str(job.id))

        job.refresh_from_db()
        self.assertEqual(job.status, "RETRY")
        self.assertEqual(job.message, "simulated upload failure")
        self.assertNotIn("RuntimeError", job.message)
        self.assertGreater(job.next_run_at, timezone.now())

    def test_overlay_rebuild_failure_is_failed_after_attempts(self):
        job = Job.enqueue(
            job_type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
            payload={"segmentation_id": str(uuid4())},
            max_attempts=1,
        )
        Job.objects.filter(id=job.id).update(attempts=1)

        with patch("quantem.jobs.registry.get_handler") as get_handler:

            def failing_handler(payload, reporter, cancel):
                raise RuntimeError("simulated overlay failure")

            get_handler.return_value = failing_handler
            run_job_in_subprocess(str(job.id))

        job.refresh_from_db()
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(job.message, "simulated overlay failure")
        self.assertNotIn("RuntimeError", job.message)

    def test_legacy_job_type_fails_that_job_without_stopping_the_runner(self):
        job = Job.enqueue(
            job_type="run_other_model",
            payload={},
            max_attempts=1,
        )
        Job.objects.filter(id=job.id).update(attempts=1)

        run_job_in_subprocess(str(job.id))

        job.refresh_from_db()
        self.assertEqual(job.status, "FAILED")
        self.assertIn("run_other_model", job.message)

    def test_successful_job_runs_handler_to_success(self):
        job = Job.enqueue(
            job_type=JOB_TYPE_REFRESH_SEGMENT_FEATURES,
            payload={"segmentation_id": str(uuid4())},
            max_attempts=1,
        )
        Job.objects.filter(id=job.id).update(status="RUNNING", attempts=1)

        with patch("quantem.jobs.registry.get_handler") as get_handler:
            get_handler.return_value = lambda payload, reporter, cancel: {"ok": True}
            run_job_in_subprocess(str(job.id))

        job.refresh_from_db()
        self.assertEqual(job.status, "SUCCESS")
        self.assertEqual(job.result_json, {"ok": True})

    def test_poll_marks_dead_running_job_as_failed(self):
        job = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            status="RUNNING",
            attempts=1,
            max_attempts=3,
            started_at=timezone.now() - timedelta(minutes=10),
            payload_json={"segmentation_id": str(uuid4())},
            message="running",
        )

        runner = JobRunner()
        runner.running[str(job.id)] = RunningJob(_DeadProcess(exitcode=1), "cpu")

        runner.poll()

        job.refresh_from_db()

        self.assertEqual(job.status, "FAILED")
        self.assertIn("exited with code 1", job.message)
        self.assertNotIn(str(job.id), runner.running)


class JobRunnerConcurrencyTests(TestCase):
    @patch.dict(
        os.environ,
        {"JOB_CPU_WORKERS": "", "JOB_UPLOAD_PIPELINE_WORKERS": ""},
        clear=False,
    )
    @patch("quantem.jobs.runner._detect_accelerator_devices", return_value=[])
    @patch("quantem.jobs.runner.get_machine_profile")
    def test_defaults_reserve_capacity_for_the_interactive_app(
        self,
        machine_profile,
        _detect_accelerators,
    ):
        machine_profile.return_value.heavy_slots = 4

        runner = JobRunner()

        self.assertEqual(runner.cpu_slots, 4)
        self.assertEqual(runner.upload_pipeline_slots, 1)
        runner.running["upload"] = RunningJob(
            _AliveProcess(),
            "cpu",
            JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
        )
        self.assertFalse(runner.can_dispatch("cpu", JOB_TYPE_UPLOAD_IMAGE_PIPELINE))
        self.assertTrue(runner.can_dispatch("cpu", JOB_TYPE_REFRESH_SEGMENT_FEATURES))

    @patch.dict(
        os.environ,
        {"JOB_CPU_WORKERS": "24", "JOB_UPLOAD_PIPELINE_WORKERS": "5"},
        clear=False,
    )
    def test_upload_pipeline_jobs_use_dedicated_cpu_cap(self):
        runner = JobRunner()
        for index in range(5):
            runner.running[f"upload-{index}"] = RunningJob(
                _AliveProcess(),
                "cpu",
                JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            )

        self.assertFalse(runner.can_dispatch("cpu", JOB_TYPE_UPLOAD_IMAGE_PIPELINE))
        self.assertTrue(runner.can_dispatch("cpu", JOB_TYPE_REFRESH_SEGMENT_FEATURES))

    @patch.dict(
        os.environ,
        {"JOB_CPU_WORKERS": "3", "JOB_UPLOAD_PIPELINE_WORKERS": "5"},
        clear=False,
    )
    def test_upload_pipeline_jobs_still_respect_total_cpu_worker_limit(self):
        runner = JobRunner()
        for index in range(3):
            runner.running[f"upload-{index}"] = RunningJob(
                _AliveProcess(),
                "cpu",
                JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            )

        self.assertFalse(runner.can_dispatch("cpu", JOB_TYPE_UPLOAD_IMAGE_PIPELINE))

    @patch.dict(os.environ, {"JOB_GPU_WORKERS": "1"}, clear=False)
    def test_accelerator_jobs_share_one_slot(self):
        runner = JobRunner()
        runner.running["full"] = RunningJob(
            _AliveProcess(),
            "gpu",
            JOB_TYPE_RUN_SEGMENTATION_FULL,
        )

        self.assertFalse(runner.can_dispatch("gpu", JOB_TYPE_RUN_SEGMENTATION_FULL))
        self.assertFalse(runner.can_dispatch("gpu", JOB_TYPE_TRAIN_ORGANELLE_ADAPTER))

    @patch.dict(os.environ, {"JOB_GPU_WORKERS": "1"}, clear=False)
    @patch("quantem.jobs.runner.PersistentJobWorker", _PersistentWorkerStub)
    def test_gpu_jobs_reuse_persistent_worker_between_jobs(self):
        runner = JobRunner()
        first_job_id = str(uuid4())
        second_job_id = str(uuid4())

        runner.start_job(first_job_id, "gpu", JOB_TYPE_RUN_SEGMENTATION_FULL)
        first_running = runner.running[first_job_id]
        first_worker = first_running.process

        self.assertIsInstance(first_worker, _PersistentWorkerStub)
        self.assertEqual(first_worker.active_job_id, first_job_id)
        self.assertEqual(len(runner.gpu_workers["gpu"]), 1)

        first_worker.completed = True
        runner.poll()

        self.assertNotIn(first_job_id, runner.running)
        self.assertIsNone(first_worker.active_job_id)

        runner.start_job(second_job_id, "gpu", JOB_TYPE_TRAIN_ORGANELLE_ADAPTER)
        second_running = runner.running[second_job_id]

        self.assertIs(second_running.process, first_worker)
        self.assertEqual(len(runner.gpu_workers["gpu"]), 1)
        self.assertEqual(first_worker.active_job_id, second_job_id)

    def test_poll_releases_completed_persistent_worker(self):
        runner = JobRunner()
        job_id = str(uuid4())
        worker = _PersistentWorkerStub(pool_key="gpu")
        worker.assign(job_id)
        worker.completed = True
        runner.gpu_workers["gpu"] = [worker]
        runner.running[job_id] = RunningJob(
            worker,
            "gpu",
            JOB_TYPE_RUN_SEGMENTATION_FULL,
        )

        runner.poll()

        self.assertNotIn(job_id, runner.running)
        self.assertIsNone(worker.active_job_id)
        self.assertIn(worker, runner.gpu_workers["gpu"])


class AcceleratorDetectionTests(TestCase):
    @patch.dict(
        os.environ,
        {"QUANTEM_CUDA_AVAILABLE": "1", "QUANTEM_GPU_COUNT": "2"},
        clear=False,
    )
    def test_cuda_devices_are_enumerated_from_the_environment_override(self):
        self.assertEqual(_detect_accelerator_devices(), ["cuda:0", "cuda:1"])

    @patch.dict(os.environ, {"QUANTEM_CUDA_AVAILABLE": "0"}, clear=False)
    def test_no_devices_when_cuda_is_disabled(self):
        self.assertEqual(_detect_accelerator_devices(), [])

    @patch.dict(
        os.environ,
        {
            "JOB_GPU_WORKERS": "2",
            "QUANTEM_CUDA_AVAILABLE": "1",
            "QUANTEM_GPU_COUNT": "2",
        },
        clear=False,
    )
    @patch("quantem.jobs.runner.PersistentJobWorker", _PersistentWorkerStub)
    def test_gpu_workers_are_assigned_explicit_cuda_devices(self):
        runner = JobRunner()

        runner.start_job(str(uuid4()), "gpu", JOB_TYPE_RUN_SEGMENTATION_FULL)
        runner.start_job(str(uuid4()), "gpu", JOB_TYPE_TRAIN_ORGANELLE_ADAPTER)

        workers = runner.gpu_workers["gpu"]
        self.assertEqual(len(workers), 2)
        self.assertEqual(workers[0].device_name, "cuda:0")
        self.assertEqual(workers[1].device_name, "cuda:1")

    @patch.dict(os.environ, {"JOB_GPU_WORKERS": "1"}, clear=False)
    @patch("quantem.jobs.runner.PersistentJobWorker", _PersistentWorkerStub)
    @patch("quantem.jobs.runner._detect_accelerator_devices", return_value=[])
    def test_cpu_only_host_leaves_the_worker_device_unset(self, _detect):
        runner = JobRunner()
        runner.start_job(str(uuid4()), "gpu", JOB_TYPE_RUN_SEGMENTATION_FULL)

        self.assertIsNone(runner.gpu_workers["gpu"][0].device_name)

    @patch.dict(os.environ, {"JOB_GPU_WORKERS": "2"}, clear=False)
    @patch("quantem.jobs.runner.PersistentJobWorker", _PersistentWorkerStub)
    @patch("quantem.jobs.runner._detect_accelerator_devices", return_value=["mps"])
    def test_apple_silicon_pins_every_worker_to_the_single_mps_device(self, _detect):
        runner = JobRunner()
        runner.start_job(str(uuid4()), "gpu", JOB_TYPE_RUN_SEGMENTATION_FULL)
        runner.start_job(str(uuid4()), "gpu", JOB_TYPE_TRAIN_ORGANELLE_ADAPTER)

        workers = runner.gpu_workers["gpu"]
        self.assertEqual([worker.device_name for worker in workers], ["mps", "mps"])
