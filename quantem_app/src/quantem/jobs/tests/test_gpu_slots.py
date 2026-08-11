"""One accelerator worker per accelerator, not one in total.

``JobRunner.gpu_slots`` defaulted to 1 no matter how many devices
``_detect_accelerator_devices`` found, so ``_next_gpu_device_name`` round-robined
across cards that ``_get_or_create_idle_gpu_worker`` would never ask for: a
two-card workstation enumerated both and used one.

One *per* card and no more. MEASURED (gpu_measure section 6): two processes
sharing a card gain 1.10x of throughput and four gain 1.20x, at 3.1x worse
per-run latency and four times the VRAM -- the SMs are already saturated by a
single stream.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from django.test import TestCase

from quantem.jobs.runner import JobRunner


class GpuSlotsFollowTheDeviceCount(TestCase):
    def _runner(self, *, gpus: int, override: str | None = None) -> JobRunner:
        env = {"QUANTEM_GPU_COUNT": str(gpus)}
        if override is not None:
            env["JOB_GPU_WORKERS"] = override
        else:
            os.environ.pop("JOB_GPU_WORKERS", None)
        with patch.dict(os.environ, env):
            runner = JobRunner()
        self.addCleanup(runner.shutdown)
        return runner

    def test_two_cards_get_two_slots(self):
        runner = self._runner(gpus=2)
        self.assertEqual(runner.gpu_devices, ["cuda:0", "cuda:1"])
        self.assertEqual(runner.gpu_slots, 2)

    def test_four_cards_get_four_slots(self):
        runner = self._runner(gpus=4)
        self.assertEqual(runner.gpu_slots, 4)

    def test_one_card_gets_one_slot(self):
        runner = self._runner(gpus=1)
        self.assertEqual(runner.gpu_slots, 1)

    def test_no_card_still_leaves_a_slot(self):
        """The pool key still has to exist: accelerator-class jobs run on the
        CPU when there is no accelerator, and a slot count of zero would stall
        the queue rather than run them slowly."""
        runner = self._runner(gpus=0)
        self.assertEqual(runner.gpu_devices, [])
        self.assertEqual(runner.gpu_slots, 1)

    def test_the_environment_still_wins(self):
        runner = self._runner(gpus=4, override="1")
        self.assertEqual(runner.gpu_slots, 1)

    def test_each_worker_is_pinned_to_a_different_card(self):
        runner = self._runner(gpus=2)
        first = runner._next_gpu_device_name()
        runner.gpu_workers.setdefault("gpu", []).append(object())
        second = runner._next_gpu_device_name()
        self.assertEqual([first, second], ["cuda:0", "cuda:1"])
