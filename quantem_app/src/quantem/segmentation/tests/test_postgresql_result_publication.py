"""Publication must serialize with a concurrent scientific mutation."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from django.db import close_old_connections, connection, connections, transaction
from shapely.geometry import box

from quantem.analysis.models import AnalysisRun
from quantem.analysis.publication import input_guard, publish, staging_directory
from quantem.segmentation.models import CompletedROI, ImageSegmentation
from quantem.segmentation.mutations import record_change
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(connection.vendor != "postgresql", reason="PostgreSQL integration test"),
]


def test_publication_waits_for_committing_mutation_then_rejects_changed_inputs():
    asset = create_small_test_image("Concurrent publication", width=32, height=32).asset
    seg = ImageSegmentation.objects.create(
        asset=asset, segmentation_type=get_or_create_mitochondria_type()
    )
    run = AnalysisRun.objects.create(segmentation=seg, params={}, status="RUNNING")
    expected = input_guard(run)
    stage = staging_directory(run)
    stage.mkdir(parents=True)
    (stage / "manifest.json").write_text("{}", encoding="utf-8")
    changed = threading.Event()
    attempted = threading.Event()
    release = threading.Event()

    def mutate():
        close_old_connections()
        try:
            with transaction.atomic():
                current = ImageSegmentation.objects.get(pk=seg.pk)
                record_change(current)
                CompletedROI.objects.create(segmentation=current, geometry=box(0, 0, 16, 16))
                changed.set()
                assert release.wait(timeout=10)
        finally:
            connections.close_all()

    def finalize():
        close_old_connections()
        try:
            attempted.set()
            publish(run, stage, expected=expected, results={})
        finally:
            connections.close_all()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            writer = pool.submit(mutate)
            assert changed.wait(timeout=10)
            finalizer = pool.submit(finalize)
            assert attempted.wait(timeout=10)
            assert not finalizer.done()
            release.set()
            writer.result(timeout=10)
            with pytest.raises(ValueError, match="Inputs changed during measurement"):
                finalizer.result(timeout=10)
        run.refresh_from_db()
        assert run.status == "RUNNING" and not run.export_dir
        assert (stage / "manifest.json").is_file()
    finally:
        release.set()
        import shutil

        shutil.rmtree(stage, ignore_errors=True)
