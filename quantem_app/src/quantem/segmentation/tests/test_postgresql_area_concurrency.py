from __future__ import annotations

import threading
from queue import Queue

import numpy as np
import pytest
from django.db import close_old_connections, connection, connections, transaction
from django.db.models import Max
from shapely.geometry import box

from quantem.segmentation.area_instances import (
    default_instance_name,
    materialize_legacy_area_instances,
)
from quantem.segmentation.global_masks import save_global_mask
from quantem.segmentation.models import AnalysisMaskObject, ImageSegmentation
from quantem.segmentation.type_service import get_or_create_cells_type
from quantem.testing import create_small_test_image

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(connection.vendor != "postgresql", reason="PostgreSQL integration test"),
]


def _run_together(function, count: int = 2) -> list[object]:
    barrier = threading.Barrier(count)
    output: Queue[object] = Queue()

    def invoke() -> None:
        close_old_connections()
        try:
            barrier.wait(timeout=10)
            output.put(function())
        except Exception as exc:  # pragma: no cover - asserted below
            output.put(exc)
        finally:
            connections.close_all()

    threads = [threading.Thread(target=invoke) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert all(not thread.is_alive() for thread in threads)
    return [output.get_nowait() for _ in threads]


def test_initial_legacy_materialization_happens_once_under_concurrency():
    image = create_small_test_image("Concurrent cells", width=48, height=32)
    cells = ImageSegmentation.objects.create(
        asset=image.asset,
        segmentation_type=get_or_create_cells_type(),
        status_stage="CANDIDATES_READY",
    )
    mask = np.zeros((32, 48), dtype=bool)
    mask[2:10, 2:10] = True
    mask[16:26, 28:40] = True
    save_global_mask(cells, mask, source="legacy")

    def materialize() -> int:
        segmentation = ImageSegmentation.objects.get(pk=cells.pk)
        return len(materialize_legacy_area_instances(segmentation))

    outcomes = _run_together(materialize)

    assert not [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert AnalysisMaskObject.objects.filter(segmentation=cells).count() == 2
    assert list(
        AnalysisMaskObject.objects.filter(segmentation=cells)
        .order_by("sort_order")
        .values_list("name", flat=True)
    ) == ["Cell 1", "Cell 2"]


def test_concurrent_initial_cell_creation_gets_distinct_identity_and_order():
    image = create_small_test_image("Concurrent cell creation", width=48, height=32)
    cells = ImageSegmentation.objects.create(
        asset=image.asset,
        segmentation_type=get_or_create_cells_type(),
        status_stage="CANDIDATES_READY",
    )

    def create_cell() -> str:
        with transaction.atomic():
            segmentation = ImageSegmentation.objects.get(pk=cells.pk)
            materialize_legacy_area_instances(segmentation)
            order = (
                AnalysisMaskObject.objects.filter(segmentation=segmentation).aggregate(
                    value=Max("sort_order")
                )["value"]
                or 0
            ) + 1
            created = AnalysisMaskObject.objects.create(
                segmentation=segmentation,
                name=default_instance_name(segmentation),
                color="#38bdf8",
                sort_order=order,
                geometry_wkb=box(order, 1, order + 0.5, 2).wkb,
            )
            return str(created.pk)

    outcomes = _run_together(create_cell)

    assert not [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(set(outcomes)) == 2
    assert list(
        AnalysisMaskObject.objects.filter(segmentation=cells)
        .order_by("sort_order")
        .values_list("name", "sort_order")
    ) == [("Cell 1", 1), ("Cell 2", 2)]
