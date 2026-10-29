"""Scientific mutation bookkeeping, separate from disposable display work.

Call these inside the transaction that writes the authoritative data. The
stable parent lock serializes revision allocation with analysis publication.
Display-only rebuilds must use overlay_ngff directly and never advance these
revisions. Geometry and state mutations deliberately have different costs.
"""

from django.db import transaction
from django.db.models import F
from django.utils import timezone

from quantem.segmentation.models import ImageSegmentation


@transaction.atomic
def record_change(segmentation, *, geometry=False, area_vectors=False):
    from quantem.analysis.superseded import mark_superseded

    ImageSegmentation.objects.select_for_update().get(pk=segmentation.pk)
    values = {"scientific_revision": F("scientific_revision") + 1, "updated_at": timezone.now()}
    if geometry:
        values["geometry_revision"] = F("geometry_revision") + 1
    if area_vectors:
        values["area_vectors_authoritative"] = True
    ImageSegmentation.objects.filter(pk=segmentation.pk).update(**values)
    segmentation.refresh_from_db(fields=list(values))
    mark_superseded(segmentation)


@transaction.atomic
def register_overlay_mutation(segmentation, **kwargs):
    from quantem.segmentation.overlay_ngff import register_overlay_mutation as display_change

    record_change(segmentation, geometry=True)
    return display_change(segmentation, **kwargs)


@transaction.atomic
def register_overlay_mutation_all_bundles(segmentation, **kwargs):
    from quantem.segmentation.overlay_ngff import (
        register_overlay_mutation_all_bundles as display_change,
    )

    record_change(segmentation, geometry=True)
    return display_change(segmentation, **kwargs)


@transaction.atomic
def register_state_mutation(segmentation, **kwargs):
    from quantem.segmentation.overlay_ngff import register_state_mutation as display_change

    record_change(segmentation)
    return display_change(segmentation, **kwargs)


@transaction.atomic
def register_label_state_mutation(segmentation, objects, **kwargs):
    from quantem.segmentation.overlay_ngff import register_label_state_mutation as display_change

    record_change(segmentation)
    return display_change(segmentation, objects, **kwargs)
