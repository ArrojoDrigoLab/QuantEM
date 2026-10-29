"""Preserve existing visible windows while separating each organelle's workspace.

Crop IDs and files, annotations, probability maps, queued jobs and revision
archives remain unchanged. Older versions exposed every image ROI to every
existing segmentation, even without an explicit membership. Materialize that
visibility before switching to membership-scoped reads. Never infer organelle
completion from the old image-wide flag.
"""

from collections import defaultdict

from django.db import migrations, models

SEGMENTATION_BATCH_SIZE = 500


def _preserve_batch(*, alias, ROI, Segmentation, Status, Membership, segmentations):
    """Materialize one bounded batch without an N+1 query per segmentation."""
    rois_by_asset = defaultdict(list)
    asset_ids = {asset_id for _, asset_id in segmentations}
    for roi in (
        ROI.objects.using(alias)
        .filter(asset_id__in=asset_ids)
        .order_by("asset_id", "created_at", "id")
        .only("pk", "asset_id", "is_active")
        .iterator(chunk_size=2000)
    ):
        rois_by_asset[roi.asset_id].append(roi)

    memberships = []
    statuses = []
    initialized_ids = []
    active_roi_ids = set()
    for segmentation_id, asset_id in segmentations:
        rois = rois_by_asset.get(asset_id, ())
        if not rois:
            continue
        active = next((roi for roi in reversed(rois) if roi.is_active), rois[-1])
        initialized_ids.append(segmentation_id)
        active_roi_ids.add(active.pk)
        memberships.extend(
            Membership(imageroi_id=roi.pk, imagesegmentation_id=segmentation_id) for roi in rois
        )
        statuses.extend(
            Status(image_roi_id=roi.pk, segmentation_id=segmentation_id) for roi in rois
        )

    if not initialized_ids:
        return
    # Batch missing rows, leaving existing completion/timestamp records
    # untouched. Large libraries must not require several queries per ROI.
    Membership.objects.using(alias).bulk_create(memberships, ignore_conflicts=True, batch_size=2000)
    Status.objects.using(alias).bulk_create(statuses, ignore_conflicts=True, batch_size=2000)
    Status.objects.using(alias).filter(segmentation_id__in=initialized_ids).update(is_active=False)
    # A correlated asset check avoids activating any malformed historical
    # cross-asset status while keeping the backfill to one UPDATE per batch.
    Status.objects.using(alias).filter(
        segmentation_id__in=initialized_ids,
        image_roi_id__in=active_roi_ids,
        image_roi__asset_id=models.F("segmentation__asset_id"),
    ).update(is_active=True)
    Segmentation.objects.using(alias).filter(pk__in=initialized_ids).update(rois_initialized=True)


def preserve_roi_workspaces(apps, schema_editor):
    alias = schema_editor.connection.alias
    ROI = apps.get_model("assets", "ImageROI")
    Segmentation = apps.get_model("segmentation", "ImageSegmentation")
    Status = apps.get_model("segmentation", "RoiSegmentationStatus")
    Membership = ROI.segmentations.through
    batch = []
    rows = (
        Segmentation.objects.using(alias)
        .exclude(asset_id=None)
        .values_list("pk", "asset_id")
        .iterator(chunk_size=SEGMENTATION_BATCH_SIZE)
    )
    for row in rows:
        batch.append(row)
        if len(batch) == SEGMENTATION_BATCH_SIZE:
            _preserve_batch(
                alias=alias,
                ROI=ROI,
                Segmentation=Segmentation,
                Status=Status,
                Membership=Membership,
                segmentations=batch,
            )
            batch = []
    if batch:
        _preserve_batch(
            alias=alias,
            ROI=ROI,
            Segmentation=Segmentation,
            Status=Status,
            Membership=Membership,
            segmentations=batch,
        )


class Migration(migrations.Migration):
    dependencies = [
        ("assets", "0008_databaseimport"),
        ("segmentation", "0012_reconcile_definitions_and_model_results"),
    ]

    operations = [
        migrations.AddField(
            model_name="imagesegmentation",
            name="rois_initialized",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="roisegmentationstatus",
            name="is_active",
            field=models.BooleanField(default=False),
        ),
        # Create the index before backfilling rows: PostgreSQL cannot build it
        # while the data updates have deferred foreign-key trigger events.
        migrations.AddConstraint(
            model_name="roisegmentationstatus",
            constraint=models.UniqueConstraint(
                fields=("segmentation",),
                condition=models.Q(is_active=True),
                name="unique_active_roi_per_segmentation",
            ),
        ),
        # Reversal preserves memberships and completion records. Older releases
        # use the untouched image-level selection flag again.
        migrations.RunPython(preserve_roi_workspaces, migrations.RunPython.noop),
    ]
