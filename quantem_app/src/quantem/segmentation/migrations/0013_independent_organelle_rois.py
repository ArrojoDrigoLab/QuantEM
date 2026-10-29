"""Preserve existing visible windows while separating each organelle's workspace.

Crop IDs and files, annotations, probability maps, queued jobs and revision
archives remain unchanged. Older versions exposed every image ROI to every
existing segmentation, even without an explicit membership. Materialize that
visibility before switching to membership-scoped reads. Never infer organelle
completion from the old image-wide flag.
"""

from django.db import migrations, models


def preserve_roi_workspaces(apps, schema_editor):
    alias = schema_editor.connection.alias
    ROI = apps.get_model("assets", "ImageROI")
    Segmentation = apps.get_model("segmentation", "ImageSegmentation")
    Status = apps.get_model("segmentation", "RoiSegmentationStatus")
    Membership = ROI.segmentations.through
    for segmentation in (
        Segmentation.objects.using(alias).exclude(asset_id=None).only("pk", "asset_id").iterator()
    ):
        rois = list(
            ROI.objects.using(alias)
            .filter(asset_id=segmentation.asset_id)
            .order_by("created_at", "id")
            .only("pk", "is_active")
        )
        if not rois:
            continue
        active = next((roi for roi in reversed(rois) if roi.is_active), rois[-1])
        # Batch missing rows, leaving existing completion/timestamp records
        # untouched. Large libraries must not require several queries per ROI.
        Membership.objects.using(alias).bulk_create(
            [Membership(imageroi_id=roi.pk, imagesegmentation_id=segmentation.pk) for roi in rois],
            ignore_conflicts=True,
            batch_size=500,
        )
        Status.objects.using(alias).bulk_create(
            [Status(image_roi_id=roi.pk, segmentation_id=segmentation.pk) for roi in rois],
            ignore_conflicts=True,
            batch_size=500,
        )
        Status.objects.using(alias).filter(segmentation_id=segmentation.pk).update(is_active=False)
        Status.objects.using(alias).filter(
            segmentation_id=segmentation.pk, image_roi_id=active.pk
        ).update(is_active=True)
        Segmentation.objects.using(alias).filter(pk=segmentation.pk).update(rois_initialized=True)


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
