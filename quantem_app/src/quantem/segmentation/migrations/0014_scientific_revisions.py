import django.utils.timezone
from django.db import migrations, models


def establish_vector_ownership(apps, schema_editor):
    segmentation = apps.get_model("segmentation", "ImageSegmentation")
    objects = apps.get_model("segmentation", "AnalysisMaskObject")
    masks = apps.get_model("segmentation", "GlobalMask")
    alias = schema_editor.connection.alias
    segmentation.objects.using(alias).filter(
        pk__in=objects.objects.using(alias).values("segmentation_id")
    ).update(area_vectors_authoritative=True)
    for mask in masks.objects.using(alias).iterator():
        metadata = mask.metadata or {}
        if metadata.get("editable_analysis_objects") or metadata.get(
            "instances_migrated_to"
        ) == str(mask.segmentation_id):
            segmentation.objects.using(alias).filter(pk=mask.segmentation_id).update(
                area_vectors_authoritative=True
            )


class Migration(migrations.Migration):
    dependencies = [("segmentation", "0013_independent_organelle_rois")]
    operations = [
        migrations.CreateModel(
            name="RetiredResultArtifact",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                ("path", models.CharField(max_length=1024, unique=True)),
                ("is_directory", models.BooleanField(default=False)),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("last_error", models.TextField(blank=True, default="")),
            ],
        ),
        migrations.AddField(
            "imagesegmentation", "scientific_revision", models.PositiveBigIntegerField(default=0)
        ),
        migrations.AddField(
            "imagesegmentation", "geometry_revision", models.PositiveBigIntegerField(default=0)
        ),
        migrations.AddField(
            "imagesegmentation", "area_vectors_authoritative", models.BooleanField(default=False)
        ),
        migrations.RunPython(establish_vector_ownership, migrations.RunPython.noop),
    ]
