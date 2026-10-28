from django.db import migrations, models
from django.db.models import Count, Q


def assert_unique_ngff_renditions(apps, schema_editor):
    Rendition = apps.get_model("assets", "Rendition")
    duplicates = list(
        Rendition.objects.using(schema_editor.connection.alias)
        .filter(type="NGFF")
        .values("asset_id")
        .annotate(row_count=Count("id"))
        .filter(row_count__gt=1)
        .values_list("asset_id", flat=True)[:20]
    )
    if duplicates:
        identifiers = ", ".join(str(value) for value in duplicates)
        raise RuntimeError(
            "Cannot add the PostgreSQL NGFF concurrency guard because these assets "
            f"have duplicate NGFF rendition rows: {identifiers}. No rows were removed."
        )


class Migration(migrations.Migration):
    dependencies = [("assets", "0006_active_assets_require_experiment")]

    operations = [
        migrations.RunPython(assert_unique_ngff_renditions, migrations.RunPython.noop),
        migrations.AddConstraint(
            model_name="rendition",
            constraint=models.UniqueConstraint(
                fields=("asset", "type"),
                condition=Q(type="NGFF"),
                name="unique_ngff_rendition_per_asset",
            ),
        ),
    ]
