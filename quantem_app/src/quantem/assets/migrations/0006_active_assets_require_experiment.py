import django.db.models.deletion
from django.db import migrations, models


def give_active_images_experiments(apps, schema_editor):
    """Give every legacy active image its own display-name experiment."""
    del schema_editor
    Asset = apps.get_model("assets", "Asset")
    Experiment = apps.get_model("library", "Experiment")

    used_names = set(Experiment.objects.values_list("name", flat=True))
    assets = Asset.objects.filter(
        lifecycle_status="ACTIVE",
        experiment__isnull=True,
    ).order_by("created_at", "id")
    for asset in assets.iterator():
        base = str(asset.display_name or "").strip() or "Untitled image"
        number = 1
        while True:
            suffix = "" if number == 1 else f" ({number})"
            candidate = f"{base[: max(1, 255 - len(suffix))]}{suffix}"
            if candidate not in used_names:
                break
            number += 1
        experiment = Experiment.objects.create(name=candidate)
        used_names.add(candidate)
        Asset.objects.filter(id=asset.id).update(experiment_id=experiment.id)


class Migration(migrations.Migration):
    dependencies = [
        ("assets", "0005_asset_datasets_asset_experiment"),
        ("library", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(give_active_images_experiments, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="asset",
            name="experiment",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                related_name="assets",
                to="library.experiment",
            ),
        ),
        migrations.AddConstraint(
            model_name="asset",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(lifecycle_status="DELETED")
                    | models.Q(experiment__isnull=False)
                ),
                name="active_asset_requires_experiment",
            ),
        ),
    ]
