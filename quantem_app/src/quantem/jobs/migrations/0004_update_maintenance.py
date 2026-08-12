from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("jobs", "0003_job_progress_units"),
    ]

    operations = [
        migrations.CreateModel(
            name="UpdateMaintenance",
            fields=[
                (
                    "id",
                    models.PositiveSmallIntegerField(
                        default=1,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                (
                    "state",
                    models.CharField(
                        choices=[("IDLE", "Idle"), ("APPLYING", "Applying update")],
                        default="IDLE",
                        max_length=16,
                    ),
                ),
                ("acquired_at", models.DateTimeField(blank=True, null=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "update maintenance state",
                "verbose_name_plural": "update maintenance state",
            },
        ),
    ]
