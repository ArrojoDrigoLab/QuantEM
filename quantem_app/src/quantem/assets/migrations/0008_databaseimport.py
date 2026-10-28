import uuid

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("assets", "0007_postgresql_concurrency_guards")]

    operations = [
        migrations.CreateModel(
            name="DatabaseImport",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("source_backend", models.CharField(default="sqlite", max_length=16)),
                ("source_sha256", models.CharField(max_length=64, unique=True)),
                ("source_schema_migrations", models.JSONField(blank=True, default=list)),
                ("snapshot_path", models.CharField(max_length=2048)),
                ("automatic", models.BooleanField(default=False)),
                ("deactivated_job_ids", models.JSONField(blank=True, default=list)),
            ],
            options={"ordering": ["-created_at"]},
        )
    ]
