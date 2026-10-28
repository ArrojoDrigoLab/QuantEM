from django.db import migrations, models
from django.db.models import Count, Q


def assert_unique_batch_positions(apps, schema_editor):
    Job = apps.get_model("jobs", "Job")
    duplicates = list(
        Job.objects.using(schema_editor.connection.alias)
        .exclude(batch_id="")
        .values("batch_id", "batch_seq")
        .annotate(row_count=Count("id"))
        .filter(row_count__gt=1)
        .values_list("batch_id", "batch_seq")[:20]
    )
    if duplicates:
        positions = ", ".join(f"{batch_id}[{sequence}]" for batch_id, sequence in duplicates)
        raise RuntimeError(
            "Cannot add the PostgreSQL job concurrency guard because duplicate batch "
            f"positions already exist: {positions}. No jobs were changed."
        )


class Migration(migrations.Migration):
    dependencies = [("jobs", "0004_update_maintenance")]

    operations = [
        migrations.RunPython(assert_unique_batch_positions, migrations.RunPython.noop),
        migrations.AddConstraint(
            model_name="job",
            constraint=models.UniqueConstraint(
                fields=("batch_id", "batch_seq"),
                condition=~Q(batch_id=""),
                name="unique_job_position_in_batch",
            ),
        ),
    ]
