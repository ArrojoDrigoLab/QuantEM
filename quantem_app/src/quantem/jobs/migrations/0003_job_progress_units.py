"""Every progress column the v2 plan needs, in one migration.

Written once, for the whole plan, rather than a column at a time as each
package lands: parallel agents authoring competing migrations against the same
app is the failure mode this avoids. ``batch_id`` and ``batch_seq`` are not read
until the set-wide packages; they are here so nobody has to add them later.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("jobs", "0002_job_progress_current_bytes_job_progress_total_bytes"),
    ]

    operations = [
        migrations.AddField(
            model_name="job",
            name="progress_units_done",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="job",
            name="progress_units_total",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="job",
            name="progress_unit_label",
            field=models.CharField(blank=True, default="", max_length=32),
        ),
        migrations.AddField(
            model_name="job",
            name="progress_stage",
            field=models.CharField(blank=True, default="", max_length=32),
        ),
        migrations.AddField(
            model_name="job",
            name="progress_detail_json",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="job",
            name="batch_id",
            field=models.CharField(blank=True, db_index=True, default="", max_length=128),
        ),
        migrations.AddField(
            model_name="job",
            name="batch_seq",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddIndex(
            model_name="job",
            index=models.Index(
                fields=["batch_id", "status"], name="jobs_job_batch_i_51b10e_idx"
            ),
        ),
    ]
