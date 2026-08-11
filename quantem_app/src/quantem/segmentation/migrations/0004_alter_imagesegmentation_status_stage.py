from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("segmentation", "0003_v2_push")]

    operations = [
        migrations.AlterField(
            model_name="imagesegmentation",
            name="status_stage",
            field=models.CharField(
                choices=[
                    ("UNSTARTED", "Unstarted"),
                    ("RUNNING_INFERENCE", "Running inference"),
                    ("THRESHOLD_READY", "Threshold ready"),
                    ("EXTRACTING_CANDIDATES", "Extracting candidates"),
                    ("CANDIDATES_READY", "Candidates ready"),
                    ("UPDATING", "Updating with feedback"),
                    ("COMPUTING_FEATURES", "Computing features"),
                    ("COMPLETED", "Completed"),
                    ("FAILED", "Failed"),
                ],
                default="UNSTARTED",
                max_length=50,
            ),
        ),
    ]
