from rest_framework import serializers

from quantem.jobs.constants import (
    ALLOWED_JOB_TYPES,
    ALLOWED_QUEUE_NAMES,
    JOB_DEFAULTS,
)
from quantem.jobs.models import Job, JobArtifact, JobLog


class JobSerializer(serializers.ModelSerializer):
    class Meta:
        model = Job
        fields = [
            "id",
            "type",
            "priority",
            "status",
            "progress",
            "progress_current_bytes",
            "progress_total_bytes",
            "message",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
            "attempts",
            "max_attempts",
            "next_run_at",
            "payload_json",
            "result_json",
            "error_traceback",
            "cancel_requested",
            "resource_class",
            "queue_name",
            "tags",
            "claimed_at",
            "heartbeat_at",
        ]


class JobCreateSerializer(serializers.Serializer):
    type = serializers.ChoiceField(choices=sorted(ALLOWED_JOB_TYPES))
    payload = serializers.JSONField(default=dict)
    priority = serializers.ChoiceField(choices=["high", "default"], required=False)
    resource_class = serializers.ChoiceField(
        choices=["cpu", "gpu"], required=False
    )
    queue_name = serializers.ChoiceField(
        choices=sorted(ALLOWED_QUEUE_NAMES),
        required=False,
    )
    max_attempts = serializers.IntegerField(default=3, min_value=0)
    tags = serializers.ListField(child=serializers.CharField(), required=False)

    def validate(self, attrs):
        job_type = attrs["type"]
        defaults = JOB_DEFAULTS[job_type]
        if attrs.get("priority") is None:
            attrs["priority"] = defaults["priority"]

        resource_class = attrs.get("resource_class")
        if resource_class is None:
            attrs["resource_class"] = defaults["resource_class"]
        elif resource_class != defaults["resource_class"]:
            raise serializers.ValidationError(
                {
                    "resource_class": (
                        f"{job_type} must use resource_class={defaults['resource_class']}."
                    )
                }
            )

        queue_name = attrs.get("queue_name")
        if queue_name is None:
            attrs["queue_name"] = defaults["queue_name"]
        elif queue_name != defaults["queue_name"]:
            raise serializers.ValidationError(
                {
                    "queue_name": (
                        f"{job_type} must use queue_name={defaults['queue_name']}."
                    )
                }
            )

        return attrs


class JobLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobLog
        fields = ["timestamp", "level", "message"]


class JobArtifactSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobArtifact
        fields = ["kind", "file_path", "metadata"]
