from django.urls import path

from quantem.jobs.views import (
    JobCancelView,
    JobClearDoneView,
    JobDetailView,
    JobQueueStatusView,
    JobRetryView,
    JobSubmitView,
    UpdateApplyLockView,
    UpdateApplyReleaseView,
)

urlpatterns = [
    path("jobs/", JobSubmitView.as_view(), name="job-submit"),
    path("jobs/queue-status/", JobQueueStatusView.as_view(), name="job-queue-status"),
    path("jobs/clear-done/", JobClearDoneView.as_view(), name="job-clear-done"),
    path("update-maintenance/acquire/", UpdateApplyLockView.as_view(), name="update-apply-lock"),
    path(
        "update-maintenance/release/", UpdateApplyReleaseView.as_view(), name="update-apply-release"
    ),
    path("jobs/<uuid:job_id>/", JobDetailView.as_view(), name="job-detail"),
    path("jobs/<uuid:job_id>/cancel/", JobCancelView.as_view(), name="job-cancel"),
    path("jobs/<uuid:job_id>/retry/", JobRetryView.as_view(), name="job-retry"),
]
