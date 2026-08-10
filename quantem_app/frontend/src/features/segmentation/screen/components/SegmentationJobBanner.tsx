import type { JobQueueItem } from "@/shared/types";

interface SegmentationJobBannerProps {
  jobs: JobQueueItem[];
}

export function SegmentationJobBanner({ jobs }: SegmentationJobBannerProps) {
  if (jobs.length === 0) {
    return null;
  }

  return (
    <div className="segmentation-job-banner">
      <div className="segmentation-job-title">Processing segmentation assets</div>
      {jobs.map((job) => (
        <div key={job.id} className="segmentation-job-row">
          <span className="segmentation-job-label">{job.task_label}</span>
          <span className="segmentation-job-status">
            {job.status === "RUNNING"
              ? `${Math.round(job.progress)}%`
              : job.status === "PENDING" || job.status === "RETRY"
                ? "Queued"
                : job.status}
          </span>
        </div>
      ))}
    </div>
  );
}
