/**
 * Sidebar showing active jobs and queued work.
 *
 * "Remove" on a queued job is the one destructive control here, and it is more
 * destructive than it looks: `DELETE /api/jobs/<id>/` is the only exit a queued
 * job has, and it deletes the row, so whatever that job was carrying -- a
 * segmentation, an analysis run, an adapter -- has no queue entry left to
 * explain it. The confirmation said only "This will not run the task", which is
 * true of the job and silent about the record. `describeJobRemoval` says what
 * the endpoint's reconciliation will actually do to it.
 *
 * The collapsed "grouped job" rows (a single summary line plus a "Cancel all"
 * that bulk-deleted a whole queue with no confirmation at all) are gone. Their
 * config list was empty -- the burst producers they existed for, SAM prompting
 * and granule auto-add, are not shipped by QuantEM -- so none of it rendered,
 * and what it would have rendered was an unguarded multi-job delete.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  cancelJob,
  clearDoneJobs,
  deleteJob,
  getJobQueueStatus,
  retryJob,
} from "@/shared/api/jobs";
import { useApiMutation } from "@/shared/hooks/useApiMutation";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { describeJobRemoval } from "@/features/library/components/jobRemovalConsequence";
import { useJobNextSteps } from "@/features/library/components/jobNextSteps";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { RunProgressList } from "@/shared/progress/RunProgressList";
import {
  buildAggregateRows,
  buildProgressRows,
  hasStructuredProgress,
  isRunJob,
} from "@/shared/progress/runProgress";
import type {
  ClearDoneJobsResponse,
  JobQueueItem,
  JobQueueStatus,
  RetryJobResponse,
} from "@/shared/types/jobs";
import "./JobQueueSidebar.css";

interface JobQueueSidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const QUEUE_STATUS_POLL_MS = 3000;
const SECTION_BATCH_SIZE = 6;

type ExpandableSectionKey = "queued" | "failed" | "completed";

interface SectionShowMoreButtonProps {
  label: string;
  remainingCount: number;
  onClick: () => void;
}

const EMPTY_STATUS: JobQueueStatus = {
  running: [],
  queues: [],
  failed: [],
  completed: [],
  worker: {
    scheduler_in_process: true,
  },
  generated_at: "",
};

function formatStatus(status: string): string {
  return status.replace("_", " ").toLowerCase();
}

function formatTimeAgo(timestamp?: string | null): string {
  if (!timestamp) return "Unknown time";
  const then = new Date(timestamp).getTime();
  if (Number.isNaN(then)) return "Unknown time";
  const diffMs = Date.now() - then;
  if (diffMs <= 0) return "Just now";

  const diffMinutes = Math.floor(diffMs / 60000);
  if (diffMinutes < 1) return "Just now";
  if (diffMinutes === 1) return "1 minute ago";
  if (diffMinutes < 60) return `${diffMinutes} minutes ago`;

  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours === 1) return "1 hour ago";
  if (diffHours < 24) return `${diffHours} hours ago`;

  const diffDays = Math.floor(diffHours / 24);
  if (diffDays === 1) return "1 day ago";
  return `${diffDays} days ago`;
}

function getImageLabel(job: JobQueueItem): string {
  if (job.image?.display_name) {
    return job.image.display_name;
  }
  return "No image context";
}

function createInitialVisibleCounts(): Record<ExpandableSectionKey, number> {
  return {
    queued: SECTION_BATCH_SIZE,
    failed: SECTION_BATCH_SIZE,
    completed: SECTION_BATCH_SIZE,
  };
}

function SectionShowMoreButton({
  label,
  remainingCount,
  onClick,
}: SectionShowMoreButtonProps) {
  const nextBatchSize = Math.min(SECTION_BATCH_SIZE, remainingCount);
  return (
    <div className="job-queue-expand">
      <button
        className="job-queue-expand-button"
        type="button"
        onClick={onClick}
      >
        <span className="job-queue-expand-arrow" aria-hidden="true" />
        <span>Show {nextBatchSize} more {label}</span>
      </button>
    </div>
  );
}

/**
 * What removing this particular queued job leaves behind.
 *
 * Names the image and the segmentation as well as the consequence, because a
 * queue row's `task_label` is a job type ("Run full-image segmentation") and
 * three of them can be identical: the thing a user needs to check before
 * confirming is *which image*.
 *
 * Silent when `describeJobRemoval` does not recognise the type. A build newer
 * than this screen must get no invented promise.
 */
function JobRemovalConsequence({ job }: { job: JobQueueItem }) {
  const consequence = describeJobRemoval(job.type);
  const target = [job.image?.display_name, job.segmentation?.name]
    .filter(Boolean)
    .join(" · ");
  if (!consequence && !target) return null;
  return (
    <>
      {target ? (
        <p className="job-queue-removal-target">{target}</p>
      ) : null}
      {consequence ? <p>{consequence}</p> : null}
    </>
  );
}

export function JobQueueSidebar({ isOpen, onClose }: JobQueueSidebarProps) {
  const pollRef = useRef<number | null>(null);
  const stableEmpty = useMemo(() => EMPTY_STATUS, []);
  const [deleteTarget, setDeleteTarget] = useState<JobQueueItem | null>(null);
  const [pendingActions, setPendingActions] = useState<
    Record<string, "cancel" | "delete" | "retry">
  >({});
  const [visibleCounts, setVisibleCounts] = useState(createInitialVisibleCounts);

  const { data, loading, refetching, error, refetch } = useApiQuery(
    () => (isOpen ? getJobQueueStatus() : Promise.resolve(stableEmpty)),
    [isOpen]
  );

  const {
    mutate: cancelJobMutation,
    error: cancelError,
  } = useApiMutation((jobId: string) => cancelJob(jobId), {
    onSuccess: () => refetch(),
  });

  const {
    mutate: deleteJobMutation,
    error: deleteError,
  } = useApiMutation((jobId: string) => deleteJob(jobId), {
    onSuccess: () => {
      setDeleteTarget(null);
      refetch();
    },
  });

  const {
    mutate: retryJobMutation,
    error: retryError,
  } = useApiMutation<string, RetryJobResponse>((jobId: string) => retryJob(jobId), {
    onSuccess: () => refetch(),
  });

  const {
    mutate: clearDoneJobsMutation,
    loading: clearingDoneJobs,
    error: clearDoneError,
  } = useApiMutation<void, ClearDoneJobsResponse>(() => clearDoneJobs(), {
    onSuccess: () => refetch(),
  });

  useEffect(() => {
    if (!isOpen) {
      if (pollRef.current !== null) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }
    pollRef.current = window.setInterval(() => {
      void refetch();
    }, QUEUE_STATUS_POLL_MS);
    return () => {
      if (pollRef.current !== null) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [isOpen, refetch]);

  useEffect(() => {
    if (!isOpen) {
      setVisibleCounts(createInitialVisibleCounts());
    }
  }, [isOpen]);

  // Only the completed rows actually rendered, and only their ids: see
  // `useJobNextSteps` for why this is a string and not an array. Computed here
  // rather than beside the markup because it feeds a hook, and the early
  // return below is not allowed to sit between the two.
  const visibleCompletedIdsKey = useMemo(
    () =>
      (data?.completed ?? [])
        .slice(0, visibleCounts.completed)
        .map((job) => job.id)
        .join(","),
    [data, visibleCounts.completed]
  );
  const nextStepsByJobId = useJobNextSteps(visibleCompletedIdsKey);

  if (!isOpen) {
    return null;
  }

  const status = data ?? stableEmpty;
  const runningJobs = status.running;
  const queuedGroups = status.queues;
  const failedJobs = status.failed;
  const completedJobs = status.completed;
  const queueGroupsWithPendingJobs = queuedGroups.filter(
    (queue) => queue.pending.length > 0
  );
  // Deliberately not surfaced as a health warning.
  //
  // `scheduler_in_process` answers "does *this* server process own the
  // scheduler thread", and it reads false in configurations where jobs are
  // demonstrably running (verified against a live server: five jobs completed
  // while the flag stayed false). A red "nothing will start" banner driven by
  // it would be a false alarm at least as often as a true one, and a false
  // alarm about the queue is worse than no banner. What this panel can observe
  // directly -- what is running, queued, failed and completed -- is reported
  // instead.
  const aggregateRows = buildAggregateRows(runningJobs);
  const doneJobsCount = failedJobs.length + completedJobs.length;
  const actionError = cancelError || deleteError || retryError || clearDoneError;
  const isPendingAction = (jobId: string) => Boolean(pendingActions[jobId]);
  const showMoreSection = (section: ExpandableSectionKey) => {
    setVisibleCounts((prev) => ({
      ...prev,
      [section]: prev[section] + SECTION_BATCH_SIZE,
    }));
  };
  const { visibleQueuedGroups, queuedRemainingCount } = (() => {
    let remainingSlots = visibleCounts.queued;
    const nextQueuedGroups: Array<
      (typeof queueGroupsWithPendingJobs)[number] & { totalPendingCount: number }
    > = [];
    for (const queue of queueGroupsWithPendingJobs) {
      if (remainingSlots <= 0) break;
      const pending = queue.pending.slice(0, remainingSlots);
      if (pending.length === 0) continue;
      nextQueuedGroups.push({
        ...queue,
        pending,
        totalPendingCount: queue.pending.length,
      });
      remainingSlots -= pending.length;
    }

    const totalQueuedRows = queueGroupsWithPendingJobs.reduce(
      (count, queue) => count + queue.pending.length,
      0
    );

    return {
      visibleQueuedGroups: nextQueuedGroups,
      queuedRemainingCount: Math.max(totalQueuedRows - visibleCounts.queued, 0),
    };
  })();
  const visibleFailedJobs = failedJobs.slice(0, visibleCounts.failed);
  const failedRemainingCount = Math.max(
    failedJobs.length - visibleCounts.failed,
    0
  );
  const visibleCompletedJobs = completedJobs.slice(0, visibleCounts.completed);
  const completedRemainingCount = Math.max(
    completedJobs.length - visibleCounts.completed,
    0
  );
  const headerSubtitle = loading
    ? "Loading queue status..."
    : error
      ? "Could not fetch queue"
      : refetching
        ? "Refreshing..."
        : "Live queue status";

  const handleCancelJob = async (job: JobQueueItem) => {
    if (job.cancel_requested || isPendingAction(job.id)) return;
    setPendingActions((prev) => ({ ...prev, [job.id]: "cancel" }));
    await cancelJobMutation(job.id);
    setPendingActions((prev) => {
      const next = { ...prev };
      delete next[job.id];
      return next;
    });
  };

  const handleDeleteJob = async (job: JobQueueItem) => {
    if (isPendingAction(job.id)) return;
    setPendingActions((prev) => ({ ...prev, [job.id]: "delete" }));
    await deleteJobMutation(job.id);
    setPendingActions((prev) => {
      const next = { ...prev };
      delete next[job.id];
      return next;
    });
  };

  const handleRetryJob = async (job: JobQueueItem) => {
    if (isPendingAction(job.id)) return;
    setPendingActions((prev) => ({ ...prev, [job.id]: "retry" }));
    await retryJobMutation(job.id);
    setPendingActions((prev) => {
      const next = { ...prev };
      delete next[job.id];
      return next;
    });
  };

  const handleClearDoneJobs = async () => {
    if (doneJobsCount === 0 || clearingDoneJobs) return;
    await clearDoneJobsMutation(undefined);
  };

  return (
    <aside className="job-queue-sidebar">
      <div className="job-queue-header">
        <div>
          <h2>Task Queues</h2>
          <div className={`job-queue-subtitle ${error ? "error" : ""}`}>
            {headerSubtitle}
          </div>
        </div>
        <div className="job-queue-header-actions">
          {/* No "Restart worker": the route does not exist, and the scheduler
              lives on a thread in this process -- there is nothing to restart
              short of relaunching the app. */}
          <button className="job-queue-close" onClick={onClose} type="button">
            Close
          </button>
        </div>
      </div>

      {loading && <div className="job-queue-state">Loading queues...</div>}
      {error && (
        <div className="job-queue-state error">
          {extractApiErrorMessage(error, "The task queue could not be loaded.")}
        </div>
      )}
      {actionError && (
        <div className="job-queue-state error">
          {extractApiErrorMessage(actionError, "That action failed.")}
        </div>
      )}

      {!loading && !error && (
        <div className="job-queue-sections">
          <section className="job-queue-section">
            <h3>Running now</h3>
            {runningJobs.length === 0 ? (
              <div className="job-queue-empty">No active tasks.</div>
            ) : (
              <>
                {/* One line per image, above the runs it covers. Every job in a
                    wave carries the same rollup, so this is deduplicated by
                    wave rather than drawn inside each run's block -- otherwise
                    "Everything on Grid2_Cell04" would appear once per
                    organelle and read as several different totals. */}
                {aggregateRows.length > 0 && (
                  <RunProgressList
                    className="job-queue-run-progress"
                    rows={aggregateRows}
                    data-testid="job-queue-aggregate-progress"
                  />
                )}
                {runningJobs.map((job) => (
                  <div key={job.id} className="job-queue-item">
                    <div className="job-queue-item-row">
                      <div className="job-queue-task">{job.task_label}</div>
                      <div className={`job-queue-status ${formatStatus(job.status)}`}>
                        {job.status}
                      </div>
                    </div>
                    <div className="job-queue-actions">
                      <button
                        className="job-queue-action job-queue-action-cancel"
                        type="button"
                        onClick={() => handleCancelJob(job)}
                        disabled={job.cancel_requested || isPendingAction(job.id)}
                        title={
                          job.cancel_requested ? "Cancellation requested" : "Cancel task"
                        }
                      >
                        {job.cancel_requested ? "Cancelling..." : "Cancel"}
                      </button>
                    </div>
                    <div className="job-queue-meta">
                      <span>{getImageLabel(job)}</span>
                      {job.segmentation?.name && (
                        <span className="job-queue-segmentation">
                          {job.segmentation.name}
                        </span>
                      )}
                    </div>
                    {/* Two ways to draw a running job, and which one is right
                        depends on whether the job can count its own work.

                        A run that reports tiles gets the structured rows: the
                        wave rollup, the run's own tiles-primary line, the
                        download. What it does *not* get is `job.message`. That
                        message is where "DINO: 57% (Tile 32/56)" came from --
                        an internal codename, and a percentage on the tiling
                        plan's divisor sitting beside a bar on the whole-job
                        divisor, disagreeing by a point. The count now comes
                        from `unit_progress` and the bar divides by the same
                        total, so there is one number.

                        Everything else -- an upload, an analysis, an overlay
                        rebuild -- has only a percentage and a sentence, and
                        keeps both. */}
                    {isRunJob(job) || hasStructuredProgress(job) ? (
                      <RunProgressList
                        className="job-queue-run-progress"
                        rows={buildProgressRows([job], {
                          includeAggregate: false,
                        })}
                        data-testid={`job-progress-${job.id}`}
                      />
                    ) : (
                      <>
                        <div className="job-queue-progress">
                          <div className="job-queue-progress-bar">
                            <div
                              className="job-queue-progress-fill"
                              style={{ width: `${Math.round(job.progress)}%` }}
                            />
                          </div>
                          <span>{Math.round(job.progress)}%</span>
                        </div>
                        {job.message && (
                          <div className="job-queue-message">{job.message}</div>
                        )}
                      </>
                    )}
                  </div>
                ))}
              </>
            )}
          </section>

          <section className="job-queue-section">
            <h3>Queued</h3>
            {queueGroupsWithPendingJobs.length === 0 ? (
              <div className="job-queue-empty">No queued tasks.</div>
            ) : (
              <>
                {visibleQueuedGroups.map((queue) => (
                  <div key={queue.queue_name} className="job-queue-group">
                    <div className="job-queue-group-header">
                      <span>{queue.display_name}</span>
                      <span className="job-queue-count">{queue.totalPendingCount}</span>
                    </div>
                    {queue.pending.length === 0 ? (
                      <div className="job-queue-empty">No tasks in queue.</div>
                    ) : (
                      queue.pending.map((job) => (
                        <div key={job.id} className="job-queue-item compact">
                          <div className="job-queue-item-row">
                            <div className="job-queue-task">{job.task_label}</div>
                            <div className={`job-queue-status ${formatStatus(job.status)}`}>
                              {job.status}
                            </div>
                          </div>
                          <div className="job-queue-actions">
                            <button
                              className="job-queue-action job-queue-action-delete"
                              type="button"
                              onClick={() => setDeleteTarget(job)}
                              disabled={isPendingAction(job.id)}
                              title="Remove queued task"
                            >
                              Remove
                            </button>
                          </div>
                          <div className="job-queue-meta">
                            <span>{getImageLabel(job)}</span>
                            {job.segmentation?.name && (
                              <span className="job-queue-segmentation">
                                {job.segmentation.name}
                              </span>
                            )}
                          </div>
                          {/* How much work is waiting, not just that something
                              is. A run's tiling plan is written when it is
                              queued, so "waiting to start · 0 of 56 tiles" is a
                              fact this row can state before any worker has
                              touched it -- and a queue whose entries have no
                              size is a queue nobody can plan around. */}
                          {isRunJob(job) && hasStructuredProgress(job) && (
                            <RunProgressList
                              className="job-queue-run-progress"
                              rows={buildProgressRows([job], {
                                includeAggregate: false,
                              })}
                              data-testid={`job-progress-${job.id}`}
                            />
                          )}
                        </div>
                      ))
                    )}
                  </div>
                ))}
                {queuedRemainingCount > 0 && (
                  <SectionShowMoreButton
                    label="queued tasks"
                    remainingCount={queuedRemainingCount}
                    onClick={() => showMoreSection("queued")}
                  />
                )}
              </>
            )}
          </section>

          <div className="job-queue-done-controls">
            <button
              className="job-queue-action job-queue-action-clear-done"
              type="button"
              onClick={() => {
                void handleClearDoneJobs();
              }}
              disabled={doneJobsCount === 0 || clearingDoneJobs}
            >
              {clearingDoneJobs ? "Clearing..." : "Clear Done"}
            </button>
          </div>

          {/* "Failed" was the heading over a list the queue fills with FAILED
              *and* CANCELLED, so a run the user stopped on purpose was filed
              under a word meaning something went wrong. The heading now covers
              both and each row says which of the two happened to it. */}
          <section className="job-queue-section">
            <h3>Stopped</h3>
            {failedJobs.length === 0 ? (
              <div className="job-queue-empty">Nothing has stopped or failed.</div>
            ) : (
              <>
                {visibleFailedJobs.map((job) => (
                  <div key={job.id} className="job-queue-item compact done">
                    <div className="job-queue-item-row">
                      <div className="job-queue-task">{job.task_label}</div>
                      <div className={`job-queue-status ${formatStatus(job.status)}`}>
                        {job.status}
                      </div>
                    </div>
                    <div className="job-queue-actions">
                      <button
                        className="job-queue-action job-queue-action-retry"
                        type="button"
                        onClick={() => {
                          void handleRetryJob(job);
                        }}
                        disabled={isPendingAction(job.id)}
                        title="Run this task again"
                      >
                        {pendingActions[job.id] === "retry" ? "Retrying..." : "Retry"}
                      </button>
                    </div>
                    <div className="job-queue-meta">
                      <span>{getImageLabel(job)}</span>
                      {job.segmentation?.name && (
                        <span className="job-queue-segmentation">
                          {job.segmentation.name}
                        </span>
                      )}
                      <span className="job-queue-time">
                        {formatTimeAgo(job.finished_at)}
                      </span>
                    </div>
                    {/* How far it got before it stopped. The run's tile columns
                        outlive the run, so the one question a user has after
                        pressing Cancel -- "how much of that did I lose?" -- is
                        answerable, and until now the answer existed on the wire
                        and on no screen: this list rendered the bare word
                        "cancelled" and nothing else. Same rows, same wording as
                        the running list above, so a run reads the same before
                        and after it stops. */}
                    {isRunJob(job) && (
                      <RunProgressList
                        className="job-queue-run-progress"
                        rows={buildProgressRows([job], { includeAggregate: false })}
                        data-testid={`job-progress-${job.id}`}
                      />
                    )}
                    {/* A failure's reason is kept -- it is the only text that
                        says what went wrong. A cancelled run's message is the
                        literal word "cancelled", which the row above already
                        says better and with a count, so it is dropped rather
                        than repeated. */}
                    {job.message && !(isRunJob(job) && job.status === "CANCELLED") && (
                      <div className="job-queue-message">{job.message}</div>
                    )}
                  </div>
                ))}
                {failedRemainingCount > 0 && (
                  <SectionShowMoreButton
                    label="stopped tasks"
                    remainingCount={failedRemainingCount}
                    onClick={() => showMoreSection("failed")}
                  />
                )}
              </>
            )}
          </section>

          <section className="job-queue-section">
            <h3>Completed</h3>
            {completedJobs.length === 0 ? (
              <div className="job-queue-empty">No completed tasks.</div>
            ) : (
              <>
                {visibleCompletedJobs.map((job) => (
                  <div key={job.id} className="job-queue-item compact done">
                    <div className="job-queue-item-row">
                      <div className="job-queue-task">{job.task_label}</div>
                      <div className={`job-queue-status ${formatStatus(job.status)}`}>
                        {job.status}
                      </div>
                    </div>
                    <div className="job-queue-meta">
                      <span>{getImageLabel(job)}</span>
                      {job.segmentation?.name && (
                        <span className="job-queue-segmentation">
                          {job.segmentation.name}
                        </span>
                      )}
                      <span className="job-queue-time">
                        {formatTimeAgo(job.finished_at)}
                      </span>
                    </div>
                    {job.message && (
                      <div className="job-queue-message">{job.message}</div>
                    )}
                    {/* The job's own advice, which the backend has always
                        written and nothing has ever shown. "completed: no new
                        objects" is true and is not enough — the sentences that
                        say why that is expected over a proofread image, and
                        what to do instead of lowering the threshold, are
                        these. */}
                    {(nextStepsByJobId[job.id]?.length ?? 0) > 0 && (
                      <ul className="job-queue-next-steps">
                        {nextStepsByJobId[job.id].map((step) => (
                          <li key={step}>{step}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
                {completedRemainingCount > 0 && (
                  <SectionShowMoreButton
                    label="completed tasks"
                    remainingCount={completedRemainingCount}
                    onClick={() => showMoreSection("completed")}
                  />
                )}
              </>
            )}
          </section>
        </div>
      )}
      {/* "This will not run the task" is true of the job row and silent about
          the record behind it. Removing a queued job deletes the only trace the
          queue has, so the segmentation, analysis run or adapter it was
          carrying has nothing left to explain it -- which is why the endpoint
          concludes them on the way out, and why the dialog has to say which
          one and what happens to it. */}
      <ConfirmDialog
        isOpen={deleteTarget !== null}
        title="Remove queued task"
        message={
          deleteTarget
            ? `Remove "${deleteTarget.task_label}" from the queue? It has not started, and removing it means it never will.`
            : ""
        }
        details={
          deleteTarget ? (
            <JobRemovalConsequence job={deleteTarget} />
          ) : null
        }
        confirmText="Remove"
        cancelText="Keep"
        onConfirm={() => {
          if (deleteTarget) {
            void handleDeleteJob(deleteTarget);
          }
        }}
        onCancel={() => setDeleteTarget(null)}
      />
    </aside>
  );
}
