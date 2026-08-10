/**
 * What removing a queued job does to the thing that job was carrying.
 *
 * Every long-running action in QuantEM is two rows: a `Job` the queue owns and a
 * domain object a screen polls — an `ImageSegmentation`, an `AnalysisRun`, an
 * `Adapter`, an `Asset`'s preprocessing. `DELETE /api/jobs/<id>/` is the only
 * exit a queued job has (`POST .../cancel/` refuses anything that is not
 * RUNNING), and it hard-deletes the row, so nothing downstream can ever explain
 * the leftover. The confirmation said only *"This will not run the task."*,
 * which is true of the job and silent about the record left behind.
 *
 * The endpoint now reconciles on the way out
 * (`quantem.jobs.failure_reconcile.reconcile_domain_objects_for_removed_job`),
 * so there is finally something definite to promise. These sentences describe
 * that reconciliation — the job type is the key on both sides, so a job type
 * gaining a domain object means adding an entry in both places.
 *
 * Keys are `quantem.jobs.constants`' job-type strings, which arrive verbatim as
 * `JobQueueItem.type`. An unrecognised type gets `null`: a build newer than this
 * screen must not have a consequence invented for it.
 */

/** Job types whose handler owns a record a screen reads. */
const CONSEQUENCE_BY_JOB_TYPE: Record<string, string> = {
  // `_reconcile_segmentation` moves the stage to FAILED with a sentence, and
  // deliberately leaves a COMPLETED segmentation alone. Objects already in it
  // are untouched: only the stage and its error text change.
  run_segmentation_full_task:
    "The segmentation this run belongs to is marked Failed with a note saying it never started. Objects already in it are kept, and nothing is deleted — start the run again from the labeling screen when you are ready.",
  run_segmentation_roi_task:
    "The segmentation this run belongs to is marked Failed with a note saying it never started. Objects already in it are kept, and nothing is deleted — start the run again from the labeling screen when you are ready.",
  // `_reconcile_analysis_run`. Without this the Analysis screen shows a history
  // row reading PENDING and the sentence "This run is pending. Results appear
  // when it finishes", about a run nothing will ever pick up.
  run_analysis:
    "The analysis run is marked Failed rather than left sitting at Pending forever. No results were written, and earlier runs are untouched — Run analysis again when you are ready.",
  // `_reconcile_adapter`. The Adapt wizard reads the adapter row to decide what
  // is in flight, so a stranded PENDING one makes the wizard permanently
  // unusable for that segmentation.
  train_organelle_adapter:
    "The adapter this run would have produced is marked Failed, so the Adapt wizard stops waiting on it and lets you start again. No weights were written and no applied adapter changes.",
  // `_reconcile_asset_preprocessing`.
  upload_image_pipeline:
    "The image's preprocessing is marked Failed. The file stays in the library, but it cannot be viewed or segmented until the processing runs again.",
  ensure_image_ngff:
    "The image's preprocessing is marked Failed. The file stays in the library, but it cannot be viewed or segmented until the processing runs again.",
  // No reconciler, because there is no record: these rebuild a derived artifact
  // from data that is already saved.
  rebuild_segmentation_overlay:
    "Nothing is lost: this only rebuilds the overlay picture from objects already saved. The overlay stays as it is until something else refreshes it.",
  refresh_segment_features:
    "Nothing is lost: this only recomputes measurements from outlines already saved. Those measurements stay as they are until something else refreshes them.",
};

/**
 * One sentence for the confirmation, or `null` when this build cannot say.
 *
 * `null` is not "nothing happens" — it is "this screen does not know", and the
 * dialog says nothing rather than promising something it cannot check.
 */
export function describeJobRemoval(jobType: string | undefined): string | null {
  if (!jobType) return null;
  return CONSEQUENCE_BY_JOB_TYPE[jobType] ?? null;
}
