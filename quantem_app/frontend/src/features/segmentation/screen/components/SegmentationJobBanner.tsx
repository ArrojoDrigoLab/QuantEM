/**
 * The run panel on the labeling screen: the owner's three indicators.
 *
 * What was here before was one line per job reading `task_label` and a bare
 * `Math.round(job.progress)%`. During a real 56-tile run a user watching this
 * screen saw "Run full-image segmentation  5%" and then a percentage that
 * climbed; the tile counts the backend had been writing since wave 0b reached
 * no screen at all, and the only tile count anywhere in the product was the
 * free-text `DINO: 57% (Tile 32/56)` in the Tasks drawer.
 *
 * Now: one tiles-primary line per organelle, and any model download as a
 * visibly different kind of row. A batch can span many images, so its total is
 * deliberately not rendered as though it belonged to the image on this
 * screen. All visible counts come from the task's own structured fields; none
 * are parsed out of a message.
 *
 * The panel also outlives the run. Pressing Cancel used to empty it within one
 * poll, taking the tile count with it, so the app's answer to "how far did it
 * get?" was nothing at all. A stopped run keeps its row, with the count it
 * reached and who stopped it, and the heading stops saying "Running".
 */

import { RunProgressList } from "@/shared/progress/RunProgressList";
import { buildProgressRows, runPanelTitle } from "@/shared/progress/runProgress";
import type { JobQueueItem } from "@/shared/types";
import "./SegmentationJobBanner.css";

interface SegmentationJobBannerProps {
  jobs: JobQueueItem[];
}

export function SegmentationJobBanner({
  jobs,
}: SegmentationJobBannerProps) {
  const rows = buildProgressRows(jobs, { includeAggregate: false });
  if (rows.length === 0) {
    return null;
  }

  return (
    <div className="segmentation-job-banner" data-testid="segmentation-run-panel">
      <div className="segmentation-job-title">{runPanelTitle(jobs)}</div>
      <RunProgressList className="segmentation-run-progress" rows={rows} />
    </div>
  );
}
