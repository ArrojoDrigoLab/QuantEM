/**
 * The workspace's "Find" panel: tick organelles, press one button, one run.
 *
 * A thin composition on purpose -- the state is
 * `useImageRunSelection` and the markup is `OrganelleRunChecklist`, both of
 * which are unit-tested on their own. This file exists so the viewer can render
 * the whole feature with one import and one prop, and so that the run panel and
 * the rail (which are a different package's) can be dropped in beside it
 * without either side reaching into the other.
 */

import { OrganelleRunChecklist } from "@/features/segmentation/components/header/RunControls";
import { useImageRunSelection } from "@/features/viewer/state/useViewerAssetState";
import "./RunOrganellesPanel.css";

export function RunOrganellesPanel({
  assetId,
  imageReady,
  onStarted,
}: {
  assetId: string | null;
  /**
   * Whether the image itself has finished being read. A run queued against an
   * image that is still being prepared fails in the worker minutes later, so
   * the button waits and says why.
   */
  imageReady: boolean;
  onStarted?: (jobId: string) => void;
}) {
  const { plan, refetchPlan, ticked, toggle, start, starting, startError } =
    useImageRunSelection(assetId);

  return (
    <div className="run-organelles-panel">
      <OrganelleRunChecklist
        plan={plan}
        ticked={ticked}
        onToggle={toggle}
        starting={starting}
        startError={startError}
        disabled={!imageReady}
        disabledReason="This image is still being read."
        onStart={() => {
          void start().then((jobId) => {
            if (!jobId) return;
            // The plan carries "already running" state, so it is refetched the
            // moment a run starts rather than after the next poll.
            void refetchPlan();
            onStarted?.(jobId);
          });
        }}
      />
    </div>
  );
}
