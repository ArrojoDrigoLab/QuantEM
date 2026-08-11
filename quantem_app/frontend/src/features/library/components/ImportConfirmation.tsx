/**
 * The receipt for an import, and the announcement of where it is about to take
 * you.
 *
 * Two reported defects live here.
 *
 * **"Seemed to have failed."** There was no confirmation of any kind. The
 * import form cleared its fields and collapsed on success, so a completed
 * import and a reset form looked identical, and nothing anywhere said the image
 * had arrived or where it had gone. This strip says both, stays until it is
 * dismissed, and names the card it is pointing at.
 *
 * **The unannounced navigation.** `LibraryPage` has always jumped into the
 * viewer once the pyramid finished, which was measured at ~100 s on a 475 MP
 * image -- a route change the user did not ask for, long after the action that
 * caused it, interrupting whatever they had moved on to. The owner does want to
 * land in the viewer (ask #3), so the behaviour stays and the surprise goes:
 * the strip says up front that it will happen, shows the preparation
 * percentage, counts down out loud when the image is ready, and can be stopped
 * at any point in that sequence. Once stopped it stays stopped, and the strip
 * becomes a plain "Open it" button.
 *
 * ## A batch has no "it"
 *
 * Importing forty images is now one trip through the panel, so this strip has
 * to answer a different question: not *where did my image go* but *how is the
 * plate doing*. It counts -- ready, still preparing, failed -- and it does not
 * navigate anywhere. There is no defensible answer to "which of the forty
 * should the app open", and jumping into one image while thirty-nine are still
 * being prepared is exactly the unannounced route change this component exists
 * to have removed. The single-image behaviour is untouched.
 */

import { Button, Panel } from "@/shared/ui/design";
import { getStageDisplay } from "@/features/library/components/imageCardUtils";
import type { HomeEntry } from "@/shared/types/images";

export interface ImportConfirmationProps {
  /** Everything imported in this session, oldest first. Never empty. */
  entries: HomeEntry[];
  /** Whether the library is still going to open this image by itself. */
  autoOpen: boolean;
  /** Seconds left on the announced countdown, or null when it is not running. */
  countdownSeconds: number | null;
  onOpenNow: () => void;
  onStayHere: () => void;
  onDismiss: () => void;
}

function hasFailed(entry: HomeEntry): boolean {
  return (
    entry.preprocess_stage === "FAILED" || entry.preprocess_stage === "CANCELLED"
  );
}

export function ImportConfirmation({
  entries,
  autoOpen,
  countdownSeconds,
  onOpenNow,
  onStayHere,
  onDismiss,
}: ImportConfirmationProps) {
  const single = entries.length === 1 ? entries[0] : null;
  const failedEntries = entries.filter(hasFailed);
  const readyCount = entries.filter(
    (entry) => entry.ngff_ready && !hasFailed(entry)
  ).length;
  const preparingCount = entries.length - readyCount - failedEntries.length;
  const failed = single !== null && hasFailed(single);
  const ready = single !== null && Boolean(single.ngff_ready);
  const anythingFailed = failedEntries.length > 0;

  return (
    <Panel
      className={
        anythingFailed
          ? "flex flex-wrap items-center justify-between gap-3 border-red-200 bg-red-50 p-4"
          : "flex flex-wrap items-center justify-between gap-3 border-cyan-200 bg-cyan-50 p-4"
      }
      // `status`, not `alert`: this is a success announcement, and an assertive
      // live region would interrupt a screen-reader user mid-sentence once a
      // second for the whole of a 100 s preparation.
      role="status"
      aria-live="polite"
      data-testid="import-confirmation"
    >
      <div className="min-w-0">
        <p className="m-0 text-sm font-semibold text-slate-900">
          {single
            ? `Imported ${single.display_name}`
            : `Imported ${entries.length} images`}
        </p>
        {single ? (
          <p className="m-0 mt-1 text-sm text-slate-700">
            {failed ? (
              <>
                It is the first card below, and it could not be prepared:{" "}
                {single.preprocess_error?.trim() ||
                  "the server did not say why. Tasks & Queues has the job."}
              </>
            ) : ready ? (
              <>
                It is ready, and it is the first card below.
                {countdownSeconds !== null
                  ? ` Opening it in ${countdownSeconds}…`
                  : autoOpen
                    ? " Opening it now."
                    : " It is not going anywhere; open it when you want."}
              </>
            ) : (
              <>
                It is the first card below — {getStageDisplay(single).toLowerCase()}.
                {autoOpen
                  ? " I will open it here when it is ready."
                  : " Staying in the library."}
              </>
            )}
          </p>
        ) : (
          <>
            <p className="m-0 mt-1 text-sm text-slate-700">
              They are the first cards below. {readyCount} ready
              {preparingCount > 0 ? ` · ${preparingCount} still preparing` : ""}
              {failedEntries.length > 0 ? ` · ${failedEntries.length} failed` : ""}
              .{" "}
              {/* No navigation for a batch, said out loud: the app is not
                  going to pick one of forty images and jump into it. */}
              {preparingCount > 0
                ? "Staying here while they finish; open any of them from its card."
                : "Open any of them from its card."}
            </p>
            {failedEntries.length > 0 ? (
              <ul className="m-0 mt-1 list-none p-0 text-sm text-red-700">
                {failedEntries.map((entry) => (
                  <li key={entry.id}>
                    {entry.display_name} could not be prepared:{" "}
                    {entry.preprocess_error?.trim() ||
                      "the server did not say why. Tasks & Queues has the job."}
                  </li>
                ))}
              </ul>
            ) : null}
          </>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {single && !failed && (
          <Button variant="primary" onClick={onOpenNow}>
            {ready ? "Open it now" : "Open it when ready"}
          </Button>
        )}
        {single && !failed && autoOpen && (
          <Button onClick={onStayHere}>Stay in the library</Button>
        )}
        <Button
          variant="ghost"
          size="icon"
          aria-label="Dismiss the import confirmation"
          onClick={onDismiss}
        >
          ✕
        </Button>
      </div>
    </Panel>
  );
}
