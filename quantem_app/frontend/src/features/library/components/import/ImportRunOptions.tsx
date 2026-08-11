/**
 * "Start a segmentation run now": the four organelle boxes, behind a closed
 * disclosure.
 *
 * Split out of `ImageUploadPanel.tsx` unchanged. Each box queues a whole-image
 * inference pass -- minutes to tens of minutes, *per image* -- and nothing is
 * ticked by default. They are behind a disclosure because the owner's ask was
 * "just having optional add-ons for the resolution and notes", and four run
 * checkboxes on the import surface are not that. They are not deleted: the plan
 * ports this exact batch pattern into the viewer, and deleting a capability
 * before its replacement lands is how users lose work.
 */

import {
  DEFAULT_PACK_FOR_ORGANELLE,
} from "@/features/models/scaleMismatch";
import { runnabilityForPackId } from "@/features/models/runnable";
import { plural } from "@/features/library/components/import/importValidation";
import { ORGANELLE_CHOICES } from "@/features/library/components/import/organelles";
import type { OrganelleId } from "@/features/library/components/import/organelles";
import type { ModelCatalogue } from "@/shared/types/finetune";

export function ImportRunOptions({
  runOptionsId,
  runOptionsOpen,
  onToggleRunOptions,
  runCount,
  fileCount,
  tickedOrganelles,
  onToggleOrganelle,
  catalogue,
}: {
  runOptionsId: string;
  runOptionsOpen: boolean;
  onToggleRunOptions: () => void;
  runCount: number;
  fileCount: number;
  tickedOrganelles: OrganelleId[];
  onToggleOrganelle: (id: OrganelleId, checked: boolean) => void;
  catalogue: ModelCatalogue | null;
}) {
  return (
    <div className="rounded-md border border-slate-200">
      <button
        type="button"
        className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm font-medium text-slate-800 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-cyan-500"
        aria-expanded={runOptionsOpen}
        aria-controls={runOptionsId}
        onClick={() => onToggleRunOptions()}
      >
        <span>Start a segmentation run now</span>
        <span className="text-xs font-normal text-slate-600">
          {runCount === 0
            ? "Nothing selected"
            : `${runCount} ${plural(runCount, "run")} selected`}
          {fileCount > 1 && runCount > 0
            ? ` (${tickedOrganelles.length} per image)`
            : ""}
          {runOptionsOpen ? " ▾" : " ▸"}
        </span>
      </button>
      {runOptionsOpen && (
        <div id={runOptionsId} className="flex flex-col gap-1 px-3 pb-3">
          {ORGANELLE_CHOICES.map((choice) => {
            const runnability = runnabilityForPackId(
              catalogue,
              DEFAULT_PACK_FOR_ORGANELLE[choice.id]
            );
            // Only a definite "blocked" disables, exactly as on the
            // labeling screen: an unanswered catalogue must not hide a
            // working model.
            const blocked = runnability.state === "blocked";
            return (
              <label
                key={choice.id}
                className="flex items-center gap-2 text-sm text-slate-800"
                htmlFor={choice.inputId}
              >
                <input
                  id={choice.inputId}
                  type="checkbox"
                  checked={tickedOrganelles.includes(choice.id)}
                  disabled={blocked}
                  onChange={(e) =>
                    onToggleOrganelle(choice.id, e.target.checked)
                  }
                />
                {choice.label}
                {blocked && (
                  <span className="text-xs text-slate-600">
                    {" — "}
                    {runnability.reason ??
                      `${DEFAULT_PACK_FOR_ORGANELLE[choice.id]} cannot run here.`}
                  </span>
                )}
              </label>
            );
          })}
          {/* A disabled control with nothing beside it reads as a bug,
              and the Models screen is the only place the blocker is
              fixable. */}
          <span className="mt-1 text-xs text-slate-600">
            Nothing is segmented unless you tick a box. Each one queues a
            separate whole-image run per image after the import — minutes
            to tens of minutes each — and you can start any of them later
            from the labeling screen instead. Models that cannot run here
            are listed but not offered — <a href="#/models">Models</a> says
            what is missing.
          </span>
        </div>
      )}
    </div>
  );
}
