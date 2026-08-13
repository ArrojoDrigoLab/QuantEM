/** Multi-organelle model controls used by the Viewer. */

import { formatBytes } from "@/shared/ui/format";
import type { RunPlan, RunPlanOrganelle } from "@/shared/types/runs";

/**
 * Tick several organelles, press one button, get one run.
 *
 * The owner's ask, verbatim: *"In the Viewer, select MULTIPLE organelles,
 * flagging any whose model needs downloading, and start them with ONE button."*
 * Before this, running four organelles meant four passes through a create
 * dialog, each of which queued its own job the moment the previous POST
 * returned.
 *
 * Three things this gets right that the dialog it replaces did not:
 *
 * * **Ticking a model that is not on this machine is allowed.** It downloads
 *   in the background and the run waits for it, so the checkbox is not a wall.
 *   The cost is stated on the row and aggregated in one line below.
 * * **The aggregate download figure is deduped.** Packs in one family share an
 *   encoder blob; adding the row figures up overstates the total by 2.62x
 *   across the eight released packs, on the one control whose job is to state
 *   a cost.
 * * **A model that cannot run here at all** — as opposed to one that is merely
 *   not downloaded — says so in the server's own words and cannot be ticked.
 *   Offering it would queue a run that dies minutes later.
 */
export function OrganelleRunChecklist({
  plan,
  ticked,
  onToggle,
  onStart,
  starting,
  startError,
  disabled = false,
  disabledReason,
}: {
  plan: RunPlan | null;
  ticked: string[];
  onToggle: (organelle: string) => void;
  onStart: () => void;
  starting: boolean;
  startError?: string | null;
  disabled?: boolean;
  disabledReason?: string;
}) {
  const entries = plan?.organelles ?? [];
  const chosen = entries.filter((entry) => ticked.includes(entry.organelle));
  const blocked = chosen.filter(
    (entry) => !entry.model_ready && entry.model_installed
  );
  const nothingChosen = chosen.length === 0;
  const cannotStart = disabled || starting || nothingChosen || blocked.length > 0;

  return (
    <section className="run-organelles" aria-labelledby="run-organelles-heading">
      <h3 id="run-organelles-heading">Find</h3>
      <ul className="run-organelles-list">
        {entries.map((entry) => (
          <OrganelleChoice
            key={entry.organelle}
            entry={entry}
            checked={ticked.includes(entry.organelle)}
            onToggle={() => onToggle(entry.organelle)}
            disabled={disabled || starting}
          />
        ))}
      </ul>

      {plan && plan.download_bytes_total > 0 && (
        <p className="run-organelles-download">
          {`Also downloading ${formatBytes(
            plan.download_bytes_total
          )} before this can start.`}
        </p>
      )}

      <FinishOrder chosen={chosen} />

      <button
        type="button"
        className="run-organelles-start"
        onClick={onStart}
        disabled={cannotStart}
        title={
          disabled
            ? disabledReason
            : nothingChosen
              ? "Tick at least one thing to find."
              : blocked.length > 0
                ? blocked[0].model_blocked_reason ?? undefined
                : undefined
        }
      >
        {starting ? "Starting..." : startButtonLabel(chosen)}
      </button>

      {blocked.length > 0 && (
        <p className="run-organelles-blocked" role="status">
          {`${blocked[0].title} cannot run on this machine. ${
            blocked[0].model_blocked_reason ?? ""
          }`}
        </p>
      )}
      {startError && (
        <p className="run-organelles-error" role="alert">
          {startError}
        </p>
      )}
    </section>
  );
}

function OrganelleChoice({
  entry,
  checked,
  onToggle,
  disabled,
}: {
  entry: RunPlanOrganelle;
  checked: boolean;
  onToggle: () => void;
  disabled: boolean;
}) {
  // Not installed and not runnable are different facts with different fixes,
  // and collapsing them is what made a downloadable pack look broken.
  const needsDownload = !entry.model_installed && entry.download_bytes > 0;
  const cannotRunHere = entry.model_installed && !entry.model_ready;
  return (
    <li className="run-organelles-item">
      <label>
        <input
          type="checkbox"
          checked={checked}
          onChange={onToggle}
          disabled={disabled || cannotRunHere}
        />
        <span className="run-organelles-name">{entry.name}</span>
        {needsDownload && (
          <span className="run-organelles-size">
            {`${formatBytes(entry.download_bytes)} to download`}
          </span>
        )}
        {cannotRunHere && (
          <span className="run-organelles-size">cannot run here</span>
        )}
      </label>
    </li>
  );
}

/**
 * Which of the ticked organelles will be ready first.
 *
 * Worth saying because it is a large cut in time-to-first-overlay for anyone
 * who ticks more than one: the nucleus pass over a given image is a tenth of
 * the mitochondria pass, so it lands while the long one is still running and
 * can be proofread immediately.
 *
 * Ordered by tile count and **not** quoted in minutes. A duration needs a
 * measured seconds-per-tile for this pack on this machine; until that constant
 * is wired through, saying "about a minute" would be an adjective dressed as a
 * number, which is what this screen exists to stop.
 */
function FinishOrder({ chosen }: { chosen: RunPlanOrganelle[] }) {
  const costed = chosen.filter(
    (entry): entry is RunPlanOrganelle & { tiles: number } => entry.tiles !== null
  );
  if (costed.length < 2) return null;
  const ordered = [...costed].sort((a, b) => a.tiles - b.tiles);
  const first = ordered[0];
  const last = ordered[ordered.length - 1];
  if (first.tiles === last.tiles) return null;
  return (
    <p className="run-organelles-order">
      {`${first.name} finishes first; it is the smallest pass. ${last.name} is the longest, and you can start checking ${first.name} while it runs.`}
    </p>
  );
}

function startButtonLabel(chosen: RunPlanOrganelle[]): string {
  if (chosen.length === 0) return "Find";
  if (chosen.length === 1) return `Find ${chosen[0].name.toLowerCase()}`;
  if (chosen.length === 2) {
    return `Find ${chosen[0].name.toLowerCase()} and ${chosen[1].name.toLowerCase()}`;
  }
  return `Find these ${chosen.length}`;
}
