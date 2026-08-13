/**
 * Whether a model can run here, as a badge plus (when it cannot) the reason.
 *
 * The reason is not decoration. On a clean install every pack reports "Not
 * installed yet."; on a machine without Meta's `dinov3` package the four
 * QuantEM packs report a sentence that names the exact command that fixes it.
 * Showing "unavailable" without that sentence would leave the user exactly
 * where the run-time failure left them.
 */

import { Badge } from "@/shared/ui/design";
import type { Runnability } from "@/features/models/runnable";

const TONE = {
  runnable: "good",
  downloadable: "info",
  blocked: "warning",
  unknown: "default",
} as const;

export function RunnabilityBadge({
  runnability,
  className,
}: {
  runnability: Runnability;
  className?: string;
}) {
  return (
    <Badge
      tone={TONE[runnability.state]}
      className={className}
      title={runnability.reason ?? undefined}
    >
      {runnability.label}
    </Badge>
  );
}

/**
 * Download preparation or a blocking reason as readable prose, or null when
 * the model is ready.
 *
 * Rendered as a paragraph rather than only a tooltip: these sentences are long
 * (one of them is a shell command) and a `title` attribute is not reachable by
 * keyboard or touch.
 */
export function RunnabilityReason({
  runnability,
  className,
}: {
  runnability: Runnability;
  className?: string;
}) {
  if (runnability.state === "runnable") return null;

  if (runnability.state === "unknown") {
    return (
      <p className={className ?? "m-0 mt-1 text-xs text-slate-500"}>
        This build did not report whether the model can run here. It may work;
        the catalogue did not say.
      </p>
    );
  }

  if (runnability.state === "downloadable") {
    return (
      <p className={className ?? "m-0 mt-1 text-xs text-blue-700"}>
        This model downloads automatically the first time you run it.
      </p>
    );
  }

  return (
    <p className={className ?? "m-0 mt-1 text-xs text-amber-800"}>
      {runnability.reason ??
        "This model cannot run on this machine, and the server did not say why."}
    </p>
  );
}
