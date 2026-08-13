import type { ModelPack } from "@/shared/types/finetune";

function modelDownloadSize(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) {
    return "size unavailable";
  }
  return `${(bytes / 1_000_000_000).toFixed(1)}GB`;
}

function modelAvailabilityTooltip(
  pack: ModelPack | null | undefined
): string {
  if (!pack) return "Model download status is still loading.";
  if (!pack.installed) {
    return `Model is not downloaded. Will automatically download (${modelDownloadSize(
      pack.download_bytes
    )}) on first run`;
  }
  if (pack.runnable === false) {
    return pack.reason || "The downloaded model cannot run on this computer.";
  }
  return "Model is downloaded and ready to run.";
}

function modelAvailabilityGlyph(
  pack: ModelPack | null | undefined
): string {
  if (!pack) return "?";
  if (!pack.installed) return "↓";
  return pack.runnable === false ? "!" : "✓";
}

/** A keyboard-accessible status icon used beside every model choice. */
export function ModelAvailabilityIcon({
  pack,
  className = "model-availability-icon",
}: {
  pack: ModelPack | null | undefined;
  className?: string;
}) {
  const tooltip = modelAvailabilityTooltip(pack);
  const state = !pack
    ? "unknown"
    : !pack.installed
      ? "download"
      : pack.runnable === false
        ? "blocked"
        : "ready";
  return (
    <span
      className={`${className} ${className}-${state}`}
      role="img"
      tabIndex={0}
      aria-label={tooltip}
      title={tooltip}
      data-model-state={state}
    >
      {modelAvailabilityGlyph(pack)}
    </span>
  );
}
