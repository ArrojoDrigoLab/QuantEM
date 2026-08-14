import { useDesktopUpdate } from "@/features/update/desktopUpdateHooks";
import "./DesktopUpdateBanner.css";

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB"];
  let current = value;
  let index = -1;
  do {
    current /= 1024;
    index += 1;
  } while (current >= 1024 && index < units.length - 1);
  return `${current >= 10 ? current.toFixed(0) : current.toFixed(1)} ${units[index]}`;
}

export function DesktopUpdateBanner() {
  const { phase, update, progress, openJobs, restartBlocker, error, upgradeNow } =
    useDesktopUpdate();

  if (!update || phase === "idle" || phase === "checking" || phase === "up-to-date") return null;

  const progressText =
    progress.total !== null
      ? `${formatBytes(progress.downloaded)} of ${formatBytes(progress.total)}`
      : formatBytes(progress.downloaded);
  const percent =
    progress.total && progress.total > 0
      ? Math.min(100, Math.round((progress.downloaded / progress.total) * 100))
      : null;

  return (
    <aside className="desktop-update-banner" aria-live="polite">
      <div className="desktop-update-banner-copy">
        <strong>QuantEM {update.version} is available.</strong>
        {phase === "available" && <span>Download it now; your data stays where it is.</span>}
        {phase === "downloading" && <span>Downloading update: {progressText}{percent !== null ? ` (${percent}%)` : ""}.</span>}
        {phase === "downloaded" && <span>Update downloaded. Preparing a safe restart.</span>}
        {phase === "waiting" && <span>Update downloaded. Waiting for {openJobs} active {openJobs === 1 ? "task" : "tasks"} to finish.</span>}
        {phase === "blocked" && <span>{restartBlocker || "Finish or clear unsaved work before QuantEM restarts."}</span>}
        {phase === "applying" && <span>Applying the verified update and restarting QuantEM.</span>}
        {phase === "error" && <span>{error || "The update could not be completed."}</span>}
      </div>
      <div className="desktop-update-banner-actions">
        {phase === "available" && <button type="button" onClick={() => void upgradeNow()}>Update</button>}
        {phase === "error" && (
          <button type="button" onClick={() => void upgradeNow()}>
            Retry
          </button>
        )}
        {update.body && (
          <details className="desktop-update-notes">
            <summary>What’s new</summary>
            <p>{update.body}</p>
          </details>
        )}
      </div>
    </aside>
  );
}
