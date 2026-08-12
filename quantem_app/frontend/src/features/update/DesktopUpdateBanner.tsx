import { useCallback, useEffect, useRef, useState } from "react";
import {
  acquireUpdateApplyLock,
  getJobQueueStatus,
  releaseUpdateApplyLock,
} from "@/shared/api/jobs";
import { getDesktopUpdater, type DesktopUpdate } from "@/features/update/desktopUpdater";
import { isDesktopTauriBuild } from "@/features/update/desktopRuntime";
import { useRestartGuard } from "@/features/update/restartGuardHooks";
import "./DesktopUpdateBanner.css";

const CHECKED_AT_KEY = "quantem.desktop-update.checked-at";
const CHECK_INTERVAL_MS = 24 * 60 * 60 * 1000;
const IDLE_POLL_MS = 2_000;

type BannerState =
  | "available"
  | "downloading"
  | "downloaded"
  | "waiting"
  | "blocked"
  | "applying"
  | "error";

interface DownloadProgress {
  downloaded: number;
  total: number | null;
}

function lastCheckedAt(): number {
  try {
    return Number(window.localStorage.getItem(CHECKED_AT_KEY) || "0");
  } catch {
    return 0;
  }
}

function rememberCheck(): void {
  try {
    window.localStorage.setItem(CHECKED_AT_KEY, String(Date.now()));
  } catch {
    // Private browsing / a locked profile should not disable updates.
  }
}

function openJobCount(status: Awaited<ReturnType<typeof getJobQueueStatus>>): number {
  return status.running.length + status.queues.reduce((count, queue) => count + queue.pending.length, 0);
}

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

function isMacOs(): boolean {
  return typeof navigator !== "undefined" && /mac/i.test(navigator.userAgent);
}

export function DesktopUpdateBanner() {
  const { blockers } = useRestartGuard();
  const [state, setState] = useState<BannerState | null>(null);
  const [update, setUpdate] = useState<DesktopUpdate | null>(null);
  const [progress, setProgress] = useState<DownloadProgress>({ downloaded: 0, total: null });
  const [openJobs, setOpenJobs] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const checkInFlight = useRef(false);
  const applyInFlight = useRef(false);
  const updateRef = useRef<DesktopUpdate | null>(null);

  const checkForUpdate = useCallback(async (force = false) => {
    if (!isDesktopTauriBuild() || checkInFlight.current || updateRef.current) {
      return;
    }
    if (!force && Date.now() - lastCheckedAt() < CHECK_INTERVAL_MS) {
      return;
    }
    checkInFlight.current = true;
    try {
      const updater = await getDesktopUpdater();
      if (!updater) return;
      // Rate-limit attempted checks as well as successful responses. A failed
      // request is deliberately quiet and will retry on the next daily check.
      rememberCheck();
      const available = await updater.check();
      if (available) {
        updateRef.current = available;
        setUpdate(available);
        setState("available");
      }
    } catch {
      // Update discovery must never interrupt scientific work. The banner
      // remains absent until the next scheduled/focused check succeeds.
    } finally {
      checkInFlight.current = false;
    }
  }, []);

  const applyWhenSafe = useCallback(async () => {
    const downloadedUpdate = updateRef.current;
    if (!downloadedUpdate || applyInFlight.current) return;
    if (blockers.length > 0) {
      setState("blocked");
      return;
    }

    try {
      const queue = await getJobQueueStatus();
      const active = openJobCount(queue);
      setOpenJobs(active);
      if (active > 0) {
        setState("waiting");
        return;
      }

      const lock = await acquireUpdateApplyLock();
      setOpenJobs(lock.open_jobs);
      if (!lock.ready) {
        setState(lock.reason === "jobs_running" ? "waiting" : "error");
        if (lock.reason !== "jobs_running") {
          setError("Another update is already being applied. Restart QuantEM if it does not reopen.");
        }
        return;
      }

      applyInFlight.current = true;
      setState("applying");
      const updater = await getDesktopUpdater();
      if (!updater) throw new Error("The desktop updater is not available in this window.");
      await downloadedUpdate.install();
      // NSIS exits this process and restarts it itself. macOS installs in place,
      // so it needs the explicit relaunch after the verified archive is applied.
      if (isMacOs()) {
        await updater.relaunch();
      }
    } catch (cause) {
      await releaseUpdateApplyLock().catch(() => undefined);
      applyInFlight.current = false;
      setError(cause instanceof Error ? cause.message : "QuantEM could not apply the update.");
      setState("error");
    }
  }, [blockers.length]);

  const downloadUpdate = useCallback(async () => {
    const available = updateRef.current;
    if (!available) return;
    setError(null);
    setProgress({ downloaded: 0, total: null });
    setState("downloading");
    try {
      await available.download((event) => {
        if (event.event === "Started") {
          setProgress({ downloaded: 0, total: event.data.contentLength ?? null });
        }
        if (event.event === "Progress") {
          setProgress((current) => ({
            ...current,
            downloaded: current.downloaded + (event.data.chunkLength ?? 0),
          }));
        }
      });
      setState("downloaded");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "QuantEM could not download the update.");
      setState("error");
    }
  }, []);

  useEffect(() => {
    void checkForUpdate();
    const onFocus = () => void checkForUpdate();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [checkForUpdate]);

  useEffect(() => {
    if (state === "downloaded" || (state === "blocked" && blockers.length === 0)) {
      void applyWhenSafe();
    }
  }, [applyWhenSafe, blockers.length, state]);

  useEffect(() => {
    if (state !== "waiting") return undefined;
    const interval = window.setInterval(() => void applyWhenSafe(), IDLE_POLL_MS);
    return () => window.clearInterval(interval);
  }, [applyWhenSafe, state]);

  if (!update || !state) return null;

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
        {state === "available" && <span>Download it now; your data stays where it is.</span>}
        {state === "downloading" && <span>Downloading update: {progressText}{percent !== null ? ` (${percent}%)` : ""}.</span>}
        {state === "downloaded" && <span>Update downloaded. Preparing a safe restart.</span>}
        {state === "waiting" && <span>Update downloaded. Waiting for {openJobs} active {openJobs === 1 ? "task" : "tasks"} to finish.</span>}
        {state === "blocked" && <span>{blockers[0]?.message || "Finish or clear unsaved work before QuantEM restarts."}</span>}
        {state === "applying" && <span>Applying the verified update and restarting QuantEM.</span>}
        {state === "error" && <span>{error || "The update could not be completed."}</span>}
      </div>
      <div className="desktop-update-banner-actions">
        {state === "available" && <button type="button" onClick={() => void downloadUpdate()}>Update</button>}
        {state === "error" && (
          <button type="button" onClick={() => void (updateRef.current && progress.downloaded > 0 ? applyWhenSafe() : downloadUpdate())}>
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
