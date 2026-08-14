import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  acquireUpdateApplyLock,
  getJobQueueStatus,
  releaseUpdateApplyLock,
} from "@/shared/api/jobs";
import { getDesktopUpdater, type DesktopUpdate } from "@/features/update/desktopUpdater";
import { isDesktopTauriBuild } from "@/features/update/desktopRuntime";
import {
  DesktopUpdateContext,
  type DesktopUpdatePhase,
  type DesktopUpdateValue,
  type DownloadProgress,
} from "@/features/update/desktopUpdateContext";
import { useRestartGuard } from "@/features/update/restartGuardHooks";

const CHECKED_AT_KEY = "quantem.desktop-update.checked-at";
const CHECK_INTERVAL_MS = 24 * 60 * 60 * 1000;
const IDLE_POLL_MS = 2_000;

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

function isMacOs(): boolean {
  return typeof navigator !== "undefined" && /mac/i.test(navigator.userAgent);
}

export function DesktopUpdateProvider({ children }: { children: ReactNode }) {
  const { blockers } = useRestartGuard();
  const enabled = isDesktopTauriBuild();
  const [phase, setPhase] = useState<DesktopUpdatePhase>("idle");
  const [update, setUpdate] = useState<DesktopUpdate | null>(null);
  const [progress, setProgress] = useState<DownloadProgress>({ downloaded: 0, total: null });
  const [openJobs, setOpenJobs] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const checkInFlight = useRef(false);
  const applyInFlight = useRef(false);
  const downloadComplete = useRef(false);
  const updateRef = useRef<DesktopUpdate | null>(null);

  const checkForUpdate = useCallback(async (force = false) => {
    if (!enabled || checkInFlight.current || updateRef.current) {
      return;
    }
    if (!force && Date.now() - lastCheckedAt() < CHECK_INTERVAL_MS) {
      return;
    }
    checkInFlight.current = true;
    setError(null);
    if (force) {
      setPhase("checking");
    }
    try {
      const updater = await getDesktopUpdater();
      if (!updater) {
        setPhase("idle");
        return;
      }
      // Rate-limit attempted automatic checks as well as successful responses.
      // A manual check always bypasses this timestamp.
      rememberCheck();
      const available = await updater.check();
      if (available) {
        updateRef.current = available;
        setUpdate(available);
        setPhase("available");
      } else {
        setPhase("up-to-date");
      }
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "QuantEM could not check for upgrades."
      );
      setPhase("error");
    } finally {
      checkInFlight.current = false;
    }
  }, [enabled]);

  const applyWhenSafe = useCallback(async () => {
    const downloadedUpdate = updateRef.current;
    if (!downloadedUpdate || applyInFlight.current || !downloadComplete.current) return;
    if (blockers.length > 0) {
      setPhase("blocked");
      return;
    }

    try {
      const queue = await getJobQueueStatus();
      const active = openJobCount(queue);
      setOpenJobs(active);
      if (active > 0) {
        setPhase("waiting");
        return;
      }

      const lock = await acquireUpdateApplyLock();
      setOpenJobs(lock.open_jobs);
      if (!lock.ready) {
        setPhase(lock.reason === "jobs_running" ? "waiting" : "error");
        if (lock.reason !== "jobs_running") {
          setError("Another update is already being applied. Restart QuantEM if it does not reopen.");
        }
        return;
      }

      applyInFlight.current = true;
      setPhase("applying");
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
      setPhase("error");
    }
  }, [blockers]);

  const downloadUpdate = useCallback(async () => {
    const available = updateRef.current;
    if (!available) return;
    downloadComplete.current = false;
    setError(null);
    setProgress({ downloaded: 0, total: null });
    setPhase("downloading");
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
      downloadComplete.current = true;
      setPhase("downloaded");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "QuantEM could not download the update.");
      setPhase("error");
    }
  }, []);

  const checkNow = useCallback(() => checkForUpdate(true), [checkForUpdate]);
  const upgradeNow = useCallback(async () => {
    if (!updateRef.current) return;
    if (downloadComplete.current) {
      await applyWhenSafe();
      return;
    }
    await downloadUpdate();
  }, [applyWhenSafe, downloadUpdate]);

  useEffect(() => {
    void checkForUpdate();
    const onFocus = () => void checkForUpdate();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [checkForUpdate]);

  useEffect(() => {
    if (phase === "downloaded" || (phase === "blocked" && blockers.length === 0)) {
      void applyWhenSafe();
    }
  }, [applyWhenSafe, blockers.length, phase]);

  useEffect(() => {
    if (phase !== "waiting") return undefined;
    const interval = window.setInterval(() => void applyWhenSafe(), IDLE_POLL_MS);
    return () => window.clearInterval(interval);
  }, [applyWhenSafe, phase]);

  const value = useMemo<DesktopUpdateValue>(
    () => ({
      enabled,
      phase,
      update,
      progress,
      openJobs,
      restartBlocker: blockers[0]?.message ?? null,
      error,
      checkNow,
      upgradeNow,
    }),
    [blockers, checkNow, enabled, error, openJobs, phase, progress, update, upgradeNow]
  );

  return <DesktopUpdateContext.Provider value={value}>{children}</DesktopUpdateContext.Provider>;
}
