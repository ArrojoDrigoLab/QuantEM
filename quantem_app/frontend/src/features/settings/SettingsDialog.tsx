import { useEffect, useId, useRef, useState } from "react";

import type { SystemStatus } from "@/shared/types/jobs";
import { Button } from "@/shared/ui/design";
import { ModelManagementSection } from "@/features/models/ModelManagementSection";
import { useDesktopUpdate } from "@/features/update/desktopUpdateHooks";
import {
  getSystemSettings,
  updateSystemSettings,
} from "@/shared/api/systemSettings";

type BackupField = "count" | "days";
type BackupSaveState = "idle" | "saving" | "saved" | "error";

const BACKUP_AUTOSAVE_DELAY_MS = 400;
const BACKUP_SAVING_INDICATOR_MS = 1000;

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function withBackupSaveState(
  current: Record<BackupField, BackupSaveState>,
  fields: BackupField[],
  state: BackupSaveState
): Record<BackupField, BackupSaveState> {
  const next = { ...current };
  for (const field of fields) next[field] = state;
  return next;
}

function BackupSaveIndicator({
  state,
  label,
}: {
  state: BackupSaveState;
  label: string;
}) {
  if (state === "idle") return <span className="block h-5 w-5" aria-hidden="true" />;
  if (state === "saving") {
    return (
      <span className="flex h-5 w-5 items-center justify-center text-cyan-700" role="status" aria-label={`Saving ${label}`}>
        <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
          <path d="M21 12a9 9 0 0 0-9-9" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="3" />
        </svg>
      </span>
    );
  }
  if (state === "saved") {
    return (
      <span className="flex h-5 w-5 items-center justify-center text-emerald-600" role="status" aria-label={`${label} saved`}>
        <svg className="h-5 w-5" viewBox="0 0 24 24" aria-hidden="true">
          <path d="m5 12 4 4L19 6" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" />
        </svg>
      </span>
    );
  }
  return (
    <span className="flex h-5 w-5 items-center justify-center font-bold text-red-600" role="status" aria-label={`${label} could not be saved`}>
      !
    </span>
  );
}

interface SettingsDialogProps {
  isOpen: boolean;
  status: SystemStatus | null;
  statusError?: Error | null;
  onClose: () => void;
  onRetryStatus?: () => void;
}

export function SettingsDialog({
  isOpen,
  status,
  statusError = null,
  onClose,
  onRetryStatus,
}: SettingsDialogProps) {
  const titleId = useId();
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const desktopUpdate = useDesktopUpdate();
  const [backupCount, setBackupCount] = useState("1");
  const [backupDays, setBackupDays] = useState("30");
  const [backupSettingsLoading, setBackupSettingsLoading] = useState(false);
  const [backupSettingsError, setBackupSettingsError] = useState<string | null>(null);
  const [backupSettingsSaved, setBackupSettingsSaved] = useState<string | null>(null);
  const [savedBackupSettings, setSavedBackupSettings] = useState<{
    count: number;
    days: number;
  } | null>(null);
  const [backupSaveState, setBackupSaveState] = useState<
    Record<BackupField, BackupSaveState>
  >({ count: "idle", days: "idle" });
  const backupSaveGeneration = useRef(0);
  const backupSaveChain = useRef<Promise<void>>(Promise.resolve());
  const lastScheduledBackupSettings = useRef<{
    count: number;
    days: number;
  } | null>(null);
  const updateBusy = ["downloading", "downloaded", "waiting", "blocked", "applying"].includes(
    desktopUpdate.phase
  );

  useEffect(() => {
    if (!isOpen) return undefined;
    let active = true;
    setBackupSettingsLoading(true);
    setBackupSettingsError(null);
    setBackupSettingsSaved(null);
    setSavedBackupSettings(null);
    lastScheduledBackupSettings.current = null;
    setBackupSaveState({ count: "idle", days: "idle" });
    void getSystemSettings()
      .then((loaded) => {
        if (!active) return;
        setBackupCount(String(loaded.pre_migration_backup_retention_count));
        setBackupDays(String(loaded.pre_migration_backup_retention_days));
        setSavedBackupSettings({
          count: loaded.pre_migration_backup_retention_count,
          days: loaded.pre_migration_backup_retention_days,
        });
        lastScheduledBackupSettings.current = {
          count: loaded.pre_migration_backup_retention_count,
          days: loaded.pre_migration_backup_retention_days,
        };
      })
      .catch((error: unknown) => {
        if (!active) return;
        setBackupSettingsError(
          error instanceof Error ? error.message : "Backup settings could not be loaded."
        );
      })
      .finally(() => {
        if (active) setBackupSettingsLoading(false);
      });
    return () => {
      active = false;
    };
  }, [isOpen]);

  const parsedBackupCount = Number(backupCount);
  const parsedBackupDays = Number(backupDays);
  const backupCountValid =
    Number.isInteger(parsedBackupCount) &&
    parsedBackupCount >= 1 &&
    parsedBackupCount <= 100;
  const backupDaysValid =
    Number.isInteger(parsedBackupDays) &&
    parsedBackupDays >= 1 &&
    parsedBackupDays <= 3650;
  const backupSettingsValid = backupCountValid && backupDaysValid;

  useEffect(() => {
    if (!isOpen || backupSettingsLoading || !savedBackupSettings) {
      return undefined;
    }
    if (!backupSettingsValid) {
      ++backupSaveGeneration.current;
      setBackupSettingsSaved(null);
      setBackupSaveState((current) => ({
        count: backupCountValid
          ? current.count === "saving"
            ? "idle"
            : current.count
          : "error",
        days: backupDaysValid
          ? current.days === "saving"
            ? "idle"
            : current.days
          : "error",
      }));
      return undefined;
    }
    const changedFields: BackupField[] = [];
    if (parsedBackupCount !== savedBackupSettings.count) changedFields.push("count");
    if (parsedBackupDays !== savedBackupSettings.days) changedFields.push("days");
    const values = { count: parsedBackupCount, days: parsedBackupDays };
    const lastScheduled = lastScheduledBackupSettings.current ?? savedBackupSettings;
    const fieldsToWrite: BackupField[] = [];
    if (values.count !== lastScheduled.count) fieldsToWrite.push("count");
    if (values.days !== lastScheduled.days) fieldsToWrite.push("days");
    if (fieldsToWrite.length === 0) {
      if (changedFields.length > 0) {
        // These values are already scheduled or in flight. Do not invalidate
        // that request merely because another render ran this effect.
        return undefined;
      }
      // A user can revert an invalid or not-yet-saved edit to the persisted
      // value. Clear the transient indicator explicitly or the field can
      // remain stuck on an error/spinner indefinitely.
      setBackupSettingsError(null);
      setBackupSaveState((current) => ({
        count:
          current.count === "error" || current.count === "saving"
            ? "idle"
            : current.count,
        days:
          current.days === "error" || current.days === "saving"
            ? "idle"
            : current.days,
      }));
      return undefined;
    }

    const generation = ++backupSaveGeneration.current;
    lastScheduledBackupSettings.current = values;
    const savingStartedAt = Date.now();
    setBackupSettingsError(null);
    setBackupSettingsSaved(null);
    setBackupSaveState((current) => {
      const next = { ...current };
      for (const field of ["count", "days"] as const) {
        if (
          !changedFields.includes(field) &&
          (next[field] === "error" || next[field] === "saving")
        ) {
          next[field] = "idle";
        }
      }
      return withBackupSaveState(next, fieldsToWrite, "saving");
    });

    let requestStarted = false;
    const timer = window.setTimeout(() => {
      requestStarted = true;
      backupSaveChain.current = backupSaveChain.current
        .catch(() => undefined)
        .then(async () => {
          try {
            const updated = await updateSystemSettings({
              pre_migration_backup_retention_count: values.count,
              pre_migration_backup_retention_days: values.days,
            });
            const remaining =
              BACKUP_SAVING_INDICATOR_MS - (Date.now() - savingStartedAt);
            if (remaining > 0) await wait(remaining);
            const saved = {
              count: updated.pre_migration_backup_retention_count,
              days: updated.pre_migration_backup_retention_days,
            };
            // Record every completed write even when a newer edit is queued.
            // The later request remains authoritative for the inputs, but its
            // comparison baseline must reflect what the server now contains.
            setSavedBackupSettings(saved);
            if (generation !== backupSaveGeneration.current) return;
            lastScheduledBackupSettings.current = saved;
            setBackupCount(String(saved.count));
            setBackupDays(String(saved.days));
            setBackupSaveState((current) =>
              withBackupSaveState(current, fieldsToWrite, "saved")
            );
            const removed = updated.removed_pre_migration_backups ?? 0;
            setBackupSettingsSaved(
              removed
                ? `${removed} older backup${removed === 1 ? " was" : "s were"} removed.`
                : null
            );
          } catch (error) {
            if (generation !== backupSaveGeneration.current) return;
            lastScheduledBackupSettings.current = savedBackupSettings;
            setBackupSettingsError(
              error instanceof Error
                ? error.message
                : "Backup settings could not be saved."
            );
            setBackupSaveState((current) =>
              withBackupSaveState(current, fieldsToWrite, "error")
            );
          }
        });
    }, BACKUP_AUTOSAVE_DELAY_MS);

    return () => {
      window.clearTimeout(timer);
      if (
        !requestStarted &&
        lastScheduledBackupSettings.current?.count === values.count &&
        lastScheduledBackupSettings.current?.days === values.days
      ) {
        lastScheduledBackupSettings.current = savedBackupSettings;
      }
    };
  }, [
    backupSettingsLoading,
    backupSettingsValid,
    backupCountValid,
    backupDaysValid,
    isOpen,
    parsedBackupCount,
    parsedBackupDays,
    savedBackupSettings,
  ]);

  useEffect(() => {
    if (!isOpen) return undefined;
    restoreFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeButtonRef.current?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      restoreFocusRef.current?.focus();
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4"
      onClick={onClose}
    >
      <section
        className="max-h-[calc(100vh-2rem)] w-full max-w-3xl overflow-y-auto rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <h2 id={titleId} className="text-xl font-semibold text-slate-950">
            Settings
          </h2>
          <button
            ref={closeButtonRef}
            type="button"
            className="rounded-lg p-1.5 text-slate-500 transition hover:bg-slate-100 hover:text-slate-800"
            aria-label="Close settings"
            onClick={onClose}
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" aria-hidden="true">
              <path
                d="m6 6 12 12M18 6 6 18"
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeWidth="2"
              />
            </svg>
          </button>
        </div>

        <dl className="mt-5 divide-y divide-slate-200 rounded-xl border border-slate-200 px-4">
          <div className="flex items-center justify-between gap-4 py-3">
            <dt className="text-sm font-medium text-slate-600">App version</dt>
            <dd className="flex flex-wrap items-center justify-end gap-3 text-sm text-slate-950">
              <span className="font-semibold">
                {status?.app_version
                  ? `v${status.app_version}`
                  : statusError
                    ? "Unavailable"
                    : "Loading…"}
              </span>
              {desktopUpdate.phase === "up-to-date" ? <span>Latest version</span> : null}
              {desktopUpdate.enabled ? (
                <Button
                  size="sm"
                  disabled={desktopUpdate.phase === "checking" || desktopUpdate.update !== null}
                  onClick={() => void desktopUpdate.checkNow()}
                >
                  {desktopUpdate.phase === "checking"
                    ? "Checking…"
                    : desktopUpdate.phase === "up-to-date"
                      ? "Check again"
                      : "Check for upgrades"}
                </Button>
              ) : null}
            </dd>
          </div>
          {desktopUpdate.update ? (
            <div className="flex flex-wrap items-center justify-between gap-3 bg-cyan-50 py-3 text-cyan-950">
              <dt className="sr-only">Upgrade available</dt>
              <dd className="flex w-full flex-wrap items-center justify-between gap-3 text-sm font-semibold">
                <span>New version v{desktopUpdate.update.version} is available.</span>
                <Button
                  variant="primary"
                  size="sm"
                  disabled={updateBusy}
                  onClick={() => void desktopUpdate.upgradeNow()}
                >
                  {desktopUpdate.phase === "downloading"
                    ? "Downloading…"
                    : desktopUpdate.phase === "applying"
                      ? "Upgrading…"
                      : updateBusy
                        ? "Waiting…"
                        : "Upgrade now"}
                </Button>
                {desktopUpdate.phase === "error" && desktopUpdate.error ? (
                  <span className="w-full font-normal text-red-700">{desktopUpdate.error}</span>
                ) : null}
              </dd>
            </div>
          ) : desktopUpdate.phase === "error" ? (
            <div className="flex items-center justify-between gap-4 py-3">
              <dt className="sr-only">Upgrade status</dt>
              <dd className="text-sm text-red-700">
                {desktopUpdate.error || "QuantEM could not check for upgrades."}
              </dd>
            </div>
          ) : null}
          <div className="flex items-center justify-between gap-4 py-3">
            <dt className="text-sm font-medium text-slate-600">GPU Acceleration</dt>
            <dd className="text-sm font-semibold text-slate-950">
              {status
                ? status.cuda_available
                  ? "CUDA-enabled"
                  : "CPU-only"
                : statusError
                  ? "Unavailable"
                  : "Loading…"}
            </dd>
          </div>
          <div className="flex items-start justify-between gap-4 py-3">
            <dt className="shrink-0 text-sm font-medium text-slate-600">
              Storage directory
            </dt>
            <dd className="min-w-0 text-right text-sm text-slate-950">
              {status ? (
                status.storage_directory ? (
                  <code className="break-all" title={status.storage_directory}>
                    {status.storage_directory}
                  </code>
                ) : (
                  "Unavailable"
                )
              ) : statusError ? (
                "Unavailable"
              ) : (
                "Loading…"
              )}
            </dd>
          </div>
        </dl>

        {statusError ? (
          <div className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-amber-200 bg-amber-50 p-3">
            <p className="m-0 text-sm text-amber-900">
              System information could not be loaded.
            </p>
            {onRetryStatus ? (
              <Button size="sm" onClick={onRetryStatus}>
                Retry
              </Button>
            ) : null}
          </div>
        ) : null}

        <div className="mt-6 border-t border-slate-200 pt-5">
          <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="m-0 text-lg font-semibold text-slate-950">Models</h2>
            </div>
          </div>
          <ModelManagementSection compact />
        </div>

        <div className="mt-6 border-t border-slate-200 pt-5">
          <h2 className="m-0 text-lg font-semibold text-slate-950">
            Database upgrade backups
          </h2>
          <p className="mt-1 text-sm text-slate-600">
            Before an app upgrade changes the database schema, QuantEM automatically
            creates a verified database backup. The SQLite recovery snapshot created
            during the PostgreSQL upgrade uses this same retained-backup slot. Manual
            backups are not affected by this policy.
          </p>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <label className="grid gap-1.5 text-sm font-medium text-slate-800">
              Maximum backups
              <span className="grid grid-cols-[minmax(0,1fr)_1.25rem] items-center gap-2">
                <input
                  className="h-10 min-w-0 rounded-lg border border-slate-300 px-3 text-slate-950 disabled:bg-slate-100"
                  type="number"
                  min={1}
                  max={100}
                  step={1}
                  value={backupCount}
                  disabled={backupSettingsLoading}
                  onChange={(event) => setBackupCount(event.target.value)}
                />
                <BackupSaveIndicator
                  state={backupSaveState.count}
                  label="maximum backups"
                />
              </span>
            </label>
            <label className="grid gap-1.5 text-sm font-medium text-slate-800">
              Delete after (days)
              <span className="grid grid-cols-[minmax(0,1fr)_1.25rem] items-center gap-2">
                <input
                  className="h-10 min-w-0 rounded-lg border border-slate-300 px-3 text-slate-950 disabled:bg-slate-100"
                  type="number"
                  min={1}
                  max={3650}
                  step={1}
                  value={backupDays}
                  disabled={backupSettingsLoading}
                  onChange={(event) => setBackupDays(event.target.value)}
                />
                <BackupSaveIndicator
                  state={backupSaveState.days}
                  label="backup retention days"
                />
              </span>
            </label>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Defaults: one backup, removed after 30 days. Cleanup runs on app startup,
            after a migration backup, and when this policy is saved.
          </p>
          {!backupSettingsValid && !backupSettingsLoading ? (
            <p className="mt-2 text-sm text-red-700">
              Enter 1–100 backups and an age of 1–3650 days.
            </p>
          ) : null}
          {backupSettingsError ? (
            <p className="mt-2 text-sm text-red-700">{backupSettingsError}</p>
          ) : null}
          {backupSettingsSaved ? (
            <p className="mt-2 text-sm text-emerald-700">{backupSettingsSaved}</p>
          ) : null}
        </div>

        <div className="mt-5 flex justify-end">
          <Button type="button" variant="secondary" onClick={onClose}>
            Close
          </Button>
        </div>
      </section>
    </div>
  );
}
