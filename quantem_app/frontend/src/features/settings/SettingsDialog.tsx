import { useEffect, useId, useRef, useState } from "react";

import type { SystemStatus } from "@/shared/types/jobs";
import { Button } from "@/shared/ui/design";
import { ModelManagementSection } from "@/features/models/ModelManagementSection";
import { useDesktopUpdate } from "@/features/update/desktopUpdateHooks";
import {
  getSystemSettings,
  updateSystemSettings,
} from "@/shared/api/systemSettings";

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
  const [backupSettingsSaving, setBackupSettingsSaving] = useState(false);
  const [backupSettingsError, setBackupSettingsError] = useState<string | null>(null);
  const [backupSettingsSaved, setBackupSettingsSaved] = useState<string | null>(null);
  const updateBusy = ["downloading", "downloaded", "waiting", "blocked", "applying"].includes(
    desktopUpdate.phase
  );

  useEffect(() => {
    if (!isOpen) return undefined;
    let active = true;
    setBackupSettingsLoading(true);
    setBackupSettingsError(null);
    setBackupSettingsSaved(null);
    void getSystemSettings()
      .then((loaded) => {
        if (!active) return;
        setBackupCount(String(loaded.pre_migration_backup_retention_count));
        setBackupDays(String(loaded.pre_migration_backup_retention_days));
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
  const backupSettingsValid =
    Number.isInteger(parsedBackupCount) &&
    parsedBackupCount >= 1 &&
    parsedBackupCount <= 100 &&
    Number.isInteger(parsedBackupDays) &&
    parsedBackupDays >= 1 &&
    parsedBackupDays <= 3650;

  const saveBackupSettings = async () => {
    if (!backupSettingsValid) return;
    setBackupSettingsSaving(true);
    setBackupSettingsError(null);
    setBackupSettingsSaved(null);
    try {
      const updated = await updateSystemSettings({
        pre_migration_backup_retention_count: parsedBackupCount,
        pre_migration_backup_retention_days: parsedBackupDays,
      });
      setBackupCount(String(updated.pre_migration_backup_retention_count));
      setBackupDays(String(updated.pre_migration_backup_retention_days));
      const removed = updated.removed_pre_migration_backups ?? 0;
      setBackupSettingsSaved(
        removed
          ? `Saved. ${removed} older backup${removed === 1 ? " was" : "s were"} removed.`
          : "Saved."
      );
    } catch (error) {
      setBackupSettingsError(
        error instanceof Error ? error.message : "Backup settings could not be saved."
      );
    } finally {
      setBackupSettingsSaving(false);
    }
  };

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
              <input
                className="h-10 rounded-lg border border-slate-300 px-3 text-slate-950 disabled:bg-slate-100"
                type="number"
                min={1}
                max={100}
                step={1}
                value={backupCount}
                disabled={backupSettingsLoading || backupSettingsSaving}
                onChange={(event) => setBackupCount(event.target.value)}
              />
            </label>
            <label className="grid gap-1.5 text-sm font-medium text-slate-800">
              Delete after (days)
              <input
                className="h-10 rounded-lg border border-slate-300 px-3 text-slate-950 disabled:bg-slate-100"
                type="number"
                min={1}
                max={3650}
                step={1}
                value={backupDays}
                disabled={backupSettingsLoading || backupSettingsSaving}
                onChange={(event) => setBackupDays(event.target.value)}
              />
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
          <div className="mt-3">
            <Button
              type="button"
              size="sm"
              disabled={
                backupSettingsLoading || backupSettingsSaving || !backupSettingsValid
              }
              onClick={() => void saveBackupSettings()}
            >
              {backupSettingsSaving ? "Saving…" : "Save backup settings"}
            </Button>
          </div>
        </div>

        <div className="mt-6 border-t border-slate-200 pt-5">
          <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="m-0 text-lg font-semibold text-slate-950">Models</h2>
            </div>
          </div>
          <ModelManagementSection compact />
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
