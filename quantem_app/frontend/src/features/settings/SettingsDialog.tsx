import { useEffect, useId, useRef } from "react";

import type { SystemStatus } from "@/shared/types/jobs";
import { Button } from "@/shared/ui/design";
import { ModelManagementSection } from "@/features/models/ModelManagementSection";

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
            <dd className="text-sm font-semibold text-slate-950">
              {status?.app_version
                ? `v${status.app_version}`
                : statusError
                  ? "Unavailable"
                  : "Loading…"}
            </dd>
          </div>
          <div className="flex items-center justify-between gap-4 py-3">
            <dt className="text-sm font-medium text-slate-600">Compute</dt>
            <dd className="text-sm font-semibold text-slate-950">
              {status
                ? status.cuda_available
                  ? "CUDA available"
                  : "CPU"
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
          <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="m-0 text-lg font-semibold text-slate-950">Models</h2>
              <p className="mt-1 max-w-3xl text-sm text-slate-600">
                Model weights are not included in the application. Select models here
                to download / remove.
              </p>
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
