/**
 * Shared confirmation dialog for primary decisions and destructive actions.
 */

import { useId, useRef, type ReactNode } from "react";
import { useModalLifecycle } from "@/shared/hooks/useModalLifecycle";
import "./ConfirmDialog.css";

export interface ConfirmDialogProps {
  isOpen: boolean;
  title: string;
  message?: string;
  /**
   * Consequences of confirming that are specific to *this* invocation --
   * counts, names, sizes. Rendered under the message and inside the dialog's
   * accessible description, so a screen reader gets the number too.
   */
  details?: ReactNode;
  /** `"warning"` when the details describe something the user may not want. */
  detailsTone?: "default" | "warning";
  confirmText?: string;
  cancelText?: string;
  confirmDisabled?: boolean;
  /** Use danger only when confirming an irreversible removal or destructive reset. */
  confirmTone?: "primary" | "danger";
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  isOpen,
  title,
  message,
  details,
  detailsTone = "default",
  confirmText = "Confirm",
  cancelText = "Cancel",
  confirmDisabled = false,
  confirmTone = "primary",
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const titleId = useId();
  const messageId = useId();
  const detailsId = useId();
  const cancelButtonRef = useRef<HTMLButtonElement | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  useModalLifecycle(panelRef, onCancel, { open: isOpen, initialFocusRef: cancelButtonRef });

  if (!isOpen) return null;

  return (
    <div className="confirm-dialog-overlay" onClick={onCancel}>
      <div
        ref={panelRef}
        className="confirm-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={
          [message ? messageId : null, details ? detailsId : null]
            .filter(Boolean)
            .join(" ") || undefined
        }
        onClick={(e) => e.stopPropagation()}
      >
        <header className="confirm-dialog-header">
          <h3 id={titleId} className="confirm-dialog-title">
            {title}
          </h3>
        </header>
        {message || details ? (
          <div className="confirm-dialog-body">
            {message ? (
              <p id={messageId} className="confirm-dialog-message">
                {message}
              </p>
            ) : null}
            {details ? (
              <div
                id={detailsId}
                className={
                  detailsTone === "warning"
                    ? "confirm-dialog-details is-warning"
                    : "confirm-dialog-details"
                }
              >
                {details}
              </div>
            ) : null}
          </div>
        ) : null}
        <footer className="confirm-dialog-actions">
          <button
            ref={cancelButtonRef}
            className="confirm-dialog-button confirm-dialog-button-cancel"
            onClick={onCancel}
          >
            {cancelText}
          </button>
          <button
            className={`confirm-dialog-button confirm-dialog-button-confirm ${confirmTone}`}
            onClick={onConfirm}
            disabled={confirmDisabled}
          >
            {confirmText}
          </button>
        </footer>
      </div>
    </div>
  );
}
