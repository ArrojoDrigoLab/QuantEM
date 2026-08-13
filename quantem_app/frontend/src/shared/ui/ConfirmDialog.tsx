/**
 * Confirmation dialog component for destructive actions.
 */

import { useEffect, useId, useRef, type ReactNode } from "react";
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
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const titleId = useId();
  const messageId = useId();
  const detailsId = useId();
  const cancelButtonRef = useRef<HTMLButtonElement | null>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const onCancelRef = useRef(onCancel);

  useEffect(() => {
    onCancelRef.current = onCancel;
  }, [onCancel]);

  useEffect(() => {
    if (!isOpen) return undefined;
    restoreFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    cancelButtonRef.current?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onCancelRef.current();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      restoreFocusRef.current?.focus();
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="confirm-dialog-overlay" onClick={onCancel}>
      <div
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
        <h3 id={titleId} className="confirm-dialog-title">
          {title}
        </h3>
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
        <div className="confirm-dialog-actions">
          <button
            ref={cancelButtonRef}
            className="confirm-dialog-button confirm-dialog-button-cancel"
            onClick={onCancel}
          >
            {cancelText}
          </button>
          <button
            className="confirm-dialog-button confirm-dialog-button-confirm"
            onClick={onConfirm}
            disabled={confirmDisabled}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}
