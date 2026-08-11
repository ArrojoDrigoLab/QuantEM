/**
 * Fit / 1:1 / Reset — the way back from a lost view.
 *
 * Until now the only way to change the view was the wheel and a drag, and
 * neither has an inverse a user can find: there was no control anywhere on the
 * canvas that says "put it back". Three buttons, always in the same corner,
 * cover the three things people actually want.
 */

interface ViewControlsProps {
  /** Show the whole image, as large as it fits. */
  onFit: () => void;
  /** One image pixel per screen pixel, about the current centre. */
  onOneToOne: () => void;
  /** The view this image opened at. */
  onReset: () => void;
  disabled?: boolean;
}

export function ViewControls({
  onFit,
  onOneToOne,
  onReset,
  disabled = false,
}: ViewControlsProps) {
  return (
    <div className="viewer-view-controls" role="group" aria-label="View">
      <button
        type="button"
        onClick={onFit}
        disabled={disabled}
        title="Show the whole image"
      >
        Fit
      </button>
      <button
        type="button"
        onClick={onOneToOne}
        disabled={disabled}
        title="One image pixel per screen pixel"
      >
        1:1
      </button>
      <button
        type="button"
        onClick={onReset}
        disabled={disabled}
        title="Back to the view this image opened at"
      >
        Reset
      </button>
    </div>
  );
}
