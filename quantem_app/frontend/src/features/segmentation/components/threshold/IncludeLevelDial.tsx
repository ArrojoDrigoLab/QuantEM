import type { IncludeLevelControls } from "./useIncludeLevelControls";
import { DIAL_TITLE, DIAL_WORKING, formatIncludeLevel, thresholdTooltip } from "./copy";
import "./IncludeLevelDial.css";

export interface IncludeLevelDialProps {
  controls: IncludeLevelControls;
}

export function IncludeLevelDial({ controls }: IncludeLevelDialProps) {
  const {
    state, loadError, settled, position, moved, blocked, working, needsPreview,
    confirming, submitting, applyJobId, submitError, previewError,
    confirmationMessage, canConfirmWholeImage, roiId,
  } = controls;
  const tooltip = thresholdTooltip(state?.can_move === true, state?.default_include_level);
  const actionLabel = confirming
    ? "Confirming…"
    : submitting || applyJobId !== null
      ? DIAL_WORKING
      : moved
        ? `Recalculate at threshold ${formatIncludeLevel(position)}`
        : needsPreview
          ? "Preview / Process"
          : "Confirm";

  return (
    <div className="include-level-dial" data-testid="include-level-dial">
      <div className="include-level-heading">
        <h4 className="include-level-title">{DIAL_TITLE}</h4>
        <output className="include-level-value" data-testid="include-level-value">
          {formatIncludeLevel(settled)}
        </output>
        <span className="include-level-help" role="img" tabIndex={0}
          aria-label={tooltip} title={tooltip}>ⓘ</span>
      </div>
      {loadError ? <p className="include-level-problem" role="alert">{loadError}</p> : null}
      {state && !loadError && <>
        <input
          id="include-level-input" className="include-level-slider" type="range"
          min={state.minimum} max={state.maximum} step={0.01} value={position}
          disabled={blocked || working} aria-label="Threshold"
          onChange={(event) => controls.changeThreshold(Number(event.target.value))}
        />
        {!blocked && moved && (
          <output className="include-level-value" data-testid="include-level-new-value" aria-live="polite">
            New threshold: {formatIncludeLevel(position)}
          </output>
        )}
        {!blocked && (needsPreview || canConfirmWholeImage) && (
          <button type="button" className="include-level-apply" data-testid="include-level-apply"
            disabled={working} onClick={() => void (needsPreview ? controls.preview() : controls.confirm())}>
            {actionLabel}
          </button>
        )}
        {submitError && !blocked ? <p className="include-level-problem" role="alert">{submitError}</p> : null}
        {previewError ? <p className="include-level-problem" role="alert">{previewError}</p> : null}
        {!needsPreview && !roiId && state.manual_roi_candidate_count > 0 ? (
          <p className="include-level-note">
            {`${state.manual_roi_candidate_count} candidate${state.manual_roi_candidate_count === 1 ? "" : "s"} inside manually annotated ROIs will remain unchanged.`}
          </p>
        ) : null}
        {confirmationMessage ? <p className="include-level-note" role="status">{confirmationMessage}</p> : null}
      </>}
    </div>
  );
}

/** Shares the dial's selected-model state; an empty successful result is still a run. */
export function RunModelButton({ controls }: IncludeLevelDialProps) {
  if (!controls.state || controls.loadError || !controls.blocked) return null;
  return (
    <div className="header-run-model">
      <button type="button" className="include-level-apply" disabled={controls.modelWorking}
        onClick={() => void controls.runModel()}>
        {controls.downloading ? "Downloading model…" : controls.modelWorking ? "Running model…" : "Run model"}
      </button>
      {controls.submitError ? <p className="include-level-problem" role="alert">{controls.submitError}</p> : null}
    </div>
  );
}
