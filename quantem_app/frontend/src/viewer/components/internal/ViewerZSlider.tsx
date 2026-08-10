import { useId } from "react";

export interface ViewerZSliderProps {
  /** Number of stored z-planes (the slider range is 0..depth-1). */
  depth: number;
  /** Current stored slice index. */
  value: number;
  onChange: (zIndex: number) => void;
  /**
   * Original source plane index for each stored slice, so the label can show
   * the true physical depth of a decimated volume (e.g. stored slice 3 -> 30).
   */
  planeIndices?: number[];
}

/**
 * Compact z-plane scrubber overlaid on the viewer for 3D volumes. Purely
 * presentational and controlled; the host clamps and persists the value.
 */
export function ViewerZSlider({ depth, value, onChange, planeIndices }: ViewerZSliderProps) {
  const inputId = useId();
  if (!Number.isFinite(depth) || depth <= 1) return null;

  const clamped = Math.min(Math.max(0, Math.floor(value)), depth - 1);
  const originalPlane = planeIndices?.[clamped];
  const planeLabel =
    typeof originalPlane === "number" ? `z=${originalPlane}` : `z ${clamped + 1}/${depth}`;

  return (
    <div className="viewer-z-slider" data-testid="viewer-z-slider">
      <label className="viewer-z-slider__label" htmlFor={inputId}>
        {planeLabel}
      </label>
      <input
        id={inputId}
        className="viewer-z-slider__input"
        type="range"
        min={0}
        max={depth - 1}
        step={1}
        value={clamped}
        aria-label="z-plane"
        aria-valuetext={planeLabel}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <span className="viewer-z-slider__count">
        {clamped + 1}/{depth}
      </span>
    </div>
  );
}
