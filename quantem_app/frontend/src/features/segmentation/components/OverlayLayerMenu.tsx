import "./OverlayLayerMenu.css";

export interface OverlayLayerControl {
  strokeWidth: number;
  fillOpacity: number;
  showBorders: boolean;
  onStrokeWidthChange: (value: number) => void;
  onFillOpacityChange: (value: number) => void;
  onShowBordersChange: (value: boolean) => void;
}

export interface OverlayLayerMenuProps {
  idPrefix: string;
  paneLabel: string;
  usesRasterOverlay: boolean;
  candidates?: OverlayLayerControl;
  confirmed: OverlayLayerControl;
}

export type PaneOverlayLayerControls = Omit<
  OverlayLayerMenuProps,
  "idPrefix" | "paneLabel"
>;

function LayerGroup({
  idPrefix,
  label,
  usesRasterOverlay,
  controls,
}: {
  idPrefix: string;
  label: "Candidates" | "Confirmed";
  usesRasterOverlay: boolean;
  controls: OverlayLayerControl;
}) {
  const accessibleLabel = label.toLowerCase();
  return (
    <section className="overlay-layer-menu-group">
      <h4>{label}</h4>
      {usesRasterOverlay ? (
        <label className="overlay-layer-menu-checkbox" htmlFor={`${idPrefix}-borders`}>
          <input
            id={`${idPrefix}-borders`}
            type="checkbox"
            checked={controls.showBorders}
            aria-label={`${label} borders`}
            onChange={(event) => controls.onShowBordersChange(event.target.checked)}
          />
          Borders
        </label>
      ) : (
        <div className="overlay-layer-menu-control">
          <label htmlFor={`${idPrefix}-stroke-width`}>Border thickness</label>
          <div className="overlay-layer-menu-slider-row">
            <input
              id={`${idPrefix}-stroke-width`}
              type="range"
              min={0.5}
              max={8}
              step={0.5}
              value={controls.strokeWidth}
              aria-label={`${label} border thickness`}
              onChange={(event) => controls.onStrokeWidthChange(Number(event.target.value))}
            />
            <span>{controls.strokeWidth.toFixed(1)}px</span>
          </div>
        </div>
      )}
      <div className="overlay-layer-menu-control">
        <label htmlFor={`${idPrefix}-fill-opacity`}>Fill opacity</label>
        <div className="overlay-layer-menu-slider-row">
          <input
            id={`${idPrefix}-fill-opacity`}
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={controls.fillOpacity}
            aria-label={`${label} fill opacity`}
            onChange={(event) => controls.onFillOpacityChange(Number(event.target.value))}
          />
          <span aria-label={`${accessibleLabel} fill opacity value`}>
            {(controls.fillOpacity * 100).toFixed(0)}%
          </span>
        </div>
      </div>
    </section>
  );
}

export function OverlayLayerMenu({
  idPrefix,
  paneLabel,
  usesRasterOverlay,
  candidates,
  confirmed,
}: OverlayLayerMenuProps) {
  return (
    <details className="overlay-layer-menu">
      <summary aria-label={`${paneLabel} overlay options`} title="Overlay options">
        <svg aria-hidden="true" viewBox="0 0 24 24" fill="none">
          <path d="M4 7h10M18 7h2M4 17h2M10 17h10M8 4v6M8 14v6M16 4v6M16 14v6" />
        </svg>
        <span>Overlays</span>
      </summary>
      <div className="overlay-layer-menu-popover">
        {candidates ? (
          <LayerGroup
            idPrefix={`${idPrefix}-candidates`}
            label="Candidates"
            usesRasterOverlay={usesRasterOverlay}
            controls={candidates}
          />
        ) : null}
        <LayerGroup
          idPrefix={`${idPrefix}-confirmed`}
          label="Confirmed"
          usesRasterOverlay={usesRasterOverlay}
          controls={confirmed}
        />
      </div>
    </details>
  );
}
