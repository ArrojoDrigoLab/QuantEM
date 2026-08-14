interface ZoomControlsProps {
  onZoomOut: () => void;
  onZoomIn: () => void;
  disabled?: boolean;
}

export function ZoomControls({ onZoomOut, onZoomIn, disabled = false }: ZoomControlsProps) {
  return (
    <div className="viewer-zoom-controls" role="group" aria-label="Zoom controls">
      <button type="button" onClick={onZoomOut} disabled={disabled} aria-label="Zoom out">
        &minus;
      </button>
      <button type="button" onClick={onZoomIn} disabled={disabled} aria-label="Zoom in">
        +
      </button>
    </div>
  );
}
