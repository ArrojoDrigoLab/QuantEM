import { useEffect, useState } from "react";
import type { ImageSegmentation } from "@/shared/types/images";
import { segmentationDisplayName } from "@/shared/segmentationNames";
import type {
  AnalysisMaskObject,
  AnalysisMaskOperation,
} from "./types";

export type AnalysisMaskTool = "brush" | "polygon";

interface ExistingMaskLayer {
  segmentation: ImageSegmentation;
  enabled: boolean;
}

interface AnalysisMaskSidebarProps {
  tool: AnalysisMaskTool;
  onToolChange: (tool: AnalysisMaskTool) => void;
  operation: AnalysisMaskOperation;
  onOperationChange: (operation: AnalysisMaskOperation) => void;
  canExclude: boolean;
  brushSize: number;
  onBrushSizeChange: (size: number) => void;
  polygonHasDraft: boolean;
  polygonCanClose: boolean;
  polygonSaving: boolean;
  onClosePolygon: () => void;
  onClearPolygon: () => void;
  navigateMode: boolean;
  onNavigateModeChange: (enabled: boolean) => void;
  objects: AnalysisMaskObject[];
  activeObjectId: string | null;
  busy: boolean;
  onEditObject: (objectId: string) => void;
  onSaveObject: () => void;
  onRenameObject: (objectId: string, name: string) => Promise<void>;
  onRequestDeleteObject: (object: AnalysisMaskObject) => void;
  existingMaskLayers: ExistingMaskLayer[];
  onToggleExistingMask: (segmentationId: string) => void;
}

const ICON_PROPS = {
  width: 16,
  height: 16,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

function PencilIcon() {
  return (
    <svg {...ICON_PROPS}>
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4z" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg {...ICON_PROPS}>
      <path d="M3 6h18" />
      <path d="M8 6V4h8v2" />
      <path d="M19 6l-1 14H6L5 6" />
      <path d="M10 11v5M14 11v5" />
    </svg>
  );
}

function BrushIcon() {
  return (
    <svg {...ICON_PROPS}>
      <path d="M12 19l7-7a2.1 2.1 0 0 0-3-3l-7 7-1 4z" />
      <path d="M8 16l-4 4" />
    </svg>
  );
}

function PolygonIcon() {
  return (
    <svg {...ICON_PROPS}>
      <polygon points="12 3 21 9.5 17.5 20 6.5 20 3 9.5" />
    </svg>
  );
}

function AnalysisMaskObjectRow({
  object,
  active,
  busy,
  onEdit,
  onSave,
  onRename,
  onRequestDelete,
}: {
  object: AnalysisMaskObject;
  active: boolean;
  busy: boolean;
  onEdit: () => void;
  onSave: () => void;
  onRename: (name: string) => Promise<void>;
  onRequestDelete: () => void;
}) {
  const [renaming, setRenaming] = useState(false);
  const [draftName, setDraftName] = useState(object.name);

  useEffect(() => setDraftName(object.name), [object.name]);

  const saveName = async () => {
    const next = draftName.trim();
    if (!next) return;
    await onRename(next);
    setRenaming(false);
  };

  return (
    <div
      className={`analysis-mask-object-row ${active ? "active" : ""}`}
      style={{ borderLeftColor: object.color }}
    >
      <div className="analysis-mask-object-name">
        {renaming ? (
          <input
            aria-label={`Name for ${object.name}`}
            value={draftName}
            maxLength={100}
            autoFocus
            onChange={(event) => setDraftName(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                void saveName();
              } else if (event.key === "Escape") {
                setDraftName(object.name);
                setRenaming(false);
              }
            }}
          />
        ) : (
          <span>{object.name}</span>
        )}
      </div>
      <div className="analysis-mask-object-actions">
        <button
          type="button"
          className="analysis-mask-icon-button"
          aria-label={renaming ? `Save name for ${object.name}` : `Rename ${object.name}`}
          title={renaming ? "Save name" : "Rename object"}
          disabled={busy || (renaming && !draftName.trim())}
          onClick={() => {
            if (renaming) void saveName();
            else setRenaming(true);
          }}
        >
          <PencilIcon />
        </button>
        <button
          type="button"
          className="analysis-mask-icon-button danger"
          aria-label={`Delete ${object.name}`}
          title="Delete object"
          disabled={busy}
          onClick={onRequestDelete}
        >
          <TrashIcon />
        </button>
        <button
          type="button"
          className="analysis-mask-edit-button"
          disabled={busy}
          onClick={active ? onSave : onEdit}
        >
          {active ? "Save" : "Edit"}
        </button>
      </div>
    </div>
  );
}

export function AnalysisMaskSidebar({
  tool,
  onToolChange,
  operation,
  onOperationChange,
  canExclude,
  brushSize,
  onBrushSizeChange,
  polygonHasDraft,
  polygonCanClose,
  polygonSaving,
  onClosePolygon,
  onClearPolygon,
  navigateMode,
  onNavigateModeChange,
  objects,
  activeObjectId,
  busy,
  onEditObject,
  onSaveObject,
  onRenameObject,
  onRequestDeleteObject,
  existingMaskLayers,
  onToggleExistingMask,
}: AnalysisMaskSidebarProps) {
  return (
    <aside className="analysis-mask-sidebar">
      <section className="analysis-mask-sidebar-section">
        <h3>Analysis Mask</h3>
        <div className="analysis-mask-operation" aria-label="Drawing operation">
          <button
            type="button"
            className={operation === "include" ? "active" : ""}
            aria-pressed={operation === "include"}
            onClick={() => onOperationChange("include")}
          >
            Include
          </button>
          <button
            type="button"
            className={operation === "exclude" ? "active exclude" : ""}
            aria-pressed={operation === "exclude"}
            disabled={!canExclude}
            title={canExclude ? "Remove area from the object being edited" : "Start a new object by including an area"}
            onClick={() => onOperationChange("exclude")}
          >
            Exclude
          </button>
        </div>
        <div className="analysis-mask-tools">
          <button
            type="button"
            className={tool === "polygon" ? "active" : ""}
            aria-label="Polygon"
            aria-pressed={tool === "polygon"}
            title="Polygon (R closes the shape)"
            onClick={() => onToolChange("polygon")}
          >
            <PolygonIcon />
          </button>
          <button
            type="button"
            className={tool === "brush" ? "active" : ""}
            aria-label="Brush"
            aria-pressed={tool === "brush"}
            title="Brush"
            onClick={() => onToolChange("brush")}
          >
            <BrushIcon />
          </button>
        </div>
        {tool === "brush" ? (
          <label className="analysis-mask-brush-size">
            Brush diameter
            <input
              type="range"
              min={4}
              max={256}
              step={1}
              value={brushSize}
              onChange={(event) => onBrushSizeChange(Number(event.target.value))}
            />
            <span>{brushSize}px</span>
          </label>
        ) : polygonHasDraft ? (
          <div className="analysis-mask-polygon-actions">
            <button type="button" disabled={!polygonCanClose} onClick={onClosePolygon}>
              {polygonSaving ? "Saving..." : "Close polygon (R)"}
            </button>
            <button type="button" disabled={polygonSaving} onClick={onClearPolygon}>
              Clear
            </button>
          </div>
        ) : (
          <p className="analysis-mask-hint">Click to place vertices, then press R to close.</p>
        )}
      </section>

      <section className="analysis-mask-sidebar-section">
        <label className="analysis-mask-toggle">
          <input
            type="checkbox"
            checked={navigateMode}
            onChange={(event) => onNavigateModeChange(event.target.checked)}
          />
          Navigate (A)
        </label>
      </section>

      <section className="analysis-mask-sidebar-section analysis-mask-objects-section">
        <h3>Objects</h3>
        {objects.length === 0 ? (
          <p className="analysis-mask-empty">Draw a polygon or brush stroke to create Object 1.</p>
        ) : (
          <div className="analysis-mask-object-list">
            {objects.map((object) => (
              <AnalysisMaskObjectRow
                key={object.id}
                object={object}
                active={object.id === activeObjectId}
                busy={busy}
                onEdit={() => onEditObject(object.id)}
                onSave={onSaveObject}
                onRename={(name) => onRenameObject(object.id, name)}
                onRequestDelete={() => onRequestDeleteObject(object)}
              />
            ))}
          </div>
        )}
      </section>

      {existingMaskLayers.length > 0 ? (
        <section className="analysis-mask-sidebar-section">
          <h3>Other Analysis Masks</h3>
          <div className="analysis-mask-existing-list">
            {existingMaskLayers.map((layer) => (
              <label key={layer.segmentation.id} className="analysis-mask-toggle">
                <input
                  type="checkbox"
                  checked={layer.enabled}
                  onChange={() => onToggleExistingMask(layer.segmentation.id)}
                />
                {segmentationDisplayName(layer.segmentation)}
              </label>
            ))}
          </div>
        </section>
      ) : null}
    </aside>
  );
}
