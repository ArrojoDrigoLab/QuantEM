import { useEffect, useId, useState } from "react";
import type { ImageSegmentation } from "@/shared/types/images";
import { segmentationDisplayName } from "@/shared/segmentationNames";
import type {
  AnalysisMaskObject,
  AnalysisMaskOperation,
  CellTag,
  CellTagSuggestion,
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
  objectNamesRequired?: boolean;
  cellsMode?: boolean;
  activeObjectId: string | null;
  metadataEditingObjectId: string | null;
  onMetadataEditingObjectChange: (objectId: string | null) => void;
  metadataEditDisabled: boolean;
  tagSuggestions: CellTagSuggestion[];
  busy: boolean;
  onEditObject: (objectId: string) => void;
  onSaveObject: () => void;
  onUpdateObjectMetadata: (
    objectId: string,
    name: string,
    tags: CellTag[]
  ) => Promise<void>;
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

function CheckIcon() {
  return (
    <svg {...ICON_PROPS}>
      <path d="m5 12 4 4L19 6" />
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
  fallbackName,
  nameRequired,
  active,
  busy,
  cellsMode,
  metadataEditing,
  metadataEditingElsewhere,
  metadataEditDisabled,
  tagSuggestions,
  onMetadataEditingChange,
  onEdit,
  onSave,
  onUpdateMetadata,
  onRequestDelete,
}: {
  object: AnalysisMaskObject;
  fallbackName: string;
  nameRequired: boolean;
  active: boolean;
  busy: boolean;
  cellsMode: boolean;
  metadataEditing: boolean;
  metadataEditingElsewhere: boolean;
  metadataEditDisabled: boolean;
  tagSuggestions: CellTagSuggestion[];
  onMetadataEditingChange: (editing: boolean) => void;
  onEdit: () => void;
  onSave: () => void;
  onUpdateMetadata: (name: string, tags: CellTag[]) => Promise<void>;
  onRequestDelete: () => void;
}) {
  const datalistId = useId().replace(/:/g, "");
  const [draftName, setDraftName] = useState(object.name);
  const [draftTags, setDraftTags] = useState<
    Array<CellTag & { rowId: string }>
  >([]);
  const [metadataError, setMetadataError] = useState<string | null>(null);
  const [metadataSaving, setMetadataSaving] = useState(false);
  const displayName = object.name || fallbackName;
  const pending = object.id.startsWith("pending-cell-");

  useEffect(() => {
    if (!metadataEditing) return;
    setDraftName(object.name);
    setDraftTags([
      ...object.tags.map((tag, index) => ({
        ...tag,
        rowId: `saved-${object.id}-${index}`,
      })),
      { name: "", value: "", rowId: `blank-${object.id}` },
    ]);
    setMetadataError(null);
  }, [metadataEditing, object.id, object.name, object.tags]);

  const saveMetadata = async () => {
    const next = draftName.trim();
    if (nameRequired && !next) {
      setMetadataError("Cell name cannot be empty.");
      return;
    }
    const tags = draftTags
      .map(({ name, value }) => ({ name: name.trim(), value: value.trim() }))
      .filter((tag) => tag.name || tag.value);
    if (tags.some((tag) => !tag.name)) {
      setMetadataError("Every tag value needs a tag name.");
      return;
    }
    const foldedNames = tags.map((tag) => tag.name.toLowerCase());
    if (new Set(foldedNames).size !== foldedNames.length) {
      setMetadataError("A cell cannot contain the same tag name twice.");
      return;
    }
    setMetadataSaving(true);
    setMetadataError(null);
    try {
      await onUpdateMetadata(next, tags);
      onMetadataEditingChange(false);
    } catch {
      // The screen owns the API error toast. Keep the editor open for retry.
    } finally {
      setMetadataSaving(false);
    }
  };

  const updateDraftTag = (
    rowId: string,
    field: "name" | "value",
    value: string
  ) => {
    setDraftTags((current) =>
      current.map((tag) => (tag.rowId === rowId ? { ...tag, [field]: value } : tag))
    );
    setMetadataError(null);
  };

  return (
    <div
      className={`analysis-mask-object-row ${active ? "active" : ""} ${metadataEditing ? "metadata-editing" : ""}`}
      style={{ borderLeftColor: object.color }}
    >
      <div className="analysis-mask-object-name">
        {metadataEditing ? (
          <input
            aria-label={`Name for ${displayName}`}
            value={draftName}
            maxLength={100}
            autoFocus
            onChange={(event) => setDraftName(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                void saveMetadata();
              } else if (event.key === "Escape") {
                setDraftName(object.name);
                onMetadataEditingChange(false);
              }
            }}
          />
        ) : (
          <span>{displayName}</span>
        )}
      </div>
      {!metadataEditing && cellsMode && object.tags.length > 0 ? (
        <div className="analysis-mask-tag-pills" aria-label={`Tags for ${displayName}`}>
          {object.tags.map((tag) => (
            <span key={tag.name.toLowerCase()}>
              {tag.name}
              {tag.value ? `: ${tag.value}` : ""}
            </span>
          ))}
        </div>
      ) : null}
      {metadataEditing && cellsMode ? (
        <div className="analysis-mask-tag-editor">
          <datalist id={`${datalistId}-tag-names`}>
            {tagSuggestions.map((suggestion) => (
              <option key={suggestion.name.toLowerCase()} value={suggestion.name} />
            ))}
          </datalist>
          {draftTags.map((tag, index) => {
            const suggestion = tagSuggestions.find(
              (candidate) =>
                candidate.name.toLowerCase() === tag.name.trim().toLowerCase()
            );
            const valuesId = `${datalistId}-tag-values-${index}`;
            return (
              <div className="analysis-mask-tag-row" key={tag.rowId}>
                <input
                  aria-label={`Tag name ${index + 1} for ${displayName}`}
                  placeholder="Tag name"
                  maxLength={64}
                  list={`${datalistId}-tag-names`}
                  value={tag.name}
                  onChange={(event) =>
                    updateDraftTag(tag.rowId, "name", event.target.value)
                  }
                />
                <input
                  aria-label={`Tag value ${index + 1} for ${displayName}`}
                  placeholder="Tag value"
                  maxLength={200}
                  list={valuesId}
                  value={tag.value}
                  onChange={(event) =>
                    updateDraftTag(tag.rowId, "value", event.target.value)
                  }
                />
                <datalist id={valuesId}>
                  {(suggestion?.values ?? []).map((value) => (
                    <option key={value.toLowerCase()} value={value} />
                  ))}
                </datalist>
                <button
                  type="button"
                  className="analysis-mask-tag-remove"
                  aria-label={`Remove tag row ${index + 1}`}
                  title="Remove tag"
                  onClick={() =>
                    setDraftTags((current) =>
                      current.filter((candidate) => candidate.rowId !== tag.rowId)
                    )
                  }
                >
                  ×
                </button>
              </div>
            );
          })}
          <button
            type="button"
            className="analysis-mask-tag-add"
            onClick={() =>
              setDraftTags((current) => [
                ...current,
                {
                  name: "",
                  value: "",
                  rowId: `new-${object.id}-${Date.now()}-${current.length}`,
                },
              ])
            }
          >
            <span aria-hidden="true">+</span> Add tag
          </button>
          {metadataError ? (
            <p className="analysis-mask-metadata-error" role="alert">
              {metadataError}
            </p>
          ) : null}
        </div>
      ) : null}
      <div className="analysis-mask-object-actions">
        <button
          type="button"
          className="analysis-mask-icon-button"
          aria-label={
            metadataEditing
              ? cellsMode
                ? `Save name and tags for ${displayName}`
                : `Save name for ${displayName}`
              : cellsMode
                ? `Edit name and tags for ${displayName}`
                : `Rename ${displayName}`
          }
          title={
            metadataEditing
              ? cellsMode
                ? "Save name and tags"
                : "Save name"
              : cellsMode
                ? "Edit name and tags"
                : "Rename"
          }
          disabled={
            busy ||
            pending ||
            metadataSaving ||
            metadataEditingElsewhere ||
            (!metadataEditing && metadataEditDisabled) ||
            (metadataEditing && nameRequired && !draftName.trim())
          }
          onClick={() => {
            if (metadataEditing) void saveMetadata();
            else onMetadataEditingChange(true);
          }}
        >
          {metadataEditing ? <CheckIcon /> : <PencilIcon />}
        </button>
        <button
          type="button"
          className="analysis-mask-icon-button danger"
          aria-label={`Delete ${displayName}`}
          title="Delete object"
          disabled={busy || pending || metadataEditing || metadataEditingElsewhere}
          onClick={onRequestDelete}
        >
          <TrashIcon />
        </button>
        <button
          type="button"
          className="analysis-mask-edit-button"
          disabled={busy || pending || metadataEditing || metadataEditingElsewhere}
          onClick={active ? onSave : onEdit}
        >
          {pending ? "Saving…" : active ? (cellsMode ? "Done" : "Save") : "Edit"}
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
  objectNamesRequired = false,
  cellsMode = false,
  activeObjectId,
  metadataEditingObjectId,
  onMetadataEditingObjectChange,
  metadataEditDisabled,
  tagSuggestions,
  busy,
  onEditObject,
  onSaveObject,
  onUpdateObjectMetadata,
  onRequestDeleteObject,
  existingMaskLayers,
  onToggleExistingMask,
}: AnalysisMaskSidebarProps) {
  const editingCell = cellsMode && activeObjectId !== null;
  const metadataEditing = metadataEditingObjectId !== null;
  const showEditingTools = !cellsMode || editingCell;

  return (
    <aside className="analysis-mask-sidebar">
      <section className="analysis-mask-sidebar-section">
        <h3>
          {metadataEditing
            ? cellsMode
              ? "Edit cell details"
              : "Rename object"
            : cellsMode
              ? editingCell
                ? "Edit cell"
                : "Draw cells"
              : "Analysis Mask"}
        </h3>
        {showEditingTools ? (
          <div className="analysis-mask-operation" aria-label="Drawing operation">
            <button
              type="button"
              className={operation === "include" ? "active" : ""}
              aria-pressed={operation === "include"}
              disabled={metadataEditing}
              onClick={() => onOperationChange("include")}
            >
              Include
            </button>
            <button
              type="button"
              className={operation === "exclude" ? "active exclude" : ""}
              aria-pressed={operation === "exclude"}
              disabled={!canExclude || metadataEditing}
              title={
                canExclude
                  ? "Remove area from the object being edited"
                  : "Start a new object by including an area"
              }
              onClick={() => onOperationChange("exclude")}
            >
              Exclude
            </button>
          </div>
        ) : null}
        {showEditingTools ? (
          <div className="analysis-mask-tools">
            <button
              type="button"
              className={tool === "polygon" ? "active" : ""}
              aria-label="Polygon"
              aria-pressed={tool === "polygon"}
              title="Polygon (R closes the shape)"
              disabled={metadataEditing}
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
              disabled={metadataEditing}
              onClick={() => onToolChange("brush")}
            >
              <BrushIcon />
            </button>
          </div>
        ) : null}
        {showEditingTools && tool === "brush" ? (
          <label className="analysis-mask-brush-size">
            Brush diameter
            <input
              type="range"
              min={4}
              max={256}
              step={1}
              value={brushSize}
              disabled={metadataEditing}
              onChange={(event) => onBrushSizeChange(Number(event.target.value))}
            />
            <span>{brushSize}px</span>
          </label>
        ) : polygonHasDraft ? (
          <div className="analysis-mask-polygon-actions">
            <button
              type="button"
              disabled={!polygonCanClose || metadataEditing}
              onClick={onClosePolygon}
            >
              {polygonSaving ? "Saving..." : "Close polygon (R)"}
            </button>
            <button
              type="button"
              disabled={polygonSaving || metadataEditing}
              onClick={onClearPolygon}
            >
              Clear
            </button>
          </div>
        ) : (
          <p className="analysis-mask-hint">
            {metadataEditing
              ? "Save the cell name and tags to resume labeling."
              : cellsMode && !editingCell
                ? "Click to outline a cell, then press R to close it. It saves automatically; your next click starts a new cell."
                : "Click to place vertices, then press R to close."}
          </p>
        )}
      </section>

      <section className="analysis-mask-sidebar-section">
        <label className="analysis-mask-toggle">
          <input
            type="checkbox"
            checked={navigateMode}
            disabled={metadataEditing}
            onChange={(event) => onNavigateModeChange(event.target.checked)}
          />
          Navigate (A)
        </label>
      </section>

      <section className="analysis-mask-sidebar-section analysis-mask-objects-section">
        <h3>{cellsMode ? "Cells" : "Objects"}</h3>
        {objects.length === 0 ? (
          <p className="analysis-mask-empty">
            {cellsMode
              ? "Outline the first cell with a polygon."
              : "Draw a polygon or brush stroke to create the first area."}
          </p>
        ) : (
          <div className="analysis-mask-object-list">
            {objects.map((object, index) => (
              <AnalysisMaskObjectRow
                key={object.id}
                object={object}
                fallbackName={`Area ${index + 1}`}
                nameRequired={objectNamesRequired}
                cellsMode={cellsMode}
                active={object.id === activeObjectId}
                busy={busy}
                metadataEditing={object.id === metadataEditingObjectId}
                metadataEditingElsewhere={
                  metadataEditingObjectId !== null &&
                  object.id !== metadataEditingObjectId
                }
                metadataEditDisabled={metadataEditDisabled}
                tagSuggestions={tagSuggestions}
                onMetadataEditingChange={(editing) =>
                  onMetadataEditingObjectChange(editing ? object.id : null)
                }
                onEdit={() => onEditObject(object.id)}
                onSave={onSaveObject}
                onUpdateMetadata={(name, tags) =>
                  onUpdateObjectMetadata(object.id, name, tags)
                }
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
