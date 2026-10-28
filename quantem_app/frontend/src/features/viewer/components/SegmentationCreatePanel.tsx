import { preferredPackId } from "@/features/models/defaultModel";
import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  createAssetSegmentation,
  createSegmentationType,
  getSegmentationTypes,
} from "@/shared/api/assets";
import { useApiMutation } from "@/shared/hooks/useApiMutation";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { useModelCatalogue } from "@/features/models/useModelCatalogue";
import { packRunnability } from "@/features/models/runnable";
import { ModelAvailabilityIcon } from "@/features/models/ModelAvailabilityIcon";
import type { ModelCatalogue, ModelPack } from "@/shared/types/finetune";
import type {
  ImageSegmentation,
  ImageSegmentationCreatePayload,
  SegmentationType,
} from "@/shared/types/images";
import "./SegmentationCreatePanel.css";

export interface SegmentationLaunch {
  sourceModel: string;
  runModel: boolean;
}

interface SegmentationCreatePanelProps {
  imageId: string;
  onCreated?: (
    segmentation: ImageSegmentation,
    launch?: SegmentationLaunch
  ) => void;
  existingSegmentationTypes?: string[];
  existingSegmentationTypeIds?: string[];
  title?: string;
  description?: string;
  imageSizeBytes?: number | null;
  pixelSizeNm?: number | null;
}

interface QuickSegmentationOption {
  id: string;
  label: string;
  dialogLabel: string;
  name: string;
}

interface DefinitionDraft {
  name: string;
  description: string;
  representation: "instances" | "semantic";
  topology: "non_overlapping" | "overlapping";
}

const MANUAL_CHOICE = "manual";
const LARGE_IMAGE_BYTES = 200 * 1024 * 1024;
const CELLS_INTERNAL_NAME = "quantem_internal_cells";

function OptionHelp({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  return (
    <span className="segmentation-option-help">
      <button
        type="button"
        className="segmentation-option-help-button"
        aria-label={`${open ? "Hide" : "About"} ${label}`}
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((current) => !current)}
      >
        {open ? "⌃" : "?"}
      </button>
      {open ? (
        <span id={panelId} className="segmentation-option-help-text">
          {children}
        </span>
      ) : null}
    </span>
  );
}

interface ModelChoice {
  id: string;
  label: string;
  pack: ModelPack | null;
}

const MODEL_FAMILIES = [
  { id: "quantem", label: "QuantEM (basic model)" },
  { id: "omniem", label: "OmniEM (large model)" },
] as const;

function packsForOrganelle(
  catalogue: ModelCatalogue | null,
  organelleId: string
): ModelChoice[] {
  return MODEL_FAMILIES.map((family) => {
    const id = `${family.id}:${organelleId}`;
    return {
      id,
      label: family.label,
      pack: catalogue?.packs.find((candidate) => candidate.id === id) ?? null,
    };
  });
}


function DefinitionDetails({
  draft,
  setDraft,
  existing,
  existingTypeIds,
  onSelect,
  category,
  creating,
}: {
  draft: DefinitionDraft;
  setDraft: (next: DefinitionDraft) => void;
  existing: SegmentationType[];
  existingTypeIds: Set<string>;
  onSelect: (definition: SegmentationType) => void;
  category: "organelle" | "area";
  creating: boolean;
}) {
  return (
    <div className="segmentation-custom-dialog">
      {existing.length > 0 ? (
        <div>
          <div className="segmentation-quick-group-title">Existing definitions</div>
          <div className="segmentation-quick-buttons">
            {existing.map((definition) => {
              const alreadyAdded = existingTypeIds.has(definition.id);
              return (
                <button
                  key={definition.id}
                  type="button"
                  disabled={creating || alreadyAdded}
                  title={alreadyAdded ? "Already added to this image" : undefined}
                  onClick={() => onSelect(definition)}
                >
                  {definition.short_name}
                </button>
              );
            })}
          </div>
        </div>
      ) : null}
      <div className="segmentation-definition-divider">Create a new definition</div>
      <label htmlFor={`new-${category}-name`}>
        <span className="segmentation-definition-label">
          Name
          <OptionHelp label={`${category} name`}>
            This application-wide name identifies the same biological structure or
            analysis area across images.
          </OptionHelp>
        </span>
        <input
          id={`new-${category}-name`}
          type="text"
          value={draft.name}
          onChange={(event) => setDraft({ ...draft, name: event.target.value })}
          placeholder={category === "organelle" ? "e.g. Vesicles" : "e.g. Lesion areas"}
          autoFocus
        />
      </label>
      <label htmlFor={`new-${category}-description`}>
        <span className="segmentation-definition-label">
          Description (optional)
          <OptionHelp label={`${category} description`}>
            Add a short biological or workflow definition so other users know what
            should and should not be included.
          </OptionHelp>
        </span>
        <textarea
          id={`new-${category}-description`}
          value={draft.description}
          onChange={(event) => setDraft({ ...draft, description: event.target.value })}
          rows={2}
        />
      </label>
      <label htmlFor={`new-${category}-representation`}>
        <span className="segmentation-definition-label">
          Representation
          <OptionHelp label="representation">
            Instances preserve separately selectable objects. A semantic mask stores
            one foreground class even when it has disconnected regions.
          </OptionHelp>
        </span>
        <select
          id={`new-${category}-representation`}
          value={draft.representation}
          onChange={(event) =>
            setDraft({
              ...draft,
              representation: event.target.value as DefinitionDraft["representation"],
            })
          }
        >
          <option value="instances">Individual instances</option>
          <option value="semantic">Semantic mask</option>
        </select>
      </label>
      <label htmlFor={`new-${category}-topology`}>
        <span className="segmentation-definition-label">
          Topology policy
          <OptionHelp label="topology policy">
            Non-overlapping instances may leave gaps but cannot cover the same pixel.
            Overlapping instances may occupy the same region independently.
          </OptionHelp>
        </span>
        <select
          id={`new-${category}-topology`}
          value={draft.topology}
          disabled={draft.representation === "semantic"}
          onChange={(event) =>
            setDraft({
              ...draft,
              topology: event.target.value as DefinitionDraft["topology"],
            })
          }
        >
          <option value="non_overlapping">Exclusive partition (non-overlapping)</option>
          <option value="overlapping">Overlapping instances</option>
        </select>
      </label>
    </div>
  );
}

export function SegmentationCreatePanel({
  imageId,
  onCreated,
  existingSegmentationTypes,
  existingSegmentationTypeIds,
  title = "Add Segmentation",
  description,
  imageSizeBytes,
}: SegmentationCreatePanelProps) {
  const [pendingOption, setPendingOption] = useState<QuickSegmentationOption | null>(null);
  const [pendingPackChoice, setPendingPackChoice] = useState<string | null>(null);
  const [pendingPackChoiceTouched, setPendingPackChoiceTouched] = useState(false);
  const [pendingArea, setPendingArea] = useState<"sample" | "cells" | null>(null);
  const [definitionDialog, setDefinitionDialog] = useState<"organelle" | "area" | null>(null);
  const [draft, setDraft] = useState<DefinitionDraft>({
    name: "",
    description: "",
    representation: "instances",
    topology: "non_overlapping",
  });
  const [definitionError, setDefinitionError] = useState<string | null>(null);
  const { catalogue } = useModelCatalogue();
  const { data: segmentationTypes, refetch: refetchTypes } = useApiQuery(
    getSegmentationTypes,
    []
  );

  const quickOptions = useMemo<QuickSegmentationOption[]>(
    () => [
      { id: "mito", label: "Mitochondria", dialogLabel: "mitochondria", name: "Mitochondria" },
      { id: "er", label: "ER", dialogLabel: "ER", name: "Endoplasmic Reticulum" },
      { id: "nucleus", label: "Nucleus", dialogLabel: "nucleus", name: "Nucleus" },
      { id: "ld", label: "Lipid Droplets", dialogLabel: "lipid droplet", name: "Lipid Droplets" },
    ],
    []
  );
  const existingNames = useMemo(
    () => new Set((existingSegmentationTypes ?? []).map((name) => name.toLowerCase())),
    [existingSegmentationTypes]
  );
  const existingIds = useMemo(
    () => new Set(existingSegmentationTypeIds ?? []),
    [existingSegmentationTypeIds]
  );
  const customDefinitions = useMemo(
    () => (segmentationTypes ?? []).filter((definition) => definition.kind === "custom"),
    [segmentationTypes]
  );
  const customOrganelles = customDefinitions.filter(
    (definition) => (definition.category ?? "organelle") === "organelle"
  );
  const customAreas = customDefinitions.filter(
    (definition) => definition.category === "area"
  );

  const {
    mutate: createSegmentation,
    loading: creatingSegmentation,
    error: createSegmentationError,
  } = useApiMutation(async (payload: ImageSegmentationCreatePayload) =>
    createAssetSegmentation(imageId, payload)
  );

  const handleCreateSegmentation = useCallback(
    async (payload: ImageSegmentationCreatePayload, launch?: SegmentationLaunch) => {
      const created = await createSegmentation(payload);
      if (!created) return null;
      onCreated?.(created, launch);
      return created;
    },
    [createSegmentation, onCreated]
  );

  const requestCreate = useCallback(
    (option: QuickSegmentationOption) => {
      setPendingPackChoiceTouched(false);
      setPendingPackChoice(preferredPackId(catalogue, option.id) ?? MANUAL_CHOICE);
      setPendingOption(option);
    },
    [catalogue]
  );

  useEffect(() => {
    if (!pendingOption || pendingPackChoiceTouched) return;
    setPendingPackChoice(preferredPackId(catalogue, pendingOption.id) ?? MANUAL_CHOICE);
  }, [catalogue, pendingOption, pendingPackChoiceTouched]);

  const pendingChoice = pendingPackChoice ?? MANUAL_CHOICE;
  const pendingPacks = pendingOption
    ? packsForOrganelle(catalogue, pendingOption.id)
    : [];

  const openDefinitionDialog = (category: "organelle" | "area") => {
    setDraft({
      name: "",
      description: "",
      representation: "instances",
      topology: category === "area" ? "overlapping" : "non_overlapping",
    });
    setDefinitionError(null);
    setDefinitionDialog(category);
  };

  const addExistingDefinition = (definition: SegmentationType) => {
    setDefinitionDialog(null);
    void handleCreateSegmentation({ segmentation_type_id: definition.id, run_inference: false });
  };

  const createDefinition = async () => {
    if (!definitionDialog || !draft.name.trim()) return;
    setDefinitionError(null);
    try {
      const definition = await createSegmentationType({
        long_name: draft.name.trim(),
        short_name: draft.name.trim(),
        description: draft.description.trim(),
        category: definitionDialog,
        representation: draft.representation,
        topology_policy: draft.topology,
      });
      await refetchTypes();
      setDefinitionDialog(null);
      await handleCreateSegmentation({ segmentation_type_id: definition.id, run_inference: false });
    } catch (error) {
      setDefinitionError(error instanceof Error ? error.message : "Could not create definition.");
    }
  };

  const cellsDefinitionId = (segmentationTypes ?? []).find(
    (type) => type.internal_name === CELLS_INTERNAL_NAME
  )?.id;

  return (
    <div className="segmentation-create-card">
      <ConfirmDialog
        isOpen={pendingOption !== null}
        title={`Start ${pendingOption?.dialogLabel ?? ""} segmentation`}
        details={
          <>
            <div className="segmentation-create-model-choices" role="radiogroup" aria-label="Segmentation method">
              <label className="segmentation-create-model-option">
                <input type="radio" name="create-source-model" value={MANUAL_CHOICE} checked={pendingChoice === MANUAL_CHOICE} onChange={() => { setPendingPackChoiceTouched(true); setPendingPackChoice(MANUAL_CHOICE); }} />
                <span>Manual segmentation</span><span className="segmentation-create-manual-icon" aria-hidden>✎</span>
              </label>
              {pendingPacks.map((choice) => {
                const blocked = choice.pack?.installed === true && packRunnability(choice.pack).state === "blocked";
                return (
                  <label key={choice.id} className="segmentation-create-model-option">
                    <input type="radio" name="create-source-model" value={choice.id} checked={pendingChoice === choice.id} disabled={blocked} onChange={() => { setPendingPackChoiceTouched(true); setPendingPackChoice(choice.id); }} />
                    <span>{choice.label}</span><ModelAvailabilityIcon pack={choice.pack} />
                  </label>
                );
              })}
            </div>
            {pendingChoice !== MANUAL_CHOICE ? (
              <p>
                The selected model will run on all tiles. Larger models may be more
                accurate, but take longer to run.
              </p>
            ) : null}
            {pendingChoice !== MANUAL_CHOICE && imageSizeBytes != null && imageSizeBytes > LARGE_IMAGE_BYTES ? <p>This image is larger than 200 MB, so inference may take several minutes.</p> : null}
          </>
        }
        confirmText={pendingChoice === MANUAL_CHOICE ? "Start manual segmentation" : "Run model"}
        onConfirm={() => {
          if (!pendingOption) return;
          const option = pendingOption;
          setPendingOption(null);
          void handleCreateSegmentation(
            { segmentation_type_name: option.name, run_inference: false },
            { sourceModel: pendingChoice, runModel: pendingChoice !== MANUAL_CHOICE }
          );
        }}
        onCancel={() => setPendingOption(null)}
      />

      <ConfirmDialog
        isOpen={pendingArea !== null}
        title={pendingArea === "sample" ? "Add Sample Area" : "Add Cells"}
        message={
          pendingArea === "sample"
            ? "Select all area containing the sample. Use this to remove background, empty spaces, or other areas you wish to exclude from analysis."
            : "Segment individual cells. Non-overlapping segmented cells can be tagged and categorized, and analyzed individually."
        }
        confirmText={pendingArea === "sample" ? "Add Sample Area" : "Add Cells"}
        onConfirm={() => {
          const area = pendingArea;
          setPendingArea(null);
          void handleCreateSegmentation({
            segmentation_type_name: area === "sample" ? "Sample Area" : "Cells",
            run_inference: false,
          });
        }}
        onCancel={() => setPendingArea(null)}
      />

      <ConfirmDialog
        isOpen={definitionDialog !== null}
        title={definitionDialog === "area" ? "Custom Area" : "Custom Organelle"}
        message="Choose an existing application-wide definition or create a new one."
        details={
          <>
            <DefinitionDetails
              draft={draft}
              setDraft={setDraft}
              existing={definitionDialog === "area" ? customAreas : customOrganelles}
              existingTypeIds={existingIds}
              onSelect={addExistingDefinition}
              category={definitionDialog ?? "organelle"}
              creating={creatingSegmentation}
            />
            {definitionError ? <p className="segmentation-error">{definitionError}</p> : null}
          </>
        }
        confirmText="Create and add"
        confirmDisabled={!draft.name.trim() || creatingSegmentation}
        onConfirm={() => void createDefinition()}
        onCancel={() => setDefinitionDialog(null)}
      />

      <h3>{title}</h3>
      {description ? <p>{description}</p> : null}

      <div className="segmentation-quick-group">
        <div className="segmentation-quick-group-title">Organelles</div>
        <div className="segmentation-quick-buttons">
          {quickOptions
            .filter((option) => !existingNames.has(option.name.toLowerCase()))
            .map((option) => (
              <button key={option.id} type="button" onClick={() => requestCreate(option)} disabled={creatingSegmentation}>
                {option.label}
              </button>
            ))}
          <button type="button" onClick={() => openDefinitionDialog("organelle")} disabled={creatingSegmentation}>
            Custom Organelle
          </button>
        </div>
      </div>

      <div className="segmentation-quick-group segmentation-areas-group">
        <div className="segmentation-quick-group-title">Areas</div>
        <div className="segmentation-quick-buttons">
          <button type="button" onClick={() => setPendingArea("sample")} disabled={creatingSegmentation || existingNames.has("sample area")}>
            Sample Area
          </button>
          <button type="button" onClick={() => setPendingArea("cells")} disabled={creatingSegmentation || existingNames.has("cells") || Boolean(cellsDefinitionId && existingIds.has(cellsDefinitionId))}>
            Cells
          </button>
          <button type="button" onClick={() => openDefinitionDialog("area")} disabled={creatingSegmentation}>
            Custom Area
          </button>
        </div>
      </div>

      {createSegmentationError ? (
        <div className="segmentation-error">
          {createSegmentationError.message || "Failed to create segmentation."}
        </div>
      ) : null}
    </div>
  );
}
