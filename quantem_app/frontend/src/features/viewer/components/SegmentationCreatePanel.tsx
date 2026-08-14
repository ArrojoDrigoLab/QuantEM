import { useCallback, useEffect, useMemo, useState } from "react";
import { useApiMutation } from "@/shared/hooks/useApiMutation";
import { createAssetSegmentation, getSegmentationTypes } from "@/shared/api/assets";
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
  /** Approximate decoded image size, used only for the large-image timing note. */
  imageSizeBytes?: number | null;
  /** Retained for callers compiled against the previous scale-warning dialog. */
  pixelSizeNm?: number | null;
}

interface QuickSegmentationOption {
  id: string;
  label: string;
  dialogLabel: string;
  name: string;
  group: "organelle";
  sourceModel?: string;
}

const MANUAL_CHOICE = "manual";
const LARGE_IMAGE_BYTES = 200 * 1024 * 1024;

const BUILTIN_SEGMENTATION_INTERNAL_NAMES = new Set([
  "quantem_internal_mito",
  "quantem_internal_er",
  "quantem_internal_nucleus",
  "quantem_internal_ld",
  "quantem_internal_tissue",
  "quantem_internal_analysis_mask",
]);

function isReusableCustomType(type: SegmentationType): boolean {
  return (
    type.kind === "custom" ||
    (!type.kind && !BUILTIN_SEGMENTATION_INTERNAL_NAMES.has(type.internal_name))
  );
}

/**
 * The released packs that can serve an organelle preset, default family first.
 *
 * Pack ids share the `family:organelle` spelling with source-model values, so
 * a pack picked here is exactly what `POST .../segmentations/` accepts as
 * `source_model`. An organelle the catalogue does not know yields an empty
 * list, and the dialog then behaves as it always did.
 */
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

/** Prefer the sole downloaded model; otherwise use OmniEM. */
function preferredPackId(
  catalogue: ModelCatalogue | null,
  organelleId: string
): string | null {
  const candidates = packsForOrganelle(catalogue, organelleId);
  const downloaded = candidates.filter((choice) => choice.pack?.installed === true);
  const usableDownloaded = downloaded.filter(
    (choice) => choice.pack && packRunnability(choice.pack).state !== "blocked"
  );
  if (downloaded.length === 1) return usableDownloaded[0]?.id ?? null;

  const omniem = candidates.find((choice) => choice.id === `omniem:${organelleId}`);
  const omniemBlocked = Boolean(
    omniem?.pack?.installed === true &&
      packRunnability(omniem.pack).state === "blocked"
  );
  if (!omniemBlocked) return omniem?.id ?? null;
  return usableDownloaded[0]?.id ?? null;
}

export function SegmentationCreatePanel({
  imageId,
  onCreated,
  existingSegmentationTypes,
  existingSegmentationTypeIds,
  title = "Create a Segmentation Type",
  description = "Choose a preset workflow or create a custom segmentation name.",
  imageSizeBytes,
}: SegmentationCreatePanelProps) {
  const [newCustomSegmentationName, setNewCustomSegmentationName] = useState("");
  const [customDialogOpen, setCustomDialogOpen] = useState(false);
  const [customIsObjectBased, setCustomIsObjectBased] = useState(true);
  const [newAnalysisMaskName, setNewAnalysisMaskName] = useState("");
  const [pendingOption, setPendingOption] =
    useState<QuickSegmentationOption | null>(null);
  // The exact model/manual choice captured when the dialog opens. `null` only
  // exists while no organelle dialog is open.
  const [pendingPackChoice, setPendingPackChoice] = useState<string | null>(null);
  const [pendingPackChoiceTouched, setPendingPackChoiceTouched] = useState(false);
  const { catalogue } = useModelCatalogue();
  const { data: segmentationTypes } = useApiQuery(getSegmentationTypes, []);

  // The released model-backed segmentations. Manual masks are presented in
  // their own named sections below so an image-specific analysis mask cannot
  // be mistaken for a reusable custom type.
  const quickSegmentationOptions = useMemo<QuickSegmentationOption[]>(
    () => [
      {
        id: "mito",
        label: "Mitochondria",
        dialogLabel: "mitochondria",
        name: "Mitochondria",
        group: "organelle",
      },
      {
        id: "er",
        label: "ER",
        dialogLabel: "ER",
        name: "Endoplasmic Reticulum",
        group: "organelle",
      },
      {
        id: "nucleus",
        label: "Nucleus",
        dialogLabel: "nucleus",
        name: "Nucleus",
        group: "organelle",
      },
      {
        id: "ld",
        label: "Lipid Droplets",
        dialogLabel: "lipid droplet",
        name: "Lipid Droplets",
        group: "organelle",
      },
    ],
    []
  );

  const filteredQuickOptions = useMemo(() => {
    const existing = new Set(
      (existingSegmentationTypes ?? []).map((n) => n.toLowerCase())
    );
    return quickSegmentationOptions.filter(
      (opt) => !existing.has(opt.name.toLowerCase())
    );
  }, [quickSegmentationOptions, existingSegmentationTypes]);

  const reusableCustomTypes = useMemo(
    () => (segmentationTypes ?? []).filter(isReusableCustomType),
    [segmentationTypes]
  );
  const existingTypeIds = useMemo(
    () => new Set(existingSegmentationTypeIds ?? []),
    [existingSegmentationTypeIds]
  );

  const {
    mutate: createSegmentation,
    loading: creatingSegmentation,
    error: createSegmentationError,
  } = useApiMutation(async (payload: ImageSegmentationCreatePayload) =>
    createAssetSegmentation(imageId, payload)
  );

  const handleCreateSegmentation = useCallback(
    async (
      payload: ImageSegmentationCreatePayload,
      launch?: SegmentationLaunch
    ) => {
      const created = await createSegmentation(payload);
      if (!created) return null;
      onCreated?.(created, launch);
      return created;
    },
    [createSegmentation, onCreated]
  );

  /** Open the organelle confirmation with its availability-based default. */
  const requestCreate = useCallback(
    (option: QuickSegmentationOption) => {
      setPendingPackChoiceTouched(false);
      setPendingPackChoice(
        preferredPackId(catalogue, option.id) ?? MANUAL_CHOICE
      );
      setPendingOption(option);
    },
    [catalogue]
  );

  // Re-resolve the default if catalogue data arrives after the dialog opens,
  // but never overwrite an explicit user choice.
  useEffect(() => {
    if (!pendingOption || pendingPackChoiceTouched) return;
    setPendingPackChoice(
      preferredPackId(catalogue, pendingOption.id) ?? MANUAL_CHOICE
    );
  }, [catalogue, pendingOption, pendingPackChoiceTouched]);

  // Manual plus both released families are always present. Catalogue state
  // changes only the icon and whether a genuinely incompatible installed pack
  // can be selected; missing weights are a downloadable state, not a blocker.
  const pendingPacks = pendingOption
    ? packsForOrganelle(catalogue, pendingOption.id)
    : [];
  const pendingChoice = pendingPackChoice ?? MANUAL_CHOICE;

  const confirmCreate = useCallback(() => {
    if (!pendingOption) return;
    const option = pendingOption;
    setPendingOption(null);
    const runModel = pendingChoice !== MANUAL_CHOICE;
    void handleCreateSegmentation(
      {
        segmentation_type_name: option.name,
        // Creation itself is always manual. A selected model is launched from
        // the labeling screen so a missing pack can download as its own first
        // progress step before inference is queued.
        run_inference: false,
        source_model: undefined,
      },
      { sourceModel: pendingChoice, runModel }
    );
  }, [handleCreateSegmentation, pendingChoice, pendingOption]);

  const confirmCreateCustom = useCallback(() => {
    const name = newCustomSegmentationName.trim();
    if (!name) return;
    setCustomDialogOpen(false);
    void handleCreateSegmentation({
      segmentation_type_name: name,
      measurement_mode: customIsObjectBased ? "objects" : "global",
    }).then((created) => {
      if (created) {
        setNewCustomSegmentationName("");
        setCustomIsObjectBased(true);
      }
    });
  }, [customIsObjectBased, handleCreateSegmentation, newCustomSegmentationName]);

  return (
    <div className="segmentation-create-card">
      <ConfirmDialog
        isOpen={pendingOption !== null}
        title={`Start ${pendingOption?.dialogLabel ?? ""} segmentation`}
        details={
          <>
            <div
              className="segmentation-create-model-choices"
              role="radiogroup"
              aria-label="Segmentation method"
            >
              <label className="segmentation-create-model-option">
                <input
                  type="radio"
                  name="create-source-model"
                  value={MANUAL_CHOICE}
                  checked={pendingChoice === MANUAL_CHOICE}
                  onChange={() => {
                    setPendingPackChoiceTouched(true);
                    setPendingPackChoice(MANUAL_CHOICE);
                  }}
                />
                <span>Manual segmentation</span>
                <span className="segmentation-create-manual-icon" aria-hidden>
                  ✎
                </span>
              </label>
              {pendingPacks.map((choice) => {
                const blocked =
                  choice.pack?.installed === true &&
                  packRunnability(choice.pack).state === "blocked";
                return (
                  <label
                    key={choice.id}
                    className="segmentation-create-model-option"
                  >
                    <input
                      type="radio"
                      name="create-source-model"
                      value={choice.id}
                      checked={pendingChoice === choice.id}
                      disabled={blocked}
                      onChange={() => {
                        setPendingPackChoiceTouched(true);
                        setPendingPackChoice(choice.id);
                      }}
                    />
                    <span>{choice.label}</span>
                    <ModelAvailabilityIcon pack={choice.pack} />
                  </label>
                );
              })}
            </div>
            {pendingChoice !== MANUAL_CHOICE ? (
              <p>
                The selected model will run on all tiles. Larger models may be
                more accurate, but take longer to run.
              </p>
            ) : null}
            {pendingChoice !== MANUAL_CHOICE &&
            imageSizeBytes !== null &&
            imageSizeBytes !== undefined &&
            imageSizeBytes > LARGE_IMAGE_BYTES ? (
              <p>
                This image is larger than 200 MB, so inference may take several
                minutes.
              </p>
            ) : null}
          </>
        }
        confirmText={
          pendingChoice === MANUAL_CHOICE
            ? "Start manual segmentation"
            : "Run model"
        }
        cancelText="Cancel"
        onConfirm={confirmCreate}
        onCancel={() => setPendingOption(null)}
      />
      <ConfirmDialog
        isOpen={customDialogOpen}
        title="Create custom segmentation"
        message="Custom segmentations are available in every image and open directly in manual labeling."
        details={
          <div className="segmentation-custom-dialog">
            <label htmlFor="new-custom-segmentation-name">
              Segmentation name
              <input
                id="new-custom-segmentation-name"
                type="text"
                value={newCustomSegmentationName}
                onChange={(event) => setNewCustomSegmentationName(event.target.value)}
                placeholder="e.g. Vesicles"
                autoFocus
              />
            </label>
            <label className="segmentation-custom-object-checkbox">
              <input
                type="checkbox"
                checked={customIsObjectBased}
                onChange={(event) => setCustomIsObjectBased(event.target.checked)}
              />
              Object-based segmentation
            </label>
            <details>
              <summary>Object-based or global?</summary>
              <p>
                Object-based segmentations keep individual objects so you can
                measure circularity, area per object, and distances to nearby
                objects. They work well for mitochondria and nuclei.
              </p>
              <p>
                Global segmentations treat the result as one overall mask. They
                suit features such as ER, where individual-object measurements
                are not needed, and are typically faster to analyse when there
                are many disconnected regions.
              </p>
            </details>
            <p className="segmentation-custom-training-note">
              Once created, this custom segmentation is available in all images.
              You can combine annotations from those images to train a model for
              the new class. New models need substantially more data than
              fine-tuning: start with at least 50 ROIs. Training may run slowly
              on machines without CUDA enabled.
            </p>
          </div>
        }
        confirmText="Create custom segmentation"
        cancelText="Cancel"
        confirmDisabled={!newCustomSegmentationName.trim() || creatingSegmentation}
        onConfirm={confirmCreateCustom}
        onCancel={() => setCustomDialogOpen(false)}
      />
      <h3>{title}</h3>
      <p>{description}</p>
      <div className="segmentation-quick-groups">
        {filteredQuickOptions.length > 0 ? (
          <div className="segmentation-quick-group">
            <div className="segmentation-quick-group-title">Built-in organelles</div>
            <div className="segmentation-quick-buttons">
              {filteredQuickOptions.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => requestCreate(option)}
                  disabled={creatingSegmentation}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        ) : null}
      </div>
      {filteredQuickOptions.length === 0 && (
        <div className="segmentation-error">
          All built-in segmentation options already exist for this image.
        </div>
      )}

      <div className="segmentation-manual-section">
        <div className="segmentation-quick-group-title">
          Analysis Segmentation Mask
          <span
            className="segmentation-info-tooltip"
            title="Make a mask with specific areas to analyze in this image (ex. tissue or cell outlines)"
            aria-label="Make a mask with specific areas to analyze in this image (ex. tissue or cell outlines)"
          >
            i
          </span>
        </div>
        <label htmlFor="new-analysis-mask-name">Mask name</label>
        <input
          id="new-analysis-mask-name"
          type="text"
          value={newAnalysisMaskName}
          onChange={(event) => setNewAnalysisMaskName(event.target.value)}
          placeholder="e.g. Tissue mask"
        />
        <button
          type="button"
          onClick={() => {
            const name = newAnalysisMaskName.trim();
            if (!name) return;
            void handleCreateSegmentation({
              segmentation_type_name: "Analysis Segmentation Mask",
              analysis_name: name,
            }).then((created) => {
              if (created) setNewAnalysisMaskName("");
            });
          }}
          disabled={creatingSegmentation || !newAnalysisMaskName.trim()}
        >
          {creatingSegmentation ? "Creating..." : "Create analysis mask"}
        </button>
      </div>

      {reusableCustomTypes.length > 0 ? (
        <div className="segmentation-manual-section">
          <div className="segmentation-quick-group-title">Custom</div>
          <div className="segmentation-quick-buttons">
            {reusableCustomTypes.map((type) => {
              const alreadyAdded = existingTypeIds.has(type.id);
              return (
                <button
                  key={type.id}
                  type="button"
                  disabled={creatingSegmentation || alreadyAdded}
                  title={alreadyAdded ? "Already added to this image" : undefined}
                  onClick={() => {
                    void handleCreateSegmentation({ segmentation_type_id: type.id });
                  }}
                >
                  {type.short_name}
                </button>
              );
            })}
          </div>
        </div>
      ) : null}

      <button
        className="segmentation-create-custom-button"
        type="button"
        onClick={() => setCustomDialogOpen(true)}
        disabled={creatingSegmentation}
      >
        {creatingSegmentation ? "Creating..." : "Create custom segmentation"}
      </button>

      {createSegmentationError && (
        <div className="segmentation-error">
          {createSegmentationError.message || "Failed to create segmentation."}
        </div>
      )}
    </div>
  );
}
