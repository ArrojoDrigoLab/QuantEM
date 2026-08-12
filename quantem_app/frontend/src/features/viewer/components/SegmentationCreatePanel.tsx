import { useCallback, useMemo, useState } from "react";
import { useApiMutation } from "@/shared/hooks/useApiMutation";
import { createAssetSegmentation, getSegmentationTypes } from "@/shared/api/assets";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { useModelCatalogue } from "@/features/models/useModelCatalogue";
import {
  describeDevice,
  packRunnability,
  runnabilityForPackId,
} from "@/features/models/runnable";
import {
  DEFAULT_PACK_FOR_ORGANELLE,
  scaleMismatchForPack,
  type ScaleMismatch,
} from "@/features/models/scaleMismatch";
import { UncalibratedScaleWarning } from "@/features/models/components/UncalibratedScaleWarning";
import type { ModelCatalogue, ModelPack } from "@/shared/types/finetune";
import type {
  ImageSegmentation,
  ImageSegmentationCreatePayload,
  SegmentationType,
} from "@/shared/types/images";
import "./SegmentationCreatePanel.css";

interface SegmentationCreatePanelProps {
  imageId: string;
  onCreated?: (segmentation: ImageSegmentation) => void;
  existingSegmentationTypes?: string[];
  existingSegmentationTypeIds?: string[];
  title?: string;
  description?: string;
  /**
   * The asset's pixel size, so the confirmation can say what running without
   * one costs. `null` means uncalibrated; `undefined` means the caller does not
   * know, and the dialog then says nothing rather than guessing.
   */
  pixelSizeNm?: number | null;
}

interface QuickSegmentationOption {
  id: string;
  label: string;
  name: string;
  group: "organelle";
  sourceModel?: string;
}

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
function packsForOrganelle(
  catalogue: ModelCatalogue | null,
  organelleId: string
): ModelPack[] {
  const packs = (catalogue?.packs ?? []).filter(
    (pack) => pack.organelle === organelleId
  );
  const defaultId = DEFAULT_PACK_FOR_ORGANELLE[organelleId];
  return [...packs].sort((a, b) => {
    if (a.id === defaultId) return -1;
    if (b.id === defaultId) return 1;
    return a.id.localeCompare(b.id);
  });
}

/**
 * The pack the dialog preselects: the default family unless it is *known*
 * blocked and another family's pack is known runnable.
 *
 * This is the paper-cut: on a machine with only OmniEM — Mitochondria
 * installed, "Create Mitochondria" led with "quantem:mito cannot run on this
 * machine — Create anyway", never mentioning the installed model one select
 * away on the labeling screen. An unknown runnability changes nothing: the
 * default stays selected rather than being second-guessed.
 */
function preferredPackId(
  catalogue: ModelCatalogue | null,
  organelleId: string
): string | null {
  const defaultId = DEFAULT_PACK_FOR_ORGANELLE[organelleId] ?? null;
  const candidates = packsForOrganelle(catalogue, organelleId);
  if (candidates.length === 0) return defaultId;
  const defaultPack = candidates.find((pack) => pack.id === defaultId);
  if (!defaultPack || packRunnability(defaultPack).state !== "blocked") {
    return defaultId;
  }
  const runnableAlternative = candidates.find(
    (pack) => pack.id !== defaultId && packRunnability(pack).state === "runnable"
  );
  return runnableAlternative?.id ?? defaultId;
}

export function SegmentationCreatePanel({
  imageId,
  onCreated,
  existingSegmentationTypes,
  existingSegmentationTypeIds,
  title = "Create a Segmentation Type",
  description = "Choose a preset workflow or create a custom segmentation name.",
  pixelSizeNm,
}: SegmentationCreatePanelProps) {
  const [newCustomSegmentationName, setNewCustomSegmentationName] = useState("");
  const [customDialogOpen, setCustomDialogOpen] = useState(false);
  const [customIsObjectBased, setCustomIsObjectBased] = useState(true);
  const [newAnalysisMaskName, setNewAnalysisMaskName] = useState("");
  const [pendingOption, setPendingOption] =
    useState<QuickSegmentationOption | null>(null);
  // The model the pending run will use. `null` means "the preferred pack" —
  // the default family, or the other family when only it can run. Reset when
  // the dialog opens so one preset's choice never leaks onto another.
  const [pendingPackChoice, setPendingPackChoice] = useState<string | null>(null);
  const { catalogue } = useModelCatalogue();
  const { data: segmentationTypes } = useApiQuery(getSegmentationTypes, []);

  // The released model-backed segmentations. Manual masks are presented in
  // their own named sections below so an image-specific analysis mask cannot
  // be mistaken for a reusable custom type.
  const quickSegmentationOptions = useMemo<QuickSegmentationOption[]>(
    () => [
      { id: "mito", label: "Mitochondria", name: "Mitochondria", group: "organelle" },
      {
        id: "er",
        label: "ER",
        name: "Endoplasmic Reticulum",
        group: "organelle",
      },
      { id: "nucleus", label: "Nucleus", name: "Nucleus", group: "organelle" },
      {
        id: "ld",
        label: "Lipid Droplets",
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
    async (payload: ImageSegmentationCreatePayload) => {
      const created = await createSegmentation(payload);
      if (!created) return null;
      onCreated?.(created);
      return created;
    },
    [createSegmentation, onCreated]
  );

  /**
   * Creating an organelle segmentation is not a free action.
   *
   * `POST /api/assets/<id>/segmentations/` enqueues a whole-image inference run
   * on the spot — roughly a minute of CPU on a 2k image — and the labeling
   * screen then disables "Run Full Segmentation" until it finishes. Nothing
   * asked, and on a machine with no installed models the whole thing existed
   * only as a progress banner stuck at 5%. So: name the cost, name the model,
   * and say up front when that model cannot run.
   *
   * Analysis and custom masks are manual-only; this confirmation belongs only
   * to a model-backed organelle run.
   */
  const requestCreate = useCallback(
    (option: QuickSegmentationOption) => {
      setPendingPackChoice(null);
      setPendingOption(option);
    },
    []
  );

  // The default pack for the pending organelle — what the server would run
  // with if the request said nothing. Shared with the import form, which
  // queues the same runs from its checkboxes.
  const defaultPackId = pendingOption
    ? DEFAULT_PACK_FOR_ORGANELLE[pendingOption.id] ?? null
    : null;
  // Both families' packs for this organelle, so the dialog can offer the
  // same choice the labeling screen's "Model to run" picker offers — instead
  // of pronouncing "cannot run on this machine" about the default while an
  // installed alternative goes unmentioned.
  const pendingPacks = pendingOption
    ? packsForOrganelle(catalogue, pendingOption.id)
    : [];
  const pendingPackId = pendingOption
    ? pendingPackChoice ?? preferredPackId(catalogue, pendingOption.id)
    : null;

  const confirmCreate = useCallback(() => {
    if (!pendingOption) return;
    const option = pendingOption;
    setPendingOption(null);
    void handleCreateSegmentation({
      segmentation_type_name: option.name,
      // Stated only when it differs from the server's own default, so a
      // build with no catalogue keeps the exact request shape it always sent.
      source_model:
        pendingPackId && pendingPackId !== defaultPackId
          ? pendingPackId
          : undefined,
    });
  }, [defaultPackId, handleCreateSegmentation, pendingOption, pendingPackId]);

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

  const pendingRunnability = runnabilityForPackId(catalogue, pendingPackId);
  const device = describeDevice(catalogue);
  // The pack declares a working resolution and this image cannot be resampled
  // to it. Shared with the labeling screen's run button, which is the other
  // door into exactly the same inference pass.
  const scaleMismatch = scaleMismatchForPack(
    catalogue,
    pendingPackId,
    pixelSizeNm
  );
  const dialogTone =
    pendingRunnability.state === "blocked" || scaleMismatch ? "warning" : "default";

  return (
    <div className="segmentation-create-card">
      <ConfirmDialog
        isOpen={pendingOption !== null}
        title={`Create ${pendingOption?.label ?? ""} and run it?`}
        message={
          "Creating this segmentation immediately queues one inference pass over the whole image. " +
          "You can leave the screen while it runs, but it holds the run button until it finishes."
        }
        details={
          <>
            {/* The labeling screen's "Model to run" choice, at the one moment
                it matters most: before the run is queued. Rendered whenever
                the catalogue knows more than one pack for this organelle, so
                a blocked default is never the end of the sentence while an
                installed alternative exists. */}
            {pendingPacks.length > 1 && (
              <label
                className="segmentation-create-model-choice"
                htmlFor="create-source-model-select"
              >
                <span>Model to run</span>
                <select
                  id="create-source-model-select"
                  aria-label="Model to run"
                  value={pendingPackId ?? ""}
                  onChange={(event) => setPendingPackChoice(event.target.value)}
                >
                  {pendingPacks.map((pack) => {
                    const runnability = packRunnability(pack);
                    return (
                      <option key={pack.id} value={pack.id}>
                        {pack.title}
                        {runnability.state === "blocked"
                          ? ` — ${runnability.label}`
                          : ""}
                      </option>
                    );
                  })}
                </select>
              </label>
            )}
            <CreateRunNotice
              packId={pendingPackId}
              defaultPackId={defaultPackId}
              packs={pendingPacks}
              runnability={pendingRunnability}
              device={device}
              scaleMismatch={scaleMismatch}
            />
          </>
        }
        detailsTone={dialogTone}
        confirmText={
          pendingRunnability.state === "blocked"
            ? "Create anyway"
            : scaleMismatch
              ? "Create and run uncalibrated"
              : "Create and run"
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

/**
 * What the queued run will cost, and whether it can succeed at all.
 *
 * The model name matters because the run is attributed to it and that
 * attribution ends up in a methods section; the device matters because CPU
 * inference on a 2k image is tens of seconds, not instant.
 */
function CreateRunNotice({
  packId,
  defaultPackId,
  packs,
  runnability,
  device,
  scaleMismatch,
}: {
  packId: string | null;
  defaultPackId: string | null;
  packs: ModelPack[];
  runnability: ReturnType<typeof runnabilityForPackId>;
  device: string | null;
  scaleMismatch: ScaleMismatch | null;
}) {
  if (runnability.state === "blocked") {
    // Only true once the alternative has been checked: when the other
    // family's pack is runnable, `preferredPackId` selects it instead and
    // this branch is never the dialog's opening line.
    return (
      <>
        <p>
          <strong>{packId} cannot run on this machine.</strong>{" "}
          {runnability.reason}
        </p>
        <p>
          The segmentation will still be created and you can annotate it by
          hand, but the run queued alongside it will fail.{" "}
          <a href="#/models">Install the model first</a> if you want it to
          produce candidates.
        </p>
      </>
    );
  }

  // Said out loud when a *blocked* default was passed over for the selected
  // model: the run is attributed to the model that made it, and a
  // substitution the user never noticed is how a methods section names the
  // wrong one. A user who picked the other family while the default runs
  // fine gets no such sentence — nothing was substituted.
  const defaultPack =
    packId !== defaultPackId && defaultPackId
      ? packs.find((pack) => pack.id === defaultPackId)
      : undefined;
  const blockedDefault =
    defaultPack && packRunnability(defaultPack).state === "blocked"
      ? defaultPack
      : undefined;
  const blockedDefaultReason = blockedDefault
    ? packRunnability(blockedDefault).reason
    : null;

  return (
    <>
      {blockedDefault ? (
        <p>
          <strong>{blockedDefault.id} cannot run on this machine</strong>
          {blockedDefaultReason
            ? ` (${blockedDefaultReason.replace(/\.$/, "")})`
            : ""}
          , so the installed {packId} is selected instead. The objects will be
          attributed to {packId}.{" "}
          <a href="#/models">Install {blockedDefault.id}</a> if you want the
          default family.
        </p>
      ) : null}
      {/* Paper-cut 8: the model and device are visual chips, which splits the
          paragraph into text fragments — read as a unit it came out as "The
          run will use on .". The sentence itself must carry the names, so the
          accessible copy is one complete string and the chip version is
          presentation only. */}
      <p>
        <span className="sr-only">
          {`The run will use ${packId ?? "the default model"}${
            device ? ` on ${device}` : ""
          }.`}
        </span>
        <span aria-hidden="true">
          The run will use <strong>{packId ?? "the default model"}</strong>
          {device ? (
            <>
              {" "}
              on <strong>{device}</strong>
            </>
          ) : null}
          .
        </span>
      </p>
      {/* The viewer badge a few pixels away already says "Pixel size not set".
          It does not say what that costs, and this is the last gate before the
          user spends the minute. */}
      {scaleMismatch ? <UncalibratedScaleWarning mismatch={scaleMismatch} /> : null}
      {device && !/CUDA|MPS/.test(device) ? (
        <p>
          On CPU this takes roughly a minute for a 2k image. Nothing is lost if
          you cancel — you can run it later from the labeling screen.
        </p>
      ) : null}
    </>
  );
}
