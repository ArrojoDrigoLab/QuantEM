import { useCallback, useMemo, useState } from "react";
import { useApiMutation } from "@/shared/hooks/useApiMutation";
import { createAssetSegmentation } from "@/shared/api/assets";
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
import type { ImageSegmentation } from "@/shared/types/images";
import "./SegmentationCreatePanel.css";

interface SegmentationCreatePanelProps {
  imageId: string;
  onCreated?: (segmentation: ImageSegmentation) => void;
  existingSegmentationTypes?: string[];
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
  group: "standard" | "masks";
  sourceModel?: string;
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
  title = "Create a Segmentation Type",
  description = "Choose a preset workflow or create a custom segmentation name.",
  pixelSizeNm,
}: SegmentationCreatePanelProps) {
  const [newSegmentationName, setNewSegmentationName] = useState("");
  const [pendingOption, setPendingOption] =
    useState<QuickSegmentationOption | null>(null);
  // The model the pending run will use. `null` means "the preferred pack" —
  // the default family, or the other family when only it can run. Reset when
  // the dialog opens so one preset's choice never leaks onto another.
  const [pendingPackChoice, setPendingPackChoice] = useState<string | null>(null);
  const { catalogue } = useModelCatalogue();

  // The built-in segmentation types the backend ships: four organelles, each
  // served by the QuantEM/OmniEM model pair, plus the manual-only tissue mask.
  const quickSegmentationOptions = useMemo<QuickSegmentationOption[]>(
    () => [
      {
        id: "mito",
        label: "Mitochondria",
        name: "Mitochondria",
        group: "standard",
      },
      {
        id: "er",
        label: "ER",
        name: "Endoplasmic Reticulum",
        group: "standard",
      },
      { id: "nucleus", label: "Nucleus", name: "Nucleus", group: "standard" },
      {
        id: "ld",
        label: "Lipid Droplets",
        name: "Lipid Droplets",
        group: "standard",
      },
      {
        id: "tissue",
        label: "Tissue Mask",
        name: "Tissue Mask",
        group: "masks",
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

  const {
    mutate: createSegmentation,
    loading: creatingSegmentation,
    error: createSegmentationError,
  } = useApiMutation(async (option: Pick<QuickSegmentationOption, "name" | "sourceModel">) => {
    return createAssetSegmentation(imageId, {
      segmentation_type_name: option.name,
      ...(option.sourceModel ? { source_model: option.sourceModel } : {}),
    });
  });

  const handleCreateSegmentation = useCallback(
    async (nameOrOption: string | Pick<QuickSegmentationOption, "name" | "sourceModel">) => {
      const option =
        typeof nameOrOption === "string"
          ? { name: nameOrOption, sourceModel: undefined }
          : nameOrOption;
      const trimmed = option.name.trim();
      if (!trimmed) return;
      const created = await createSegmentation({ ...option, name: trimmed });
      if (!created) return;
      setNewSegmentationName("");
      onCreated?.(created);
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
   * Manual-only types (the tissue mask) queue nothing and are created directly.
   */
  const requestCreate = useCallback(
    (option: QuickSegmentationOption) => {
      if (option.group === "masks") {
        void handleCreateSegmentation(option);
        return;
      }
      setPendingPackChoice(null);
      setPendingOption(option);
    },
    [handleCreateSegmentation]
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
      name: option.name,
      // Stated only when it differs from the server's own default, so a
      // build with no catalogue keeps the exact request shape it always sent.
      sourceModel:
        pendingPackId && pendingPackId !== defaultPackId
          ? pendingPackId
          : undefined,
    });
  }, [defaultPackId, handleCreateSegmentation, pendingOption, pendingPackId]);

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
      <h3>{title}</h3>
      <p>{description}</p>
      <div className="segmentation-quick-groups">
        {(["standard", "masks"] as const).map((group) => {
          const options = filteredQuickOptions.filter((option) => option.group === group);
          if (options.length === 0) return null;
          const groupTitle = group === "standard" ? "Standard" : "Masks";
          return (
            <div className="segmentation-quick-group" key={group}>
              <div className="segmentation-quick-group-title">{groupTitle}</div>
              <div className="segmentation-quick-buttons">
                {options.map((option) => (
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
          );
        })}
      </div>
      {filteredQuickOptions.length === 0 && (
        <div className="segmentation-error">
          All built-in segmentation options already exist for this image.
        </div>
      )}

      <label htmlFor="new-segmentation-name">Custom segmentation name</label>
      <input
        id="new-segmentation-name"
        type="text"
        value={newSegmentationName}
        onChange={(event) => setNewSegmentationName(event.target.value)}
        placeholder="e.g. Vesicles"
      />
      <button
        type="button"
        onClick={() => {
          void handleCreateSegmentation(newSegmentationName);
        }}
        disabled={creatingSegmentation || !newSegmentationName.trim()}
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
