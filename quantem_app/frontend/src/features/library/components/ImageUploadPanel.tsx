/**
 * Import an image into the local library.
 *
 * The accepted file types come from `/api/system/status/`
 * (`supported_upload_formats`), never from a list hard-coded here: the server
 * validates against `UPLOAD_SUFFIXES` and the field exists precisely so the
 * picker cannot drift from it. PNG was accepted by the server and rejected by
 * this form for exactly that reason.
 *
 * This is also the third — and busiest — door into an uncalibrated inference
 * run. Each ticked "Segment ..." box queues a whole-image pass through
 * `default_source_model_for_organelle`, exactly the run the create-segmentation
 * dialog and "Run Full Segmentation" both stop to warn about. This form used to
 * say only that units stay in pixels, which reads as a reporting detail you can
 * fix later, so images got segmented uncalibrated on first contact and the
 * object sets were never revisited. The same warning the other two doors use now
 * appears here, as soon as the combination that causes it exists.
 *
 * Two things about *when* it appears were wrong, and both cost more than they
 * saved:
 *
 * **It fired on the correct workflow.** The warning was decided from the typed
 * box alone, while the helper text directly above that box says "Leave blank to
 * use the value in the file". So a TIFF declaring 5 nm/px, imported the way the
 * form tells you to import it, produced the full warning and a button reading
 * "Import and segment uncalibrated" over an import that came back
 * `pixel_size_nm: 5.0`. A warning that fires on the correct workflow is one
 * people learn to click through, and this is the warning that must still land
 * when it is right. `probeFileDeclaredPixelSize` reads the file's own header
 * before the form claims anything, and the wording hedges when it cannot.
 *
 * **All four boxes were ticked.** A machine with two packs queued four runs on a
 * first import and two of them FAILED; restricting the default to installed
 * packs fixed the two failures and left the real problem, which was the number
 * of runs. A full install ticked all four, so importing one image queued four
 * whole-image CPU passes -- tens of minutes each, on organelles nobody had said
 * they were interested in, with the Library's queue sidebar the only place to
 * stop them. Nothing is ticked now: importing an image imports an image, and
 * every run is asked for. That is what the other two doors already do.
 *
 * **And the button said "Upload" while four runs were about to start.** The
 * caption only mentioned segmentation when the run would be *uncalibrated*, so
 * a calibrated image -- the case where the runs certainly happen -- got the
 * plainest wording on the form. It now counts the runs in every case.
 */

import { useEffect, useMemo, useState } from "react";
import { uploadAsset } from "@/shared/api/assets";
import { getSystemStatus } from "@/shared/api/jobs";
import { useApiMutation } from "@/shared/hooks/useApiMutation";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { formatPixelSizeNm, parsePixelSizeInput } from "@/shared/pixelSize";
import {
  probeFileDeclaredPixelSize,
  type FileDeclaredPixelSize,
} from "@/shared/fileDeclaredPixelSize";
import { useModelCatalogue } from "@/features/models/useModelCatalogue";
import {
  DEFAULT_PACK_FOR_ORGANELLE,
  scaleMismatchesForOrganelles,
} from "@/features/models/scaleMismatch";
import { runnabilityForPackId } from "@/features/models/runnable";
import { UncalibratedImportWarning } from "@/features/models/components/UncalibratedScaleWarning";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type { AssetDetail, UploadImageOptions } from "@/shared/types/images";
import "./ImageUploadPanel.css";

/**
 * Used only until the first `/api/system/status/` response lands (or if it
 * fails). Kept minimal on purpose: it is a stopgap, not a second source of
 * truth.
 */
const FALLBACK_UPLOAD_FORMATS = [".tif", ".tiff", ".png"];

/**
 * The four runs this form can queue, in the order they are offered.
 *
 * `id` matches `DEFAULT_PACK_FOR_ORGANELLE`, which mirrors the server's
 * `default_source_model_for_organelle`, so the pack named in a warning is the
 * pack the import will actually use.
 */
const ORGANELLE_CHOICES = [
  { id: "mito", inputId: "segment-mito", label: "Segment mitochondria" },
  { id: "er", inputId: "segment-er", label: "Segment ER" },
  { id: "nucleus", inputId: "segment-nucleus", label: "Segment nucleus" },
  { id: "ld", inputId: "segment-ld", label: "Segment lipid droplets" },
] as const;

type OrganelleId = (typeof ORGANELLE_CHOICES)[number]["id"];

/**
 * What to tick before the user has touched anything: nothing.
 *
 * Each box is one whole-image inference pass on the CPU, minutes to tens of
 * minutes, and the only way to stop one once it is queued is the Library's job
 * sidebar. A default of "everything runnable" made that four passes on a full
 * install, chosen by what happens to be installed rather than by what the image
 * is of -- and the person importing has not yet seen the image, let alone
 * decided which organelles they care about.
 *
 * Restricting the default to *runnable* packs was the previous fix and it
 * addressed a different complaint (two of the four runs FAILED on a partial
 * install). The runs that succeeded were never the smaller problem.
 *
 * There is no lost capability here: every organelle is one tick away on this
 * form, and "Run Full Segmentation" on the labeling screen queues the identical
 * pass once the image is open and calibrated -- which is the point at which the
 * choice can actually be made well.
 */
const NO_ORGANELLES_TICKED: OrganelleId[] = [];

function normaliseExtension(value: string): string {
  const trimmed = value.trim().toLowerCase();
  if (!trimmed) return "";
  return trimmed.startsWith(".") ? trimmed : `.${trimmed}`;
}

/** `[".tif", ".tiff"]` -> `".tif, .tiff"` for prose. */
function formatExtensionList(extensions: string[]): string {
  return extensions.join(", ");
}

function stripKnownExtension(filename: string, extensions: string[]): string {
  const lower = filename.toLowerCase();
  const match = extensions.find((extension) => lower.endsWith(extension));
  return match ? filename.slice(0, -match.length) : filename;
}

interface ImageUploadPanelProps {
  onUploaded?: (asset: AssetDetail) => void;
  /** Open on mount -- the library's empty state points at this panel. */
  defaultExpanded?: boolean;
}

export function ImageUploadPanel({
  onUploaded,
  defaultExpanded = false,
}: ImageUploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [displayName, setDisplayName] = useState("");
  const [pixelSizeText, setPixelSizeText] = useState("");
  const [notesText, setNotesText] = useState("");
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [fileError, setFileError] = useState<string | null>(null);
  const [pixelSizeError, setPixelSizeError] = useState<string | null>(null);
  const [tickedOrganelles, setTickedOrganelles] =
    useState<OrganelleId[]>(NO_ORGANELLES_TICKED);
  /** What the chosen file declares, once it has been read. Null while pending. */
  const [fileScale, setFileScale] = useState<FileDeclaredPixelSize | null>(null);

  const { data: systemStatus } = useApiQuery(() => getSystemStatus(), []);
  const { catalogue } = useModelCatalogue();

  /**
   * Read the file's own pixel size as soon as one is chosen.
   *
   * Header only -- a few hundred bytes through `Blob.slice` -- so this costs
   * nothing on a 40k x 40k image. `cancelled` matters because a user who picks
   * two files quickly would otherwise get the first file's answer applied to
   * the second.
   */
  useEffect(() => {
    setFileScale(null);
    if (!file) return undefined;
    let cancelled = false;
    void probeFileDeclaredPixelSize(file).then((result) => {
      if (!cancelled) setFileScale(result);
    });
    return () => {
      cancelled = true;
    };
  }, [file]);

  /**
   * What the typed pixel size is *right now*, by the same rule the submit uses.
   *
   * Read from the draft text rather than from the saved asset, because the
   * whole point is to warn while the field is still empty and still editable.
   * A half-typed or invalid value counts as no pixel size: it is what would be
   * sent, and `parsePixelSizeInput` is the same check the server applies.
   */
  const draftPixelSizeNm = useMemo(() => {
    const parsed = parsePixelSizeInput(pixelSizeText);
    return parsed.error ? null : parsed.value;
  }, [pixelSizeText]);

  const declaredPixelSizeNm =
    fileScale?.state === "declared" ? fileScale.pixelSizeNm : null;

  /**
   * The pixel size the import will end up with, as best this form can tell.
   *
   * A typed value wins, exactly as it does server-side (`create_asset` only
   * falls back to the file's metadata when none was posted). `null` means
   * either "the file was read and says nothing" or "the file could not be
   * read"; `uncalibratedIsCertain` is what separates those two.
   */
  const effectivePixelSizeNm = draftPixelSizeNm ?? declaredPixelSizeNm;

  /**
   * The packs that would resample and cannot. Empty when the image is
   * calibrated, when nothing that declares a working resolution is ticked (ER
   * runs at native scale by design), or when the catalogue has not answered —
   * an unknown pack must not be claimed to declare a resolution.
   */
  const scaleMismatches = useMemo(
    () =>
      scaleMismatchesForOrganelles(
        catalogue,
        tickedOrganelles,
        effectivePixelSizeNm
      ),
    [catalogue, tickedOrganelles, effectivePixelSizeNm]
  );
  const willRunUncalibrated = scaleMismatches.length > 0;
  /** How many whole-image passes pressing the button would queue. */
  const runCount = tickedOrganelles.length;
  /**
   * Whether "this will run uncalibrated" is a fact or a possibility.
   *
   * Only a file that was read and found silent earns the flat statement. A file
   * that could not be parsed here (BigTIFF, an unfamiliar layout) or no file at
   * all leaves it a conditional -- which is weaker, and correct.
   */
  const uncalibratedIsCertain =
    draftPixelSizeNm === null && fileScale?.state === "silent";

  const acceptedExtensions = useMemo(() => {
    const fromServer = (systemStatus?.supported_upload_formats ?? [])
      .map(normaliseExtension)
      .filter(Boolean);
    return fromServer.length > 0 ? fromServer : FALLBACK_UPLOAD_FORMATS;
  }, [systemStatus]);

  const { mutate, loading, error, reset } = useApiMutation<
    { file: File; options: UploadImageOptions },
    AssetDetail
  >(async ({ file, options }) => {
    return uploadAsset(file, options);
  });

  useEffect(() => {
    if (!expanded) {
      setFileError(null);
      setPixelSizeError(null);
    }
  }, [expanded]);

  const toggleOrganelle = (id: OrganelleId, checked: boolean) => {
    setTickedOrganelles((current) =>
      checked
        ? current.includes(id)
          ? current
          : [...current, id]
        : current.filter((value) => value !== id)
    );
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;
    const lower = selectedFile.name.toLowerCase();
    if (!acceptedExtensions.some((extension) => lower.endsWith(extension))) {
      // Inline, not `alert()`: a modal dialog that steals focus is not a
      // validation message, and it cannot be read by anything but a human.
      setFileError(
        `${selectedFile.name} is not a supported format. This build accepts ${formatExtensionList(
          acceptedExtensions
        )}.`
      );
      setFile(null);
      e.target.value = "";
      return;
    }
    setFileError(null);
    setFile(selectedFile);
    if (!displayName) {
      setDisplayName(stripKnownExtension(selectedFile.name, acceptedExtensions));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setFileError("Choose a file to import.");
      return;
    }
    const pixelSize = parsePixelSizeInput(pixelSizeText);
    if (pixelSize.error) {
      setPixelSizeError(pixelSize.error);
      return;
    }
    setPixelSizeError(null);

    reset();
    const notes = notesText.trim();
    const result = await mutate({
      file,
      options: {
        displayName: displayName.trim() || undefined,
        pixelSizeNm: pixelSize.value,
        notes: notes.length > 0 ? notes : undefined,
        segmentMito: tickedOrganelles.includes("mito"),
        segmentEr: tickedOrganelles.includes("er"),
        segmentNucleus: tickedOrganelles.includes("nucleus"),
        segmentLd: tickedOrganelles.includes("ld"),
      },
    });

    if (result && onUploaded) {
      // Clear form
      setFile(null);
      setDisplayName("");
      setPixelSizeText("");
      setNotesText("");
      setTickedOrganelles(NO_ORGANELLES_TICKED);
      setExpanded(false);
      // Reset file input
      const fileInput = document.getElementById(
        "file-input"
      ) as HTMLInputElement;
      if (fileInput) fileInput.value = "";
      onUploaded(result);
    }
  };

  return (
    <div className="upload-panel">
      <button
        type="button"
        className="upload-toggle"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
      >
        {expanded ? "▼" : "▶"} Import image
      </button>

      {expanded && (
        <form className="upload-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="file-input">
              Image file ({formatExtensionList(acceptedExtensions)}):
            </label>
            <input
              id="file-input"
              type="file"
              accept={acceptedExtensions.join(",")}
              onChange={handleFileChange}
              aria-describedby="file-input-help"
              required
            />
            <span className="upload-help" id="file-input-help">
              Formats this build can read, reported by the server.
            </span>
            {file && (
              <span className="file-name">Selected: {file.name}</span>
            )}
            {fileError && (
              <span className="error-message" role="alert">
                {fileError}
              </span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="display-name">Display Name (optional):</label>
            <input
              id="display-name"
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Enter display name..."
            />
          </div>

          <div className="form-group">
            <label htmlFor="upload-pixel-size">Pixel size, nm per pixel (optional):</label>
            <input
              id="upload-pixel-size"
              type="number"
              min="0"
              step="any"
              inputMode="decimal"
              value={pixelSizeText}
              onChange={(e) => setPixelSizeText(e.target.value)}
              placeholder="e.g. 4.2"
              aria-describedby="upload-pixel-size-help"
            />
            <span className="upload-help" id="upload-pixel-size-help">
              Leave blank to use the value in the file. Many EM exports carry
              none; without a pixel size, areas and distances stay in pixels and
              analysis cannot report µm².{" "}
              {/* "You can set or change it later" is true of the number and
                  false of everything downstream of it once a run has been
                  queued, and read on its own it is what made an uncalibrated
                  import look reversible. */}
              {willRunUncalibrated
                ? "You can set or change it later, but that does not re-run anything segmented now."
                : "You can set or change it later."}
            </span>
            {pixelSizeError && (
              <span className="error-message" role="alert">
                {pixelSizeError}
              </span>
            )}
            {/* Answering the question the box asks, from the file itself. The
                form told people to leave this blank and then warned them for
                doing it; saying what the file declares is what makes "blank"
                a legible choice rather than a gamble. */}
            {declaredPixelSizeNm !== null && (
              <span className="upload-declared-scale" role="status">
                {file?.name ?? "This file"} declares{" "}
                {formatPixelSizeNm(declaredPixelSizeNm)}
                {draftPixelSizeNm === null
                  ? ", which this import will use."
                  : ". The value typed above is used instead."}
              </span>
            )}
          </div>

          {/* This was "Tags (comma-separated, optional)". It collected text,
              posted it as `tag_names`, and the server never read it -- there is
              no tag field on `Asset` and no tag anywhere in the Python tree, so
              typing "PV" here did nothing at all and the library went on
              showing no tags. A field that accepts input and discards it is how
              people come to believe their images are grouped when they are not.

              Notes is the field that already exists: it is on `Asset`, it is
              PATCHable through `update_asset`, and `_filtered_asset_queryset`
              includes it in the library search alongside the display name and
              the filename. So the word typed here is findable, which is what
              the tag box was reached for and never delivered. */}
          <div className="form-group">
            <label htmlFor="notes">Notes (optional):</label>
            <input
              id="notes"
              type="text"
              value={notesText}
              onChange={(e) => setNotesText(e.target.value)}
              placeholder="e.g. PV, day 14, control"
              aria-describedby="notes-help"
            />
            <span className="upload-help" id="notes-help">
              Saved with the image. The library&apos;s search box matches notes
              as well as names and filenames, so a word typed here is how you
              find this image again.
            </span>
          </div>

          <div className="form-group">
            {ORGANELLE_CHOICES.map((choice) => {
              const runnability = runnabilityForPackId(
                catalogue,
                DEFAULT_PACK_FOR_ORGANELLE[choice.id]
              );
              // Only a definite "blocked" disables, exactly as on the labeling
              // screen: an unanswered catalogue must not hide a working model.
              const blocked = runnability.state === "blocked";
              return (
                <label key={choice.id} htmlFor={choice.inputId}>
                  <input
                    id={choice.inputId}
                    type="checkbox"
                    checked={tickedOrganelles.includes(choice.id)}
                    disabled={blocked}
                    onChange={(e) =>
                      toggleOrganelle(choice.id, e.target.checked)
                    }
                  />
                  {choice.label}
                  {blocked && (
                    <span className="upload-organelle-blocked">
                      {" — "}
                      {runnability.reason ??
                        `${DEFAULT_PACK_FOR_ORGANELLE[choice.id]} cannot run here.`}
                    </span>
                  )}
                </label>
              );
            })}
            {/* A disabled control with nothing beside it reads as a bug, and
                the Models screen is the only place the blocker is fixable.

                The first sentence is why nothing starts ticked: four ticked
                boxes were four whole-image CPU passes queued by an import,
                before the person had seen the image. */}
            <span className="upload-help">
              Nothing is segmented unless you tick a box. Each one queues a
              separate whole-image run after the import — minutes to tens of
              minutes each — and you can start any of them later from the
              labeling screen instead. Models that cannot run here are listed
              but not offered — <a href="#/models">Models</a> says what is
              missing.
            </span>
            {/* Directly under the boxes that cause it, not at the top of the
                form: this is the moment the choice is being made, and the
                sentence is about what those four boxes will produce. */}
            <UncalibratedImportWarning
              mismatches={scaleMismatches}
              certain={uncalibratedIsCertain}
              unreadableFileName={
                fileScale?.state === "unknown" ? (file?.name ?? null) : null
              }
            />
          </div>

          {error && (
            <div className="error-message" role="alert">
              {extractApiErrorMessage(error, "The image could not be imported.")}
            </div>
          )}

          <div className="form-actions">
            {/* What pressing this actually does, counted.

                The caption used to name segmentation only when the run would be
                *uncalibrated*, so the one case where the runs were certain to
                happen -- a calibrated image with boxes ticked -- read plain
                "Upload". The run count leads now, in every case.

                "Uncalibrated" is still only asserted when it is a fact. A button
                that says it over a file that turns out to declare 5 nm/px is how
                a warning stops being believed, and this is the warning that has
                to land when it is right. */}
            <button type="submit" disabled={loading || !file}>
              {loading
                ? "Uploading..."
                : runCount === 0
                  ? "Upload"
                  : willRunUncalibrated && uncalibratedIsCertain
                    ? `Import and start ${runCount} uncalibrated run${runCount === 1 ? "" : "s"}`
                    : `Import and start ${runCount} segmentation run${runCount === 1 ? "" : "s"}`}
            </button>
            <button
              type="button"
              onClick={() => {
                setExpanded(false);
                reset();
              }}
              disabled={loading}
            >
              Cancel
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
