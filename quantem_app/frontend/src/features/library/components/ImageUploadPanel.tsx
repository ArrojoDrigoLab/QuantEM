/**
 * Import images into the local library.
 *
 * ## What this panel used to do, and why none of it survived
 *
 * It was a collapsed accordion whose header button read "▶ Import image". The
 * library's own empty state offered a button called "Import an image" which
 * *expanded the accordion* and left the user looking at a `<input type=file>`
 * they still had to click. There was no drop target anywhere in the
 * application. And on success `handleSubmit` cleared every field and called
 * `setExpanded(false)`, so a completed import and a reset form were pixel-for-
 * pixel identical -- which is exactly what the owner reported as "seemed to
 * have failed (reverted back to upload button)".
 *
 * Three changes answer that:
 *
 * 1. **The picker opens on the first click.** The drop zone is a `<label>` for
 *    the file input, so a click anywhere in it opens the OS dialog through the
 *    browser's own native path -- no JavaScript, nothing to go wrong, and it
 *    works with the keyboard. {@link ImageUploadPanelHandle.openFilePicker}
 *    gives the Library's own "Import an image" button the same behaviour.
 * 2. **The same area takes a drop**, through the same validation and the same
 *    state, with a visible hover state while a file is over it.
 * 3. **Success is not a reset.** The panel does not clear itself and vanish;
 *    it hands each created asset to `onUploaded`, and the Library turns those
 *    into highlighted cards. The panel only clears
 *    the files that landed, so the next import can start.
 *
 * ## A plate, not a picture
 *
 * The zone says **"Choose files…"** and it means it. The input took exactly one
 * file for as long as that label has been on screen: the picker refused the
 * second file, and a drop of forty imported one and said so. A postdoc coming
 * off a session has a plate, so the queue is the normal case --
 *
 *   - the picker is `multiple`, and a drop of any number is accepted;
 *   - **each file is its own import**, posted one at a time to
 *     `POST /api/assets/upload/` (which takes one `file`), with its own row and
 *     its own state: *Waiting* → *Uploading…* → *Imported*, or a named failure
 *     in place;
 *   - **one file failing does not lose the other thirty-nine.** The queue
 *     carries on, the failures stay listed with the server's own words, and the
 *     button becomes "Try these 3 again";
 *   - the optional pixel size and notes **apply to the whole batch**, and every
 *     row states the pixel size it will actually be imported with, so "what is
 *     about to be applied to what" is never a guess.
 *
 * Sequential, not parallel: a plate of EM mosaics is 250 MB to 2 GB each, and
 * forty simultaneous multipart POSTs would compete for the same disk while
 * making the first image land last. The queue is also what makes per-file
 * progress mean anything.
 *
 * This is deliberately **not** the Set/Condition sheet from plan §1.2 -- no
 * groups, no sampling unit, no `ImageSet` (that is package 4.1, and it needs a
 * migration). It is only the part that stops forty images costing forty trips
 * through this panel.
 *
 * ## What is still asked, and what is not
 *
 * Only one question can silently ruin every number downstream, and that is the
 * pixel size -- packs resample before they look for anything, so the scale
 * decides which objects exist, not just the units they are reported in. So the
 * pixel size is *asked* when the file does not declare one and merely *stated*
 * when it does. `probeFileDeclaredPixelSize` reads each file's own header
 * (a few hundred bytes through `Blob.slice`) before the form claims anything.
 *
 * **Nothing is pre-filled except the display name**, which comes from the
 * filename and is inert. A pixel size harvested from anywhere other than this
 * file's own header would be a fabricated calibration -- and with a batch that
 * matters more, not less: a typed value reaches only the images that declare
 * nothing unless the user explicitly says otherwise, so one number typed over a
 * mixed folder cannot silently overwrite twelve real calibrations.
 *
 * ## Where the parts live
 *
 * This file was 1 441 lines, the largest in the app, and the first thing the
 * owner complained about. It is now the queue's state and the layout; each
 * responsibility it used to inline is its own module under `./import/`:
 *
 * * `import/importValidation.ts` — what counts as an importable file, and the
 *   prose helpers the panel words its refusals with;
 * * `import/useImportScale.ts` — the pixel size each queued file will actually
 *   be imported with, and the census the form describes the batch from;
 * * `import/ImportDropZone.tsx` — the empty state: drop target and picker;
 * * `import/ImportQueue.tsx` — the file rows and their per-file progress;
 * * `import/ImportDetailsFields.tsx` — the optional fields;
 *
 * Nothing moved changed: same DOM, same `data-testid`s, same words.
 */

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  exceedsUploadLimit,
  readMaxUploadBytes,
} from "@/shared/api/assets";
import { getSystemStatus } from "@/shared/api/jobs";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { parsePixelSizeInput } from "@/shared/pixelSize";
import { probeFileDeclaredPixelSize } from "@/shared/fileDeclaredPixelSize";
import { Button, Panel } from "@/shared/ui/design";
import { cx } from "@/shared/ui/cx";
import { formatBytes } from "@/shared/ui/format";
import {
  FALLBACK_UPLOAD_FORMATS,
  formatExtensionList,
  isSameFile,
  normaliseExtension,
  stripKnownExtension,
  type ChosenFile,
} from "@/features/library/components/import/importValidation";
import { useImportRun } from "@/features/library/components/import/useImportRun";
import type { ImportBatchPosition } from "@/features/library/components/import/useImportRun";
import { useImportScale } from "@/features/library/components/import/useImportScale";
import { ImportDropZone } from "@/features/library/components/import/ImportDropZone";
import { ImportQueue } from "@/features/library/components/import/ImportQueue";
import { ImportDetailsFields } from "@/features/library/components/import/ImportDetailsFields";
import { useExperiments } from "@/features/library/components/grouping/useExperiments";
import {
  chosenId,
  chosenName,
  isChosen,
  type GroupingChoice,
} from "@/features/library/components/grouping/groupingChoices";
import type { AssetDetail, UploadImageOptions } from "@/shared/types/images";

function defaultExperimentName(date = new Date()): string {
  const dateString = [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
  ].join("-");
  return `New Experiment ${dateString}`;
}

function newExperimentChoice(): GroupingChoice {
  return { kind: "new", name: defaultExperimentName() };
}

function newDatasetChoice(): GroupingChoice {
  return { kind: "new", name: "Dataset 1" };
}

export interface ImageUploadPanelHandle {
  /**
   * Open the OS file dialog. This is what the Library's "Import an image"
   * button calls: owner ask #1 is that the button opens the picker, not an
   * accordion.
   */
  openFilePicker: () => void;
  /**
   * Take files the user dropped somewhere else on the page. Same validation,
   * same state, same code path as the picker.
   */
  acceptDroppedFiles: (files: FileList | File[]) => void;
}

/**
 * Where one finished import sat in the batch that produced it.
 *
 * Declared with the queue that produces it and re-exported here, because
 * `LibraryPage` imports it from this module and that import is not worth
 * moving for a type.
 */
export type { ImportBatchPosition } from "@/features/library/components/import/useImportRun";

interface ImageUploadPanelProps {
  /**
   * Called once per image, the moment it lands, with its place in the batch.
   *
   * The position is not decoration: the Library opens an import by itself when
   * it is the only one (owner ask #3), and it must not do that in the middle of
   * a plate of forty. A small PNG can be ready in a second while a 2 GB mosaic
   * is still uploading, so "is this the whole batch?" cannot be inferred from
   * what has arrived so far -- only the panel running the queue knows.
   */
  onUploaded?: (asset: AssetDetail, batch: ImportBatchPosition) => void;
  /**
   * True while a drag is anywhere over the Library, so the drop zone can light
   * up before the pointer reaches it.
   */
  pageDragActive?: boolean;
  /** Use the short drop target once the home page already has experiments. */
  compact?: boolean;
}

export const ImageUploadPanel = forwardRef<
  ImageUploadPanelHandle,
  ImageUploadPanelProps
>(function ImageUploadPanel(
  { onUploaded, pageDragActive = false, compact = false },
  ref
) {
  const {
    imports,
    setImports,
    batchSummary,
    setBatchSummary,
    batchTotal,
    importing,
    runImport,
    retryDeferredProcessing,
    mountedRef,
  } = useImportRun({ onUploaded });
  const [files, setFiles] = useState<ChosenFile[]>([]);
  const [displayName, setDisplayName] = useState("");
  const [pixelSizeText, setPixelSizeText] = useState("");
  const [notesText, setNotesText] = useState("");
  /**
   * Whether a typed pixel size also replaces the values files declare.
   *
   * Off by default and only offered for a batch: with one file the typed box is
   * the documented way to correct a wrong tag, and the help text above it says
   * so. With forty, one number typed over a mixed folder would silently
   * overwrite every real calibration in it -- the exact "fabricates 43
   * calibrations" failure the plan's §1.2 is built to avoid.
   */
  const [replaceDeclaredPixelSizes, setReplaceDeclaredPixelSizes] =
    useState(false);
  /**
   * Files that were refused, and why -- one sentence each.
   *
   * A list rather than a string because a drop of forty can contain three
   * different problems, and "some files were not added" is not an error
   * message.
  */
  const [rejections, setRejections] = useState<string[]>([]);
  const [pixelSizeError, setPixelSizeError] = useState<string | null>(null);
  /**
   * Every new import starts organised, but the user can still choose a
   * different experiment or dataset before it is submitted.
   */
  const [experimentChoice, setExperimentChoice] =
    useState<GroupingChoice>(newExperimentChoice);
  const [datasetChoice, setDatasetChoice] =
    useState<GroupingChoice>(newDatasetChoice);
  const [dropActive, setDropActive] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  /**
   * `dragenter`/`dragleave` fire for every child element the pointer crosses,
   * so a boolean set by `dragleave` flickers off as soon as the pointer moves
   * from the zone onto the text inside it. Counting entries and leaves is the
   * standard fix and the only one that does not need layout maths.
   */
  const dragDepthRef = useRef(0);
  /** Monotonic, so two identically named files from different folders differ. */
  const fileKeyCounterRef = useRef(0);
  /**
   * Whether the next picker result replaces the queue or adds to it.
   *
   * "Choose a different file" means *different*; "Add more files" means *more*.
   * The input cannot tell the two apart, so the caller says which it meant.
   */
  const pickerReplacesRef = useRef(false);
  /**
   * The queue as it stands, readable from callbacks that must not be
   * re-created every time it changes -- the imperative handle and the drop
   * handlers hold on to `acceptFiles`, and it needs the current list to spot a
   * file that is already queued and to append rather than replace.
   */
  const filesRef = useRef<ChosenFile[]>([]);

  const { data: systemStatus } = useApiQuery(() => getSystemStatus(), []);
  // The experiments this library already has, so the picker can offer them.
  // An empty list is the ordinary starting state and renders as "No
  // experiment" plus "New experiment…", which is all a first import needs.
  const { experiments, reload: reloadExperiments } = useExperiments();

  const acceptedExtensions = useMemo(() => {
    const fromServer = (systemStatus?.supported_upload_formats ?? [])
      .map(normaliseExtension)
      .filter(Boolean);
    return fromServer.length > 0 ? fromServer : FALLBACK_UPLOAD_FORMATS;
  }, [systemStatus]);

  /**
   * The largest upload this server accepts, or null while it has not said.
   *
   * Until `/api/system/status/` carried this, a file over the limit was refused
   * by waitress from the request headers and the socket was closed while the
   * browser was still streaming -- which the browser reports as a plain network
   * error, after however long the doomed upload took. Knowing the number is
   * what turns that into an instant, local, named refusal.
   */
  const maxUploadBytes = readMaxUploadBytes(systemStatus);

  /**
   * Read each newly chosen file's own pixel size.
   *
   * Header only -- a few hundred bytes through `Blob.slice` -- so this costs
   * nothing on a 40k x 40k image, and forty of them cost forty times nothing.
   * The result is applied by key, so a file removed while its probe was in
   * flight simply drops the answer, and a second file with the same name gets
   * its own.
   */
  const probeFile = useCallback((entry: ChosenFile) => {
    void probeFileDeclaredPixelSize(entry.file).then((scale) => {
      if (!mountedRef.current) return;
      setFiles((current) =>
        current.map((candidate) =>
          candidate.key === entry.key ? { ...candidate, scale } : candidate
        )
      );
    });
  }, [mountedRef]);

  /**
   * The one place files become *the* files, whether they arrived by picker or
   * by drop.
   *
   * Every path gets the same three checks -- `accept` is a hint the picker may
   * honour and a drop ignores entirely, the size limit is the server's and is
   * unknowable to the file dialog, and a file already in the queue must not be
   * imported twice. Anything refused is named, with its reason, and the rest
   * are still added: refusing forty files because one of them is a `.txt` is
   * not validation, it is an obstacle.
   */
  const acceptFiles = useCallback(
    (incoming: FileList | File[], { replace }: { replace: boolean }) => {
      const list = Array.from(incoming);
      if (list.length === 0) return;

      const reasons: string[] = [];
      const accepted: ChosenFile[] = [];
      const existing = replace ? [] : filesRef.current;

      for (const candidate of list) {
        const lower = candidate.name.toLowerCase();
        if (!acceptedExtensions.some((extension) => lower.endsWith(extension))) {
          // Inline, not `alert()`: a modal dialog that steals focus is not a
          // validation message, and it cannot be read by anything but a human.
          reasons.push(
            `${candidate.name} is not a supported format. This build accepts ${formatExtensionList(
              acceptedExtensions
            )}.`
          );
          continue;
        }
        if (exceedsUploadLimit(candidate.size, maxUploadBytes)) {
          // Refused here, in the picker, naming both numbers -- rather than
          // after a multi-minute upload that the server was always going to
          // drop on the floor.
          reasons.push(
            `${candidate.name} is ${formatBytes(candidate.size)}. This build imports files up to ${formatBytes(
              maxUploadBytes
            )}, so it was not added.`
          );
          continue;
        }
        const alreadyHere =
          existing.some((chosen) => isSameFile(chosen.file, candidate)) ||
          accepted.some((chosen) => isSameFile(chosen.file, candidate));
        if (alreadyHere) {
          reasons.push(`${candidate.name} is already in this list.`);
          continue;
        }
        fileKeyCounterRef.current += 1;
        accepted.push({
          key: `${candidate.name}:${candidate.size}:${candidate.lastModified}:${fileKeyCounterRef.current}`,
          file: candidate,
          scale: null,
        });
      }

      setRejections(reasons);
      if (accepted.length === 0 && !replace) return;

      setBatchSummary(null);
      setFiles(replace ? accepted : [...existing, ...accepted]);
      if (replace) {
        setImports({});
        setDisplayName("");
      }
      // One file: the display name is derived from the filename and editable.
      // Many: each keeps its own, and there is no single name to show.
      const nextCount = (replace ? 0 : existing.length) + accepted.length;
      if (nextCount === 1 && accepted.length === 1) {
        setDisplayName(
          stripKnownExtension(accepted[0].file.name, acceptedExtensions)
        );
      } else if (nextCount > 1) {
        setDisplayName("");
      }
      for (const entry of accepted) probeFile(entry);
    },
    // `setBatchSummary` and `setImports` come from `useImportRun` rather than
    // from a local `useState`, so they have to be named; both are `useState`
    // setters and never change identity, so this is the same callback it was.
    [acceptedExtensions, maxUploadBytes, probeFile, setBatchSummary, setImports]
  );

  useEffect(() => {
    filesRef.current = files;
  }, [files]);

  const acceptDroppedFiles = useCallback(
    (dropped: FileList | File[]) => {
      if (importing) return;
      acceptFiles(dropped, { replace: false });
    },
    [acceptFiles, importing]
  );

  const openFilePicker = useCallback((replace = false) => {
    pickerReplacesRef.current = replace;
    fileInputRef.current?.click();
  }, []);

  useImperativeHandle(
    ref,
    () => ({ openFilePicker: () => openFilePicker(false), acceptDroppedFiles }),
    [acceptDroppedFiles, openFilePicker]
  );

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const chosen = event.target.files;
    if (chosen && chosen.length > 0) {
      acceptFiles(chosen, { replace: pickerReplacesRef.current });
    }
    pickerReplacesRef.current = false;
    // So picking the same file again after removing it still fires `change`.
    event.target.value = "";
  };

  const removeFile = (key: string) => {
    setFiles((current) => current.filter((entry) => entry.key !== key));
    setImports((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
    setRejections([]);
  };

  const clearChosenFiles = () => {
    setFiles([]);
    setImports({});
    setBatchSummary(null);
    setDisplayName("");
    setPixelSizeText("");
    setNotesText("");
    setReplaceDeclaredPixelSizes(false);
    setPixelSizeError(null);
    setRejections([]);
    setExperimentChoice(newExperimentChoice());
    setDatasetChoice(newDatasetChoice());
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const scale = useImportScale({
    files,
    pixelSizeText,
    replaceDeclaredPixelSizes,
  });
  const { appliedPixelSize } = scale;

  const handleDragEnter = (event: React.DragEvent) => {
    if (!Array.from(event.dataTransfer?.types ?? []).includes("Files")) return;
    event.preventDefault();
    dragDepthRef.current += 1;
    setDropActive(true);
  };

  const handleDragOver = (event: React.DragEvent) => {
    if (!Array.from(event.dataTransfer?.types ?? []).includes("Files")) return;
    // Without this the browser navigates to the dropped file and the whole app
    // is replaced by the raw image.
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    setDropActive(true);
  };

  const handleDragLeave = () => {
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) setDropActive(false);
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    dragDepthRef.current = 0;
    setDropActive(false);
    const dropped = event.dataTransfer?.files;
    if (!dropped || dropped.length === 0) return;
    acceptDroppedFiles(dropped);
  };

  /**
   * Start the queue: validate, then hand it to `useImportRun`.
   *
   * The loop itself is in that hook; what stays here is the two answers only
   * this form has — the options each file is posted with, and what to do with
   * the queue when the batch settles.
   */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (files.length === 0) {
      setRejections(["Choose a file to import."]);
      return;
    }
    const pixelSize = parsePixelSizeInput(pixelSizeText);
    if (pixelSize.error) {
      setPixelSizeError(pixelSize.error);
      return;
    }
    setPixelSizeError(null);
    setRejections([]);
    setBatchSummary(null);

    const queue = files.filter(
      (entry) => imports[entry.key]?.kind !== "imported"
    );
    if (queue.length === 0) return;

    const notes = notesText.trim();

    await runImport({
      queue,
      buildOptions: (entry, queueLength): UploadImageOptions => {
        const applied = appliedPixelSize(entry);
        return {
          // Derived from the filename, exactly as the single-file form does it,
          // so a batch does not land forty cards reading "Grid2_Cell04.tif".
          displayName:
            (queueLength === 1
              ? displayName.trim()
              : stripKnownExtension(entry.file.name, acceptedExtensions)) ||
            undefined,
          // Only send a number the user typed *for this file*. Sending the value
          // a file already declares would restamp a file-declared calibration as
          // user-entered, which is a provenance lie the rest of the app then
          // repeats forever.
          pixelSizeNm: applied.source === "typed" ? applied.valueNm : null,
          notes: notes.length > 0 ? notes : undefined,
          // Whole-batch, like the pixel size and the notes. A typed name is
          // sent on every file in the queue and the server resolves it to the
          // same row each time, so forty images land in one experiment rather
          // than in forty identically named ones.
          experimentId: chosenId(experimentChoice) || undefined,
          experimentName: chosenName(experimentChoice) || undefined,
          datasetId: chosenId(datasetChoice) || undefined,
          datasetName: chosenName(datasetChoice) || undefined,
        };
      },
      onSettled: (failedKeys) => {
        // The imports that worked leave the queue; the ones that did not stay
        // put, with their reason, so the button can retry exactly those. A
        // completely clean batch also clears the optional fields, which is what
        // makes the panel ready for the next plate -- but it does not
        // *collapse*. The highlighted cards the Library draws from the assets
        // handed up above provide the durable status. A cleared form that looks exactly like a reset is
        // what the owner read as a failure.
        setFiles((current) =>
          current.filter((entry) => failedKeys.has(entry.key))
        );
        // A batch that named a new experiment or dataset has just created it.
        // Re-reading the catalogue is what lets the next import pick it from
        // the list instead of typing the same name again and relying on the
        // server to match it.
        if (isChosen(experimentChoice) || isChosen(datasetChoice)) {
          void reloadExperiments();
        }
        if (failedKeys.size === 0) {
          setImports({});
          setDisplayName("");
          setPixelSizeText("");
          setNotesText("");
          setReplaceDeclaredPixelSizes(false);
          setExperimentChoice(newExperimentChoice());
          setDatasetChoice(newDatasetChoice());
          if (fileInputRef.current) fileInputRef.current.value = "";
        }
      },
    });
  };

  const highlightDropZone = dropActive || pageDragActive;
  const acceptAttribute = acceptedExtensions.join(",");
  const totalBytes = files.reduce((sum, entry) => sum + entry.file.size, 0);
  /** Files this batch has finished with, either way. */
  const settledCount = files.filter((entry) => {
    const kind = imports[entry.key]?.kind;
    return kind === "imported" || kind === "failed";
  }).length;
  /** Every remaining row failed last time, so the button is a retry. */
  const retrying =
    batchSummary !== null &&
    files.length > 0 &&
    files.every((entry) => imports[entry.key]?.kind === "failed");

  /** The caption on the one button that starts the import queue. */
  const submitLabel = (() => {
    if (importing) {
      if (batchTotal <= 1) return "Uploading...";
      return `Importing ${Math.min(settledCount + 1, batchTotal)} of ${batchTotal}…`;
    }
    if (retrying) {
      return files.length === 1 ? "Try it again" : `Try these ${files.length} again`;
    }
    if (files.length === 1) return "Import image";
    return `Import ${files.length} images`;
  })();

  return (
    <Panel
      className={cx("p-4", compact && files.length === 0 && "py-3")}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      data-testid="import-panel"
    >
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
        Import an image
      </h2>

      {/* The input is the mechanism for both paths and is never hidden from
          assistive technology: `sr-only` keeps it focusable and labelled, and
          the visible drop zone is its `<label>`, so one click on the zone opens
          the OS dialog through the browser's own native handling. `multiple`
          because the zone says "Choose files…" and because a plate is the
          normal case. */}
      <input
        ref={fileInputRef}
        id="file-input"
        className="peer sr-only"
        type="file"
        multiple
        // A stable accessible name in both states. The drop zone is the input's
        // `<label>` only while no file is chosen; without this the control
        // would lose its name the moment one is.
        aria-label="Image file"
        accept={acceptAttribute}
        onChange={handleFileChange}
        aria-describedby={files.length === 0 ? "file-input-help" : undefined}
      />

      {/* Panel level, not inside the form: the commonest way to see these is to
          pick unsupported files when nothing is chosen yet, and the form does
          not exist in that state. */}
      {rejections.map((reason) => (
        <p className="mt-2 text-sm text-red-700" role="alert" key={reason}>
          {reason}
        </p>
      ))}

      {files.length === 0 ? (
        <ImportDropZone
          acceptedExtensions={acceptedExtensions}
          highlightDropZone={highlightDropZone}
          batchSummary={batchSummary}
          compact={compact}
        />
      ) : (
        <form
          className="mt-3 flex flex-col gap-4"
          onSubmit={handleSubmit}
          data-testid="import-form"
        >
          <ImportQueue
            files={files}
            imports={imports}
            importing={importing}
            highlightDropZone={highlightDropZone}
            batchSummary={batchSummary}
            totalBytes={totalBytes}
            appliedPixelSize={appliedPixelSize}
            onAddMoreFiles={() => openFilePicker(false)}
            onChooseDifferentFile={() => openFilePicker(true)}
            onClearChosenFiles={clearChosenFiles}
            onRemoveFile={removeFile}
            onRetryProcessing={() => void retryDeferredProcessing()}
          />

          {/* The current files still own the upload connection at this point.
              A second drop cannot be accepted until every request has either
              landed or failed, so do not leave a control on screen that looks
              available but silently ignores the files. Once the server has
              accepted the batch and its encoding jobs have been started,
              `importing` becomes false and the fresh drop zone returns. */}
          {!importing ? (
            <ImportDropZone
              acceptedExtensions={acceptedExtensions}
              highlightDropZone={highlightDropZone}
              batchSummary={null}
              variant="additional"
            />
          ) : null}

          <ImportDetailsFields
            files={files}
            scale={scale}
            displayName={displayName}
            onDisplayNameChange={setDisplayName}
            pixelSizeText={pixelSizeText}
            onPixelSizeTextChange={setPixelSizeText}
            pixelSizeError={pixelSizeError}
            notesText={notesText}
            onNotesTextChange={setNotesText}
            replaceDeclaredPixelSizes={replaceDeclaredPixelSizes}
            onReplaceDeclaredPixelSizesChange={setReplaceDeclaredPixelSizes}
            experiments={experiments}
            experimentChoice={experimentChoice}
            datasetChoice={datasetChoice}
            onExperimentChoiceChange={setExperimentChoice}
            onDatasetChoiceChange={setDatasetChoice}
          />

          <div className="flex items-center gap-2">
            {/* What pressing this actually does, counted. "Uncalibrated" is
                only asserted when it is a fact: a button that says it over a
                file that turns out to declare 5 nm/px is how a warning stops
                being believed. */}
            <Button variant="primary" type="submit" disabled={importing}>
              {submitLabel}
            </Button>
            <Button onClick={clearChosenFiles} disabled={importing}>
              Cancel
            </Button>
          </div>
        </form>
      )}
    </Panel>
  );
});
