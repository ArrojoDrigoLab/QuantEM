import { useCallback, useEffect, useId, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

import {
  deleteAsset,
  getExperiment,
  getHomeEntryPage,
  updateAssetLibraryDetails,
  updateDataset,
  updateExperiment,
} from "@/shared/api/assets";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { parsePixelSizeInput } from "@/shared/pixelSize";
import type { Dataset, Experiment } from "@/shared/types/common";
import { UNASSIGNED_FILTER, type HomeEntry } from "@/shared/types/images";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { Button, PageState, Panel } from "@/shared/ui/design";
import { ExportDialog } from "@/features/export/ExportDialog";
import { ImageCard } from "@/features/library/components/ImageCard";
import { extractApiErrorMessage } from "@/utils/apiErrors";

const PREVIEW_LIMIT = 5;
const PAGE_LIMIT = 200;

async function getGroupEntries(
  experimentId: string,
  datasetId: string,
  expanded: boolean
): Promise<{ entries: HomeEntry[]; total: number }> {
  const first = await getHomeEntryPage({
    experiment: experimentId,
    dataset: datasetId,
    ordering: "created_at",
    limit: expanded ? PAGE_LIMIT : PREVIEW_LIMIT,
    offset: 0,
  });
  if (!expanded || !first.has_more) {
    return { entries: first.results, total: first.total };
  }
  const entries = [...first.results];
  let offset = entries.length;
  let hasMore: boolean = first.has_more;
  while (hasMore) {
    const page = await getHomeEntryPage({
      experiment: experimentId,
      dataset: datasetId,
      ordering: "created_at",
      limit: PAGE_LIMIT,
      offset,
    });
    entries.push(...page.results);
    offset += page.results.length;
    hasMore = page.has_more && page.results.length > 0;
  }
  return { entries, total: first.total };
}

function PencilIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
      <path
        d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  );
}

function GroupEditor({
  name,
  notes,
  level,
  onSave,
}: {
  name: string;
  notes: string;
  level: "experiment" | "dataset";
  onSave: (updates: { name: string; notes: string }) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [draftName, setDraftName] = useState(name);
  const [draftNotes, setDraftNotes] = useState(notes);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (editing) return;
    setDraftName(name);
    setDraftNotes(notes);
  }, [editing, name, notes]);

  if (editing) {
    return (
      <div className="flex max-w-3xl flex-1 flex-col gap-2">
        <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          {level === "experiment" ? "Experiment name" : "Dataset name"}
          <input
            className="mt-1 h-10 w-full rounded-md border border-slate-300 px-3 text-sm text-slate-950 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            value={draftName}
            onChange={(event) => setDraftName(event.target.value)}
            autoFocus
          />
        </label>
        <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Notes
          <textarea
            className="mt-1 min-h-20 w-full rounded-md border border-slate-300 px-3 py-2 text-sm normal-case text-slate-950 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            value={draftNotes}
            onChange={(event) => setDraftNotes(event.target.value)}
            placeholder="Add notes"
          />
        </label>
        {error ? <p className="m-0 text-sm text-red-700">{error}</p> : null}
        <div className="flex gap-2">
          <Button
            variant="primary"
            disabled={saving || !draftName.trim()}
            onClick={() => {
              setSaving(true);
              setError(null);
              void onSave({ name: draftName.trim(), notes: draftNotes })
                .then(() => setEditing(false))
                .catch((saveError) =>
                  setError(extractApiErrorMessage(saveError, "That could not be saved."))
                )
                .finally(() => setSaving(false));
            }}
          >
            Confirm
          </Button>
          <Button disabled={saving} onClick={() => setEditing(false)}>
            Cancel
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-w-0 flex-1">
      <div className="flex items-center gap-2">
        {level === "experiment" ? (
          <h1 className="m-0 truncate text-2xl font-semibold text-slate-950">{name}</h1>
        ) : (
          <h2 className="m-0 truncate text-lg font-semibold text-slate-950">{name}</h2>
        )}
        <Button
          size="icon"
          variant="secondary"
          aria-label={`Edit ${level}`}
          title={`Edit ${level}`}
          onClick={() => setEditing(true)}
        >
          <PencilIcon />
        </Button>
      </div>
      {notes.trim() ? (
        <p className="mt-2 whitespace-pre-wrap text-sm text-slate-600">{notes}</p>
      ) : (
        <p className="mt-2 text-sm italic text-slate-500">No notes</p>
      )}
    </div>
  );
}

function DatasetSection({
  experimentId,
  dataset,
  refreshKey,
  onEditDataset,
  onEditImage,
  onExportImage,
  onDeleteImage,
}: {
  experimentId: string;
  dataset: Dataset | null;
  refreshKey: number;
  onEditDataset: (dataset: Dataset, updates: { name: string; notes: string }) => Promise<void>;
  onEditImage: (image: HomeEntry) => void;
  onExportImage: (image: HomeEntry) => void;
  onDeleteImage: (image: HomeEntry) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const datasetId = dataset?.id ?? UNASSIGNED_FILTER;
  const { data, loading, error } = useApiQuery(
    () => getGroupEntries(experimentId, datasetId, expanded),
    [datasetId, expanded, experimentId, refreshKey]
  );
  if (!loading && !error && data?.total === 0) return null;

  return (
    <Panel className="p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        {dataset ? (
          <GroupEditor
            name={dataset.name}
            notes={dataset.notes}
            level="dataset"
            onSave={(updates) => onEditDataset(dataset, updates)}
          />
        ) : (
          <div>
            <h2 className="m-0 text-lg font-semibold text-slate-950">
              Not in a dataset
            </h2>
            <p className="mt-2 text-sm text-slate-600">
              Images assigned to this experiment but not to one of its datasets.
            </p>
          </div>
        )}
        {data ? (
          <span className="text-sm text-slate-600">
            {data.total} {data.total === 1 ? "image" : "images"}
          </span>
        ) : null}
      </div>

      {loading ? <p className="mt-4 text-sm text-slate-600">Loading images…</p> : null}
      {error ? (
        <p className="mt-4 text-sm text-red-700">
          {extractApiErrorMessage(error, "These images could not be loaded.")}
        </p>
      ) : null}
      {data?.entries.length ? (
        <div className="mt-4 flex flex-wrap gap-4">
          {data.entries.map((image) => (
            <div key={image.id} className="h-[320px] w-[230px]">
              <ImageCard
                image={image}
                onDelete={onDeleteImage}
                onEdit={onEditImage}
                onExport={onExportImage}
                useActionMenu
              />
            </div>
          ))}
        </div>
      ) : null}
      {data && data.total > PREVIEW_LIMIT ? (
        <Button className="mt-4" onClick={() => setExpanded((current) => !current)}>
          {expanded ? "Show first 5" : `Show all ${data.total}`}
        </Button>
      ) : null}
    </Panel>
  );
}

function ImageEditDialog({
  image,
  experiment,
  saving,
  error,
  onClose,
  onSave,
}: {
  image: HomeEntry | null;
  experiment: Experiment;
  saving: boolean;
  error: string | null;
  onClose: () => void;
  onSave: (updates: {
    displayName: string;
    pixelSizeNm: number | null;
    datasetId: string;
    notes: string;
  }) => void;
}) {
  const [displayName, setDisplayName] = useState("");
  const [pixelSize, setPixelSize] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [notes, setNotes] = useState("");
  const [pixelError, setPixelError] = useState<string | null>(null);
  const titleId = useId();
  const nameInputRef = useRef<HTMLInputElement | null>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const savingRef = useRef(saving);

  useEffect(() => {
    savingRef.current = saving;
  }, [saving]);

  const requestClose = useCallback(() => {
    if (!savingRef.current) onClose();
  }, [onClose]);

  useEffect(() => {
    if (!image) return;
    setDisplayName(image.display_name);
    setPixelSize(image.pixel_size_nm ? String(image.pixel_size_nm) : "");
    setDatasetId(
      image.dataset_ids?.find((id) => experiment.datasets.some((row) => row.id === id)) ?? ""
    );
    setNotes(image.notes ?? "");
    setPixelError(null);
  }, [experiment.datasets, image]);

  useEffect(() => {
    if (!image) return undefined;
    restoreFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    nameInputRef.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        requestClose();
      }
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("keydown", closeOnEscape);
      restoreFocusRef.current?.focus();
    };
  }, [image, requestClose]);

  if (!image) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4"
      onClick={requestClose}
    >
      <section
        className="w-full max-w-lg rounded-2xl bg-white p-6 shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(event) => event.stopPropagation()}
      >
        <h2 id={titleId} className="m-0 text-xl font-semibold text-slate-950">
          Edit image
        </h2>
        <div className="mt-5 flex flex-col gap-4">
          <label className="text-sm font-medium text-slate-700">
            Display name
            <input
              ref={nameInputRef}
              className="mt-1 h-10 w-full rounded-md border border-slate-300 px-3 text-slate-950 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
            />
          </label>
          <label className="text-sm font-medium text-slate-700">
            Resolution (nm/px)
            <input
              className="mt-1 h-10 w-full rounded-md border border-slate-300 px-3 text-slate-950 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              type="number"
              min="0"
              step="any"
              value={pixelSize}
              placeholder="Unknown"
              onChange={(event) => setPixelSize(event.target.value)}
            />
          </label>
          <label className="text-sm font-medium text-slate-700">
            Dataset
            <select
              className="mt-1 h-10 w-full rounded-md border border-slate-300 bg-white px-3 text-slate-950 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              value={datasetId}
              onChange={(event) => setDatasetId(event.target.value)}
            >
              <option value="">Not in a dataset</option>
              {experiment.datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm font-medium text-slate-700">
            Image notes
            <textarea
              className="mt-1 min-h-28 w-full rounded-md border border-slate-300 px-3 py-2 text-slate-950 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              value={notes}
              onChange={(event) => setNotes(event.target.value)}
              placeholder="Add notes specific to this image"
            />
          </label>
          {pixelError || error ? (
            <p className="m-0 text-sm text-red-700" role="alert">
              {pixelError || error}
            </p>
          ) : null}
        </div>
        <div className="mt-6 flex justify-end gap-2">
          <Button disabled={saving} onClick={requestClose}>Cancel</Button>
          <Button
            variant="primary"
            disabled={saving || !displayName.trim()}
            onClick={() => {
              const parsed = parsePixelSizeInput(pixelSize);
              if (parsed.error) {
                setPixelError(parsed.error);
                return;
              }
              setPixelError(null);
              onSave({
                displayName: displayName.trim(),
                pixelSizeNm: parsed.value,
                datasetId,
                notes,
              });
            }}
          >
            {saving ? "Saving…" : "Save changes"}
          </Button>
        </div>
      </section>
    </div>
  );
}

export function ExperimentPage() {
  const { experimentId = "" } = useParams();
  const { data, loading, error, refetch } = useApiQuery(
    () => getExperiment(experimentId),
    [experimentId]
  );
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [editingImage, setEditingImage] = useState<HomeEntry | null>(null);
  const [exportingImage, setExportingImage] = useState<HomeEntry | null>(null);
  const [deleteImage, setDeleteImage] = useState<HomeEntry | null>(null);
  const [savingImage, setSavingImage] = useState(false);
  const [imageEditError, setImageEditError] = useState<string | null>(null);
  const [pageActionError, setPageActionError] = useState<string | null>(null);

  useEffect(() => {
    if (data) setExperiment(data);
  }, [data]);

  const refreshGroups = useCallback(() => {
    setRefreshKey((current) => current + 1);
    void refetch();
  }, [refetch]);
  const closeImageEditor = useCallback(() => setEditingImage(null), []);

  if (loading && !experiment) return <PageState title="Loading experiment…" />;
  if (error && !experiment) {
    return (
      <PageState
        title="Experiment could not be loaded"
        detail={extractApiErrorMessage(error, "Return home and try again.")}
        tone="error"
      />
    );
  }
  if (!experiment) return null;

  return (
    <main className="min-h-screen px-5 py-5 text-slate-900 lg:px-8">
      <div className="mx-auto flex max-w-[1500px] flex-col gap-5">
        <header className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <Link className="text-sm font-medium text-cyan-800 hover:underline" to="/">
            Back to Home
          </Link>
          <div className="mt-4 flex items-start justify-between gap-4">
            <GroupEditor
              name={experiment.name}
              notes={experiment.notes}
              level="experiment"
              onSave={async (updates) => {
                const updated = await updateExperiment(experiment.id, updates);
                setExperiment(updated);
              }}
            />
            <span className="text-sm text-slate-600">
              {experiment.asset_count} {experiment.asset_count === 1 ? "image" : "images"}
            </span>
          </div>
        </header>

        {pageActionError ? (
          <p
            className="m-0 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800"
            role="alert"
          >
            {pageActionError}
          </p>
        ) : null}

        {experiment.datasets.map((dataset) => (
          <DatasetSection
            key={dataset.id}
            experimentId={experiment.id}
            dataset={dataset}
            refreshKey={refreshKey}
            onEditDataset={async (current, updates) => {
              const updated = await updateDataset(current.id, updates);
              setExperiment((value) =>
                value
                  ? {
                      ...value,
                      datasets: value.datasets.map((row) =>
                        row.id === updated.id ? updated : row
                      ),
                    }
                  : value
              );
            }}
            onEditImage={(image) => {
              setImageEditError(null);
              setEditingImage(image);
            }}
            onExportImage={setExportingImage}
            onDeleteImage={setDeleteImage}
          />
        ))}
        <DatasetSection
          experimentId={experiment.id}
          dataset={null}
          refreshKey={refreshKey}
          onEditDataset={async () => undefined}
          onEditImage={(image) => {
            setImageEditError(null);
            setEditingImage(image);
          }}
          onExportImage={setExportingImage}
          onDeleteImage={setDeleteImage}
        />

        <ImageEditDialog
          image={editingImage}
          experiment={experiment}
          saving={savingImage}
          error={imageEditError}
          onClose={closeImageEditor}
          onSave={(updates) => {
            if (!editingImage) return;
            setSavingImage(true);
            setImageEditError(null);
            void updateAssetLibraryDetails(editingImage.id, {
              display_name: updates.displayName,
              pixel_size_nm: updates.pixelSizeNm,
              notes: updates.notes,
              datasets: updates.datasetId ? [updates.datasetId] : [],
            })
              .then(() => {
                setEditingImage(null);
                refreshGroups();
              })
              .catch((saveError) => {
                setImageEditError(
                  extractApiErrorMessage(
                    saveError,
                    "The image could not be updated."
                  )
                );
                refreshGroups();
              })
              .finally(() => setSavingImage(false));
          }}
        />
        {exportingImage ? (
          <ExportDialog
            asset={{
              id: exportingImage.id,
              displayName: exportingImage.display_name,
            }}
            onClose={() => setExportingImage(null)}
          />
        ) : null}
        <ConfirmDialog
          isOpen={deleteImage !== null}
          title="Delete image?"
          message={
            deleteImage
              ? `Delete “${deleteImage.display_name}” and all of its derived files? This cannot be undone.`
              : undefined
          }
          confirmText="Delete"
          cancelText="Cancel"
          onConfirm={() => {
            const image = deleteImage;
            setDeleteImage(null);
            if (!image) return;
            setPageActionError(null);
            void deleteAsset(image.id)
              .then(refreshGroups)
              .catch((deleteError) =>
                setPageActionError(
                  extractApiErrorMessage(
                    deleteError,
                    "The image could not be deleted."
                  )
                )
              );
          }}
          onCancel={() => setDeleteImage(null)}
        />
      </div>
    </main>
  );
}
