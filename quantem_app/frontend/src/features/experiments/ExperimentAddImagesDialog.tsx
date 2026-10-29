import { useCallback, useEffect, useRef, useState } from "react";

import { ImageUploadPanel } from "@/features/library/components/ImageUploadPanel";
import { useModalLifecycle } from "@/shared/hooks/useModalLifecycle";
import type { Experiment } from "@/shared/types/common";
import { Button } from "@/shared/ui/design";

export function ExperimentAddImagesDialog({
  open,
  experiment,
  onClose,
  onImagesAdded,
}: {
  open: boolean;
  experiment: Experiment;
  onClose: () => void;
  onImagesAdded: () => void;
}) {
  const panelRef = useRef<HTMLElement | null>(null);
  const [addedCount, setAddedCount] = useState(0);
  const [importing, setImporting] = useState(false);

  useEffect(() => {
    if (open) setAddedCount(0);
  }, [open]);

  const close = useCallback(() => {
    // Unmounting the importer in the middle of a multi-file batch stops the
    // remaining uploads and, more importantly, can skip the single deferred
    // pipeline-start call for files already accepted by the server.
    if (importing) return;
    if (addedCount > 0) onImagesAdded();
    onClose();
  }, [addedCount, importing, onClose, onImagesAdded]);

  useModalLifecycle(panelRef, close, { open, closeOnEscape: !importing });

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4"
      data-testid="add-images-overlay"
    >
      <section
        ref={panelRef}
        className="max-h-[92vh] w-full max-w-3xl overflow-y-auto rounded-2xl bg-white shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby="add-images-title"
      >
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white px-5 py-4">
          <div>
            <h2 id="add-images-title" className="m-0 text-xl font-semibold text-slate-950">
              Add images
            </h2>
            <p className="m-0 mt-1 text-sm text-slate-600">
              Add images to {experiment.name}, then choose one new dataset or one or more existing datasets.
            </p>
          </div>
          <Button
            size="icon"
            variant="ghost"
            aria-label="Close add images"
            disabled={importing}
            onClick={close}
          >
            <svg
              width="19"
              height="19"
              viewBox="0 0 20 20"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.25"
              strokeLinecap="round"
              aria-hidden="true"
            >
              <path d="M4 4l12 12M16 4L4 16" />
            </svg>
          </Button>
        </div>

        <div className="p-5">
          <ImageUploadPanel
            fixedExperiment={experiment}
            onUploaded={() => setAddedCount((current) => current + 1)}
            onImportingChange={setImporting}
          />
          {addedCount > 0 ? (
            <p className="mb-0 mt-3 text-sm font-medium text-emerald-700" role="status">
              Added {addedCount} {addedCount === 1 ? "image" : "images"}.
            </p>
          ) : null}
          <div className="mt-4 flex justify-end">
            <Button variant="primary" disabled={importing} onClick={close}>
              {importing ? "Adding images…" : "Done"}
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}
