/**
 * Running the import queue, one POST at a time, and handing each image up as
 * it lands.
 *
 * Split out of `ImageUploadPanel.tsx` unchanged. Each file is handed up the
 * moment it lands, not at the end: the Library pins imported cards as they
 * arrive, so a plate of forty appears one row at a time instead of all at once
 * after ten minutes. A failure is recorded against its own row and the loop
 * continues -- the whole point of a queue is that image 12 being corrupt does
 * not cost the user images 13 to 40.
 *
 * Sequential, not parallel: a plate of EM mosaics is 250 MB to 2 GB each, and
 * forty simultaneous multipart POSTs would compete for the same disk while
 * making the first image land last. The queue is also what makes per-file
 * progress mean anything.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  startUploadedAssetPipelines,
  uploadAsset,
} from "@/shared/api/assets";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type {
  BatchSummary,
  ChosenFile,
  FileImportState,
} from "@/features/library/components/import/importValidation";
import type {
  AssetDetail,
  UploadImageOptions,
  UploadPipelineStart,
} from "@/shared/types/images";

/** Where one finished import sat in the batch that produced it. */
export interface ImportBatchPosition {
  /** 1-based, in queue order. */
  index: number;
  /** How many files this batch is importing in total. */
  total: number;
}

export function useImportRun({
  onUploaded,
}: {
  onUploaded?: (asset: AssetDetail, batch: ImportBatchPosition) => void;
}) {
  const [imports, setImports] = useState<Record<string, FileImportState>>({});
  const [batchSummary, setBatchSummary] = useState<BatchSummary | null>(null);
  /** How many files the running batch is working through, for "3 of 40". */
  const [batchTotal, setBatchTotal] = useState(0);
  const [importing, setImporting] = useState(false);
  const mountedRef = useRef(true);
  const deferredRetryRef = useRef<{
    pipelines: UploadPipelineStart[];
    attempted: number;
    imported: number;
    failedKeys: Set<string>;
    onSettled: (failedKeys: Set<string>) => void;
  } | null>(null);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const runImport = useCallback(
    async ({
      queue,
      buildOptions,
      onSettled,
    }: {
      queue: ChosenFile[];
      /** The upload options for one file, given the size of the batch. */
      buildOptions: (entry: ChosenFile, queueLength: number) => UploadImageOptions;
      /** What the panel does with the outcome: keep failures, clear the rest. */
      onSettled: (failedKeys: Set<string>) => void;
    }) => {
      setImporting(true);
      setBatchTotal(queue.length);
      setImports(() =>
        Object.fromEntries(
          queue.map((entry) => [entry.key, { kind: "waiting" } as FileImportState])
        )
      );

      const failedKeys = new Set<string>();
      let imported = 0;
      let position = 0;
      const deferredPipelines: UploadPipelineStart[] = [];

      for (const entry of queue) {
        position += 1;
        if (!mountedRef.current) return;
        setImports((current) => ({
          ...current,
          [entry.key]: { kind: "uploading" },
        }));
        const options = {
          ...buildOptions(entry, queue.length),
          deferProcessing: queue.length > 1,
        };
        try {
          const asset = await uploadAsset(entry.file, options);
          if (!mountedRef.current) return;
          imported += 1;
          setImports((current) => ({
            ...current,
            [entry.key]: { kind: "imported" },
          }));
          if (options.deferProcessing) {
            deferredPipelines.push({
              assetId: asset.id,
              segmentMito: options.segmentMito,
              segmentEr: options.segmentEr,
              segmentNucleus: options.segmentNucleus,
              segmentLd: options.segmentLd,
            });
          }
          onUploaded?.(asset, { index: position, total: queue.length });
        } catch (err) {
          if (!mountedRef.current) return;
          failedKeys.add(entry.key);
          setImports((current) => ({
            ...current,
            [entry.key]: {
              kind: "failed",
              message: extractApiErrorMessage(
                err,
                `${entry.file.name} could not be imported.`
              ),
            },
          }));
        }
      }

      if (!mountedRef.current) return;
      // This call happens after the upload loop, including after any failures.
      // The server commits every job row together, so no encoder can claim an
      // early image while a later file is still importing or being queued.
      if (deferredPipelines.length > 0) {
        try {
          await startUploadedAssetPipelines(deferredPipelines);
        } catch (err) {
          // The imports themselves succeeded and must not be offered as upload
          // retries (that would create duplicates). Keep their cards in the
          // queued state and put an actionable batch-level failure in the
          // summary surface. The endpoint is idempotent, so the recovery
          // button can safely resubmit after a lost response.
          if (!mountedRef.current) return;
          setImporting(false);
          setBatchSummary({
            attempted: queue.length,
            imported,
            failed: failedKeys.size,
            processingError: extractApiErrorMessage(
              err,
              "The images were imported, but processing could not be started."
            ),
          });
          deferredRetryRef.current = {
            pipelines: deferredPipelines,
            attempted: queue.length,
            imported,
            failedKeys,
            onSettled,
          };
          return;
        }
      }
      deferredRetryRef.current = null;
      setImporting(false);
      setBatchSummary({
        attempted: queue.length,
        imported,
        failed: failedKeys.size,
      });
      onSettled(failedKeys);
    },
    [onUploaded]
  );

  const retryDeferredProcessing = useCallback(async () => {
    const pending = deferredRetryRef.current;
    if (!pending) return;
    setImporting(true);
    try {
      await startUploadedAssetPipelines(pending.pipelines);
      if (!mountedRef.current) return;
      deferredRetryRef.current = null;
      setBatchSummary({
        attempted: pending.attempted,
        imported: pending.imported,
        failed: pending.failedKeys.size,
      });
      pending.onSettled(pending.failedKeys);
    } catch (err) {
      if (!mountedRef.current) return;
      setBatchSummary({
        attempted: pending.attempted,
        imported: pending.imported,
        failed: pending.failedKeys.size,
        processingError: extractApiErrorMessage(
          err,
          "The images were imported, but processing still could not be started."
        ),
      });
    } finally {
      if (mountedRef.current) setImporting(false);
    }
  }, []);

  return {
    imports,
    setImports,
    batchSummary,
    setBatchSummary,
    batchTotal,
    importing,
    runImport,
    retryDeferredProcessing,
    /**
     * Whether the panel is still mounted. Shared with the header-probe
     * callback, which must not `setState` after an unmount either.
     */
    mountedRef,
  };
}
