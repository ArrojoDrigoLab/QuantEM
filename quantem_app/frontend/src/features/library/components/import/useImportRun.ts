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
import { uploadAsset } from "@/shared/api/assets";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type {
  BatchSummary,
  ChosenFile,
  FileImportState,
} from "@/features/library/components/import/importValidation";
import type { AssetDetail, UploadImageOptions } from "@/shared/types/images";

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

      for (const entry of queue) {
        position += 1;
        if (!mountedRef.current) return;
        setImports((current) => ({
          ...current,
          [entry.key]: { kind: "uploading" },
        }));
        const options = buildOptions(entry, queue.length);
        try {
          const asset = await uploadAsset(entry.file, options);
          if (!mountedRef.current) return;
          imported += 1;
          setImports((current) => ({
            ...current,
            [entry.key]: { kind: "imported" },
          }));
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

  return {
    imports,
    setImports,
    batchSummary,
    setBatchSummary,
    batchTotal,
    importing,
    runImport,
    /**
     * Whether the panel is still mounted. Shared with the header-probe
     * callback, which must not `setState` after an unmount either.
     */
    mountedRef,
  };
}
