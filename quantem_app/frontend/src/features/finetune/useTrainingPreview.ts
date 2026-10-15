import { useEffect, useState } from "react";
import { fineTuneErrorMessage, previewFineTuneScope } from "@/shared/api/finetune";
import type { FineTunePreviewResponse } from "@/shared/types/finetune";
import { isSelectionEmpty, selectionKey, toSelectionPayload, type ScopeSelection } from "./scopeTree";

/** Training eligibility is derived only from the currently selected annotations. */
export function useTrainingPreview({ typeId, selection, baseModel, parentId, includePrevious, enabled }: {
  typeId: string; selection: ScopeSelection; baseModel: string; parentId: string;
  includePrevious: boolean; enabled: boolean;
}) {
  const [preview, setPreview] = useState<FineTunePreviewResponse | null>(null);
  const [loadedKey, setLoadedKey] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const key = `${selectionKey(typeId, selection)}|${baseModel}|${parentId}|${includePrevious}`;
  const empty = isSelectionEmpty(selection);
  useEffect(() => {
    setPreview(null); setError(null);
    if (!enabled || !typeId || empty) { setPending(false); return; }
    let current = true;
    setPending(true);
    const timer = window.setTimeout(() => {
      void previewFineTuneScope({ ...toSelectionPayload(typeId, selection), base_model: baseModel || undefined,
        parent_adapter_id: parentId || undefined, include_previous_annotations: includePrevious })
        .then((value) => { if (current) { setPreview(value); setLoadedKey(key); } })
        .catch((cause) => { if (current) setError(fineTuneErrorMessage(cause, "Could not check this selection.")); })
        .finally(() => { if (current) setPending(false); });
    }, 250);
    return () => { current = false; clearTimeout(timer); };
  }, [typeId, selection, baseModel, parentId, includePrevious, enabled, empty, key]);
  return { livePreview: loadedKey === key && enabled ? preview : null, previewPending: pending, previewError: error };
}
