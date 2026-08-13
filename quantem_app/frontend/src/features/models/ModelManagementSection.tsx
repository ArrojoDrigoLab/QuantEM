import { useCallback, useEffect, useMemo, useState } from "react";

import {
  getModelCatalogue,
  installModelPack,
  removeModelPack,
} from "@/shared/api/finetune";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import type {
  ModelCatalogue,
  ModelFamily,
  ModelPack,
  OrganelleKey,
} from "@/shared/types/finetune";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { Badge, Button, PageState, Panel } from "@/shared/ui/design";
import { extractApiErrorMessage } from "@/utils/apiErrors";

const FAMILIES: Array<{ id: ModelFamily; title: string }> = [
  { id: "omniem", title: "OmniEM (Large model)" },
  { id: "quantem", title: "QuantEM (Basic model)" },
];

const ORGANELLES: Array<{ id: OrganelleKey; label: string }> = [
  { id: "mito", label: "Mitochondria" },
  { id: "er", label: "Endoplasmic reticulum" },
  { id: "nucleus", label: "Nuclei" },
  { id: "ld", label: "Lipid droplets" },
];

/** The model catalogue and its install/remove actions, shared by both surfaces. */
export function ModelManagementSection({ compact = false }: { compact?: boolean }) {
  const { data: catalogue, error, loading, refetch } = useApiQuery<ModelCatalogue>(
    () => getModelCatalogue(),
    []
  );
  const [busyPackId, setBusyPackId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<{
    packId: string;
    message: string;
  } | null>(null);
  const [removeTarget, setRemoveTarget] = useState<ModelPack | null>(null);

  const packsByKey = useMemo(
    () =>
      new Map(
        (catalogue?.packs ?? []).map((pack) => [
          `${pack.family}:${pack.organelle}`,
          pack,
        ])
      ),
    [catalogue]
  );
  const hasActiveDownload = Boolean(
    catalogue?.packs.some((pack) => pack.active_install)
  );

  useEffect(() => {
    if (!hasActiveDownload) return undefined;
    const intervalId = window.setInterval(() => void refetch(), 1500);
    return () => clearInterval(intervalId);
  }, [hasActiveDownload, refetch]);

  const download = useCallback(
    async (pack: ModelPack) => {
      setBusyPackId(pack.id);
      setActionError(null);
      try {
        await installModelPack(pack.id);
        await refetch();
      } catch (requestError) {
        setActionError({
          packId: pack.id,
          message: extractApiErrorMessage(
            requestError,
            "The model could not be downloaded."
          ),
        });
      } finally {
        setBusyPackId(null);
      }
    },
    [refetch]
  );

  const remove = useCallback(async () => {
    const pack = removeTarget;
    if (!pack) return;
    setRemoveTarget(null);
    setBusyPackId(pack.id);
    setActionError(null);
    try {
      await removeModelPack(pack.id);
      await refetch();
    } catch (requestError) {
      setActionError({
        packId: pack.id,
        message: extractApiErrorMessage(
          requestError,
          "The model could not be removed."
        ),
      });
    } finally {
      setBusyPackId(null);
    }
  }, [refetch, removeTarget]);

  if (loading && !catalogue) return <PageState title="Loading models…" />;
  if (error && !catalogue) {
    return (
      <PageState
        title="Models could not be loaded"
        detail={extractApiErrorMessage(error, "Try again.")}
        tone="error"
      />
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {FAMILIES.map((family) => (
        <Panel key={family.id} className="overflow-hidden p-0">
          <h3
            className={`border-b border-slate-200 font-semibold text-slate-950 ${
              compact ? "px-4 py-3 text-sm" : "px-5 py-4 text-lg"
            }`}
          >
            {family.title}
          </h3>
          <div className="divide-y divide-slate-200">
            {ORGANELLES.map((organelle) => {
              const pack = packsByKey.get(`${family.id}:${organelle.id}`);
              if (!pack) return null;
              const downloading = Boolean(pack.active_install);
              const thisActionBusy = busyPackId === pack.id;
              const mutationsDisabled = busyPackId !== null || hasActiveDownload;
              return (
                <div
                  key={pack.id}
                  className={`flex flex-wrap items-center justify-between gap-3 ${
                    compact ? "px-4 py-3" : "px-5 py-4"
                  }`}
                >
                  <div>
                    <p className="m-0 text-sm font-semibold text-slate-950">
                      {organelle.label}
                    </p>
                    <div className="mt-2 flex items-center gap-2">
                      <Badge tone={pack.installed ? "good" : "default"}>
                        {pack.installed ? "Downloaded" : "Not downloaded"}
                      </Badge>
                      {downloading ? (
                        <span className="text-xs text-slate-600">Downloading…</span>
                      ) : null}
                    </div>
                    {actionError?.packId === pack.id ? (
                      <p className="mt-2 max-w-xl text-sm text-red-700" role="alert">
                        {actionError.message}
                      </p>
                    ) : null}
                  </div>
                  {pack.installed ? (
                    <Button
                      size={compact ? "sm" : undefined}
                      disabled={mutationsDisabled}
                      onClick={() => setRemoveTarget(pack)}
                    >
                      {thisActionBusy ? "Removing…" : "Remove"}
                    </Button>
                  ) : (
                    <Button
                      size={compact ? "sm" : undefined}
                      variant="primary"
                      disabled={mutationsDisabled}
                      onClick={() => void download(pack)}
                    >
                      {thisActionBusy || downloading ? "Downloading…" : "Download"}
                    </Button>
                  )}
                </div>
              );
            })}
          </div>
        </Panel>
      ))}

      <ConfirmDialog
        isOpen={removeTarget !== null}
        title="Remove downloaded model?"
        message={
          removeTarget
            ? `Remove ${removeTarget.title} from this machine? You can download it again later.`
            : undefined
        }
        confirmText="Remove"
        cancelText="Cancel"
        onConfirm={() => void remove()}
        onCancel={() => setRemoveTarget(null)}
      />
    </div>
  );
}
