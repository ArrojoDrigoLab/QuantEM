import { getJob } from "@/shared/api/jobs";
import {
  getModelCatalogue,
  installModelPack,
} from "@/shared/api/finetune";
import type { Job } from "@/shared/types/jobs";
import type { ModelPack } from "@/shared/types/finetune";

const DEFAULT_POLL_INTERVAL_MS = 1000;

interface EnsureModelInstalledOptions {
  onDownloadQueued?: (jobId: string) => void | Promise<void>;
  onInstalled?: (pack: ModelPack) => void | Promise<void>;
  pollIntervalMs?: number;
}

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function waitForInstallJob(
  jobId: string,
  pollIntervalMs: number
): Promise<Job> {
  for (;;) {
    const job = await getJob(jobId);
    if (job.status === "SUCCESS") return job;
    if (job.status === "FAILED" || job.status === "CANCELLED") {
      throw new Error(
        job.message?.trim() ||
          (job.status === "CANCELLED"
            ? "The model download was cancelled."
            : "The model could not be downloaded.")
      );
    }
    await wait(pollIntervalMs);
  }
}

/**
 * Make one selected pack runnable, downloading and verifying it when needed.
 *
 * The install remains its own background job. That gives both progress
 * surfaces a byte-based "Download model…" row before the tile-based inference
 * row, and lets an install already started from the Models screen be reused.
 */
export async function ensureModelInstalled(
  packId: string,
  options: EnsureModelInstalledOptions = {}
): Promise<ModelPack> {
  const catalogue = await getModelCatalogue();
  let pack = catalogue.packs.find((candidate) => candidate.id === packId);
  if (!pack) throw new Error(`This build does not know model ${packId}.`);

  if (!pack.installed) {
    let jobId = pack.active_install?.job_id
      ? String(pack.active_install.job_id)
      : null;
    if (!jobId) {
      const install = await installModelPack(packId);
      jobId = install.job_id ? String(install.job_id) : null;
    }
    if (jobId) {
      await options.onDownloadQueued?.(jobId);
      await waitForInstallJob(
        jobId,
        options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS
      );
    }

    const refreshed = await getModelCatalogue();
    pack = refreshed.packs.find((candidate) => candidate.id === packId);
    if (!pack?.installed) {
      throw new Error("The model download finished but the model is not installed.");
    }
  }

  if (pack.runnable === false) {
    throw new Error(pack.reason || "The downloaded model cannot run on this computer.");
  }
  await options.onInstalled?.(pack);
  return pack;
}
