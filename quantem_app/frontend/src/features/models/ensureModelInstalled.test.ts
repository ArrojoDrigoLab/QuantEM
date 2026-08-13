import { beforeEach, describe, expect, it, vi } from "vitest";

import { ensureModelInstalled } from "@/features/models/ensureModelInstalled";
import { getModelCatalogue, installModelPack } from "@/shared/api/finetune";
import { getJob } from "@/shared/api/jobs";
import type { ModelCatalogue, ModelPack } from "@/shared/types/finetune";

vi.mock("@/shared/api/finetune", () => ({
  getModelCatalogue: vi.fn(),
  installModelPack: vi.fn(),
}));
vi.mock("@/shared/api/jobs", () => ({ getJob: vi.fn() }));

function pack(installed: boolean): ModelPack {
  return {
    id: "quantem:mito",
    family: "quantem",
    organelle: "mito",
    title: "QuantEM — Mitochondria",
    installed,
    download_bytes: 2_500_000_000,
    canonical_nm: 8,
    tile_size: 512,
    default_threshold: 0.5,
    decoder: "decoder",
    neck: "neck",
    adapt: "adapt",
    licence: "licence",
    notes: "",
    runnable: installed,
    reason: installed ? null : "Not installed yet.",
  };
}

function catalogue(model: ModelPack): ModelCatalogue {
  return { packs: [model], adapted: [], device: null };
}

describe("ensureModelInstalled", () => {
  beforeEach(() => vi.clearAllMocks());

  it("downloads a missing model, waits for verification, then returns it", async () => {
    vi.mocked(getModelCatalogue)
      .mockResolvedValueOnce(catalogue(pack(false)))
      .mockResolvedValueOnce(catalogue(pack(true)));
    vi.mocked(installModelPack).mockResolvedValue({
      job_id: "download-1",
      status: "PENDING",
    });
    vi.mocked(getJob).mockResolvedValue({
      id: "download-1",
      type: "install_model_pack",
      priority: "default",
      status: "SUCCESS",
      progress: 100,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:01Z",
      attempts: 1,
      max_attempts: 3,
      next_run_at: "2026-01-01T00:00:00Z",
      payload_json: {},
      cancel_requested: false,
      resource_class: "cpu",
      queue_name: "default",
      tags: [],
    });
    const onDownloadQueued = vi.fn();
    const onInstalled = vi.fn();

    const installed = await ensureModelInstalled("quantem:mito", {
      onDownloadQueued,
      onInstalled,
      pollIntervalMs: 0,
    });

    expect(installModelPack).toHaveBeenCalledWith("quantem:mito");
    expect(onDownloadQueued).toHaveBeenCalledWith("download-1");
    expect(onInstalled).toHaveBeenCalledWith(installed);
    expect(installed.installed).toBe(true);
  });

  it("does not start a download when the model is already ready", async () => {
    vi.mocked(getModelCatalogue).mockResolvedValue(catalogue(pack(true)));

    await ensureModelInstalled("quantem:mito");

    expect(installModelPack).not.toHaveBeenCalled();
    expect(getJob).not.toHaveBeenCalled();
  });
});
