/**
 * Starting one run over one image, across several organelles.
 *
 * Two shapes live here and they answer two different questions:
 *
 * * {@link RunPlan} — *before* the run. What each ticked organelle will cost in
 *   tiles, whether its model is on this machine, and how many bytes have to come
 *   down first. The server computes it from the tiling arithmetic, not from an
 *   adjective, and computing it queues nothing.
 * * {@link RunLeg} — *during* the run. One job row covers every organelle, so
 *   the per-organelle lines the run panel draws come from this list rather than
 *   from one job each.
 */

/** One ticked organelle, costed. */
export interface RunPlanOrganelle {
  /** `"mito"`, `"er"`, `"nucleus"`, `"ld"`. */
  organelle: string;
  /** What a person calls it: "Mitochondria". */
  name: string;
  /** The model pack that would run it, e.g. `"quantem:mito"`. */
  pack_id: string;
  /** The pack's own title, e.g. `"QuantEM — Mitochondria"`. */
  title: string;
  /**
   * Windows the run will walk. Exact — the same arithmetic the loop counts
   * with — and available for a pack that has not been downloaded yet. Null only
   * when this build cannot work it out at all.
   */
  tiles: number | null;
  model_installed: boolean;
  model_ready: boolean;
  /** The server's sentence when the model cannot run here. */
  model_blocked_reason: string | null;
  /** Bytes to fetch before this one can start. Zero once it is installed. */
  download_bytes: number;
  segmentation_id: string | null;
}

/** What the whole run will cost, before it starts. */
export interface RunPlan {
  asset_id: string;
  pixel_size_nm: number | null;
  /**
   * Every organelle this image can be run for — including the ones not ticked,
   * because otherwise there would be no way to tick them.
   */
  organelles: RunPlanOrganelle[];
  /** The subset the totals below are about. */
  selected: string[];
  /**
   * Every organelle's tiles added up, or null when one of them could not be
   * costed. Never a short total: a denominator missing one organelle's tiles
   * fills the bar with work still to do.
   */
  tiles_total: number | null;
  /**
   * Deduped. Packs in one family share an encoder blob, so adding their
   * download figures up overstates the cost — measured at 2.62x across the
   * eight released packs.
   */
  download_bytes_total: number;
  packs_to_download: string[];
}

/** One organelle inside a run that covers several. */
export interface RunLeg {
  segmentation_id: string;
  name: string;
  status: "PENDING" | "RUNNING" | "SUCCESS" | "FAILED" | "CANCELLED";
  units_done: number;
  units_total: number;
  unit_label: string;
  percent: number | null;
}

export interface StartRunResponse {
  job_id: string;
  plan: RunPlan;
}
