# API contract — analysis, guided fine-tuning, models, deletion

The shared spec for the backend endpoints and the frontend that calls them.
Written before either side, so they cannot drift while being built in parallel.
Everything is loopback-only and unauthenticated.

Conventions: UUIDs are strings. Times are ISO-8601 UTC. Long work returns
`202 Accepted` with a `job_id` and is polled through the existing
`GET /api/jobs/<job_id>/`. Errors are `{"error": "<human-readable sentence>"}`.

---

## Deleting a segmentation

### `GET /api/segmentations/<seg_id>/`
The serialized segmentation plus `delete_preview` — the live counts a deletion
confirm dialog must quote. Read fresh when the dialog opens, to the Mark-Done
standard; the list payload's `segment_counts` can be a poll behind.

```jsonc
{
  // ...the ordinary ImageSegmentation payload...
  "delete_preview": {
    "segmentation_id": "...",
    "segmentation_type": "Mitochondria",
    "object_count": 12,                      // every object, whatever its label
    "objects_by_label_state": {"CONFIRMED": 3, "EXCLUDED": 2,
                               "CANDIDATE": 5, "INFERRED": 2},
    "probability_map_count": 1,
    "overlay_count": 2,
    "adapter_count": 1,
    "analysis_run_count": 2,                 // KEPT, not deleted — see below
    "locked": false
  }
}
```

### `DELETE /api/segmentations/<seg_id>/`
Deletes the segmentation and everything it owns: its objects (confirmed,
rejected and unreviewed alike), overlay rasters, probability maps, adapters
(including trained head weights on disk), completed-ROI record, feedback and
config. Nothing is archived; there is no undo short of running the model again.
Afterwards the organelle's preset is creatable again — the create endpoint is
get_or_create per (asset, type), so the deleted row was what blocked it.

Optional `acknowledged_object_count` (JSON body or query parameter): the object
count the user was shown. Same contract as Mark Image Done's discard — a
mismatch (usually a run that finished while the dialog was open) is refused
with **409** carrying a fresh `delete_preview`, and nothing is deleted.

`200` response:

```jsonc
{"deleted": { /* the delete_preview that was destroyed */ },
 "analysis_runs_kept": 2}
```

Refusals, both **409** and both naming the reason:

* **A job is active on the segmentation** (queued, running or retrying — a
  full/ROI run, an analysis, an adapter training). Body carries `detail`
  naming the job, plus `job_id` / `job_type` / `job_status` and how to clear
  it. Pulling rows out from under a worker is a crash, so the job goes first.
* **The completion lock is on** (`locked: true`, the standard locked payload).
  "Done" stays final until unlocked; deletion is the strongest mutation there
  is.

**Analysis runs survive deletion — decided, not accidental.** A run's numbers
and its export bundle are the record of an analysis that happened, possibly
already cited; deleting the segmentation does not un-happen it. The FK is
`SET_NULL`: every surviving run keeps its `results`, `params`, `export_dir`
and its place in the group rollup, with `segmentation_id: null` and
`segmentation_deleted: true` in both the run detail and the run-list payloads.
Every run is created *with* a segmentation, so a null reference has exactly
one meaning. What is honestly lost: the run can no longer be re-run or traced
back to live objects, which is why `segmentation_deleted` must be rendered
wherever such a run is shown.

---

## Models

### `GET /api/models/`
The eight released models plus any the user has adapted.

```jsonc
{
  "packs": [
    {
      "id": "quantem:mito",
      "family": "quantem",            // quantem | omniem
      "organelle": "mito",            // mito | er | nucleus | ld
      "title": "QuantEM — Mitochondria",
      "installed": true,
      "download_bytes": 364227368,
      "canonical_nm": 8.0,            // null = run at native resolution
      "tile_size": 512,
      "default_threshold": 0.5,
      "decoder": "affinity_mws",
      "neck": "naive_1x1",
      "adapt": "last_n",
      "licence": "see NOTICE",
      "notes": "",
      // The install already in flight for this pack, or null. Present on
      // every pack entry. `status` is QUEUED (the job is waiting — PENDING or
      // RETRY underneath) or RUNNING (bytes are moving). `job_id` is the job
      // UUID **as a string**, the same id `GET /api/jobs/<job_id>/` polls.
      // The byte counts are null until the download's first progress sample
      // lands, and stay null for jobs that never report bytes.
      "active_install": null // | {"job_id": "<uuid>", "status": "QUEUED"|"RUNNING",
                             //    "progress_current_bytes": 214000000, // int|null
                             //    "progress_total_bytes": 1243000000}  // int|null
    }
  ],
  "adapted": [
    {
      "id": "adapted:<uuid>",
      "base": "quantem:mito",
      "name": "mito @ my-liver-set",
      "created_at": "...",
      "calibrated_threshold": 0.35,
      "heldout_dice": 0.87,
      "split_mode": "image-disjoint"   // image-disjoint | within-image | no-heldout
    }
  ],
  "device": {"kind": "cpu", "name": "CPU", "cuda": false, "mps": false},
  "registry": {                        // where a not-installed pack downloads from
    "repo_id": "ArrojoeDrigoLab/quantem",
    "revision": "<pinned commit sha>",
    "url": "https://huggingface.co/ArrojoeDrigoLab/quantem"
  }
}
```

Each pack's `download_bytes` is what a fresh install of it must fetch (heads
sharing an encoder count it once per family, not per pack).

### `POST /api/models/<pack_id>/install/`
Make a pack usable. Sources, in order: an already-installed copy, a local path
given as `{"source_path": "..."}`, then the remote registry — the QuantEM
Hugging Face repository named in the list body's `registry` block.

**No body (or no `source_path`)** downloads the pack. The response is a `202`
with a *queued* job:

```jsonc
{
  "job_id": "...",                 // poll GET /api/jobs/<job_id>/
  "pack_id": "quantem:mito",
  "status": "PENDING",
  "source": "huggingface",
  "repo_id": "ArrojoeDrigoLab/quantem",
  "revision": "<pinned commit sha>",
  "download_bytes": 364227368,     // known before the first byte moves
  "detail": "..."
}
```

The job (`type: "install_model_pack"`) reports real progress: 2–80% is the
download with a byte count in `message`, then verify/convert/export. Every
artifact's sha256 is verified against the pinned revision **before** anything
is installed; a mismatch or an offline machine fails the job with the exact
reason (both digests, or the repo URL plus the offline bundle route). Cancel
and queue-removal behave like every other job; an aborted install never leaves
a half-installed pack.

**With `source_path`** the copy runs inline and the `202` arrives when the pack
is already usable; its job row is written terminal, so the first poll returns
`SUCCESS`. `{"force": true}` reinstalls either way.

An already-installed pack returns `200` with `"detail": "Already installed."`
and no job.

**A pack whose install is already queued or running returns `409`** instead of
queueing a duplicate download — the same guard the first-launch
(installer-requested) queueing applies:

```jsonc
{
  "error": "An install of omniem:mito is already running as job <uuid> (status RUNNING). ...",
  "job_id": "<uuid>",              // the existing job to watch or cancel
  "pack_id": "omniem:mito",
  "status": "PENDING|RUNNING|RETRY",  // the existing job's raw status
  "active_install": {              // same shape as the list body's field
    "job_id": "<uuid>", "status": "QUEUED|RUNNING",
    "progress_current_bytes": 214000000, "progress_total_bytes": 1243000000
  }
}
```

---

## Analysis

### `POST /api/segmentations/<seg_id>/analysis/`
Start an analysis run. `202` + `{"job_id", "analysis_run_id"}`.

```jsonc
{
  "compartments": {"nucleus": "<seg_id>", "mito": "<seg_id>"},  // organelle -> segmentation
  "tissue_segmentation_id": "<seg_id>|null",   // null = whole image, and say so
  "points_source": "centroids|csv|null",
  "points_csv": "x,y\n10,20\n...",             // when points_source == "csv"
  "distance_target": "mito|null",
  "band_edges_nm": [0, 50, 100, 200],
  "replicates": 20,
  "seed": 12345,
  "group": "fasted"
}
```

### `GET /api/analysis/<run_id>/`
```jsonc
{
  "id": "...", "status": "PENDING|RUNNING|SUCCESS|FAILED",
  "created_at": "...", "group": "fasted",
  "calibrated": true, "pixel_size_nm": 5.0,
  "composition": {"tissue_px": 0, "tissue_um2": 0.0, "area_fractions": {}},
  "objects": {"n": 0, "summary": {}, "density": {}},
  "points": {"n_total": 0, "n_on_tissue": 0, "n_off_tissue": 0,
             "counts": {}, "fractions": {}, "enrichment": {}},
  "distances": {"target": "mito", "band_labels": [], "band_counts": [],
                "band_fractions": [], "median_nm": 0.0, "n_inside": 0},
  "monte_carlo": {"replicates": 20, "seed": 12345, "observed": {},
                  "null_mean": {}, "null_sd": {}, "z": {}, "p_two_sided": {}},
  "monte_carlo_self_check": {                 // internal control, see below
    "n_points": 25600, "smallest_compartment_fraction": 0.049,
    "enrichment": {}, "max_abs_deviation": 0.021,
    "skipped_reason": null
  },
  "caveats": ["..."],
  "export_dir": "<abs path>"
}
```

`monte_carlo` and `monte_carlo_self_check` are both `null` when there are no
points, and both are `null` when the tissue mask is empty — there is nowhere to
scatter a simulated point. That case is a **caveat, not a failure**: the run
succeeds and names the cause.

**`monte_carlo_self_check`** scatters uniform random points inside the tissue and
re-measures enrichment. Uniform points must recover ~1.0 in every compartment, so
`max_abs_deviation` is the bias in the normalisation *for the user's own
geometry* — it is a control worth showing, not an implementation detail. The draw
is sized to give the smallest compartment ~2,000 points and is bounded by the
tissue area and by 50,000, so it is proportionate rather than fixed.
`skipped_reason` is a sentence when the check could not run and `null` otherwise;
`max_abs_deviation` is then `null`, never `0.0`.

### `GET /api/analysis/<run_id>/export/<name>`
`objects.csv`, `image_summary.csv`, `manifest.json`. `text/csv` or
`application/json`, `Content-Disposition: attachment`.

Both CSVs always carry a header row, including when the run confirmed no objects.
A zero-byte file is not an empty table — `pandas.read_csv` rejects it.

### `GET /api/segmentations/<seg_id>/analysis/`
List runs for a segmentation, newest first.

### `GET /api/analysis/groups/?segmentation=<seg_id>&group=<label>`
Group-level rollup over completed runs. Both query parameters are optional:
`segmentation` restricts to one segmentation's runs, `group` to one label.
Only `SUCCESS` runs are included.

```jsonc
{
  "aggregation_rule": "Unweighted mean over experimental units, ...",
  "unit": "one analysis run (one image)",
  "scope": {"segmentation": "<seg_id>|null", "group": "fasted|null",
            "status": "SUCCESS", "n_runs": 3},
  "metrics": ["area_fraction_mito", "enrichment_mito", "n_objects", "z_..."],
  "groups": [
    {
      "group": "fasted",
      "n_units": 3,
      "unit": "one analysis run (one image)",
      "run_ids": ["..."],
      "image_keys": ["..."],
      "pixel_sizes_nm": [5.0],              // distinct values, never a mean
      "warnings": ["3 runs cover only 1 distinct image. ..."],
      "metrics": {
        "enrichment_mito": {"n_units": 3, "mean": 1.8, "sd": 0.3,
                            "sem": 0.17, "values": [1.5, 1.8, 2.1]}
      }
    }
  ]
}
```

`sd` and `sem` are `null` for a single unit — one observation has no spread and
the payload will not invent one. `warnings` names anything that would mislead if
left unsaid: repeated runs of the same image (pseudo-replication), runs with no
group label, and groups mixing calibrated and uncalibrated images. The UI must
show `aggregation_rule` next to any number from this endpoint (honesty rule 4).

---

## Guided fine-tuning

### `GET /api/segmentations/<seg_id>/adapt/crops/`
What the user has annotated, and whether it is enough to proceed.

```jsonc
{
  "crops": [
    {"id": "...", "name": "roi-1", "image_key": "<asset_id>",
     "width": 1024, "height": 1024, "n_objects": 12, "annotated_px": 1048576}
  ],
  "split_mode": "image-disjoint",
  "n_images": 2,
  "ready": true,
  "blockers": [],                       // e.g. ["No completed ROI on this image"]
  "warnings": ["Only one image is annotated; the held-out score is within-image."]
}
```

A **completed ROI** is required: inside it, anything not a confirmed object is
true background; outside it is *ignore*. Without that, Dice is meaningless — this
is the one hard blocker.

### `POST /api/segmentations/<seg_id>/adapt/`
`202` + `{"job_id", "adapter_id"}`.

```jsonc
{
  "base_model": "quantem:mito",
  "mode": "threshold_only|head",   // threshold_only is CPU-cheap and always offered
  "steps": 300,
  "lr": 0.0001,
  "seed": 0,
  "name": "mito @ my-liver-set"
}
```

### `GET /api/adapters/<adapter_id>/`
```jsonc
{
  "id": "...", "base_model": "quantem:mito", "name": "...",
  "status": "PENDING|RUNNING|SUCCESS|FAILED", "mode": "head",
  "steps": 300, "trainable_params": 5775000,
  "split_mode": "image-disjoint",
  "train_crop_names": [], "heldout_crop_names": [],
  "sweep": {
    "thresholds": [], "train_dice": [],
    "calibrated_threshold": 0.35,
    "train_dice_at_default": 0.0, "train_dice_at_calibrated": 0.0,
    "heldout_dice_at_default": 0.0, "heldout_dice_at_calibrated": 0.0,
    "heldout_oracle": 0.0, "improvement": 0.0,
    "per_crop": {}
  },
  "verified_reload": true,     // the saved head was reloaded and re-scored
  "caveats": ["The threshold was fit on the training crops only."]
}
```

### `POST /api/adapters/<adapter_id>/apply/`
Use this adapter for subsequent runs on the segmentation. `200`.

---

## Honesty rules the UI must honour

These are requirements, not suggestions. They exist because the numbers here go
into papers.

1. **Never present a held-out score without its split mode.** `within-image` and
   `image-disjoint` measure different things.
2. **Badge the crops the threshold was fit on** in any per-crop table.
3. **Show the oracle as a ceiling**, never as an achievable target.
4. **State the aggregation rule** next to any group-level number: unweighted
   mean over units, never weighted by point count.
5. **Show `n_off_tissue`** whenever it is non-zero — excluded points change every
   fraction.
6. **Say when pixel size is unset**, and do not render µm units in that state.
