"""Canonical queue and job contracts for the DB worker system.

**Adding a job type is a two-step change, and the steps may land apart.**
``registry.job_handler`` refuses to register a handler for a type that is not in
:data:`ALLOWED_JOB_TYPES`, so the type must be declared here before the package
that implements it can register anything. Declaring types for several packages
at once -- which is what the two v2-push types below are -- means this file
knows about work whose handler may not have landed yet. A type with no handler
cannot be enqueued -- the serializer refuses it -- which is what makes it safe to
declare the whole push's types in one edit. ``run_segmentation_for_image`` has a
handler; ``reextract_at_include_level`` does not yet.

That gap is a hazard, not merely untidy: ``JobCreateSerializer`` validates
``type`` against this list, so a declared-but-unimplemented type would validate,
be written to the queue, and then die at dispatch with nothing the user can do
about it. It is closed in that serializer, which refuses any type with no
handler registered. This file may therefore declare freely; only a type that can
actually be run is enqueueable.
"""

from __future__ import annotations

JOB_TYPE_ENSURE_IMAGE_NGFF = "ensure_image_ngff"
JOB_TYPE_UPLOAD_IMAGE_PIPELINE = "upload_image_pipeline"
JOB_TYPE_RUN_SEGMENTATION_ROI = "run_segmentation_roi_task"
JOB_TYPE_RUN_SEGMENTATION_FULL = "run_segmentation_full_task"
JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY = "rebuild_segmentation_overlay"
JOB_TYPE_REFRESH_SEGMENT_FEATURES = "refresh_segment_features"
JOB_TYPE_TRAIN_ORGANELLE_ADAPTER = "train_organelle_adapter"
JOB_TYPE_RUN_ANALYSIS = "run_analysis"
JOB_TYPE_INSTALL_MODEL_PACK = "install_model_pack"

# --- v2 push -------------------------------------------------------------
# Declared here, ahead of their handlers, so that the packages implementing
# them each touch only their own ``handlers/*.py``. Until a handler is
# registered neither can be enqueued; see the module docstring.

#: One run over one image, covering every organelle the user asked for, instead
#: of one job per organelle. GPU-bound and long, so it sits on the background
#: queue beside the full-image run it replaces. The image is decoded once, the
#: organelles are walked one after another in a single worker, and their tile
#: counts add up to one denominator on one row. Four separate jobs paid four
#: cold model loads -- 45.6 s measured -- and gave the wave rollup four moving
#: parts to reconcile.
JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE = "run_segmentation_for_image"

#: Re-derive the objects from the stored probability map at a new include
#: level. No model runs: it thresholds bytes that are already on disk and
#: re-extracts, which is CPU work of a few seconds, and the user is watching the
#: dial -- so it belongs on the interactive queue and nowhere else.
JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL = "reextract_at_include_level"

ALLOWED_JOB_TYPES = frozenset(
    {
        JOB_TYPE_ENSURE_IMAGE_NGFF,
        JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
        JOB_TYPE_RUN_SEGMENTATION_ROI,
        JOB_TYPE_RUN_SEGMENTATION_FULL,
        JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
        JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
        JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
        JOB_TYPE_REFRESH_SEGMENT_FEATURES,
        JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
        JOB_TYPE_RUN_ANALYSIS,
        JOB_TYPE_INSTALL_MODEL_PACK,
    }
)

# Job types that no longer have a handler. Rows carrying one of these must not
# crash the runner: it fails that single job with a legible message and keeps
# draining the queue (see ``quantem.jobs.registry.get_handler``).
LEGACY_JOB_TYPES = frozenset(
    {
        # Pre-queue-unification types.
        "generate_dzi",
        "encode_full_png",
        "train_pixel_classifier",
        "organelle_inference",
        "organelle_retrain",
        "organelle_apply_global",
        # Dropped in QuantEM: SAM prompting, cell/granule pipelines, comparator
        # models, the membrane classifier, and mito interactive refinement.
        "process_user_feedback",
        "process_auto_sam_prompt_request",
        "process_granule_auto_add_request",
        "run_proposed_refinement",
        "finalize_cell_boundary_refinement",
        "run_other_model",
        "train_membrane_classifier",
        "apply_membrane_classifier",
        "process_mito_feature_cache",
        "train_mito_source_adapter",
    }
)

QUEUE_P1_INTERACTIVE = "p1_interactive"
QUEUE_P2_UPLOAD = "p2_upload"
QUEUE_P3_ROI = "p3_roi"
QUEUE_P4_FULL = "p4_full"

ALLOWED_QUEUE_NAMES = frozenset(
    {
        QUEUE_P1_INTERACTIVE,
        QUEUE_P2_UPLOAD,
        QUEUE_P3_ROI,
        QUEUE_P4_FULL,
    }
)

QUEUE_PRIORITY_ORDER = {
    QUEUE_P1_INTERACTIVE: 0,
    QUEUE_P2_UPLOAD: 1,
    QUEUE_P3_ROI: 2,
    QUEUE_P4_FULL: 3,
}

QUEUE_DISPLAY_NAMES = {
    QUEUE_P1_INTERACTIVE: "P1 Interactive",
    QUEUE_P2_UPLOAD: "P2 Upload",
    QUEUE_P3_ROI: "P3 ROI",
    QUEUE_P4_FULL: "P4 Background",
}

#: How the Tasks & Queues panel names each kind of work. These are read by a
#: biologist, so they are plain English and never the type string; the copy gate
#: enforces that.
JOB_TYPE_LABELS = {
    JOB_TYPE_ENSURE_IMAGE_NGFF: "Build image NGFF",
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE: "Process upload",
    JOB_TYPE_RUN_SEGMENTATION_ROI: "Run ROI segmentation",
    JOB_TYPE_RUN_SEGMENTATION_FULL: "Run full-image segmentation",
    # Plain English, because ``registry/tests/copy_gate.py`` gates these as
    # user-visible copy and the Tasks drawer prints them verbatim.
    JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE: "Segment this image",
    JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL: "Redo objects at a new include level",
    JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY: "Rebuild segmentation overlay",
    JOB_TYPE_REFRESH_SEGMENT_FEATURES: "Refresh segment features",
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER: "Adapt model to your data",
    JOB_TYPE_RUN_ANALYSIS: "Run analysis",
    JOB_TYPE_INSTALL_MODEL_PACK: "Download model pack",
}

JOB_DEFAULTS = {
    JOB_TYPE_ENSURE_IMAGE_NGFF: {
        "priority": "high",
        "resource_class": "cpu",
        "queue_name": QUEUE_P2_UPLOAD,
    },
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE: {
        "priority": "high",
        "resource_class": "cpu",
        "queue_name": QUEUE_P2_UPLOAD,
    },
    JOB_TYPE_RUN_SEGMENTATION_ROI: {
        "priority": "high",
        "resource_class": "gpu",
        "queue_name": QUEUE_P3_ROI,
    },
    JOB_TYPE_RUN_SEGMENTATION_FULL: {
        "priority": "default",
        "resource_class": "gpu",
        "queue_name": QUEUE_P4_FULL,
    },
    # One job covering every organelle the user ticked. Same shape as the
    # full-image run it supersedes -- GPU, background queue -- because it is
    # the same work with one denominator instead of four.
    JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE: {
        "priority": "default",
        "resource_class": "gpu",
        "queue_name": QUEUE_P4_FULL,
    },
    # Thresholding a stored map and re-extracting: seconds of CPU, with the
    # user holding the dial. On the interactive queue so it never waits behind
    # a half-hour segmentation, and ``cpu`` so it does not take a GPU slot it
    # has no use for.
    JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL: {
        "priority": "high",
        "resource_class": "cpu",
        "queue_name": QUEUE_P1_INTERACTIVE,
    },
    JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY: {
        "priority": "high",
        "resource_class": "cpu",
        "queue_name": QUEUE_P1_INTERACTIVE,
    },
    JOB_TYPE_REFRESH_SEGMENT_FEATURES: {
        "priority": "high",
        "resource_class": "cpu",
        "queue_name": QUEUE_P1_INTERACTIVE,
    },
    # Guided fine-tuning. ``resource_class="gpu"`` routes it through the
    # persistent worker pool, which is what we want even with no CUDA device:
    # the pool is one slot wide, so a long adaptation never runs alongside a
    # full-image segmentation on the same CPU. Device pinning degrades to
    # "whatever torch picks" on a CPU-only or Apple Silicon machine.
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER: {
        "priority": "default",
        "resource_class": "gpu",
        "queue_name": QUEUE_P4_FULL,
    },
    # Quantitative analysis is pure CPU numerics over segments already in the
    # database, so it belongs on the background queue and never blocks the
    # interactive one.
    JOB_TYPE_RUN_ANALYSIS: {
        "priority": "default",
        "resource_class": "cpu",
        "queue_name": QUEUE_P4_FULL,
    },
    # Downloading a model pack from Hugging Face. Network- and disk-bound, so
    # ``cpu``; on the upload queue rather than P4 because the user who clicked
    # Install is waiting to *run* the model, and parking the download behind a
    # half-hour full-image segmentation would read as a hang.
    JOB_TYPE_INSTALL_MODEL_PACK: {
        "priority": "default",
        "resource_class": "cpu",
        "queue_name": QUEUE_P2_UPLOAD,
    },
}

#: Job types that hold a segmentation while they run. Membership wires a type
#: into the failure/retry reconcilers in :mod:`quantem.jobs.failure_reconcile`,
#: which read ``payload_json["segmentation_id"]`` -- so a type here whose
#: payload does not carry that key reconciles nothing, and a run that dies
#: leaves its segmentation showing as still running.
ACTIVE_SEGMENTATION_JOB_TYPES = frozenset(
    {
        JOB_TYPE_RUN_SEGMENTATION_ROI,
        JOB_TYPE_RUN_SEGMENTATION_FULL,
        JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
        JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
    }
)

# Failures here are deterministic (bad payload, unusable image, out of memory on
# this machine), so an unattended retry only delays the error the user is
# watching for.
NO_RETRY_JOB_TYPES = frozenset(
    {
        JOB_TYPE_RUN_SEGMENTATION_FULL,
        # A retry would re-run every organelle, including the ones that already
        # produced objects, on a failure that will not get better unattended.
        JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
        # Nothing about a missing or unreadable stored map improves on a second
        # attempt, and the user is holding the dial waiting for an answer.
        JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
        JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
        # A failed download names its cause (offline, digest mismatch, disk);
        # none of those get better unattended, and the user is watching.
        JOB_TYPE_INSTALL_MODEL_PACK,
    }
)
