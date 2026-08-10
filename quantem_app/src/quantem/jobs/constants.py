"""Canonical queue and job contracts for the DB worker system."""

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

ALLOWED_JOB_TYPES = frozenset(
    {
        JOB_TYPE_ENSURE_IMAGE_NGFF,
        JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
        JOB_TYPE_RUN_SEGMENTATION_ROI,
        JOB_TYPE_RUN_SEGMENTATION_FULL,
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

JOB_TYPE_LABELS = {
    JOB_TYPE_ENSURE_IMAGE_NGFF: "Build image NGFF",
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE: "Process upload",
    JOB_TYPE_RUN_SEGMENTATION_ROI: "Run ROI segmentation",
    JOB_TYPE_RUN_SEGMENTATION_FULL: "Run full-image segmentation",
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

ACTIVE_SEGMENTATION_JOB_TYPES = frozenset(
    {
        JOB_TYPE_RUN_SEGMENTATION_ROI,
        JOB_TYPE_RUN_SEGMENTATION_FULL,
    }
)

# Failures here are deterministic (bad payload, unusable image, out of memory on
# this machine), so an unattended retry only delays the error the user is
# watching for.
NO_RETRY_JOB_TYPES = frozenset(
    {
        JOB_TYPE_RUN_SEGMENTATION_FULL,
        JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
        # A failed download names its cause (offline, digest mismatch, disk);
        # none of those get better unattended, and the user is watching.
        JOB_TYPE_INSTALL_MODEL_PACK,
    }
)
