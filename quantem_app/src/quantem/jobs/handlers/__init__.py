"""Job handlers, one module per kind of work.

This was a single ``handlers.py``. It is a package so that the packages that
add a job type each own one file, and two of them adding a handler never edit
the same one. Nothing about the registry changed: the job-type keys, the
handler functions and the module path ``quantem.jobs.handlers`` are exactly what
they were, and every name the old module exposed is re-exported here.

**Every submodule must be imported eagerly, right here.** ``@job_handler``
registers by import side effect and ``jobs.apps.JobsConfig.ready`` autodiscovers
only ``quantem.jobs.handlers`` itself, so a submodule left to lazy import would
drop its job type out of the registry -- silently at boot, and visibly only when
a queued row of that type fails in a frozen build. ``jobs/tests/
test_handler_registry_complete.py`` is the guard.
"""

from quantem.jobs.handlers.analysis import handle_run_analysis
from quantem.jobs.handlers.common import _as_bool, _asset_for_payload
from quantem.jobs.handlers.models import (
    _model_display_name,
    handle_install_model_pack,
    handle_train_organelle_adapter,
)
from quantem.jobs.handlers.rethreshold import handle_reextract_at_include_level
from quantem.jobs.handlers.segmentation import (
    _segmentation_run_outcome,
    _unmeasured_segment_ids,
    _validate_segmentation_payload,
    handle_rebuild_segmentation_overlay,
    handle_refresh_segment_features,
    handle_run_segmentation_full_task,
    handle_run_segmentation_roi_task,
)
from quantem.jobs.handlers.upload import (
    _UPLOAD_ORGANELLE_CHOICES,
    VOLUME_SEGMENTATION_SKIP_MESSAGE,
    VOLUME_SEGMENTATION_SKIP_REASON,
    _requested_upload_organelles,
    _run_volume_upload_pipeline,
    handle_ensure_image_ngff,
    handle_upload_image_pipeline,
)

__all__ = [
    "VOLUME_SEGMENTATION_SKIP_MESSAGE",
    "VOLUME_SEGMENTATION_SKIP_REASON",
    "handle_ensure_image_ngff",
    "handle_install_model_pack",
    "handle_reextract_at_include_level",
    "handle_rebuild_segmentation_overlay",
    "handle_refresh_segment_features",
    "handle_run_analysis",
    "handle_run_segmentation_full_task",
    "handle_run_segmentation_roi_task",
    "handle_train_organelle_adapter",
    "handle_upload_image_pipeline",
    # Private, but imported by name elsewhere in the tree (tests included), so
    # they are part of this package's surface whether or not they are public.
    "_UPLOAD_ORGANELLE_CHOICES",
    "_as_bool",
    "_asset_for_payload",
    "_model_display_name",
    "_requested_upload_organelles",
    "_run_volume_upload_pipeline",
    "_segmentation_run_outcome",
    "_unmeasured_segment_ids",
    "_validate_segmentation_payload",
]
