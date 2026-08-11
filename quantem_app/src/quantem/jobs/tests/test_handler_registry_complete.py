"""Every shipped job type still has a handler registered.

``@job_handler`` registers by import side effect, and ``quantem.jobs.handlers``
is a package: if one of its submodules stopped being imported by
``handlers/__init__.py``, that submodule's job types would vanish from the
registry with no error anywhere. The failure would surface only as a queued row
that cannot be dispatched -- in a frozen build, on a user's machine.

The expected set is written out as literal strings rather than derived from
``ALLOWED_JOB_TYPES``, on purpose: these strings are stored in the ``job_type``
column of rows that already exist in people's databases, so a rename is a
migration, not a refactor, and this test is where that shows up.
"""

from quantem.jobs.registry import _HANDLERS

SHIPPED_JOB_TYPES = frozenset(
    {
        "ensure_image_ngff",
        "upload_image_pipeline",
        "run_segmentation_roi_task",
        "run_segmentation_full_task",
        # One run over one image, covering every organelle the user ticked.
        "run_segmentation_for_image",
        # The include-level dial: re-derive the objects from the stored
        # probability map without running the model.
        "reextract_at_include_level",
        "rebuild_segmentation_overlay",
        "refresh_segment_features",
        "train_organelle_adapter",
        "run_analysis",
        "install_model_pack",
    }
)


def test_registry_holds_exactly_the_shipped_job_types():
    assert set(_HANDLERS) == SHIPPED_JOB_TYPES


def test_every_registered_handler_is_callable():
    for job_type, handler in _HANDLERS.items():
        assert callable(handler), job_type
