"""
Helper module for managing asset preprocessing status.

This module provides a centralized API for updating preprocessing stage, progress,
and error fields on asset-backed objects. All preprocessing status updates should
go through this module to ensure consistency and validation.

Usage:
    from .preprocess_status import set_stage

    image = Asset.objects.get(id=some_id)

    # Set stage with progress
    set_stage(image, "ENCODING", progress=25.5)

    # Set stage with progress and error
    set_stage(image, "FAILED", progress=50.0, error="Conversion failed")

    # Clear error when moving to a new stage
    set_stage(image, "ENCODING", progress=0.0, error="")
"""

from .models import PREPROCESS_STAGE_CHOICES, Asset, ImageROI

# Extract valid stage values from the model's choices
VALID_STAGES = {choice[0] for choice in PREPROCESS_STAGE_CHOICES}


def set_stage(
    image: Asset | ImageROI,
    stage: str,
    progress: float | None = None,
    error: str | None = None,
) -> None:
    """
    Update preprocessing stage, progress, and/or error for an asset-backed object.

    This function validates the stage name and updates preprocessing fields
    atomically. All preprocessing status updates should use this function to
    ensure consistency.

    Args:
        image: The asset-backed instance to update
        stage: One of the valid PREPROCESS_STAGE_CHOICES values
        progress: Optional progress value (0-100). If None, progress is not updated.
        error: Optional error message. If None, error is not updated.
               Pass empty string "" to clear an existing error.

    Raises:
        ValueError: If stage is not one of the valid choices

    Example:
        # Start encoding
        set_stage(image, "ENCODING", progress=0.0)

        # Update progress during encoding
        set_stage(image, "ENCODING", progress=15.5)

        # Continue encoding / view-asset generation
        set_stage(image, "ENCODING", progress=20.0)

        # Mark as failed with error
        set_stage(image, "FAILED", progress=45.0, error="TIFF read error")

        # Clear error and restart
        set_stage(image, "ENCODING", progress=0.0, error="")
    """
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Invalid preprocess stage: {stage}. "
            f"Valid stages are: {', '.join(sorted(VALID_STAGES))}"
        )

    image.preprocess_stage = stage

    if progress is not None:
        image.preprocess_progress = progress

    if error is not None:
        image.preprocess_error = error

    # Use update_fields to only update the relevant fields
    update_fields = ["preprocess_stage"]
    if progress is not None:
        update_fields.append("preprocess_progress")
    if error is not None:
        update_fields.append("preprocess_error")

    try:
        image.save(update_fields=update_fields)
    except (ImageROI.DoesNotExist, Asset.DoesNotExist):
        # Object was deleted while a background task was running.
        return
