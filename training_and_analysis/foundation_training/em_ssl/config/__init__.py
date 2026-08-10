"""Experiment configuration schema + loader (the user-facing contract).

The DINOv3 training entry point, the configuration resolver and the post-run diagnostics all read
the same normalized `ExperimentSpec`, so one experiment YAML fixes the data bundle, augmentation,
model, and training schedule for an arm.
"""

from .schema import (  # noqa: F401
    AugmentationSpec,
    CheckpointingSpec,
    CropStage,
    DataSpec,
    ExperimentSpec,
    LoggingSpec,
    ModelSpec,
    OptimSpec,
    TrainSpec,
    load_experiment,
    resolve_data_paths,
)
