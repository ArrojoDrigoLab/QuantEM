"""Edge refinement utilities: snap a polygon boundary onto image gradients."""

from .constraints import ConstraintEnforcementError
from .service import refine_mask_with_edges

__all__ = [
    "ConstraintEnforcementError",
    "refine_mask_with_edges",
]
