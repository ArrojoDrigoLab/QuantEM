"""EM FINO layer — metadata-guided foundation training on top of the DINOv3 ``FINO`` branch.

FINO is fine-tuning with no labels: continued pretraining that makes one metadata factor either more
or less decodable from the representation. A factor marked ``M+`` gets an auxiliary head that predicts
it; a factor marked ``M-`` gets the same head behind a gradient reversal, so the representation is
pushed to discard it. The reported arms continue for 30,000 steps from the base arm's teacher export
and condition on one factor each — physical scale and imaging modality, in both directions — against a
control that continues the same 30,000 steps with no guide at all.
(Gardes, E. et al. Who needs labels? Adapting vision foundation models with the metadata you already
have. arXiv:2606.05107.)

The official metadata-guidance machinery comes directly from the pinned DINOv3 FINO branch
(``dinov3.train.guided_ssl_meta_arch.GuidedSSLMetaArch`` + ``dinov3.train.metadata_utils``:
classification/regression/prototypical heads, gradient-reversal, sigmoid lambda schedule). This
package adds only the EM-specific pieces:

  * :mod:`factors` — the EM factors a guide head may target (``log(effective_nm_per_px)``,
    ``modality``, ``organ``), the allow/deny objective guard, vocabulary canonicalization, log-scale
    standardisation, and the picklable per-sample :class:`~factors.EMTileMetadata` payload;
  * :mod:`guide_losses` — mask-aware classification/regression losses (partial-metadata EM
    extension over the upstream complete-metadata assumption);
  * :mod:`meta_arch_patch` — :func:`~meta_arch_patch.apply_fino_grafts` grafting the masking
    onto ``GuidedSSLMetaArch._compute_guide_losses``.

"""

from __future__ import annotations

from .factors import (
    ALLOWED_OBJECTIVES,
    DENIED_OBJECTIVE_FIELDS,
    EMTileMetadata,
    FinoFactorSpec,
    FinoRuntime,
    FinoTargetTransform,
    encode_tile_metadata,
    factor_from_dict,
    factors_from_config,
    fino_factors_fingerprint,
)
from .meta_arch_patch import apply_fino_grafts

__all__ = [
    "ALLOWED_OBJECTIVES",
    "DENIED_OBJECTIVE_FIELDS",
    "EMTileMetadata",
    "FinoFactorSpec",
    "FinoRuntime",
    "FinoTargetTransform",
    "encode_tile_metadata",
    "factor_from_dict",
    "factors_from_config",
    "fino_factors_fingerprint",
    "apply_fino_grafts",
]
