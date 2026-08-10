"""Label canonicalisation: raw multi-class / instance ``_label.tif`` -> binary organelle mask.

The single source of truth is the per-crop ``organelles_present`` map from the dataset manifest
(not the dataset-level ``organelle_classes`` map), because several datasets renumber organelles per
volume (e.g. empiar_12885 mito in {1,2}, ER in {6,8}; empiar_10994 ER in {2,3} + subtypes). Matching
is by name, reading each match's integer ``value`` per crop; instance-encoded entries (which carry
``instances`` but no ``value``) binarise as ``label != 0``.

OpenOrganelle is handled separately in ``openorganelle.py`` (per-class sidecar seg files).
"""

from __future__ import annotations

import numpy as np

from ..constants import ORGANELLES


def _matching_entries(organelles_present: dict, organelle: str) -> list[tuple[str, dict]]:
    """Return ``(name, entry)`` for every organelles_present entry that maps to this target.

    A name matches when (lowercased) it is not in ``exclude`` and (it is in ``include`` or starts with
    any ``include_prefix``). The prefix rule makes ER/mito robust to per-volume renaming, so a genuine
    target organelle is always mapped to foreground rather than background.
    """
    spec = ORGANELLES[organelle]
    include = set(spec["include"])  # type: ignore[arg-type]
    exclude = set(spec["exclude"])  # type: ignore[arg-type]
    prefixes = tuple(spec.get("include_prefix", ()))  # type: ignore[arg-type]
    out = []
    for name, entry in (organelles_present or {}).items():
        lname = str(name).lower()
        if lname in exclude:
            continue
        if lname in include or (prefixes and lname.startswith(prefixes)):
            out.append((name, entry))
    return out


def canonical_mask(label: np.ndarray, organelles_present: dict, organelle: str) -> np.ndarray:
    """Binary foreground mask (bool) for ``organelle`` from a raw label array + its crop manifest.

    Foreground = union over matching organelle entries of ``(label == entry.value)``; for an
    instance-encoded entry (no ``value``) the whole label is that organelle so ``label != 0``.
    Returns an all-False mask if the organelle is absent from this crop.
    """
    if organelle not in ORGANELLES:
        raise ValueError(f"Unknown organelle {organelle!r}; expected one of {tuple(ORGANELLES)}")
    label = np.asarray(label)
    matches = _matching_entries(organelles_present, organelle)
    matched_names = {n for n, _ in matches}
    fg = np.zeros(label.shape, dtype=bool)
    used_instance = False
    for _name, entry in matches:
        val = entry.get("value") if isinstance(entry, dict) else None
        if val is None:
            # instance / binary-nonzero encoding (entry has 'instances' or is otherwise value-less)
            fg |= label != 0
            used_instance = True
        else:
            fg |= label == int(val)
    # Defensive guard: ``label != 0`` is only correct for single-organelle (instance) label files. If
    # an instance-encoded target ever co-occurs with other value-bearing organelles in the same label
    # map, their values are subtracted so no other organelle is pulled into the mask. (No-op on the
    # current corpus, where instance-target crops are single-organelle.)
    if used_instance:
        other_vals = [
            int(e["value"]) for n, e in (organelles_present or {}).items()
            if n not in matched_names and isinstance(e, dict) and e.get("value") is not None
        ]
        if other_vals:
            fg &= ~np.isin(label, other_vals)
    return fg


def has_target(organelles_present: dict, organelle: str) -> bool:
    """Whether this crop's manifest declares the target organelle at all."""
    return len(_matching_entries(organelles_present, organelle)) > 0


def is_instance_encoded(organelles_present: dict, organelle: str) -> bool:
    """True if the target organelle is stored as instance ids (a matching entry without a ``value``)."""
    for _n, e in _matching_entries(organelles_present, organelle):
        if not (isinstance(e, dict) and e.get("value") is not None):
            return True
    return False


def instance_map(label: np.ndarray, organelles_present: dict, organelle: str) -> tuple:
    """Return (instance-id map int32, gt_is_instance) for ``organelle``.

    Real ids where the source is instance-encoded; otherwise connected-components 'pseudo-instances'
    of the binary mask (flagged ``gt_is_instance=False``). Non-foreground pixels are 0.
    """
    from scipy import ndimage as ndi

    fg = canonical_mask(label, organelles_present, organelle)
    if is_instance_encoded(organelles_present, organelle):
        return (np.asarray(label).astype(np.int32) * fg, True)
    lab, _ = ndi.label(fg)
    return (lab.astype(np.int32), False)
