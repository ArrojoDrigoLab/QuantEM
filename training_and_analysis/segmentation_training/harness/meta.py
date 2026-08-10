"""Metadata vocabulary for image-style conditioning — string manifest fields -> integer ids.

Used by (a) the gradient-reversed source adversary, for both its class counts and its per-sample
targets, (b) source-id pooling (``style_scope`` = source/dataset), and (c) the source-aware MixStyle
permutation (``mixstyle_mix`` = crossdomain). Built from the train records so the same vocabulary is
shared by train + eval and saved into ``head.pt`` (a held-out source's ``dataset`` maps to 0='unknown',
which is correct — its identity was never a training class).

Index 0 is reserved for 'unknown'/missing on every field. Pure stdlib (no GPU needed).
"""

from __future__ import annotations


class MetaVocab:
    def __init__(self, fields: list[str], maps: dict[str, dict[str, int]]):
        self.fields = list(fields)
        self.maps = {f: dict(m) for f, m in maps.items()}

    @classmethod
    def build(cls, records: list[dict], fields: list[str]) -> "MetaVocab":
        maps: dict[str, dict[str, int]] = {}
        for f in fields:
            vals = sorted({str(r.get(f)) for r in records if r.get(f) is not None})
            maps[f] = {v: i + 1 for i, v in enumerate(vals)}  # 0 reserved for unknown
        return cls(fields, maps)

    def sizes(self) -> dict[str, int]:
        """Field -> adversary class count (values + 1 for the unknown slot)."""
        return {f: len(self.maps[f]) + 1 for f in self.fields}

    def encode(self, record: dict) -> dict[str, int]:
        return {f: self.maps[f].get(str(record.get(f)), 0) for f in self.fields}

    def to_dict(self) -> dict:
        return {"fields": self.fields, "maps": self.maps}

    @classmethod
    def from_dict(cls, d: dict | None) -> "MetaVocab | None":
        if not d:
            return None
        return cls(list(d.get("fields", [])), dict(d.get("maps", {})))


def conditioning_fields(cond) -> list[str]:
    """The manifest fields the conditioner needs: the source id (``dataset``) plus the adversary's targets."""
    fields = {"dataset"}
    fields.update(getattr(cond, "adv_targets", []) or [])
    return sorted(fields)
