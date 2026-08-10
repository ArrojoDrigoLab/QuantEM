"""A mixed multi-organelle training dataset (mito + ER [+ LD / nucleus]).

Yields crops drawn from several organelles, each tagged with a per-crop organelle code (the one-hot the
DoDNet controller consumes / the FiLM-MoE gate reads). Under the hood it reuses ``SegTrainDataset`` per
organelle (each organelle has its own derived data root + manifest, resampled to that organelle's canonical
nm/px), so all the segmentation training crop/pad/augment/normalise logic is inherited unchanged.

Design (matches the study's binary-per-organelle head): each organelle is a binary foreground task
(``num_classes=2``); the organelle code selects which task the shared head solves. So a mixed batch mixes
"is-this-pixel-mito" and "is-this-pixel-ER" crops, and the DoDNet controller generates the right final filter
per crop from its code. Item dict: ``{image, target[, inst], org_idx, org_code}`` — the extra keys
``org_idx`` (LongTensor scalar) and ``org_code`` (``[K]`` float one-hot) collate into ``[B]`` / ``[B, K]``.

The per-organelle sub-datasets can carry different tasks (mito=instance, ER=semantic). Each sub-dataset's
own ``inst`` policy is honoured: a crop only emits ``inst`` if its organelle's task is instance. To keep the
default-collate batch keys uniform across a mixed batch, ``MixedOrganelleDataset`` provides a ``collate`` that
pads the missing ``inst`` with a zero map for semantic crops (so mito's instance-head loss still gets a real
map on mito crops and a harmless all-background map on ER crops, which its fg term ignores).

Runs without a GPU: imports only torch (numpy arrives transitively via SegTrainDataset, which also pulls
scipy lazily for instance labels). The test suite exercises it by indexing the dataset and calling
``collate`` directly, without a DataLoader; the sub-datasets expose ``reseed`` so the module-level
worker-init in the trainer reseeds them all.
"""

from __future__ import annotations

import copy

import torch
from torch.utils.data import Dataset

# Canonical multi-organelle order -> the fixed index each organelle occupies in the K-dim code. The
# per-organelle baseline recipes are documented in config_templates.ORG_RECIPE.
ORGANELLE_ORDER = ("mito", "er", "ld", "nucleus")

# The instance-target ignore value (matches the ``ignore_index`` default of
# models.instance_targets.seg_to_affinities, and reads as non-foreground to seg_to_center_offset): a crop
# whose inst is all this value contributes nothing to the instance loss (ignored, not zero-supervised).
INSTANCE_IGNORE = -100


def organelle_index(organelle: str, order=ORGANELLE_ORDER) -> int:
    o = organelle.lower()
    if o not in order:
        raise ValueError(f"unknown organelle {organelle!r}; known: {order}")
    return order.index(o)


def subset_code_map(organelles, order=ORGANELLE_ORDER) -> dict:
    """The subset-local one-hot slot per organelle: 0..len-1, ordered by ORGANELLE_ORDER (deterministic
    regardless of dict/CLI order). The code dimension K = the number of selected organelles, so a non-prefix
    subset (e.g. mito+ld) does not index past K, which the fixed ORGANELLE_ORDER index would.
    Both training (MixedOrganelleDataset) and eval (evaluate_multi) use this so the codes agree."""
    sel = sorted({o.lower() for o in organelles}, key=lambda o: organelle_index(o, order))
    return {o: i for i, o in enumerate(sel)}


def one_hot(idx: int, k: int) -> torch.Tensor:
    v = torch.zeros(k, dtype=torch.float32)
    v[int(idx)] = 1.0
    return v


class MixedOrganelleDataset(Dataset):
    """Concatenate per-organelle ``SegTrainDataset``s + tag each crop with its organelle code.

    Args:
        per_organelle: ``{organelle: (records, data_root)}`` — one derived root + manifest per organelle.
        cfg:           a baseline-shaped ``SegConfig`` (tile_size / augment / min_fg_frac_keep / seed). It is
                       deep-copied per organelle with ``data.organelle`` + ``data.task`` set (mito=instance,
                       ER=semantic) so each sub-dataset inherits the correct per-organelle crop/inst policy.
        mean, std:     encoder EM norm stats.
        n_organelles:  K, the code dimension (default = the number of organelles in ``per_organelle``).
        tasks:         optional ``{organelle: 'instance'|'semantic'}`` override (default: mito/ld instance,
                       er/nucleus semantic, mirroring ``config_templates.ORG_RECIPE``).
    """

    def __init__(self, per_organelle: dict, cfg, mean: float, std: float, *, patch_size: int = 16,
                 n_organelles: int | None = None, tasks: dict | None = None, balance: str = "raw"):
        from ...harness.dataset import SegTrainDataset

        self.organelles = list(per_organelle.keys())
        # Subset-local code slot (0..len-1) so K = #selected organelles and no code indexes past K.
        self.code_slot = subset_code_map(self.organelles)
        # Default task per organelle: mito/ld = instance, er/nucleus = semantic (mirrors config_templates.ORG_RECIPE).
        default_task = {"mito": "instance", "ld": "instance", "er": "semantic", "nucleus": "semantic"}
        self.tasks = {o: (tasks or {}).get(o, default_task.get(o, "semantic")) for o in self.organelles}
        self.k = int(n_organelles) if n_organelles is not None else len(self.code_slot)
        self.balance = str(balance)                          # 'raw' | 'balanced'
        self.subsets: list[Dataset] = []
        raw_index: list[tuple[int, int]] = []                # (subset_i, local_idx)
        self.org_idx: list[int] = []                         # subset-local one-hot slot (0..K-1) per subset
        self.raw_counts: dict[str, int] = {}                 # per-organelle raw crop count (the imbalance)
        for si, org in enumerate(self.organelles):
            records, data_root = per_organelle[org]
            sub_cfg = copy.deepcopy(cfg)
            sub_cfg.data.organelle = org
            sub_cfg.data.task = self.tasks[org]
            ds = SegTrainDataset(records, data_root, sub_cfg, mean, std, patch_size=patch_size)
            self.subsets.append(ds)
            self.org_idx.append(self.code_slot[org.lower()])
            self.raw_counts[org] = len(ds)
            for li in range(len(ds)):
                raw_index.append((si, li))
        # Balance: 'raw' concatenates (ER is about 7x mito, so the shared head is dominated by ER gradients and
        # a sharing effect cannot be separated from an imbalance artifact). 'balanced' oversamples each
        # organelle to the largest count (cycling; discards no data) so each organelle gets ~equal exposure,
        # i.e. matched per-organelle gradient steps. This confound lies outside the seed comparator, so the
        # sampling mode and the raw counts are reported alongside the verdict.
        self.index = self._balanced_index(raw_index) if self.balance == "balanced" else raw_index

    def _balanced_index(self, raw_index):
        """Oversample each organelle's crops (cycling) to the largest per-organelle count -> ~equal exposure."""
        import collections
        by_sub = collections.defaultdict(list)
        for si, li in raw_index:
            by_sub[si].append((si, li))
        target = max((len(v) for v in by_sub.values()), default=0)
        out = []
        for si, items in by_sub.items():
            for j in range(target):
                out.append(items[j % len(items)])
        return out

    def ratio_report(self) -> dict:
        """The organelle imbalance (raw counts + fractions) and the sampling mode, reported alongside any
        shared-vs-specialist verdict: at ER ~7x mito raw, 'sharing hurts mito' can be an imbalance artifact."""
        tot = sum(self.raw_counts.values()) or 1
        return {"balance": self.balance, "raw_counts": dict(self.raw_counts),
                "raw_fractions": {o: c / tot for o, c in self.raw_counts.items()},
                "sampled_len": len(self.index),
                "max_over_min_ratio": (max(self.raw_counts.values()) / max(1, min(self.raw_counts.values())))
                if self.raw_counts else None}

    def __len__(self) -> int:
        return len(self.index)

    def reseed(self, salt: int) -> None:
        for i, ds in enumerate(self.subsets):
            if hasattr(ds, "reseed"):
                ds.reseed(salt * 131 + i)

    def __getitem__(self, i: int):
        si, li = self.index[i]
        item = dict(self.subsets[si][li])       # {image, target[, inst]}
        oidx = self.org_idx[si]
        item["org_idx"] = torch.tensor(oidx, dtype=torch.long)
        item["org_code"] = one_hot(oidx, self.k)
        item["_task"] = self.tasks[self.organelles[si]]
        return item

    def collate(self, batch: list[dict]) -> dict:
        """Uniform-key collate for a mixed batch.

        For crops without an instance task (ER/nucleus), ``inst`` is filled with ``INSTANCE_IGNORE`` (-100 =
        the ``seg_to_affinities`` ignore value), not a zero (all-background) map.
        This is load-bearing: a zero-inst target would supervise the shared instance head to 'predict no
        instances' on ER crops (the majority under raw imbalance), so the shared instance head would learn
        to under-call and mito instance recall would degrade. Ignoring rather than zero-supervising is the
        correct target. (The DoDNet head is semantic-only, so no instance loss consumes ``inst``;
        ``inst_task`` marks the instance crops.)"""
        images = torch.stack([b["image"] for b in batch])
        targets = torch.stack([b["target"] for b in batch])
        org_idx = torch.stack([b["org_idx"] for b in batch])
        org_code = torch.stack([b["org_code"] for b in batch])
        hw = targets.shape[-2:]
        insts, inst_task = [], []
        for b in batch:
            is_inst = ("inst" in b) and (b.get("_task") == "instance")
            inst_task.append(is_inst)
            if is_inst:
                insts.append(b["inst"].long())
            else:
                insts.append(torch.full(hw, INSTANCE_IGNORE, dtype=torch.long))  # ignored, not zero-supervised
        out = {"image": images, "target": targets, "org_idx": org_idx, "org_code": org_code,
               "inst_task": torch.tensor(inst_task, dtype=torch.bool)}
        if any(inst_task):
            out["inst"] = torch.stack(insts)
        return out


def build_mixed_dataset(per_organelle: dict, cfg, mean: float, std: float, *, patch_size: int = 16,
                        n_organelles: int | None = None, tasks: dict | None = None,
                        balance: str = "raw") -> MixedOrganelleDataset:
    return MixedOrganelleDataset(per_organelle, cfg, mean, std, patch_size=patch_size,
                                 n_organelles=n_organelles, tasks=tasks, balance=balance)


def load_per_organelle(data_roots: dict, split: str = "train") -> dict:
    """``{organelle: data_root}`` -> ``{organelle: (records, data_root)}`` by loading each organelle's manifest
    for ``split`` (group = ``group2_<organelle>``). Used by the runner to assemble the mixed training set."""
    from ...harness.dataset import load_manifest

    out = {}
    for org, root in data_roots.items():
        recs = load_manifest(root, f"group2_{org}", split)
        out[org] = (recs, root)
    return out
