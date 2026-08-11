"""Model registry manifest — the contract between a QuantEM release and its weights.

Weights are never bundled in an installer and never committed to git. They are
fetched on first use from a versioned manifest, verified by SHA-256, and cached
under the user data directory.

Design notes
------------
* **Content-addressed blobs.** ``quantem:mito``, ``quantem:ld`` and
  ``quantem:nucleus`` all reference the *same* 525,781,487 B ViT-B encoder.
  Storing per-model would cost 3x; addressing by digest costs 1x.
* **Upstream vs mirrored.** ``BlobRef.url`` may point at our own host *or* at the
  upstream publisher. The OmniEM encoder is a third party's artifact; if
  redistribution cannot be evidenced, its entry links upstream instead of being
  mirrored, and no schema change is needed. See NOTICE.
* **Licence is per pack and is shown before download.** The repository is MIT;
  the weights are not necessarily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = 1

Organelle = Literal["mito", "er", "nucleus", "ld"]
Family = Literal["quantem", "omniem"]

#: Default foreground probability threshold. 0.5 for every organelle and both
#: families -- the setting behind every benchmark in the manuscript, so a value a
#: user reproduces here is comparable to a published one. Guided fine-tuning may
#: calibrate a per-model threshold on top of this; it does not change the default.
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class BlobRef:
    """One downloadable file, addressed by digest."""

    sha256: str
    size_bytes: int
    url: str
    #: True when ``url`` is the original publisher rather than a QuantEM mirror.
    upstream: bool = False

    @property
    def cache_key(self) -> str:
        return f"{self.sha256[:2]}/{self.sha256}"


@dataclass(frozen=True)
class ModelPack:
    """One installable (family, organelle) model."""

    id: str  # e.g. "quantem:mito"
    family: Family
    organelle: Organelle
    version: str
    title: str
    encoder: BlobRef | None  # None when the head embeds a full fine-tuned encoder
    head: BlobRef
    #: Input is resampled to this pixel size before inference. None = native
    #: resolution. From the released resolved_config.yaml: 8.0 for mito and LD,
    #: 25.0 for nucleus, null for ER.
    canonical_nm: float | None
    tile_size: int  # 512 (patch 16) or 518 (patch 14)
    default_threshold: float = DEFAULT_THRESHOLD
    licence: str = "see NOTICE"
    licence_url: str | None = None
    citation: str | None = None
    min_app_version: str = "0.1.0"
    notes: str = ""

    @property
    def download_bytes(self) -> int:
        return self.head.size_bytes + (self.encoder.size_bytes if self.encoder else 0)


@dataclass
class Manifest:
    schema_version: int = SCHEMA_VERSION
    generated_at: str = ""
    packs: list[ModelPack] = field(default_factory=list)

    def get(self, pack_id: str) -> ModelPack:
        for p in self.packs:
            if p.id == pack_id:
                return p
        raise KeyError(f"unknown model pack: {pack_id!r}")

    def for_organelle(self, organelle: Organelle) -> list[ModelPack]:
        return [p for p in self.packs if p.organelle == organelle]

    def unique_blobs(self) -> dict[str, BlobRef]:
        """Deduplicate by digest -- this is what makes the encoder shared."""
        blobs: dict[str, BlobRef] = {}
        for p in self.packs:
            for b in (p.encoder, p.head):
                if b is not None:
                    blobs.setdefault(b.sha256, b)
        return blobs

    def total_bytes(self, pack_ids: list[str] | None = None) -> int:
        packs = self.packs if pack_ids is None else [self.get(i) for i in pack_ids]
        seen: dict[str, int] = {}
        for p in packs:
            for b in (p.encoder, p.head):
                if b is not None:
                    seen[b.sha256] = b.size_bytes
        return sum(seen.values())


# ---------------------------------------------------------------------------
# Released artifact inventory.
#
# Sizes below were measured on disk and are correct. The sha256 values are NOT
# yet computed -- every upstream ``checkpoint_index.json`` carries
# ``"sha256": null``, so these must be generated when the artifacts are uploaded.
# `quantem registry hash` writes them. Until then the loader refuses to install.
# ---------------------------------------------------------------------------
MEASURED_SIZES: dict[str, int] = {
    # The published safetensors on Hugging Face (ArrojoeDrigoLab/quantem at the
    # pinned revision, byte sizes from the repo's own files metadata) -- because
    # that is what an install actually downloads. The first version of this
    # table measured the *local research artifacts* instead, and the QuantEM
    # encoder entry was the 525.8 MB fp32 ``encoder.pth`` while the published
    # trunk is 227.7 MB fp16: the Models screen advertised "631.7 MB to
    # install" for a 364 MB download -- a 74% overstatement on the one screen
    # whose job is to tell the user the cost before they commit.
    "quantem_vitb_encoder": 227_685_512,  # quantem-vitb-trunk.safetensors
    "omniem_vitl_encoder": 1_217_509_768,  # omniem-vitl.safetensors
    "mito_quantem_head": 136_541_856,
    "ld_quantem_head": 136_541_848,
    "nucleus_quantem_head": 136_541_864,
    "er_quantem_head": 465_028_184,  # adapt: full -- embeds a whole fine-tuned ViT-B
    "mito_omniem_head": 25_730_696,  # LoRA r=8
    "ld_omniem_head": 25_730_688,
    "nucleus_omniem_head": 25_730_704,
    "er_omniem_head": 135_200_976,
}

#: Per-model architecture facts, transcribed from the eight released
#: ``resolved_config.yaml`` files. Verified to match the manuscript exactly.
ARCHITECTURE: dict[str, dict[str, Any]] = {
    "quantem:mito": {
        "neck": "naive_1x1",
        "decoder": "affinity_mws",
        "adapt": "last_n",
        "canonical_nm": 8.0,
        "tile": 512,
    },
    "quantem:ld": {
        "neck": "naive_1x1",
        "decoder": "affinity_mws",
        "adapt": "last_n",
        "canonical_nm": 8.0,
        "tile": 512,
    },
    "quantem:nucleus": {
        "neck": "naive_1x1",
        "decoder": "affinity_mws",
        "adapt": "last_n",
        "canonical_nm": 25.0,
        "tile": 512,
    },
    "quantem:er": {
        "neck": "resnet34_detail",
        "decoder": "upernet",
        "adapt": "full",
        "canonical_nm": None,
        "tile": 512,
    },
    "omniem:mito": {
        "neck": "naive_1x1",
        "decoder": "affinity_mws",
        "adapt": "lora8",
        "canonical_nm": 8.0,
        "tile": 518,
    },
    "omniem:ld": {
        "neck": "naive_1x1",
        "decoder": "affinity_mws",
        "adapt": "lora8",
        "canonical_nm": 8.0,
        "tile": 518,
    },
    "omniem:nucleus": {
        "neck": "naive_1x1",
        "decoder": "affinity_mws",
        "adapt": "lora8",
        "canonical_nm": 25.0,
        "tile": 518,
    },
    "omniem:er": {
        "neck": "resnet34_detail",
        "decoder": "dpt",
        "adapt": "lora8",
        "canonical_nm": None,
        "tile": 518,
    },
}

#: Encoder normalisation, from each family's ``checkpoint_index.json``. These do
#: NOT appear in any resolved_config.yaml -- inference is not reproducible from
#: the eight YAMLs alone, which is why they are pinned here.
ENCODER_NORM: dict[Family, tuple[float, float]] = {
    "quantem": (0.583175, 0.244468),
    "omniem": (0.595446, 0.211906),
}
