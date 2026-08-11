"""Per-organelle and per-model inference specs for the eight released models.

Two independent sources are merged here:

* **Organelle post-processing constants** (``default_min_area``,
  ``close_radius``, the registry name and generated-flag) are properties of the
  *organelle*, identical for both model families, and are expressed in **native
  image pixels**.
* **Architecture facts** (``neck``, ``decoder``, ``adapt``, ``canonical_nm``,
  ``tile``) come from :data:`quantem.registry.manifest.ARCHITECTURE`, which was
  transcribed from the eight released ``resolved_config.yaml`` files. Encoder
  normalisation comes from :data:`quantem.registry.manifest.ENCODER_NORM`;
  those numbers appear in no YAML and are not recoverable from the checkpoints,
  which is why they are pinned in the manifest.

Nothing here touches the filesystem. Where the weights for a pack live is the
model registry's business (:mod:`quantem.registry`), not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quantem.registry.manifest import ARCHITECTURE, DEFAULT_THRESHOLD, ENCODER_NORM
from quantem.segmentation.type_definitions import (
    ER,
    LIPID_DROPLETS,
    MITOCHONDRIA,
    NUCLEUS,
)

Family = Literal["quantem", "omniem"]
OrganelleKey = Literal["mito", "er", "ld", "nucleus"]

FAMILIES: tuple[str, ...] = ("quantem", "omniem")
DEFAULT_FAMILY: str = "quantem"


# --- Families ---------------------------------------------------------------


@dataclass(frozen=True)
class FamilySpec:
    """One foundation-encoder family.

    Two normalisations, and they are not the same thing:

    * ``image_mean`` / ``image_std`` -- the encoder's **EM corpus statistics**,
      from :data:`quantem.registry.manifest.ENCODER_NORM`. Never ImageNet.
    * ``input_mean`` / ``input_std`` -- what the **tensor handed to the model**
      must already be scaled by. For QuantEM the two coincide: its training
      pipeline standardised the tile with the corpus stats and gave the result
      straight to a 1-channel ViT. For OmniEM they do not: that pipeline handed
      the encoder a raw ``[0, 1]`` tile and applied the corpus stats *inside*
      the encoder, after replicating grey to three channels.

    Using the corpus stats as the input scaling for OmniEM would double-normalise
    the encoder and, for ``omniem:er``, also feed the wrong distribution to the
    ``resnet34_detail`` neck's raw-image branch, which reads the same tensor.
    :func:`quantem.inference.engine.load_model` checks the built encoder's
    contract against these values, so a drift fails loudly instead of quietly
    degrading a published model.
    """

    key: str
    label: str
    encoder: str  # encoder run name, as published
    patch_size: int  # ViT patch edge; tile sizes are multiples of it
    image_mean: float  # EM corpus normalisation -- never ImageNet
    image_std: float
    input_mean: float  # scaling of the tensor the model is called with
    input_std: float


FAMILY_SPECS: dict[str, FamilySpec] = {
    "quantem": FamilySpec(
        key="quantem",
        label="QuantEM",
        encoder="m1_dinov3_vitb",
        patch_size=16,
        image_mean=ENCODER_NORM["quantem"][0],
        image_std=ENCODER_NORM["quantem"][1],
        # dinov3 path: the model input is the standardised tile.
        input_mean=ENCODER_NORM["quantem"][0],
        input_std=ENCODER_NORM["quantem"][1],
    ),
    "omniem": FamilySpec(
        key="omniem",
        label="OmniEM",
        encoder="omniem_emdino_vitl",
        patch_size=14,
        image_mean=ENCODER_NORM["omniem"][0],
        image_std=ENCODER_NORM["omniem"][1],
        # timm path: the model input is the raw [0, 1] tile; the encoder
        # normalises internally after channel replication.
        input_mean=0.0,
        input_std=1.0,
    ),
}

FAMILY_LABELS: dict[str, str] = {k: v.label for k, v in FAMILY_SPECS.items()}


# --- Organelles -------------------------------------------------------------


@dataclass(frozen=True)
class OrganelleSpec:
    """Organelle-level facts, shared by both families.

    ``default_min_area`` and ``close_radius`` are in **native image pixels**;
    post-processing runs at native scale so these keep their meaning whatever a
    given asset's pixel size is.
    """

    key: str
    internal_name: str  # segmentation-type internal name
    segmenter_name: str  # registry key, e.g. "dino_mito"
    generated_flag: str  # feature marker, e.g. "mito_generated"
    default_min_area: int
    #: Disk radius for the morphological closing applied to the thresholded
    #: mask before labeling, to consolidate a compact object fragmented by
    #: internal probability texture (large for nuclei, small for thin ER).
    #: Enclosed holes are filled afterwards.
    close_radius: int = 2


ORGANELLES: dict[str, OrganelleSpec] = {
    "mito": OrganelleSpec(
        key="mito",
        internal_name=MITOCHONDRIA.internal_name,
        segmenter_name="dino_mito",
        generated_flag="mito_generated",
        default_min_area=60,
        close_radius=3,
    ),
    "er": OrganelleSpec(
        key="er",
        internal_name=ER.internal_name,
        segmenter_name="dino_er",
        generated_flag="er_generated",
        default_min_area=100,
        close_radius=1,
    ),
    "ld": OrganelleSpec(
        key="ld",
        internal_name=LIPID_DROPLETS.internal_name,
        segmenter_name="dino_ld",
        generated_flag="ld_generated",
        default_min_area=40,
        close_radius=2,
    ),
    "nucleus": OrganelleSpec(
        key="nucleus",
        internal_name=NUCLEUS.internal_name,
        segmenter_name="dino_nucleus",
        generated_flag="nucleus_generated",
        default_min_area=8000,
        close_radius=12,
    ),
}

#: Reverse map: registry name -> organelle key.
SEGMENTER_NAME_TO_ORGANELLE: dict[str, str] = {
    spec.segmenter_name: key for key, spec in ORGANELLES.items()
}


# --- Models (family x organelle) --------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Everything inference needs to know about one released model pack."""

    pack_id: str  # "quantem:mito"
    family: str
    organelle: str
    neck: str  # "naive_1x1" | "resnet34_detail"
    decoder: str  # "affinity_mws" | "upernet" | "dpt"
    adapt: str  # "last_n" | "full" | "lora8"
    canonical_nm: float | None  # None = run at the asset's native resolution
    tile_size: int  # 512 (patch 16) or 518 (patch 14)
    patch_size: int
    image_mean: float  # encoder EM corpus stats; see FamilySpec
    image_std: float
    input_mean: float  # scaling of the tensor the model is called with
    input_std: float
    threshold: float = DEFAULT_THRESHOLD

    @property
    def organelle_spec(self) -> OrganelleSpec:
        return ORGANELLES[self.organelle]

    @property
    def family_spec(self) -> FamilySpec:
        return FAMILY_SPECS[self.family]

    @property
    def encoder(self) -> str:
        return self.family_spec.encoder

    @property
    def label(self) -> str:
        return f"{self.family_spec.label} {self.organelle}"

    @property
    def embeds_encoder(self) -> bool:
        """True when the head is a fully fine-tuned model, not encoder + adapter.

        ``quantem:er`` was adapted with ``adapt: full``, so its 465 MB head file
        embeds a whole ViT-B and the shared encoder blob is not loaded for it.
        """
        return self.adapt == "full"


def _build_model_specs() -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {}
    for pack_id, arch in ARCHITECTURE.items():
        family, organelle = pack_id.split(":", 1)
        family_spec = FAMILY_SPECS[family]
        tile_size = int(arch["tile"])
        if tile_size % family_spec.patch_size != 0:
            raise ValueError(
                f"{pack_id}: tile {tile_size} is not a multiple of patch {family_spec.patch_size}"
            )
        canonical = arch["canonical_nm"]
        specs[pack_id] = ModelSpec(
            pack_id=pack_id,
            family=family,
            organelle=organelle,
            neck=str(arch["neck"]),
            decoder=str(arch["decoder"]),
            adapt=str(arch["adapt"]),
            canonical_nm=float(canonical) if canonical is not None else None,
            tile_size=tile_size,
            patch_size=family_spec.patch_size,
            image_mean=family_spec.image_mean,
            image_std=family_spec.image_std,
            input_mean=family_spec.input_mean,
            input_std=family_spec.input_std,
        )
    return specs


#: All eight released models, keyed by pack id ("quantem:mito").
MODEL_SPECS: dict[str, ModelSpec] = _build_model_specs()


# --- Source-model parsing ---------------------------------------------------


def source_model_value(family: str, organelle: str) -> str:
    """Catalog value for a (family, organelle) pair, e.g. ``'quantem:mito'``."""
    return f"{family}:{organelle}"


def parse_family(source_model: str | None, default: str = DEFAULT_FAMILY) -> str:
    """Extract the family ('quantem'/'omniem') from a source-model value."""
    if source_model:
        head = source_model.split(":", 1)[0].strip().lower()
        if head in FAMILIES:
            return head
    return default


def get_model_spec(family: str, organelle: str) -> ModelSpec:
    """Look up a model spec, with a clear error for an unreleased combination."""
    pack_id = source_model_value(family, organelle)
    try:
        return MODEL_SPECS[pack_id]
    except KeyError as exc:
        raise ValueError(f"No released model for {pack_id!r}") from exc


def get_organelle_spec(organelle: str) -> OrganelleSpec:
    try:
        return ORGANELLES[organelle]
    except KeyError as exc:
        raise ValueError(f"Unknown organelle: {organelle!r}") from exc
