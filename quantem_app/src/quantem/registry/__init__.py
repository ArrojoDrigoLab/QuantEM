"""Model registry: manifest, local cache, checksum verification, installation.

``manifest`` is the release contract (what packs exist, what they cost, what
licence they carry). ``cache`` is where an installed pack lives on this machine
and is what :mod:`quantem.inference.engine` resolves against. ``install`` puts
packs there. ``catalogue`` joins the three into the read model behind
``GET /api/models/`` and adds the one fact none of them holds -- whether a pack
can actually be *run* here -- and ``views``/``urls`` serve it.

Only ``manifest`` is imported here: ``cache`` reads the user data directory,
``install`` pulls in torch through the pack config, and ``views`` needs Django
REST framework. Importing this package must stay free for Django startup.
"""

from .manifest import (
    ARCHITECTURE,
    DEFAULT_THRESHOLD,
    ENCODER_NORM,
    MEASURED_SIZES,
    SCHEMA_VERSION,
    BlobRef,
    Manifest,
    ModelPack,
)

__all__ = [
    "ARCHITECTURE",
    "DEFAULT_THRESHOLD",
    "ENCODER_NORM",
    "MEASURED_SIZES",
    "SCHEMA_VERSION",
    "BlobRef",
    "Manifest",
    "ModelPack",
]
