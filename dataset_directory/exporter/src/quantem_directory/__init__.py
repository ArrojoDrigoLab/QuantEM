"""Export the QuantEM EM corpus into the static artifacts the directory site reads.

The site has no server and no database. Everything it needs is a handful of JSON
files plus a directory of thumbnails, all produced by this package from a
read-only extract of the corpus database.

See ``README.md`` for the pipeline and ``../data/SCHEMA.md`` for the published
data contract.
"""
from __future__ import annotations

SCHEMA_VERSION = "1.0.0"

__all__ = ["SCHEMA_VERSION"]
