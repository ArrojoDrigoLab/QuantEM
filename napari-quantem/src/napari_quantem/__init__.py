"""QuantEM organelle segmentation for napari.

Import-light on purpose: napari imports plugin top-levels during manifest discovery, so nothing
here may pull in torch or quantem_em's model layer. Widgets import what they need when they are
constructed, and a broken torch install therefore fails when a widget is opened -- with a readable
message -- rather than preventing napari from starting.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
