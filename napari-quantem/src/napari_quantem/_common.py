"""Shared widget helpers: layer discovery, pixel size, model choice, and the run estimate."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

ORGANELLE_ORDER = ("mito", "er", "nucleus", "ld")


def organelle_labels() -> dict:
    from quantem_em.registry import ORGANELLE_LABELS

    return ORGANELLE_LABELS


def models_for(organelle: str) -> list:
    from quantem_em.registry import REGISTRY

    return [s for s in REGISTRY.values() if s.organelle == organelle]


def default_model_id(organelle: str) -> str:
    from quantem_em.registry import DEFAULT_MODEL_FOR_ORGANELLE

    return DEFAULT_MODEL_FOR_ORGANELLE[organelle]


def family_label(spec) -> str:
    """What the model picker shows. No benchmark numbers -- users judge on their own data."""
    enc = "ViT-B" if spec.family == "quantem" else "ViT-L"
    return f"{'QuantEM' if spec.family == 'quantem' else 'OmniEM'} {enc}"


# --------------------------------------------------------------------------- layers


def image_layers(viewer):
    from napari.layers import Image

    return [ly for ly in viewer.layers if isinstance(ly, Image) and ly.data.ndim in (2, 3)]


def labels_layers(viewer):
    from napari.layers import Labels

    return [ly for ly in viewer.layers if isinstance(ly, Labels)]


def shapes_layers(viewer):
    from napari.layers import Shapes

    return [ly for ly in viewer.layers if isinstance(ly, Shapes)]


def current_plane(layer, viewer) -> tuple[np.ndarray, int | None]:
    """The 2-D array a user is looking at, plus its z index when the layer is a stack."""
    data = np.asarray(layer.data)
    if data.ndim == 2:
        return data, None
    z = int(viewer.dims.current_step[0]) if viewer.dims.ndim >= 3 else 0
    z = max(0, min(z, data.shape[0] - 1))
    return data[z], z


def shapes_to_valid(shapes_layer, shape_hw) -> np.ndarray:
    """Rasterise a Shapes layer to the boolean "reviewed region" mask.

    A user-drawn polygon inside which the annotation is complete, so background there is real
    background rather than "not looked at".
    """
    if shapes_layer is None or len(shapes_layer.data) == 0:
        return np.zeros(shape_hw, dtype=bool)
    lab = shapes_layer.to_labels(labels_shape=tuple(shape_hw))
    return np.asarray(lab) > 0


# --------------------------------------------------------------------------- widgets


def hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFrameShadow(QFrame.Sunken)
    return f


def note(text: str = "", *, warn: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setTextFormat(Qt.PlainText)
    lbl.setStyleSheet("color: #d08770;" if warn else "color: palette(mid); font-size: 11px;")
    return lbl


def row(*widgets) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        lay.addWidget(x)
    return w


class PixelSizeBox(QWidget):
    """Pixel size entry. Always visible, empty by default, never inferred.

    Owner ruling: the plugin does not read pixel size from file metadata. When the field is empty,
    nothing is rescaled and the note says so, because running a model at native resolution when it
    was trained at a fixed nm/px is a real accuracy difference the user should see.
    """

    def __init__(self, on_change=None):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.enabled = QComboBox()
        self.enabled.addItems(["not specified", "specify nm/px"])
        self.value = QDoubleSpinBox()
        self.value.setRange(0.01, 100000.0)
        self.value.setDecimals(3)
        self.value.setValue(2.0)
        self.value.setSuffix(" nm/px")
        self.value.setEnabled(False)

        lay.addWidget(row(QLabel("Pixel size"), self.enabled, self.value))
        self.note = note("")
        lay.addWidget(self.note)

        self.enabled.currentIndexChanged.connect(self._toggle)
        if on_change:
            self.enabled.currentIndexChanged.connect(lambda *_: on_change())
            self.value.valueChanged.connect(lambda *_: on_change())

    def _toggle(self, idx):
        self.value.setEnabled(idx == 1)

    def value_nm(self) -> float | None:
        return float(self.value.value()) if self.enabled.currentIndex() == 1 else None

    def update_note(self, spec) -> None:
        if spec is None:
            self.note.setText("")
            return
        if spec.canonical_nm is None:
            self.setEnabled(False)
            self.note.setText("This model runs at native resolution — no pixel size needed.")
            return
        self.setEnabled(True)
        v = self.value_nm()
        if v is None:
            self.note.setText(
                f"No pixel size given, so the image is used as-is. This model was trained at "
                f"{spec.canonical_nm:g} nm/px; supplying a pixel size usually improves results."
            )
        else:
            f = v / spec.canonical_nm
            # The core treats |factor - 1| < 1e-3 as a no-op, so say that rather than "1.00x
            # upsample", which reads as though work is being done.
            if abs(f - 1.0) < 1e-3:
                self.note.setText(f"Already at {spec.canonical_nm:g} nm/px — no rescaling needed.")
                return
            kind = "downsample" if f < 1 else "upsample"
            self.note.setText(
                f"{v:g} → {spec.canonical_nm:g} nm/px  ({f:.2f}× {kind})"
                + ("   ⚠ large upsamples degrade accuracy" if f > 2.0 else "")
            )
