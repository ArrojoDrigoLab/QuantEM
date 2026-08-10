"""Per-object morphometrics and per-image summaries.

The numbers come from ``quantem_em.measure``, not from this file, so the plugin and the desktop
application report identical values for the same mask.

Results are written into the Labels layer's ``features`` table, which is where napari expects
per-object data to live — so they show up in napari's own table view and follow the layer around.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ._common import PixelSizeBox, hline, image_layers, labels_layers, note, row


class MeasureWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._columns = None
        self._summary = None

        lay = QVBoxLayout(self)
        self.labels_box = QComboBox()
        self.image_box = QComboBox()
        lay.addWidget(row(QLabel("Labels"), self.labels_box))
        lay.addWidget(row(QLabel("Intensity image"), self.image_box))
        lay.addWidget(
            note("The intensity image is optional; it adds per-object intensity columns.")
        )

        self.tissue_box = QComboBox()
        lay.addWidget(row(QLabel("Tissue mask"), self.tissue_box))
        lay.addWidget(
            note(
                "Optional. When set, the area fraction is measured against the tissue area rather "
                "than the whole image."
            )
        )

        self.pixel = PixelSizeBox()
        lay.addWidget(self.pixel)
        lay.addWidget(
            note(
                "Without a pixel size, measurements are reported in pixels and the column names say so."
            )
        )

        lay.addWidget(hline())
        self.run_btn = QPushButton("Measure")
        self.run_btn.clicked.connect(self._measure)
        lay.addWidget(self.run_btn)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(170)
        lay.addWidget(self.summary)

        self.export_btn = QPushButton("Export objects to CSV…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export)
        lay.addWidget(self.export_btn)
        lay.addStretch(1)

        self.viewer.layers.events.inserted.connect(self._refresh)
        self.viewer.layers.events.removed.connect(self._refresh)
        self._refresh()

    def _refresh(self, *_):
        for box, getter, blank in (
            (self.labels_box, labels_layers, False),
            (self.image_box, image_layers, True),
            (self.tissue_box, labels_layers, True),
        ):
            cur = box.currentText()
            box.blockSignals(True)
            box.clear()
            if blank:
                box.addItem("(none)")
            box.addItems([ly.name for ly in getter(self.viewer)])
            i = box.findText(cur)
            if i >= 0:
                box.setCurrentIndex(i)
            box.blockSignals(False)

    def _named(self, box, getter):
        name = box.currentText()
        if name in ("", "(none)"):
            return None
        for ly in getter(self.viewer):
            if ly.name == name:
                return ly
        return None

    def _plane(self, layer):
        if layer is None:
            return None
        data = np.asarray(layer.data)
        if data.ndim == 2:
            return data
        z = int(self.viewer.dims.current_step[0]) if self.viewer.dims.ndim >= 3 else 0
        return data[max(0, min(z, data.shape[0] - 1))]

    def _measure(self):
        from quantem_em.measure import measure_objects, summarize

        lab_layer = self._named(self.labels_box, labels_layers)
        if lab_layer is None:
            self.summary.setPlainText("Choose a labels layer.")
            return
        labels = self._plane(lab_layer)
        image = self._plane(self._named(self.image_box, image_layers))
        tissue = self._plane(self._named(self.tissue_box, labels_layers))
        px = self.pixel.value_nm()

        try:
            cols = measure_objects(labels, image, pixel_size_nm=px)
            summ = summarize(
                labels, pixel_size_nm=px, tissue_mask=(tissue > 0) if tissue is not None else None
            )
        except Exception as e:
            # Drop the previous result too: leaving Export enabled after a failure lets the user
            # save the PREVIOUS layer's table under the new layer's name.
            self._columns, self._summary = None, None
            self.export_btn.setEnabled(False)
            self.summary.setPlainText(f"failed: {e}")
            return

        self._columns, self._summary = cols, summ
        n = len(cols.get("label", []))
        if n:
            try:
                lab_layer.features = {k: np.asarray(v) for k, v in cols.items()}
            except Exception:
                pass  # older napari, or a feature-table shape mismatch: the CSV still works

        unit = summ["units"]
        lines = [
            f"objects:            {summ['n_objects']}",
            f"total object area:  {summ[f'total_object_area_{unit}2']:,.1f} {unit}²",
            f"image area:         {summ[f'image_area_{unit}2']:,.1f} {unit}²"
            f"  ({summ['area_fraction_denominator'].replace('_', ' ')})",
            f"area fraction:      {summ['area_fraction'] * 100:.2f} %",
            f"mean object area:   {summ[f'mean_object_area_{unit}2']:,.1f} {unit}²",
            f"median object area: {summ[f'median_object_area_{unit}2']:,.1f} {unit}²",
            "",
            f"{len(cols)} per-object columns written to the layer's feature table.",
        ]
        if px is None:
            lines.append("Units are PIXELS — set a pixel size for physical units.")
        self.summary.setPlainText("\n".join(lines))
        self.export_btn.setEnabled(n > 0)

    def _export(self):
        from quantem_em.measure import to_csv

        if not self._columns:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export measurements", "objects.csv", "CSV (*.csv)"
        )
        if path:
            to_csv(self._columns, path)
            self.summary.append(f"\nwrote {path}")
