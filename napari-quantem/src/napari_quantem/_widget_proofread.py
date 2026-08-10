"""Proofreading tools on top of napari's own Labels editing.

napari already gives you paint, erase, fill, pick, brush size and contour mode. What it does not
give you is the object-level surgery that segmentation clean-up actually needs, so that is what
this adds — split, merge, delete, filter by size, remove edge-touching objects, and the same
morphological operations the inference post-processing uses.

Everything here writes back into the same Labels layer the fine-tuning widget reads, so the
correct-then-adapt loop closes without an export step.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._common import hline, labels_layers, note, row


class ProofreadWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        lay = QVBoxLayout(self)

        self.labels_box = QComboBox()
        lay.addWidget(row(QLabel("Labels"), self.labels_box))
        self.info = note("")
        lay.addWidget(self.info)
        lay.addWidget(hline())

        # -- object surgery -------------------------------------------------
        g1 = QGroupBox("Objects")
        l1 = QVBoxLayout(g1)
        b_merge = QPushButton("Merge selected + touching")
        b_merge.setToolTip(
            "Give every object that touches the selected label the selected label's id."
        )
        b_split = QPushButton("Split selected (watershed)")
        b_split.setToolTip("Split the selected object at its narrowest necks.")
        b_del = QPushButton("Delete selected")
        for b in (b_merge, b_split, b_del):
            l1.addWidget(b)
        b_merge.clicked.connect(self._merge)
        b_split.clicked.connect(self._split)
        b_del.clicked.connect(self._delete)
        lay.addWidget(g1)

        # -- filters --------------------------------------------------------
        g2 = QGroupBox("Filter")
        l2 = QVBoxLayout(g2)
        self.min_area = QSpinBox()
        self.min_area.setRange(0, 10_000_000)
        self.min_area.setValue(100)
        self.min_area.setSuffix(" px")
        b_small = QPushButton("Remove objects smaller than")
        b_small.clicked.connect(self._remove_small)
        l2.addWidget(row(b_small, self.min_area))
        b_edge = QPushButton("Remove objects touching the border")
        b_edge.clicked.connect(self._remove_edge)
        l2.addWidget(b_edge)
        lay.addWidget(g2)

        # -- morphology -----------------------------------------------------
        g3 = QGroupBox("Shape")
        l3 = QVBoxLayout(g3)
        self.radius = QSpinBox()
        self.radius.setRange(1, 50)
        self.radius.setValue(2)
        self.radius.setSuffix(" px")
        self.op = QComboBox()
        self.op.addItems(["fill holes", "close", "open", "dilate", "erode"])
        b_apply = QPushButton("Apply")
        b_apply.clicked.connect(self._morph)
        l3.addWidget(row(self.op, self.radius, b_apply))
        lay.addWidget(g3)

        b_compact = QPushButton("Renumber labels 1..N")
        b_compact.clicked.connect(self._compact)
        lay.addWidget(b_compact)

        self.status = note("")
        lay.addWidget(self.status)
        lay.addStretch(1)

        self.viewer.layers.events.inserted.connect(self._refresh)
        self.viewer.layers.events.removed.connect(self._refresh)
        self.labels_box.currentIndexChanged.connect(self._refresh_info)
        self.labels_box.currentIndexChanged.connect(self._rewire_data_events)
        self._watched: list = []
        self._refresh()
        self._rewire_data_events()

    def _rewire_data_events(self, *_):
        """Track the chosen layer's contents, so the object count reflects the user's painting.

        The count is the only feedback this widget gives while someone paints or erases; leaving it
        frozen until they change the layer selection makes it worse than no count at all.
        """
        for ly, cb in getattr(self, "_watched", []):
            try:
                ly.events.data.disconnect(cb)
            except Exception:  # layer already gone, or never connected
                pass
        self._watched = []
        ly = self._layer()
        if ly is not None:
            try:
                ly.events.data.connect(self._refresh_info)
                self._watched.append((ly, self._refresh_info))
            except Exception:
                pass

    # -- helpers -------------------------------------------------------------
    def _refresh(self, *_):
        cur = self.labels_box.currentText()
        self.labels_box.blockSignals(True)
        self.labels_box.clear()
        self.labels_box.addItems([ly.name for ly in labels_layers(self.viewer)])
        i = self.labels_box.findText(cur)
        if i >= 0:
            self.labels_box.setCurrentIndex(i)
        self.labels_box.blockSignals(False)
        self._refresh_info()

    def _layer(self):
        name = self.labels_box.currentText()
        for ly in labels_layers(self.viewer):
            if ly.name == name:
                return ly
        return None

    def _refresh_info(self, *_):
        ly = self._layer()
        if ly is None:
            self.info.setText("")
            return
        data = np.asarray(ly.data)
        n = int(data.max())
        self.info.setText(f"{n} object(s) · selected label {ly.selected_label}")

    def _plane(self, ly):
        """Return (2-D view, setter). Proofreading acts on the visible slice for stacks."""
        data = np.asarray(ly.data)
        if data.ndim == 2:
            return data, lambda new: setattr(ly, "data", new)
        z = int(self.viewer.dims.current_step[0]) if self.viewer.dims.ndim >= 3 else 0
        z = max(0, min(z, data.shape[0] - 1))

        def setter(new):
            d = np.asarray(ly.data).copy()
            d[z] = new
            ly.data = d

        return data[z], setter

    def _apply(self, fn, msg):
        ly = self._layer()
        if ly is None:
            return
        plane, setter = self._plane(ly)
        try:
            new = fn(np.asarray(plane).astype(np.int32))
        except Exception as e:
            self.status.setText(f"failed: {e}")
            return
        setter(new)
        self.status.setText(msg() if callable(msg) else msg)
        self._refresh_info()

    # -- operations ----------------------------------------------------------
    def _merge(self):
        from scipy import ndimage as ndi

        ly = self._layer()
        if ly is None:
            return
        sel = int(ly.selected_label)

        merged = []

        def fn(lab):
            if sel == 0 or not (lab == sel).any():
                raise ValueError("select an existing label first")
            # 8-connectivity and two steps, because connected components are labelled with
            # 4-connectivity: no two distinct labels in a segmentation can ever be 4-adjacent, so a
            # single cross-shaped dilation could only ever find the object itself. Two steps of a
            # 3x3 also bridge the one-pixel seam that splits an object in the first place.
            structure = np.ones((3, 3), bool)
            grown = ndi.binary_dilation(lab == sel, structure=structure, iterations=2)
            touching = set(np.unique(lab[grown])) - {0, sel}
            for t in touching:
                lab[lab == t] = sel
            merged.extend(sorted(int(t) for t in touching))
            return lab

        self._apply(
            fn,
            lambda: (
                f"merged {len(merged)} label(s) into {sel}: {merged}"
                if merged
                else f"nothing adjacent to label {sel} to merge"
            ),
        )

    def _split(self):
        from scipy import ndimage as ndi
        from skimage.segmentation import watershed

        ly = self._layer()
        if ly is None:
            return
        sel = int(ly.selected_label)

        def fn(lab):
            m = lab == sel
            if not m.any():
                raise ValueError("select an existing label first")
            dist = ndi.distance_transform_edt(m)
            from skimage.feature import peak_local_max

            coords = peak_local_max(dist, labels=m, min_distance=7, exclude_border=False)
            if len(coords) < 2:
                raise ValueError("no obvious split point in this object")
            markers = np.zeros_like(lab)
            nxt = int(lab.max())
            for i, (r, c) in enumerate(coords):
                markers[r, c] = sel if i == 0 else nxt + i
            parts = watershed(-dist, markers, mask=m)
            lab[m] = parts[m]
            return lab

        self._apply(fn, f"split label {sel}")

    def _delete(self):
        ly = self._layer()
        if ly is None:
            return
        sel = int(ly.selected_label)
        self._apply(lambda lab: np.where(lab == sel, 0, lab), f"deleted label {sel}")

    def _remove_small(self):

        thr = int(self.min_area.value())

        def fn(lab):
            # Filter the labels the user actually has. label_objects(lab > 0) would re-derive
            # components from the binary union, silently fusing anything the user had split apart
            # and measuring merged blobs against the threshold instead of individual objects.
            ids, counts = np.unique(lab[lab > 0], return_counts=True)
            drop = ids[counts < thr]
            if len(drop):
                lab[np.isin(lab, drop)] = 0
            return lab

        self._apply(fn, f"removed objects smaller than {thr} px")

    def _remove_edge(self):
        def fn(lab):
            edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
            edge.discard(0)
            for e in edge:
                lab[lab == e] = 0
            return lab

        self._apply(fn, "removed border-touching objects")

    def _morph(self):
        from quantem_em.inference.postprocess import _disk, label_objects
        from scipy import ndimage as ndi

        op = self.op.currentText()
        r = int(self.radius.value())
        st = _disk(r)

        def fn(lab):
            m = lab > 0
            if op == "fill holes":
                m = ndi.binary_fill_holes(m)
            elif op == "close":
                m = ndi.binary_closing(m, structure=st)
            elif op == "open":
                m = ndi.binary_opening(m, structure=st)
            elif op == "dilate":
                m = ndi.binary_dilation(m, structure=st)
            elif op == "erode":
                m = ndi.binary_erosion(m, structure=st)
            new, _ = label_objects(m, min_area=0)
            return new

        self._apply(fn, f"{op} (radius {r})")

    def _compact(self):
        def fn(lab):
            # Renumber, not re-segment. Deriving components from lab > 0 would merge every pair of
            # touching labels -- undoing each watershed split the user just made.
            ids = np.unique(lab)
            ids = ids[ids > 0]
            out = np.zeros_like(lab)
            for new_id, old_id in enumerate(ids, start=1):
                out[lab == old_id] = new_id
            return out

        self._apply(fn, "renumbered")
