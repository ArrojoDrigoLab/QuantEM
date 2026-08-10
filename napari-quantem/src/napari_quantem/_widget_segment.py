"""The segmentation widget — pick an image, pick an organelle, run."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._common import (
    ORGANELLE_ORDER,
    PixelSizeBox,
    current_plane,
    default_model_id,
    family_label,
    hline,
    image_layers,
    models_for,
    note,
    organelle_labels,
    row,
)
from ._models import confirm_download, download_reporter, get_model, status_for, unavailable_message


class SegmentWidget(QWidget):
    #: Progress and status arrive from a worker thread. Touching a QWidget from a non-GUI
    #: thread is undefined behaviour in Qt -- it corrupts paint state and can hard-crash.
    #: Signals cross the thread boundary safely: Qt queues them onto the GUI thread.
    progressed = Signal(int)
    said = Signal(str)

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker = None
        self._cancel = False

        lay = QVBoxLayout(self)

        # -- input ----------------------------------------------------------
        self.layer_box = QComboBox()
        lay.addWidget(row(QLabel("Image"), self.layer_box))

        # -- organelle ------------------------------------------------------
        og = QGroupBox("Organelle")
        ogl = QVBoxLayout(og)
        labels = organelle_labels()
        self.organelle_buttons = {}
        for key in ORGANELLE_ORDER:
            rb = QRadioButton(labels[key])
            self.organelle_buttons[key] = rb
            ogl.addWidget(rb)
        # Signals are connected at the end of __init__: setChecked fires `toggled`, and the handler
        # needs every widget it touches to exist already.
        self.organelle_buttons["mito"].setChecked(True)
        lay.addWidget(og)

        # -- model ----------------------------------------------------------
        self.model_box = QComboBox()
        lay.addWidget(row(QLabel("Model"), self.model_box))
        self.model_note = note("")
        lay.addWidget(self.model_note)

        lay.addWidget(hline())

        # -- pixel size -----------------------------------------------------
        self.pixel = PixelSizeBox(on_change=self._refresh_estimate)
        lay.addWidget(self.pixel)

        # -- options --------------------------------------------------------
        opts = QGroupBox("Options")
        ol = QVBoxLayout(opts)
        self.invert = QCheckBox("Invert contrast (dark structures on light background)")
        ol.addWidget(self.invert)
        self.prob_layer = QCheckBox("Also add the probability map")
        ol.addWidget(self.prob_layer)

        self.slices = QComboBox()
        self.slices.addItems(["current slice", "all slices", "z range"])
        self.z_from, self.z_to = QSpinBox(), QSpinBox()
        for sb in (self.z_from, self.z_to):
            sb.setRange(0, 100000)
        self.slice_row = row(QLabel("Stack"), self.slices)
        ol.addWidget(self.slice_row)
        self.z_row = row(QLabel("z from"), self.z_from, QLabel("to"), self.z_to)
        ol.addWidget(self.z_row)
        self.link_z = QCheckBox("Join labels through z after running (not 3-D segmentation)")
        ol.addWidget(self.link_z)
        lay.addWidget(opts)

        # -- run ------------------------------------------------------------
        self.estimate = note("")
        lay.addWidget(self.estimate)
        self.run_btn = QPushButton("Run")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        lay.addWidget(row(self.run_btn, self.stop_btn))
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.status = note("")
        lay.addWidget(self.status)
        lay.addStretch(1)

        # -- signals, now that every widget exists --------------------------
        self.run_btn.clicked.connect(self._run)
        self.stop_btn.clicked.connect(self._stop)
        self.progressed.connect(self.bar.setValue)
        self.said.connect(self.status.setText)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.layer_box.currentIndexChanged.connect(self._refresh_estimate)
        self.model_box.currentIndexChanged.connect(self._refresh_estimate)
        self.slices.currentIndexChanged.connect(self._refresh_estimate)
        self.z_from.valueChanged.connect(self._refresh_estimate)
        self.z_to.valueChanged.connect(self._refresh_estimate)
        for rb in self.organelle_buttons.values():
            rb.toggled.connect(self._organelle_changed)

        self._refresh_layers()
        self._organelle_changed()

    # -- state ---------------------------------------------------------------
    def _refresh_layers(self, *_):
        cur = self.layer_box.currentText()
        self.layer_box.blockSignals(True)
        self.layer_box.clear()
        self._layers = image_layers(self.viewer)
        self.layer_box.addItems([ly.name for ly in self._layers])
        if cur:
            i = self.layer_box.findText(cur)
            if i >= 0:
                self.layer_box.setCurrentIndex(i)
        self.layer_box.blockSignals(False)
        self._refresh_estimate()

    def _organelle(self) -> str:
        for k, rb in self.organelle_buttons.items():
            if rb.isChecked():
                return k
        return "mito"

    def _organelle_changed(self, *_):
        org = self._organelle()
        specs = models_for(org)
        default = default_model_id(org)
        self.model_box.blockSignals(True)
        self.model_box.clear()
        self._specs = sorted(specs, key=lambda s: s.model_id != default)
        for s in self._specs:
            self.model_box.addItem(f"{family_label(s)} — {status_for(s)}")
        self.model_box.blockSignals(False)
        self._refresh_estimate()

    def _spec(self):
        i = self.model_box.currentIndex()
        return self._specs[i] if 0 <= i < len(getattr(self, "_specs", [])) else None

    def _layer(self):
        i = self.layer_box.currentIndex()
        return self._layers[i] if 0 <= i < len(self._layers) else None

    def _busy(self) -> bool:
        return getattr(self, "_worker", None) is not None

    def _refresh_estimate(self, *_):
        spec = self._spec()
        layer = self._layer()
        self.pixel.update_note(spec)
        if spec is None or layer is None:
            self.estimate.setText("")
            self.run_btn.setEnabled(False)
            return
        # Never re-enable Run while a worker is live. This method is wired to layer events, so
        # adding a layer mid-run used to re-arm the button -- and a second worker would then share
        # the cached model, which is explicitly not thread-safe.
        self.run_btn.setEnabled(not self._busy())

        # .ndim/.shape only: np.asarray() here would materialise a lazily-loaded dask/zarr volume
        # on the GUI thread, on every keystroke in the pixel-size box.
        data = layer.data
        is_stack = data.ndim == 3
        self.slice_row.setVisible(is_stack)
        self.link_z.setVisible(is_stack)
        self.z_row.setVisible(is_stack and self.slices.currentIndex() == 2)
        if is_stack:
            nz = int(data.shape[0])
            for sb in (self.z_from, self.z_to):
                sb.setMaximum(nz - 1)
            if self.z_to.value() == 0:
                self.z_to.setValue(nz - 1)
        hw = tuple(data.shape[-2:])

        try:
            from quantem_em.api import QuantEMModel  # noqa: F401
            from quantem_em.inference.predict import plan_resample
            from quantem_em.inference.tiling import round_up, stride_for, window_count

            factors, info = plan_resample(spec, hw, self.pixel.value_nm(), allow_extreme=True)
            work = info.get("working_shape", hw) if factors else hw
            t = round_up(spec.tile_size, spec.encoder.patch_size)
            padded = (max(work[0], t), max(work[1], t))
            n = window_count(padded, t, stride_for(t, spec.overlap))
            peak = padded[0] * padded[1] * 17 / 1e9
            if is_stack and self.slices.currentIndex() == 1:
                n_slices = data.shape[0]
            elif is_stack and self.slices.currentIndex() == 2:
                n_slices = abs(self.z_to.value() - self.z_from.value()) + 1
            else:
                n_slices = 1
            self.estimate.setText(
                f"{n * n_slices} tile(s) · working size {work[0]}×{work[1]} · "
                f"peak memory ≈ {peak:.2f} GB" + (f" · {n_slices} slices" if n_slices > 1 else "")
            )
        except Exception as e:  # pragma: no cover - defensive
            self.estimate.setText(f"(estimate unavailable: {e})")

    # -- run -----------------------------------------------------------------
    def _stop(self):
        self._cancel = True
        self.status.setText("stopping…")

    def _run(self):
        spec = self._spec()
        layer = self._layer()
        if spec is None or layer is None:
            return
        if not confirm_download(self, [spec]):
            return

        from napari.qt.threading import thread_worker

        data = np.asarray(layer.data)
        is_stack = data.ndim == 3
        all_slices = is_stack and self.slices.currentIndex() >= 1
        pixel_nm = self.pixel.value_nm()
        invert = self.invert.isChecked()
        self._cancel = False

        if is_stack and self.slices.currentIndex() == 2:
            lo = min(self.z_from.value(), self.z_to.value())
            hi = max(self.z_from.value(), self.z_to.value())
            hi = min(hi, data.shape[0] - 1)
            planes = [(z, data[z]) for z in range(lo, hi + 1)]
        elif is_stack and not all_slices:
            plane, z = current_plane(layer, self.viewer)
            planes = [(z or 0, plane)]
        elif is_stack:
            planes = list(enumerate(data))
        else:
            planes = [(None, data)]

        # Captured now, used by _done. Re-reading the combo boxes when the result arrives means
        # a user who switches organelle mid-run gets the new organelle's name and min_area stamped
        # onto the old organelle's data.
        self._active = (spec, layer)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.bar.setVisible(True)
        self.bar.setValue(0)

        @thread_worker
        def work():
            model = get_model(
                spec.model_id,
                progress=download_reporter(self.progressed.emit, self.said.emit),
            )
            total_planes = len(planes)
            out_labels, out_prob, contract = None, None, None
            for pi, (_z, plane) in enumerate(planes):

                def prog(done, tot, _pi=pi):
                    frac = (_pi + done / max(tot, 1)) / total_planes
                    self.progressed.emit(int(frac * 100))

                res = model.segment(
                    plane,
                    pixel_size_nm=pixel_nm,
                    invert=invert,
                    progress=prog,
                    cancel=lambda: self._cancel,
                )
                contract = res.contract
                if total_planes == 1:
                    out_labels, out_prob = res.labels, res.probability
                else:
                    if out_labels is None:
                        out_labels = np.zeros((total_planes, *res.labels.shape), np.int32)
                        out_prob = np.zeros((total_planes, *res.probability.shape), np.float32)
                    out_labels[pi] = res.labels
                    out_prob[pi] = res.probability
            return out_labels, out_prob, contract, (total_planes > 1)

        w = work()
        w.returned.connect(self._done)
        w.errored.connect(self._failed)
        w.finished.connect(self._reset)
        self._worker = w
        w.start()

    def _done(self, payload):
        labels, prob, contract, stacked = payload
        if labels is None:
            return
        spec, layer = getattr(self, "_active", (None, None))
        if spec is None or layer is None:
            return
        name = f"{spec.organelle} · {'QuantEM' if spec.family == 'quantem' else 'OmniEM'}"

        if stacked and self.link_z.isChecked():
            from quantem_em.inference.postprocess import link_across_z

            labels, n = link_across_z(labels > 0, min_area=spec.min_area)
            contract = dict(contract, z_linked=True, n_objects_3d=n)

        kw = {}
        if layer is not None:
            if len(layer.scale) == labels.ndim:
                kw["scale"] = layer.scale
            if len(layer.translate) == labels.ndim:
                kw["translate"] = layer.translate
        self.viewer.add_labels(labels, name=name, metadata={"quantem": contract}, **kw)
        if self.prob_layer.isChecked():
            self.viewer.add_image(
                prob,
                name=f"{name} probability",
                colormap="magma",
                contrast_limits=(0, 1),
                blending="additive",
                metadata={"quantem": contract},
                **kw,
            )
        n = int(labels.max())
        self.status.setText(
            f"{n} object(s) · threshold {contract.get('threshold', '?')} "
            f"· {contract.get('reason', 'resampled' if contract.get('resampled') else '')}"
        )

    def _failed(self, exc):
        from quantem_em.weights.fetch import WeightsUnavailableError

        msg = unavailable_message(exc) if isinstance(exc, WeightsUnavailableError) else str(exc)
        self.status.setText(f"failed: {msg}")
        try:
            from napari.utils.notifications import show_error

            show_error(f"QuantEM: {msg}")
        except Exception:
            pass

    def _reset(self):
        self._worker = None  # cleared first: _refresh_estimate consults it
        self.stop_btn.setEnabled(False)
        self.bar.setVisible(False)
        self._refresh_estimate()
        self._organelle_changed()
