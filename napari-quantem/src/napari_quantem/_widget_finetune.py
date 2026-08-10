"""Fine-tuning on the user's own annotations.

The workflow this implements, and why:

1. Segment normally, and look at where the result is wrong.
2. Draw a rectangle or polygon around one of those places, in a Shapes layer.
3. Make the labels inside that shape correct and complete — paint what was missed, erase what was
   invented. Everything inside the shape must be right; nothing outside it is looked at.
4. Fine-tune.

Step 2 is not a formality. Outside the drawn shape the labels are treated as *unknown*, not as
background. Without it, a user who corrects one corner of a large image is implicitly asserting the
whole rest of the image is empty, and training on that assertion makes the model worse. This is the
same contract that produced the published label-efficiency curve.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
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
    family_label,
    hline,
    image_layers,
    labels_layers,
    models_for,
    note,
    organelle_labels,
    row,
    shapes_layers,
    shapes_to_valid,
)
from ._models import confirm_download, download_reporter, get_model, unavailable_message

MIN_CV_REGIONS = 3


class FineTuneWidget(QWidget):
    #: Progress and status arrive from a worker thread. Touching a QWidget from a non-GUI
    #: thread is undefined behaviour in Qt -- it corrupts paint state and can hard-crash.
    #: Signals cross the thread boundary safely: Qt queues them onto the GUI thread.
    progressed = Signal(int)
    said = Signal(str)

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._cancel = False
        self._model = None

        lay = QVBoxLayout(self)
        lay.addWidget(
            note(
                "Correct the segmentation inside a region you draw, then adapt the model to it. "
                "Only what is inside your region is used for training."
            )
        )
        lay.addWidget(hline())

        self.image_box = QComboBox()
        self.labels_box = QComboBox()
        self.shapes_box = QComboBox()
        lay.addWidget(row(QLabel("Image"), self.image_box))
        lay.addWidget(row(QLabel("Corrected labels"), self.labels_box))
        lay.addWidget(row(QLabel("Reviewed region"), self.shapes_box))

        self.new_shapes_btn = QPushButton("New reviewed-region layer")
        self.new_shapes_btn.clicked.connect(self._new_shapes)
        lay.addWidget(self.new_shapes_btn)

        scope = QGroupBox("What did you annotate?")
        sl = QVBoxLayout(scope)
        self.scope_shapes = QRadioButton("Everything inside the shapes I drew")
        self.scope_image = QRadioButton("Everything in this entire image")
        self.scope_shapes.setChecked(True)
        sl.addWidget(self.scope_shapes)
        sl.addWidget(self.scope_image)
        lay.addWidget(scope)
        # Outside the group box: a wrapped label inside one gets clipped when the group sizes to
        # its buttons.
        lay.addWidget(
            note(
                "Outside the annotated area, labels are treated as unknown rather than as "
                "background — so a correction in one corner never teaches the model that the rest "
                "of the image is empty."
            )
        )

        lay.addWidget(hline())

        og = QGroupBox("Model")
        ogl = QVBoxLayout(og)
        self.organelle = QComboBox()
        labels = organelle_labels()
        for k in ORGANELLE_ORDER:
            self.organelle.addItem(labels[k], k)
        self.model_box = QComboBox()
        ogl.addWidget(row(QLabel("Organelle"), self.organelle))
        ogl.addWidget(row(QLabel("Base model"), self.model_box))
        self.organelle.currentIndexChanged.connect(self._organelle_changed)
        lay.addWidget(og)

        self.pixel = PixelSizeBox()
        lay.addWidget(self.pixel)

        self.steps = QSpinBox()
        self.steps.setRange(20, 5000)
        self.steps.setValue(300)
        lay.addWidget(row(QLabel("Training steps"), self.steps))

        lay.addWidget(hline())
        self.estimate = note("")
        lay.addWidget(self.estimate)

        self.run_btn = QPushButton("Fine-tune")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        lay.addWidget(row(self.run_btn, self.stop_btn))

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.status = note("")
        lay.addWidget(self.status)

        lay.addWidget(hline())
        self.cv_btn = QPushButton("Check accuracy (leave-one-region-out)")
        self.cv_btn.setEnabled(False)
        self.cv_btn.clicked.connect(self._cross_validate)
        lay.addWidget(self.cv_btn)
        self.cv_note = note(
            f"Needs {MIN_CV_REGIONS}+ regions. Trains one model per region, each time holding that "
            "region out, so the score is genuinely held out. Slow — especially without a GPU."
        )
        lay.addWidget(self.cv_note)

        self.save_btn = QPushButton("Save adapted model…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        self.load_btn = QPushButton("Load adapted model…")
        self.load_btn.clicked.connect(self._load)
        lay.addWidget(row(self.save_btn, self.load_btn))
        lay.addStretch(1)

        self.run_btn.clicked.connect(self._run)
        self.model_box.currentIndexChanged.connect(self._refresh_estimate)
        self.progressed.connect(self.bar.setValue)
        self.said.connect(self.status.setText)
        self.stop_btn.clicked.connect(lambda: setattr(self, "_cancel", True))
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        for box in (self.image_box, self.labels_box, self.shapes_box):
            box.currentIndexChanged.connect(self._refresh_estimate)
            box.currentIndexChanged.connect(self._rewire_data_events)
        self.scope_shapes.toggled.connect(self._refresh_estimate)

        #: Layers we are currently subscribed to, so the subscriptions can be moved when the
        #: selection changes instead of piling up.
        self._watched: list = []

        self._refresh_layers()
        self._rewire_data_events()
        self._organelle_changed()

    def _rewire_data_events(self, *_):
        """Follow the *contents* of the chosen shapes and labels layers, not just their existence.

        Drawing a region is the central interaction of this widget, and ``layers.events.inserted``
        does not fire for it -- that event is about a layer appearing in the list, not about its
        data changing. Without this, drawing the third region left the leave-one-region-out button
        disabled and the region count stale until the user happened to touch another control.
        """
        for ly, cb in getattr(self, "_watched", []):
            try:
                ly.events.data.disconnect(cb)
            except Exception:  # layer already gone, or never connected
                pass
        self._watched = []
        for box, getter in ((self.shapes_box, shapes_layers), (self.labels_box, labels_layers)):
            ly = self._named(box, getter)
            if ly is None:
                continue
            cb = self._refresh_estimate
            try:
                ly.events.data.connect(cb)
                self._watched.append((ly, cb))
            except Exception:
                pass

    # -- state ---------------------------------------------------------------
    def _new_shapes(self):
        ly = self.viewer.add_shapes(
            name="reviewed region",
            shape_type="rectangle",
            edge_color="#00d0ff",
            face_color="transparent",
            edge_width=4,
        )
        ly.mode = "add_rectangle"
        self._refresh_layers()
        i = self.shapes_box.findText(ly.name)
        if i >= 0:
            self.shapes_box.setCurrentIndex(i)

    def _refresh_layers(self, *_):
        for box, getter in (
            (self.image_box, image_layers),
            (self.labels_box, labels_layers),
            (self.shapes_box, shapes_layers),
        ):
            cur = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems([ly.name for ly in getter(self.viewer)])
            i = box.findText(cur)
            if i >= 0:
                box.setCurrentIndex(i)
            box.blockSignals(False)
        self._refresh_estimate()

    def _organelle_changed(self, *_):
        org = self.organelle.currentData()
        self._specs = models_for(org)
        from quantem_em.registry import DEFAULT_MODEL_FOR_ORGANELLE

        default = DEFAULT_MODEL_FOR_ORGANELLE[org]
        self._specs = sorted(self._specs, key=lambda s: s.model_id != default)
        self.model_box.clear()
        self.model_box.addItems([family_label(s) for s in self._specs])
        self._refresh_estimate()

    def _spec(self):
        i = self.model_box.currentIndex()
        return self._specs[i] if 0 <= i < len(getattr(self, "_specs", [])) else None

    def _named(self, box, getter):
        layers = getter(self.viewer)
        name = box.currentText()
        for ly in layers:
            if ly.name == name:
                return ly
        return None

    def _build_examples(self):
        """Assemble ``Example`` objects from the chosen layers. Returns ``(examples, problem)``."""
        from quantem_em.adapt import Example

        img_layer = self._named(self.image_box, image_layers)
        lab_layer = self._named(self.labels_box, labels_layers)
        if img_layer is None or lab_layer is None:
            return [], "Choose an image layer and a corrected-labels layer."

        image, _ = current_plane(img_layer, self.viewer)
        labels = np.asarray(lab_layer.data)
        if labels.ndim == 3:
            z = int(self.viewer.dims.current_step[0]) if self.viewer.dims.ndim >= 3 else 0
            labels = labels[max(0, min(z, labels.shape[0] - 1))]
        if labels.shape != image.shape:
            return [], f"Labels {labels.shape} and image {image.shape} have different shapes."

        px = self.pixel.value_nm()

        if self.scope_image.isChecked():
            valid = np.ones(image.shape, bool)
            return [Example(image, labels, valid, name="whole image", pixel_size_nm=px)], None

        sh = self._named(self.shapes_box, shapes_layers)
        if sh is None or len(sh.data) == 0:
            return [], "Draw at least one reviewed region, or say you annotated the whole image."

        # Rasterise once, not once per shape: to_labels paints every shape on each call, so doing
        # it inside the loop was O(n^2) full-image rasterisations -- and this now runs on every
        # stroke the user draws.
        try:
            stamped = np.asarray(sh.to_labels(labels_shape=image.shape))
        except Exception:
            stamped = None

        examples = []
        for i in range(len(sh.data)):
            if stamped is not None:
                one = stamped == (i + 1)
            else:
                one = shapes_to_valid(sh, image.shape)
            if not one.any():
                continue
            examples.append(Example(image, labels, one, name=f"region {i + 1}", pixel_size_nm=px))
        if not examples:
            return [], "The reviewed regions are empty."
        return examples, None

    def _refresh_estimate(self, *_):
        spec = self._spec()
        self.pixel.update_note(spec)
        examples, problem = self._build_examples()
        n = len(examples)
        self.cv_btn.setEnabled(n >= MIN_CV_REGIONS and self._model is not None)
        if problem:
            self.estimate.setText(problem)
            self.run_btn.setEnabled(False)
            return
        self.run_btn.setEnabled(True)
        area = sum(int(e.valid.sum()) for e in examples)
        self.estimate.setText(
            f"{n} region(s), {area:,} px annotated. Head-only training updates the neck and decoder."
        )

    # -- run -----------------------------------------------------------------
    def _run(self):
        spec = self._spec()
        examples, problem = self._build_examples()
        if spec is None or problem:
            return
        if not confirm_download(self, [spec]):
            return
        from napari.qt.threading import thread_worker

        steps = int(self.steps.value())
        self._cancel = False
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.bar.setVisible(True)

        @thread_worker
        def work():
            model = get_model(
                spec.model_id,
                progress=download_reporter(self.progressed.emit, self.said.emit),
            )

            def prog(done, total, sps):
                self.progressed.emit(int(100 * done / max(total, 1)))
                if done in (3, 5) or done % 25 == 0:
                    from quantem_em.device import format_duration

                    left = (total - done) * sps
                    self.said.emit(
                        f"step {done}/{total} · {sps * 1000:.0f} ms/step · "
                        f"{format_duration(left)} remaining"
                    )

            report = model.finetune(
                examples, steps=steps, progress=prog, cancel=lambda: self._cancel
            )
            return model, report

        w = work()
        w.returned.connect(self._done)
        w.errored.connect(self._failed)
        w.finished.connect(self._reset)
        w.start()

    def _done(self, payload):
        model, report = payload
        self._model = model
        cal = report.get("calibration", {})
        self.status.setText(
            f"trained {report['steps']} steps on {report['n_regions']} region(s) "
            f"({report['n_windows']} windows) in {report['train_seconds']:.1f}s · "
            f"threshold {cal.get('threshold', model.spec.fg_threshold)} "
            f"(Dice {cal.get('dice_at_threshold', float('nan')):.3f} on your training regions — "
            "not held out)"
        )
        self.save_btn.setEnabled(True)
        self._refresh_estimate()

    def _cross_validate(self):
        spec = self._spec()
        examples, problem = self._build_examples()
        if problem or len(examples) < MIN_CV_REGIONS:
            return
        # cross_validate re-loads the base model for every fold, so this can be the first thing
        # that touches the network -- and the picker may have moved since the fine-tune ran.
        if not confirm_download(self, [spec]):
            return
        from napari.qt.threading import thread_worker
        from quantem_em.api import load_model

        steps = int(self.steps.value())
        self._cancel = False
        self.cv_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)  # CV is k fine-tunes; it must be interruptible
        self.bar.setVisible(True)

        @thread_worker
        def work():
            from quantem_em.adapt import cross_validate

            def prog(fold, nfolds, done, total, sps):
                frac = (fold + done / max(total, 1)) / nfolds
                self.progressed.emit(int(100 * frac))
                self.said.emit(f"fold {fold + 1}/{nfolds} · step {done}/{total}")

            return cross_validate(
                lambda: load_model(spec.model_id),
                examples,
                steps=steps,
                progress=prog,
                cancel=lambda: self._cancel,
            )

        w = work()
        w.returned.connect(self._cv_done)
        w.errored.connect(self._failed)
        w.finished.connect(self._reset)
        w.start()

    def _cv_done(self, cv):
        if cv.get("mean_dice") is None:
            self.status.setText("cross-validation produced no score")
            return
        per = "  ".join(f"{f['held_out']}: {f['dice']:.3f}" for f in cv["folds"] if f["dice"])
        self.status.setText(
            f"held-out Dice {cv['mean_dice']:.3f} ± {cv['std_dice']:.3f} "
            f"over {cv['n_folds']} folds    [{per}]"
        )

    def _save(self):
        if self._model is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save adapted model",
            f"{self._model.spec.organelle}_adapted.safetensors",
            "safetensors (*.safetensors)",
        )
        if path:
            self._model.save_adapted(path)
            self.status.setText(f"saved {path}")

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load adapted model", "", "safetensors (*.safetensors)"
        )
        if not path:
            return
        spec = self._spec()
        if spec is None:
            return
        # An adapted head is only the neck+decoder; the base encoder still has to be present, so
        # this can trigger the first download. Ask, and do it off the GUI thread -- hashing 1.2 GB
        # in a button slot freezes napari with no progress and no cancel.
        if not confirm_download(self, [spec]):
            return

        from napari.qt.threading import thread_worker

        self.load_btn.setEnabled(False)
        self.said.emit("loading adapted head…")

        @thread_worker
        def work():
            from quantem_em.adapt import load_adapted_head

            model = get_model(
                spec.model_id,
                progress=download_reporter(self.progressed.emit, self.said.emit),
            )
            return model, load_adapted_head(model, path)

        def done(payload):
            model, meta = payload
            self._model = model
            self.status.setText(
                f"loaded adapted head for {meta.get('base_model_id')} "
                f"(threshold {meta.get('threshold')})"
            )
            self.save_btn.setEnabled(True)

        w = work()
        w.returned.connect(done)
        w.errored.connect(self._failed)
        w.finished.connect(lambda: self.load_btn.setEnabled(True))
        w.start()

    def _failed(self, exc):
        from quantem_em.weights.fetch import WeightsUnavailableError

        msg = unavailable_message(exc) if isinstance(exc, WeightsUnavailableError) else str(exc)
        self.status.setText(f"failed: {msg}")

    def _reset(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.bar.setVisible(False)
        self._refresh_estimate()
