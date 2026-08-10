"""Batch processing: point at a folder, segment everything, write labels and one measurements CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ._common import (
    ORGANELLE_ORDER,
    PixelSizeBox,
    family_label,
    hline,
    models_for,
    note,
    organelle_labels,
    row,
)
from ._models import confirm_download, download_reporter, get_model, unavailable_message

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


class BatchWidget(QWidget):
    #: Progress and status arrive from a worker thread. Touching a QWidget from a non-GUI
    #: thread is undefined behaviour in Qt -- it corrupts paint state and can hard-crash.
    #: Signals cross the thread boundary safely: Qt queues them onto the GUI thread.
    progressed = Signal(int)
    said = Signal(str)

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._cancel = False

        lay = QVBoxLayout(self)

        self.in_dir = QLineEdit()
        b_in = QPushButton("Choose…")
        b_in.clicked.connect(lambda: self._pick(self.in_dir))
        lay.addWidget(row(QLabel("Input folder"), self.in_dir, b_in))

        self.out_dir = QLineEdit()
        b_out = QPushButton("Choose…")
        b_out.clicked.connect(lambda: self._pick(self.out_dir))
        lay.addWidget(row(QLabel("Output folder"), self.out_dir, b_out))

        self.pattern = QLineEdit("*")
        lay.addWidget(row(QLabel("Filename pattern"), self.pattern))
        self.found = note("")
        lay.addWidget(self.found)

        lay.addWidget(hline())

        self.organelle = QComboBox()
        labels = organelle_labels()
        for k in ORGANELLE_ORDER:
            self.organelle.addItem(labels[k], k)
        self.model_box = QComboBox()
        lay.addWidget(row(QLabel("Organelle"), self.organelle))
        lay.addWidget(row(QLabel("Model"), self.model_box))
        self.organelle.currentIndexChanged.connect(self._organelle_changed)

        self.pixel = PixelSizeBox()
        lay.addWidget(self.pixel)
        lay.addWidget(note("One pixel size is applied to every image in the batch."))

        self.save_prob = QCheckBox("Also write the probability map")
        self.measure = QCheckBox("Write a combined measurements CSV")
        self.measure.setChecked(True)
        self.skip_existing = QCheckBox("Skip images that already have an output")
        self.skip_existing.setChecked(True)
        for c in (self.save_prob, self.measure, self.skip_existing):
            lay.addWidget(c)

        lay.addWidget(hline())
        self.run_btn = QPushButton("Run batch")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        lay.addWidget(row(self.run_btn, self.stop_btn))
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log)

        self.run_btn.clicked.connect(self._run)
        self.progressed.connect(self.bar.setValue)
        self.said.connect(self._set_last_line)
        self.stop_btn.clicked.connect(lambda: setattr(self, "_cancel", True))
        self.in_dir.textChanged.connect(self._scan)
        self.pattern.textChanged.connect(self._scan)
        self._organelle_changed()

    def _set_last_line(self, text):
        """Replace the last log line rather than appending, so a byte-by-byte download does not
        scroll thousands of near-identical lines past the user."""
        lines = self.log.toPlainText().splitlines()
        if lines and lines[-1].startswith("Downloading "):
            lines[-1] = text
        else:
            lines.append(text)
        self.log.setPlainText("\n".join(lines))
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _pick(self, line):
        d = QFileDialog.getExistingDirectory(self, "Choose folder")
        if d:
            line.setText(d)

    def _organelle_changed(self, *_):
        org = self.organelle.currentData()
        from quantem_em.registry import DEFAULT_MODEL_FOR_ORGANELLE

        self._specs = sorted(
            models_for(org), key=lambda s: s.model_id != DEFAULT_MODEL_FOR_ORGANELLE[org]
        )
        self.model_box.clear()
        self.model_box.addItems([family_label(s) for s in self._specs])

    def _spec(self):
        i = self.model_box.currentIndex()
        return self._specs[i] if 0 <= i < len(self._specs) else None

    def _files(self):
        d = Path(self.in_dir.text().strip() or ".")
        if not d.is_dir():
            return []
        pat = self.pattern.text().strip() or "*"
        return sorted(p for p in d.glob(pat) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

    def _scan(self, *_):
        n = len(self._files())
        self.found.setText(f"{n} image file(s) found" if n else "no matching image files")

    def _run(self):
        spec = self._spec()
        files = self._files()
        out = Path(self.out_dir.text().strip() or "")
        if spec is None or not files or not out:
            self.log.append("Choose an input folder with images and an output folder.")
            return
        if not confirm_download(self, [spec]):
            return

        from napari.qt.threading import thread_worker

        out.mkdir(parents=True, exist_ok=True)
        root = Path(self.in_dir.text().strip() or ".")
        px = self.pixel.value_nm()
        save_prob = self.save_prob.isChecked()
        do_measure = self.measure.isChecked()
        skip = self.skip_existing.isChecked()
        self._cancel = False
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.bar.setVisible(True)
        self.log.clear()

        @thread_worker
        def work():
            import tifffile
            from quantem_em.measure import measure_objects, summarize, to_csv

            model = get_model(
                spec.model_id,
                progress=download_reporter(self.progressed.emit, self.said.emit),
            )
            rows: dict[str, list] = {}
            summaries = []
            n_ok = n_fail = 0

            for i, path in enumerate(files):
                if self._cancel:
                    yield f"stopped after {i} file(s)"
                    break
                # Flatten the path relative to the input root, not just the stem: a recursive
                # pattern over s1/img.tif and s2/img.tif would otherwise write both to
                # img_mito_labels.tif -- and with "skip existing" on (the default) the second
                # image is silently skipped entirely, which reads in the log as a clean resume.
                try:
                    rel = path.relative_to(root)
                except ValueError:
                    rel = Path(path.name)
                stem = "__".join(rel.with_suffix("").parts)
                dest = out / f"{stem}_{spec.organelle}_labels.tif"
                if skip and dest.exists():
                    yield f"skip  {path.name} (output exists)"
                    continue
                try:
                    img = (
                        tifffile.imread(str(path))
                        if path.suffix.lower() in (".tif", ".tiff")
                        else _imread_any(path)
                    )
                    if img.ndim == 3 and img.shape[-1] not in (3, 4):
                        img = img[img.shape[0] // 2]  # a stack: take the middle plane
                    res = model.segment(img, pixel_size_nm=px, cancel=lambda: self._cancel)
                    tifffile.imwrite(str(dest), res.labels.astype(np.int32))
                    if save_prob:
                        tifffile.imwrite(
                            str(out / f"{stem}_{spec.organelle}_prob.tif"),
                            res.probability.astype(np.float32),
                        )
                    if do_measure:
                        cols = measure_objects(
                            res.labels, img if img.ndim == 2 else None, pixel_size_nm=px
                        )
                        n = len(cols.get("label", []))
                        if n:
                            cols = {"image": np.array([str(rel)] * n), **cols}
                            for k, v in cols.items():
                                rows.setdefault(k, []).extend(list(v))
                        s = summarize(res.labels, pixel_size_nm=px)
                        summaries.append({"image": str(rel), **s})
                    n_ok += 1
                    yield f"ok    {path.name}  ({res.n_objects} objects)"
                except Exception as e:
                    n_fail += 1
                    yield f"FAIL  {path.name}: {e}"
                self.progressed.emit(int(100 * (i + 1) / len(files)))

            if do_measure and rows:
                to_csv({k: np.asarray(v) for k, v in rows.items()}, out / "objects.csv")
                yield f"wrote {out / 'objects.csv'}"
            if do_measure and summaries:
                keys = list(summaries[0])
                to_csv(
                    {k: np.asarray([s.get(k) for s in summaries]) for k in keys},
                    out / "per_image_summary.csv",
                )
                yield f"wrote {out / 'per_image_summary.csv'}"
            yield f"done: {n_ok} succeeded, {n_fail} failed"

        w = work()
        w.yielded.connect(self.log.append)
        w.errored.connect(lambda e: self.log.append(unavailable_message(e)))
        w.finished.connect(self._reset)
        w.start()

    def _reset(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.bar.setVisible(False)


def _imread_any(path):
    from imageio.v3 import imread as _imread  # napari always brings imageio

    return np.asarray(_imread(str(path)))
