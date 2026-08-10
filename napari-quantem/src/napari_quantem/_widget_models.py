"""Model manager: what is cached, what a download would cost, and where files live."""

from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ._common import hline, note, row
from ._models import confirm_download, download_reporter, forget, unavailable_message


class ModelManagerWidget(QWidget):
    #: Emitted from the download worker; Qt queues them onto the GUI thread.
    progressed = Signal(int)
    said = Signal(str)

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        lay = QVBoxLayout(self)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Model", "Encoder", "Size", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table)

        self.total = note("")
        lay.addWidget(self.total)

        self.dl_btn = QPushButton("Download everything")
        self.dl_btn.clicked.connect(self._download_all)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh)
        lay.addWidget(row(self.dl_btn, self.refresh_btn))

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)

        lay.addWidget(hline())
        lay.addWidget(QLabel("This machine"))
        self.about = QTextEdit()
        self.about.setReadOnly(True)
        self.about.setMaximumHeight(150)
        lay.addWidget(self.about)

        self.status = note("")
        lay.addWidget(self.status)

        self._refresh()

    def _refresh(self):
        from quantem_em.registry import REGISTRY
        from quantem_em.weights import fetch

        from ._models import status_for

        self.table.setRowCount(0)
        for spec in REGISTRY.values():
            plan = fetch.download_plan([spec])
            r = self.table.rowCount()
            self.table.insertRow(r)
            enc = "ViT-B (86M)" if spec.family == "quantem" else "ViT-L (302M)"
            size = sum(a["bytes"] or 0 for a in plan["artifacts"])
            marginal = plan["download_bytes"]
            size_text = fetch.format_bytes(size)
            if 0 < marginal < size:
                size_text += f"  (+{fetch.format_bytes(marginal)} to add)"
            for c, val in enumerate([spec.model_id, enc, size_text, status_for(spec)]):
                self.table.setItem(r, c, QTableWidgetItem(str(val)))

        whole = fetch.download_plan(list(REGISTRY.values()))
        cached = sum(1 for a in whole["artifacts"] if a["cached"])
        self.total.setText(
            f"{cached}/{len(whole['artifacts'])} files cached · "
            f"{fetch.format_bytes(whole['download_bytes'])} to download for everything"
        )
        self._describe()

    def _describe(self):
        lines = []
        try:
            from quantem_em.device import describe

            d = describe()
            lines.append(d["summary"])
            lines.append(f"torch {d['torch']}")
            if not d["accelerated"]:
                lines.append(
                    "No GPU detected. Segmentation works but is slower, and fine-tuning "
                    "noticeably so — the estimate before each run is measured, not guessed."
                )
        except Exception as e:
            lines.append(f"torch unavailable: {e}")
            lines.append("Install PyTorch to use the models.")
        try:
            from quantem_em.weights import fetch

            d = fetch.local_dir()
            lines.append(
                f"model directory: {d}"
                if d
                else "model cache: huggingface_hub default (set QUANTEM_MODEL_DIR to override)"
            )
            if fetch.offline():
                lines.append("offline mode is ON — no downloads will be attempted")
        except Exception:
            pass
        self.about.setPlainText("\n".join(lines))

    def _download_all(self):
        from napari.qt.threading import thread_worker
        from quantem_em.registry import REGISTRY

        specs = list(REGISTRY.values())
        if not confirm_download(self, specs):
            return
        self.progressed.connect(self.bar.setValue)
        self.said.connect(self.status.setText)
        self.dl_btn.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)

        @thread_worker
        def work():
            from quantem_em.weights import fetch

            names = []
            for s in specs:
                for a in fetch.artifacts_for(s):
                    if a not in names:
                        names.append(a)
            fetch.ensure(
                names,
                progress=download_reporter(self.progressed.emit, self.said.emit),
            )
            return len(names)

        w = work()
        w.returned.connect(lambda n: self.status.setText(f"{n} artifact(s) available"))
        w.errored.connect(lambda e: self.status.setText(unavailable_message(e)))
        w.finished.connect(self._after_download)
        w.start()

    def _after_download(self):
        self.dl_btn.setEnabled(True)
        self.bar.setVisible(False)
        self.bar.setRange(0, 100)
        forget()
        self._refresh()
