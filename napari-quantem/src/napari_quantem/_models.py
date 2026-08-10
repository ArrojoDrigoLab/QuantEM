"""Model loading, caching and the download-consent flow.

Nothing here touches the network until the user has seen exactly what will be downloaded and
agreed to it. That is deliberate: across the comparable plugins, none asks consent and none shows
the size up front, and the failure mode — a 1.2 GB fetch starting as a side effect of clicking
"Run" — is the one people complain about.
"""

from __future__ import annotations

from qtpy.QtWidgets import QMessageBox

#: One loaded model per id. Models are not thread-safe, so a worker must own its model for the
#: duration of a run; these are handed out to one worker at a time by the widgets.
_CACHE: dict[str, object] = {}


def loaded(model_id: str):
    return _CACHE.get(model_id)


def forget(model_id: str | None = None) -> None:
    if model_id is None:
        _CACHE.clear()
    else:
        _CACHE.pop(model_id, None)


def status_for(spec) -> str:
    """``Loaded`` / ``Cached`` / ``Download required (N MB)`` for the picker."""
    from quantem_em.weights import fetch

    if spec.model_id in _CACHE:
        return "Loaded"
    plan = fetch.download_plan([spec])
    if plan["all_present"]:
        return "Cached"
    return f"Download required ({fetch.format_bytes(plan['download_bytes'])})"


def describe_download(specs) -> dict:
    from quantem_em.weights import fetch

    return fetch.download_plan(list(specs))


def download_summary(plan) -> tuple[str, str]:
    """The (headline, detail) shown before anything is fetched.

    Kept separate from the dialog so the wording can be tested without a display.
    """
    from quantem_em.weights import fetch

    lines = []
    for a in plan["missing"]:
        lines.append(f"  • {a['filename']}   {fetch.format_bytes(a['bytes'])}")
        if a.get("description"):
            lines.append(f"      {a['description']}")
        # The closing paragraph tells the user the licence is listed here, so it has to be.
        lines.append(f"      Licence: {a.get('license') or 'see the model repository'}")
    repo = plan["missing"][0]["repo"] if plan["missing"] else ""

    headline = (
        f"{len(plan['missing'])} file(s) totalling "
        f"{fetch.format_bytes(plan['download_bytes'])} need to be downloaded before this model "
        "can run."
    )
    detail = (
        "\n".join(lines)
        + f"\n\nSource: huggingface.co/{repo}"
        + "\nEach file is verified against a published SHA-256 after download."
        + "\n\nThey are cached, so this happens once."
        + "\n\nThe code is BSD-3-Clause. The weights carry their own licence, listed above and "
        "declared in full on the model's Hugging Face page."
    )
    return headline, detail


def confirm_download(parent, specs) -> bool:
    """Show exactly what will be fetched, from where, and how big. Returns True if the user agrees."""
    plan = describe_download(specs)
    if plan["all_present"]:
        return True

    headline, detail = download_summary(plan)

    box = QMessageBox(parent)
    box.setWindowTitle("Download model files")
    box.setIcon(QMessageBox.Question)
    box.setText(headline)
    box.setInformativeText(detail)
    box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    box.button(QMessageBox.Ok).setText("Download")
    return box.exec_() == QMessageBox.Ok


def download_reporter(on_percent, on_text):
    """Build a ``fetch.ensure`` progress callback that drives a widget's signals.

    Reports bytes, not file counts. Users judge whether a download is progressing or wedged from
    the number moving, and hub's own bar writes to stderr -- which a GUI napari launch discards.
    """
    from quantem_em.weights import fetch

    def report(done, total, label):
        if total:
            on_percent(int(100 * done / total))
        on_text(f"Downloading {label} — {fetch.format_bytes(done)} of {fetch.format_bytes(total)}")

    return report


def get_model(model_id: str, *, device: str = "auto", progress=None):
    """Load (or return the cached) model. Assumes consent has already been obtained."""
    m = _CACHE.get(model_id)
    if m is not None:
        return m
    from quantem_em.api import load_model

    m = load_model(model_id, device=device, progress=progress)
    _CACHE[model_id] = m
    return m


#: Substrings that identify a specific, fixable cause. Matched on text rather than exception class
#: because huggingface_hub wraps several of these in its own hierarchy, and that hierarchy has
#: changed between majors -- text is the stable surface.
_HINTS = (
    (
        "proxy",
        "A proxy rejected the connection. On an institutional network, set HTTPS_PROXY, or "
        "download the files on another machine (see below).",
    ),
    (
        "401",
        "The download was refused as unauthorised. If HF_TOKEN is set in your environment and has "
        "expired, unset it: these files are public and need no token.",
    ),
    (
        "403",
        "The download was refused. If HF_TOKEN is set and has expired, unset it — these files are "
        "public and need no token.",
    ),
    ("429", "Hugging Face is rate-limiting this machine. Wait a few minutes and try again."),
    ("no space", "The disk filled up during the download. Free some space and try again."),
    ("errno 28", "The disk filled up during the download. Free some space and try again."),
    ("permission", "A cache directory could not be written. Set HF_HOME to a directory you own."),
    (
        "failed verification",
        "A cached file is corrupt. Delete the file named above and run again; it will be "
        "re-downloaded and re-checked.",
    ),
)

_SIDELOAD = (
    "If this machine has no internet access, download the files on a connected machine with\n"
    "    python -m quantem_em.weights download --all --to <folder>\n"
    "copy that folder across, and set QUANTEM_MODEL_DIR to it before starting napari."
)


def unavailable_message(exc) -> str:
    """Turn any download or verification failure into something a user can act on.

    A raw httpx or hub traceback in a status label tells a microscopist nothing. Every branch ends
    with a route forward, because the side-load path always exists.
    """
    text = str(exc)
    low = text.lower()
    for needle, hint in _HINTS:
        if needle in low:
            return f"{text}\n\n{hint}\n\n{_SIDELOAD}"
    return f"{text}\n\n{_SIDELOAD}"
