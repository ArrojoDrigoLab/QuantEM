"""A viewer fixture that does not need OpenGL.

napari's own ``make_napari_viewer`` builds a full Qt window with a live OpenGL canvas. That is the
wrong dependency for this suite. None of these widgets draw anything: they read ``viewer.layers``,
add layers, and drive plain Qt controls. Requiring a GL context makes the suite fail on headless
CI, over RDP, on locked desktop sessions, and in VMs without a GPU -- none of which say anything
about whether the plugin works. It failed here for exactly that reason: bare ``napari.Viewer()``
raised ``GLError 1282`` on an idle GPU with no other process competing for it.

:class:`napari.components.ViewerModel` is napari's GUI-free layer model -- the same ``layers``,
``add_image``, ``add_labels`` and ``add_shapes`` API, minus the canvas. Qt widgets themselves need
no GL, so the widgets are exercised exactly as written, including ``show()``.

``make_napari_viewer_gui`` is the real thing, for the few checks that want a window. It skips
itself when no GL context can be created, so a machine that cannot render never reports a failure
it cannot fix.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def make_napari_viewer(qapp):
    """Shadows napari's fixture of the same name with a GL-free equivalent.

    Takes and ignores the same keyword arguments so existing calls keep working.
    """
    from napari.components import ViewerModel

    created = []

    def factory(*_args, **_kwargs):
        v = ViewerModel()
        created.append(v)
        return v

    yield factory

    for v in created:
        v.layers.clear()


@pytest.fixture
def make_napari_viewer_gui(qapp):
    """The real windowed viewer. Skips when the session has no usable GL context."""
    napari = pytest.importorskip("napari")
    created = []

    def factory(*args, **kwargs):
        kwargs.setdefault("show", False)
        try:
            v = napari.Viewer(*args, **kwargs)
        except Exception as exc:  # GLError, and whatever else a broken driver raises
            pytest.skip(f"no usable OpenGL context in this session: {type(exc).__name__}")
        created.append(v)
        return v

    yield factory

    for v in created:
        v.close()
