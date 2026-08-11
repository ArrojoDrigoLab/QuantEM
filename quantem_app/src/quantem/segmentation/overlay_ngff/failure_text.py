"""Turning an overlay-build exception into a sentence a user can act on.

The overlay's failure reason is not a log line. It is recorded on
``SegmentationOverlayState.last_error``, served on the manifest, and rendered
verbatim in front of the person using the application -- and the whole value of
it is the path it names, because the two failures that actually happen in the
field are "something is sitting where the staging directory belongs" and
"another program is holding an overlay file open". A user has to be able to
select that path and paste it into Explorer.

``str(OSError)`` cannot be that string. It formats the filename through
``repr()``, so on Windows every separator in it is **doubled** and the whole
path is wrapped in quotes::

    [WinError 183] Cannot create a file when that file already exists:
    '<overlay directory, with every backslash doubled>'

What reached the screen was a path in which each separator appeared twice --
honest, unpasteable, and noisy in exactly the place the user is meant to look.

So the two facts are pulled out of the exception and joined by hand:
``strerror`` (the OS's own description, without the numeric code) and
``filename`` (the real path). Nothing is invented and nothing is paraphrased.
"""

from __future__ import annotations

__all__ = ["describe_failure", "describe_os_error"]


def _clean(value: object) -> str:
    return str(value).strip() if value is not None else ""


def describe_os_error(exc: OSError, *, operation_path: object | None = None) -> str:
    """``"<what went wrong>: <real path>"`` for a filesystem failure.

    Both halves are optional in principle -- an :class:`OSError` raised by hand
    may carry neither -- so every combination degrades to something true rather
    than to a half-built sentence.

    ``filename2`` is included when the operation had two of them (a rename, a
    copy): saying only the source of a failed rename hides half the problem.
    ``operation_path`` supplies the full path when an fd-based stdlib operation
    records only a child name in ``filename``.
    """
    reason = _clean(getattr(exc, "strerror", None))
    if operation_path is None:
        filename = _clean(getattr(exc, "filename", None))
        filename2 = _clean(getattr(exc, "filename2", None))
    else:
        # fd-based shutil operations may report only the child name ("0") in
        # OSError.filename. Its callback carries the complete failing path.
        filename = _clean(operation_path)
        filename2 = ""

    if filename and filename2 and filename2 != filename:
        where = f"{filename} -> {filename2}"
    else:
        where = filename or filename2

    if reason and where:
        return f"{reason}: {where}"
    if reason:
        return reason
    if where:
        # No description from the OS, but the path is still the actionable half.
        # It used to be introduced by ``type(exc).__name__``, which put a Python
        # class in front of whoever is labeling -- I-12's ``exception-class``,
        # and invisible to the copy gate until the sweep learned to follow
        # ``last_error=describe_failure(exc)`` into this module. The class name
        # told the reader nothing they could act on; the path is the whole
        # message, so it gets a plain sentence in front of it instead.
        return f"Could not complete this file operation: {where}"
    return _clean(exc) or type(exc).__name__


def describe_failure(exc: BaseException) -> str:
    """One line describing ``exc``, fit to be shown to whoever is labeling.

    Non-``OSError`` exceptions keep ``str(exc)``: their messages are written by
    us and already read as English. The class name is used only when the
    exception carries no message at all, because "the overlay build failed" with
    an empty reason underneath is worse than a bare ``KeyError``.
    """
    if isinstance(exc, OSError):
        return describe_os_error(exc)
    return _clean(exc) or type(exc).__name__
