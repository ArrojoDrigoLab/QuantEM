"""The failure catalogue: one name per class of thing that can go wrong.

**Why a catalogue at all.** Until this module existed, every failure reached the
user as a sentence and nothing else. A sentence is written once, by whoever was
in that function that day, and it drifts: the same underlying problem -- no
model installed -- was phrased three different ways on three screens, one of
them with a shell command in it. Worse, a sentence cannot carry an *action*. The
client had no way to know that "the pack is not installed" is the one failure
with a button that fixes it, so it rendered every failure the same way: red
text, no way forward, and a user who concludes the fault is theirs.

A code fixes both. The backend says *which class of failure this is*; the client
looks the class up in ``frontend/src/shared/copy/failures.ts`` and renders the
wording **and the in-app control** that belongs to it. One place to change the
words, one place to decide what the button does.

Who writes what
---------------
This module is the **catalogue**, and ``failures.ts`` is the **copy**. Neither
raises anything. The *emit sites* belong to whichever module owns that failure:
the segmentation task owner attaches the code when inference dies, the assets
owner attaches it when an image will not decode. That split is deliberate --
one file naming every failure would have to be edited by every package at once
-- and it is enforced from both ends by
``quantem/core/tests/test_error_codes.py``:

* every ``error_code`` value written anywhere in ``src/quantem`` is a member of
  :class:`ErrorCode`, so a code cannot be invented at a call site and reach a
  client that has never heard of it;
* every member of :class:`ErrorCode` has an entry in ``failures.ts`` with an
  action, so a code cannot be added here and render as a blank space.

How an emitter uses it
----------------------
Attach the code beside the sentence, never instead of it. The sentence stays
because it is the only text that can name *this* run's particulars; the code is
what lets the client add a control::

    from quantem.core.error_codes import ErrorCode, classify_exception

    return Response(
        {"error": str(exc), "error_code": ErrorCode.IMAGE_UNREADABLE},
        status=400,
    )

    # or, when the exception is all you have:
    code = classify_exception(exc)          # ErrorCode | None

:func:`classify_exception` matches on exception **class names**, walking the
MRO, rather than importing the classes. :mod:`quantem.seg_core` and this module
must keep working on an install with no torch and no model layer, and an
``except ImportError`` wrapped around an import that exists only to classify an
error is worse than a name check that cannot fail. It is the same technique, and
the same reasoning, as
:data:`quantem.seg_core.model_errors.MODEL_UNAVAILABLE_CLASS_NAMES`, which this
module reuses rather than restates.

The codes are wire values, not prose
------------------------------------
An ``ErrorCode`` is a lowercase snake_case identifier that travels in a JSON
field of its own. It is **never** interpolated into a sentence: invariant I-12
forbids internal names in copy, and a user shown ``probability_map_missing``
learns nothing. The gate in ``quantem/registry/tests/copy_gate.py`` treats a
field carrying only the datum as data and a sentence containing it as a defect,
which is exactly the line this module has to stay on the right side of.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "ERROR_CODE_FIELD",
    "ErrorCode",
    "classify_exception",
    "with_error_code",
]

#: The JSON field an error code travels in, everywhere. Named once so a
#: serialiser, a test and a client cannot disagree about it by a typo.
ERROR_CODE_FIELD = "error_code"


class ErrorCode(StrEnum):
    """Every class of failure this application can put in front of a person.

    A member is added here when a failure needs *different words or a different
    control* from the ones already listed -- not for every distinct exception.
    Two exceptions that leave the user with the same thing to do are the same
    code.
    """

    #: The pack the run needs is not installed, or is installed and cannot be
    #: built into a runnable module on this machine. The one failure with a
    #: button that fixes it: the Models screen installs or repairs it.
    MODEL_NOT_INSTALLED = "model_not_installed"

    #: The machine ran out of memory -- host RAM or VRAM -- while the model was
    #: running. Recoverable by asking for less at once: a region instead of the
    #: whole image, or nothing else running alongside.
    OUT_OF_MEMORY = "out_of_memory"

    #: The image has no pixel size, and the thing being asked for cannot be
    #: done without one. Every released head bar the ER one was trained at a
    #: fixed nm/px and has to resample the image to reach it, so this is a
    #: refusal *before* the run rather than a bad result after it.
    NO_PIXEL_SIZE = "no_pixel_size"

    #: The image bytes could not be decoded: truncated, not the format the
    #: extension claims, or a variant no reader here supports. The user's file
    #: is the subject, so the message names the file and the app takes no blame
    #: it has not earned.
    IMAGE_UNREADABLE = "image_unreadable"

    #: A write failed for want of space on the volume the data directory lives
    #: on. Distinct from a permission failure and from a locked file: the
    #: remedy is free space, and the app can say which folder it was writing to.
    DISK_FULL = "disk_full"

    #: Somebody stopped it on purpose. Not a fault, and the copy must not
    #: apologise for it or imply data was lost; work already saved is still
    #: saved.
    CANCELLED = "cancelled"

    #: A threshold-only re-run was asked for and the stored probability map is
    #: not there -- never written, or reclaimed to save disk. Under R11 the
    #: stored native-coordinate map is what every threshold reads, so without
    #: it the dial cannot move and inference has to run again.
    PROBABILITY_MAP_MISSING = "probability_map_missing"

    #: The image is too large to process the way it was asked for on this
    #: machine's memory budget. Distinct from :attr:`OUT_OF_MEMORY` because it
    #: is known *before* anything is attempted, so the app can offer the
    #: smaller way of doing it rather than reporting a crash.
    IMAGE_TOO_LARGE_FOR_MEMORY = "image_too_large_for_memory"

    #: These exact bytes are already in the library. Not an error in the user's
    #: work -- the right response is to open the copy that is already there,
    #: which is why the payload also carries its identity.
    #: Emitted by :class:`quantem.assets.models.DuplicateImportError`.
    DUPLICATE_IMAGE = "duplicate_image"

    #: The upload is bigger than this install will accept in one request.
    #: Emitted by the request-size guard in :mod:`quantem.cli`.
    UPLOAD_TOO_LARGE = "upload_too_large"


#: Exception class names that mean each code, walked over the MRO.
#:
#: Name-matched rather than imported, for the reason in the module docstring.
#: The model-unavailable set is *imported* from
#: :mod:`quantem.seg_core.model_errors` at classification time rather than
#: copied: that module is already the authority on which exceptions mean "no
#: model can run here", and two lists would drift the first time one grew.
_CLASS_NAMES: dict[ErrorCode, frozenset[str]] = {
    ErrorCode.OUT_OF_MEMORY: frozenset(
        {
            "OutOfMemoryError",  # torch.OutOfMemoryError / torch.cuda.OutOfMemoryError
            "CudaOutOfMemoryError",
            "MemoryError",
        }
    ),
    ErrorCode.CANCELLED: frozenset(
        {"JobCancelled", "CancelledError", "KeyboardInterrupt"}
    ),
    ErrorCode.DUPLICATE_IMAGE: frozenset({"DuplicateImportError"}),
    ErrorCode.PROBABILITY_MAP_MISSING: frozenset(
        {"ProbabilityMapMissing", "ProbabilityMapNotStored"}
    ),
}

#: ``OSError.errno`` values that mean the volume is full. ``ENOSPC`` is POSIX
#: and Windows both; ``EDQUOT`` is a quota, which the user experiences the same
#: way and can act on the same way.
_DISK_FULL_ERRNOS: frozenset[int] = frozenset({28, 122})


def _model_unavailable_names() -> frozenset[str]:
    """The model layer's own vocabulary, read rather than duplicated."""
    try:
        from quantem.seg_core.model_errors import (  # noqa: PLC0415
            MODEL_UNAVAILABLE_CLASS_NAMES,
        )
    except Exception:  # pragma: no cover - seg_core always ships with the app
        return frozenset()
    return frozenset(MODEL_UNAVAILABLE_CLASS_NAMES)


def classify_exception(exc: BaseException) -> ErrorCode | None:
    """Which class of failure ``exc`` is, or ``None`` when it is not one of them.

    ``None`` is a real answer and the common one: most exceptions are bugs, and
    a bug has no user-facing remedy to offer. An emitter that gets ``None``
    should send its sentence with no code, and the client renders the sentence
    on its own -- which is what every failure surface did before this module
    existed, so ``None`` degrades to the old behaviour rather than to a blank.

    Never guesses from message text. A substring match on "memory" would
    classify "not enough memory to open the settings file" as a model OOM and
    offer the wrong control, and a wrong control is worse than none.
    """
    names = {klass.__name__ for klass in type(exc).__mro__}

    if names & _model_unavailable_names():
        return ErrorCode.MODEL_NOT_INSTALLED
    for code, class_names in _CLASS_NAMES.items():
        if names & class_names:
            return code
    if isinstance(exc, OSError) and exc.errno in _DISK_FULL_ERRNOS:
        return ErrorCode.DISK_FULL
    return None


def with_error_code(payload: dict, code: ErrorCode | None) -> dict:
    """``payload`` plus its code, or ``payload`` unchanged when there is none.

    A convenience so an emitter never writes the field name by hand, and so a
    ``None`` from :func:`classify_exception` cannot put ``"error_code": null``
    on the wire -- a client checking for the key's presence would then see a
    code that is not one.
    """
    if code is None:
        return payload
    return {**payload, ERROR_CODE_FIELD: str(code)}
