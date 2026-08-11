"""Turning "this model cannot run" into something an end user can act on.

The model layer's own exceptions are written for whoever has to fix the build.
:class:`quantem.inference.engine.ModelArchitectureUnavailable` carries, verbatim,
the advice to point ``QUANTEM_DINOV3_PATH`` at a checkout of
``github.com/facebookresearch/dinov3`` and run ``python -m
quantem.inference.export``. That is correct, and it is the right thing to say to
a maintainer building a release -- but it travelled straight through
:mod:`quantem.segmentation.organelle_tasks` into the job's error message and the
segmentation's ``status_error``, where an end user read it as their instructions.

The Models screen gives the right answer to that user
(:data:`quantem.registry.cache.INSTALL_HINT`: install the pack from that screen,
or from a folder you unzipped a release into), and this module makes the job
queue give the same one, from the same string. The maintainer's text is not lost
-- it is logged at ``exception`` level with the traceback, where a maintainer
will look and a user will not.

**App copy, not terminal copy.** The hint used here is deliberately
:data:`~quantem.registry.cache.INSTALL_HINT` and never
:data:`~quantem.registry.cache.INSTALL_INSTRUCTIONS`: what this function returns
is written to a segmentation's ``status_error`` and rendered verbatim in the
labeling header and the viewer's overlay card, where a shell command is an
instruction the reader cannot follow (invariant I-12). It used to return the
terminal text, and the terminal text was on screen.

Detection is deliberately by class *name*, walking the MRO, rather than by
importing the exception classes: :mod:`quantem.seg_core` must keep working on an
install with no torch and no model layer at all, and an ``except ImportError``
around an import that only exists to classify an error is worse than a name
check that cannot fail.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Exception class names meaning "no model can run here right now". Every one is
#: a *state of the installation*, not a bug in the run: weights absent, encoder
#: unbuildable, pack record missing.
MODEL_UNAVAILABLE_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "ModelUnavailableError",       # quantem.inference.engine, base
        "ModelWeightsNotInstalled",    # ... weights absent
        "ModelArchitectureUnavailable",  # ... weights present, nothing can build them
        "EncoderUnavailable",          # quantem.inference.encoders
        "PackNotInstalled",            # quantem.registry.cache
    }
)


def is_model_unavailable(exc: BaseException) -> bool:
    """True when ``exc`` means "the model is not runnable on this install"."""
    return any(
        klass.__name__ in MODEL_UNAVAILABLE_CLASS_NAMES
        for klass in type(exc).__mro__
    )


def _install_hint() -> str:
    """The one thing a user is ever told to do to obtain models.

    Imported here rather than at module scope so ``seg_core`` does not depend on
    the registry app being importable in every context, and so there is still a
    sensible sentence if it is not. The fallback obeys the same rule as the real
    string: a screen and a button, never a command.
    """
    try:
        from quantem.registry.cache import INSTALL_HINT  # noqa: PLC0415
    except Exception:  # pragma: no cover -- registry is always present in-app
        return (
            "Install it on the Models screen. With no internet, unzip a QuantEM "
            'model release onto this machine and use "Install from a local '
            'folder" on the same screen.'
        )
    return INSTALL_HINT


def user_facing_model_error(
    exc: BaseException,
    *,
    pack_id: str | None = None,
) -> str | None:
    """An end-user message for ``exc``, or None if it is not a model-availability error.

    None means "this is not mine to rewrite" -- an ordinary failure keeps its own
    message, because replacing a real error with install advice would be its own
    kind of lie.
    """
    if not is_model_unavailable(exc):
        return None

    named = f"Model pack {pack_id!r}" if pack_id else "The model for this run"
    if type(exc).__name__ == "ModelWeightsNotInstalled" or (
        type(exc).__name__ == "PackNotInstalled"
    ):
        headline = f"{named} is not installed."
    else:
        # Weights are there but nothing here can turn them into a runnable
        # model: an install from something other than a release bundle, or a
        # partial one. Reinstalling from a bundle is the fix, and it is the only
        # action a user of the desktop app can take.
        headline = (
            f"{named} is installed but this copy of QuantEM cannot load it. "
            "The packs in a QuantEM release bundle carry everything needed to "
            "run them; reinstalling from one replaces a partial install."
        )
    return f"{headline}\n{_install_hint()}"


def translate_model_error(
    exc: BaseException,
    *,
    pack_id: str | None = None,
    log_context: str = "",
) -> str:
    """``user_facing_model_error`` with a fallback to the exception's own text.

    Always returns something printable, so a caller can use it unconditionally
    when writing ``status_error`` or a job message.
    """
    message = user_facing_model_error(exc, pack_id=pack_id)
    if message is None:
        return str(exc)
    logger.info(
        "Reporting a model-availability failure to the user%s: %s",
        f" ({log_context})" if log_context else "",
        type(exc).__name__,
    )
    return message


__all__ = [
    "MODEL_UNAVAILABLE_CLASS_NAMES",
    "is_model_unavailable",
    "translate_model_error",
    "user_facing_model_error",
]
