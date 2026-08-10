"""A user who cannot run a model must be told something they can act on.

``ModelArchitectureUnavailable`` carries maintainer instructions -- clone Meta's
``dinov3``, run ``python -m quantem.inference.export`` -- and those reached the
job's error message and the segmentation's ``status_error``, where an end user
read them as their own next steps. They are not: the user's fix is to reinstall
from a release bundle, which is what the CLI and the Models screen already say.
"""

from __future__ import annotations

import pytest

from quantem.registry.cache import INSTALL_COMMAND, INSTALL_INSTRUCTIONS
from quantem.seg_core.model_errors import (
    is_model_unavailable,
    translate_model_error,
    user_facing_model_error,
)


class _ModelUnavailableError(RuntimeError):
    """Stand-in with the real class name; detection is by name, not import."""


class ModelArchitectureUnavailable(_ModelUnavailableError):
    pass


class ModelWeightsNotInstalled(_ModelUnavailableError):
    pass


class PackNotInstalled(FileNotFoundError):
    pass


MAINTAINER_TEXT = (
    "quantem:mito: The QuantEM family's encoder is a DINOv3 ViT-B and needs Meta's "
    "`dinov3` package, which QuantEM does not redistribute.\n"
    "  1. point QUANTEM_DINOV3_PATH at a checkout of "
    "https://github.com/facebookresearch/dinov3\n"
    "  2. python -m quantem.inference.export <pack-id>"
)


def test_the_real_exception_types_are_recognised():
    # Guards the name list against a rename in quantem.inference.engine.
    from quantem.inference.engine import ModelArchitectureUnavailable as RealArchitecture
    from quantem.inference.engine import ModelUnavailableError as RealBase
    from quantem.inference.engine import ModelWeightsNotInstalled as RealWeights

    assert is_model_unavailable(RealBase("x"))
    assert is_model_unavailable(RealArchitecture("x"))
    assert is_model_unavailable(RealWeights("x"))


def test_an_ordinary_error_is_left_alone():
    # Replacing a real failure with install advice is its own kind of lie.
    assert not is_model_unavailable(ValueError("bad ROI"))
    assert user_facing_model_error(ValueError("bad ROI")) is None
    assert translate_model_error(ValueError("bad ROI")) == "bad ROI"


@pytest.mark.parametrize(
    "exc",
    [
        ModelArchitectureUnavailable(MAINTAINER_TEXT),
        ModelWeightsNotInstalled("quantem:mito: not installed"),
        PackNotInstalled("quantem:mito: no pack.json"),
    ],
)
def test_maintainer_advice_never_reaches_the_user(exc):
    message = translate_model_error(exc, pack_id="quantem:mito")

    assert "dinov3" not in message
    assert "facebookresearch" not in message
    assert "quantem.inference.export" not in message
    assert "QUANTEM_DINOV3_PATH" not in message


@pytest.mark.parametrize(
    "exc",
    [
        ModelArchitectureUnavailable(MAINTAINER_TEXT),
        ModelWeightsNotInstalled("quantem:mito: not installed"),
        PackNotInstalled("quantem:mito: no pack.json"),
    ],
)
def test_every_model_failure_gives_the_one_shared_answer(exc):
    message = translate_model_error(exc, pack_id="quantem:mito")

    # The same strings the CLI and the Models screen use, so a user is never
    # given two different answers to the same question.
    assert INSTALL_COMMAND in message
    assert INSTALL_INSTRUCTIONS in message
    assert "quantem:mito" in message


def test_missing_weights_and_an_unbuildable_pack_read_differently():
    absent = translate_model_error(
        ModelWeightsNotInstalled("x"), pack_id="quantem:mito"
    )
    unbuildable = translate_model_error(
        ModelArchitectureUnavailable(MAINTAINER_TEXT), pack_id="quantem:mito"
    )

    assert "not installed" in absent
    assert "cannot load it" in unbuildable
    assert absent != unbuildable


def test_it_still_says_something_useful_without_a_pack_id():
    message = translate_model_error(ModelWeightsNotInstalled("x"))

    assert "The model for this run" in message
    assert INSTALL_COMMAND in message
