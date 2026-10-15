"""Temporary fixture cleanup tolerates sharing locks but still reports failures."""

from unittest.mock import patch

import pytest

from quantem.testing import _unlink_fixture


def _windows_error(code):
    error = PermissionError("fixture is unavailable")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [32, 33])
def test_cleanup_retries_a_transient_sharing_lock(tmp_path, code):
    with (
        patch("pathlib.Path.unlink", side_effect=[_windows_error(code), None]) as unlink,
        patch("quantem.testing.time.sleep") as sleep,
    ):
        _unlink_fixture(tmp_path / "fixture.tif")
    assert unlink.call_count == 2
    unlink.assert_called_with(missing_ok=True)
    sleep.assert_called_once_with(0.05)


def test_cleanup_does_not_retry_unrelated_permission_errors(tmp_path):
    with (
        patch("pathlib.Path.unlink", side_effect=_windows_error(5)) as unlink,
        patch("quantem.testing.time.sleep") as sleep,
        pytest.raises(PermissionError),
    ):
        _unlink_fixture(tmp_path / "fixture.tif")
    unlink.assert_called_once_with(missing_ok=True)
    sleep.assert_not_called()


def test_cleanup_reports_a_persistent_sharing_lock(tmp_path):
    with (
        patch("pathlib.Path.unlink", side_effect=_windows_error(32)) as unlink,
        patch("quantem.testing.time.sleep") as sleep,
        pytest.raises(PermissionError),
    ):
        _unlink_fixture(tmp_path / "fixture.tif")
    assert unlink.call_count == 6
    assert sleep.call_count == 5
