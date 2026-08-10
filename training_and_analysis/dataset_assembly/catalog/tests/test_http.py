from __future__ import annotations

import io
import urllib.error
import unittest
from unittest import mock

from catalog.http import HttpFetchError, get_json


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _http_error(status: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.org/api",
        status,
        "error",
        hdrs={},
        fp=io.BytesIO(body),
    )


class HttpTests(unittest.TestCase):
    def test_get_json_retries_zenodo_style_429(self) -> None:
        error = _http_error(429, b'{"message":"30 per 1 minute","status":429}')
        with (
            mock.patch("urllib.request.urlopen", side_effect=[error, _Response(b'{"ok": true}')]),
            mock.patch("time.sleep") as sleep,
        ):
            self.assertEqual(get_json("https://example.org/api", max_retries=1), {"ok": True})

        sleep.assert_called_once()

    def test_get_json_does_not_retry_non_rate_limit_error(self) -> None:
        error = _http_error(400, b"bad request")
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(HttpFetchError) as ctx:
                get_json("https://example.org/api", max_retries=1)

        self.assertEqual(ctx.exception.status, 400)


if __name__ == "__main__":
    unittest.main()
