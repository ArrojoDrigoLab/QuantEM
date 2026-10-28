from __future__ import annotations

from quantem.cli import _ShutdownApplication


class _Timer:
    started = False

    def __init__(self, interval, callback):
        self.interval = interval
        self.callback = callback

    def start(self):
        _Timer.started = True


def _call(application, *, token: str, method: str = "POST"):
    response = {}

    def start_response(status, headers):
        response["status"] = status
        response["headers"] = headers

    body = application(
        {
            "PATH_INFO": "/__quantem_shutdown__",
            "REQUEST_METHOD": method,
            "HTTP_X_QUANTEM_SHUTDOWN_TOKEN": token,
        },
        start_response,
    )
    return response, b"".join(body)


def test_shutdown_route_requires_the_private_token(monkeypatch):
    monkeypatch.setattr("quantem.cli.threading.Timer", _Timer)
    application = _ShutdownApplication(lambda *_: [b"app"], "expected")

    response, body = _call(application, token="wrong")

    assert response["status"] == "404 Not Found"
    assert body == b"Not found"
    assert not _Timer.started


def test_shutdown_route_interrupts_after_accepting_response(monkeypatch):
    _Timer.started = False
    monkeypatch.setattr("quantem.cli.threading.Timer", _Timer)
    application = _ShutdownApplication(lambda *_: [b"app"], "expected")

    response, body = _call(application, token="expected")

    assert response["status"] == "202 Accepted"
    assert body == b"Shutting down"
    assert _Timer.started
