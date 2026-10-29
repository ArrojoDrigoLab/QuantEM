from rest_framework.test import APIRequestFactory

from quantem.assets.views import SystemHandshakeView


def _get(*, token: str = ""):
    headers = {"HTTP_X_QUANTEM_STARTUP_TOKEN": token} if token else {}
    request = APIRequestFactory().get("/api/system/handshake/", **headers)
    return SystemHandshakeView.as_view()(request)


def test_desktop_handshake_requires_the_per_launch_token(monkeypatch):
    monkeypatch.setenv("QUANTEM_STARTUP_TOKEN", "launch-secret")
    assert _get().status_code == 404
    assert _get(token="incorrect").status_code == 404
    response = _get(token="launch-secret")

    assert response.status_code == 200
    assert response.data == {"status": "ok", "message": "handshake-ok"}


def test_unmanaged_handshake_remains_available(monkeypatch):
    monkeypatch.delenv("QUANTEM_STARTUP_TOKEN", raising=False)

    response = _get()

    assert response.status_code == 200
