"""Serving the built frontend from the local server.

This is what makes ``pip install quantem && quantem`` a whole application rather
than an API with no UI, so its edges are worth pinning.
"""

from __future__ import annotations

import json

from django.test import Client, TestCase, override_settings

from quantem.core import spa


class SpaRoutingTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_unknown_api_path_is_404_not_the_app(self):
        """The catch-all must not answer a mistyped endpoint with index.html.

        A caller expecting JSON would get HTML and fail on parse, several layers
        from the actual mistake.
        """
        for path in (
            "/api/does-not-exist/",
            "/ngff/nope",
            "/segmentation-overlays/nope",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)

    def test_traversal_outside_dist_is_refused(self):
        response = self.client.get("/../../etc/passwd")
        self.assertIn(response.status_code, (400, 404))

    def test_unmatched_api_post_is_404_never_a_csrf_403(self):
        """A POST that falls through to the catch-all must 404 like a GET does.

        The concrete trap: ``.../roi/<id>/complete`` is registered without a
        trailing slash, so ``.../complete/`` resolves to the SPA catch-all --
        and CsrfViewMiddleware used to reject the POST with a 403 CSRF HTML
        page *before* the view's own 404 could run. Nothing in QuantEM ever
        sets a CSRF cookie, so that 403 was unresolvable for every caller.
        ``enforce_csrf_checks`` makes the test client run the real middleware;
        without ``csrf_exempt`` on ``serve_frontend`` these are 403s.
        """
        client = Client(enforce_csrf_checks=True)
        seg = "11111111-1111-1111-1111-111111111111"
        roi = "22222222-2222-2222-2222-222222222222"
        for path in (
            f"/api/segmentations/{seg}/roi/{roi}/complete/",  # slashed form
            f"/api/segmentations/{seg}/complete/",
            "/api/does-not-exist/",
        ):
            with self.subTest(path=path):
                response = client.post(path)
                self.assertEqual(response.status_code, 404)


@override_settings(DEBUG=False)
class RuntimeConfigInjectionTests(TestCase):
    """``window.__APP_CONFIG__`` is the shell-injection seam.

    ``frontend/src/config.ts`` has always read it; nothing used to set it.
    For the pip channel the server fills it in so the UI knows its own origin.
    """

    def test_origin_is_injected(self):
        if not spa.frontend_available():
            self.skipTest("frontend not built")
        html = self.client.get("/").content.decode()
        self.assertIn("window.__APP_CONFIG__", html)
        payload = html.split("window.__APP_CONFIG__=", 1)[1].split(";</script>", 1)[0]
        config = json.loads(payload)
        # The absolute origin, not "": the viewer does `new URL(path, base)`
        # and an empty base throws "Invalid URL".
        self.assertTrue(config["apiBaseUrl"].startswith("http://"))
        self.assertFalse(config["apiBaseUrl"].endswith("/"))
        self.assertFalse(config["dev"])
        # There is no authentication in QuantEM.
        self.assertNotIn("authToken", config)

    def test_index_is_never_cached(self):
        if not spa.frontend_available():
            self.skipTest("frontend not built")
        response = self.client.get("/")
        self.assertIn("no-store", response["Cache-Control"])

    def test_missing_build_gives_an_actionable_error(self):
        with override_settings(QUANTEM_FRONTEND_DIST="/nonexistent/dist"):
            self.assertEqual(self.client.get("/").status_code, 404)


class LocalOnlyMiddlewareTests(TestCase):
    """No auth -- but the server must still only answer this machine.

    Binding loopback and ALLOWED_HOSTS cover the socket and the Host header. This
    covers the third door: a page you happen to have open in a browser making
    cross-origin requests to the QuantEM port.
    """

    def test_no_origin_is_allowed(self):
        """The app's own navigations, the CLI and in-process clients."""
        self.assertEqual(self.client.get("/api/system/status/").status_code, 200)

    def test_loopback_origin_is_allowed_on_any_port(self):
        for origin in (
            "http://127.0.0.1:8734",
            "http://localhost:5173",
            "http://[::1]:9000",
        ):
            with self.subTest(origin=origin):
                response = self.client.get(
                    "/api/system/status/", headers={"origin": origin}
                )
                self.assertEqual(response.status_code, 200)

    def test_foreign_origin_is_refused(self):
        response = self.client.get(
            "/api/system/status/", headers={"origin": "https://evil.example.com"}
        )
        self.assertEqual(response.status_code, 403)
