from django.http import HttpResponse
from django.test import RequestFactory

from quantem.core.performance import _TIMINGS, PerformanceTimingMiddleware, timed_stage


def test_stage_headers_and_context_cleanup():
    @timed_stage("decode")
    def decode():
        pass

    def response(request):
        decode()
        decode()
        return HttpResponse("ok")

    result = PerformanceTimingMiddleware(response)(RequestFactory().get("/api/example/"))
    assert result["Server-Timing"].count("decode;dur=") == 1
    assert "app;dur=" in result["Server-Timing"]
    assert len(result["X-QuantEM-Trace"]) == 32
    assert _TIMINGS.get() is None


def test_failed_request_does_not_leak_timing_context():
    def response(request):
        raise ValueError("failed")

    try:
        PerformanceTimingMiddleware(response)(RequestFactory().get("/api/example/"))
    except ValueError:
        pass
    assert _TIMINGS.get() is None
