"""Segmentation-type catalogue.

The organelle picker in the UI populates from this endpoint, so it is product
code, not corpus code.

QuantEM ships neither SAM nor a corpus catalogue, so this viewset lives in its
own module rather than alongside those endpoints.
"""

from __future__ import annotations

from rest_framework import viewsets

from quantem.segmentation.models import SegmentationType
from quantem.segmentation.serializers import SegmentationTypeSerializer


class SegmentationTypeViewSet(viewsets.ModelViewSet):
    """List and create segmentation types (mitochondria, ER, nucleus, lipid droplet, tissue)."""

    queryset = SegmentationType.objects.all()
    serializer_class = SegmentationTypeSerializer
    http_method_names = ["get", "post"]

    def get_queryset(self):
        return SegmentationType.objects.all().order_by("long_name")
