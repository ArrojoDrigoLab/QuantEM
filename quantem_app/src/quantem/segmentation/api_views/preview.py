"""Live-preview routes.

Empty on purpose. This module exists so the live-preview package can add its
views and routes without opening :mod:`quantem.segmentation.urls`, which four
other packages are adding to at the same time. ``urlpatterns`` is spliced into
that module's list; an empty list adds nothing to the resolver.
"""

from django.urls.resolvers import URLPattern

urlpatterns: list[URLPattern] = []
