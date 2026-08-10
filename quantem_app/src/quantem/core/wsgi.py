"""
WSGI config for core project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'quantem.core.settings')

# Ensure directories exist before Django fully initializes
from quantem.core.config import ensure_directories

ensure_directories()

application = get_wsgi_application()
