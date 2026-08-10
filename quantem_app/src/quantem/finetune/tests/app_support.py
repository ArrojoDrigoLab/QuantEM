"""Make ``quantem.finetune`` an installed app for the duration of a test.

``core/settings.py`` does not list ``"quantem.finetune"`` yet (that file belongs
to another package and is not edited from here), so importing
:mod:`quantem.finetune.models` raises and its table does not exist. Rather than
skip every model and API test until the one-line settings change lands, these
tests install the app themselves and create the table with the schema editor.

Once the settings entry is added this becomes a no-op: the app is already
installed and ``migrate --run-syncdb`` creates the table when the test database
is built.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from django.apps import apps
from django.conf import settings
from django.db import connection
from django.test import TestCase, override_settings

APP_LABEL = "quantem.finetune"
#: A urlconf that mounts only this package's routes, so the API tests do not
#: depend on ``core/urls.py`` having been wired up yet either.
TEST_URLCONF = "quantem.finetune.tests.urls"


@contextlib.contextmanager
def finetune_app() -> Iterator[type]:
    """Yield the ``Adapter`` model with its table present and its routes mounted.

    ``ROOT_URLCONF`` is overridden whether or not the app is installed, so these
    tests never depend on ``core/urls.py`` and only one settings override is ever
    in play — see :class:`FinetuneAppTestCase` for why a second one would be a
    trap.
    """
    installed = apps.is_installed(APP_LABEL)
    overrides: dict[str, object] = {"ROOT_URLCONF": TEST_URLCONF}
    if not installed:
        overrides["INSTALLED_APPS"] = [*settings.INSTALLED_APPS, APP_LABEL]

    with override_settings(**overrides):
        from quantem.finetune.models import Adapter

        if installed:
            # Built with the test database by ``migrate --run-syncdb``.
            yield Adapter
            return

        with connection.schema_editor() as editor:
            editor.create_model(Adapter)
        try:
            yield Adapter
        finally:
            with contextlib.suppress(Exception), connection.schema_editor() as editor:
                editor.delete_model(Adapter)


class FinetuneAppTestCase(TestCase):
    """A ``TestCase`` with the ``Adapter`` table present as ``self.Adapter``.

    Two ordering constraints, both learned the hard way:

    * The table is created **before** the class transaction opens. SQLite
      refuses schema edits inside an atomic block, so this cannot be done from
      ``setUp``.
    * The context is closed through ``addClassCleanup``, not ``tearDownClass``.
      Django registers its own settings overrides as class *cleanups*, which run
      after ``tearDownClass``; unwinding from ``tearDownClass`` therefore
      restores the settings first and lets Django's cleanup put them back
      afterwards — which leaks ``ROOT_URLCONF`` into every test that follows.
      Registering here keeps the whole stack LIFO.
    """

    @classmethod
    def setUpClass(cls):
        cls._finetune_app = finetune_app()
        cls.Adapter = cls._finetune_app.__enter__()
        cls.addClassCleanup(cls._finetune_app.__exit__, None, None, None)
        super().setUpClass()
