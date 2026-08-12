import logging
import os
import sys
import threading

from django.apps import AppConfig
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db.backends.signals import connection_created
from django.utils.module_loading import autodiscover_modules

_scheduler_started = False


def scheduler_is_running() -> bool:
    """True when this process owns the in-process job scheduler thread."""
    return _scheduler_started


def _should_autostart_scheduler() -> bool:
    if os.environ.get("QUANTEM_DISABLE_JOB_AUTOSTART") == "1":
        return False
    # A spawned job worker is not the server. It re-imports the whole app and
    # inherits QUANTEM_AUTOSTART_JOBS=1, so it would start a scheduler of its
    # own and race the real one for jobs. Exactly one process dispatches.
    if os.environ.get("QUANTEM_JOB_WORKER") == "1":
        return False
    if os.environ.get("QUANTEM_AUTOSTART_JOBS") != "1":
        return False
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "runserver":
        # The dev server's autoreloader runs the app twice; only the reloaded
        # child should own the scheduler thread.
        return os.environ.get("RUN_MAIN") == "true"
    return True


def start_scheduler_if_needed() -> None:
    global _scheduler_started
    if _scheduler_started:
        return
    if not _should_autostart_scheduler():
        return
    try:
        default_db = settings.DATABASES.get("default", {})
    except ImproperlyConfigured:
        return
    engine = default_db.get("ENGINE") if default_db else None
    if not engine or engine == "django.db.backends.dummy":
        return
    from quantem.jobs.scheduler import JobScheduler

    scheduler = JobScheduler()
    thread = threading.Thread(target=scheduler.run_forever, name="JobScheduler", daemon=True)
    thread.start()
    _scheduler_started = True
    logging.getLogger(__name__).info("Job scheduler autostarted in-process.")


def _on_connection_created(sender, connection, **kwargs) -> None:
    if connection.alias != "default":
        return
    start_scheduler_if_needed()


class JobsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "quantem.jobs"

    def ready(self):
        autodiscover_modules("handlers")
        connection_created.connect(
            _on_connection_created, dispatch_uid="quantem.jobs.scheduler.autostart"
        )
