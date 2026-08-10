from django.apps import AppConfig


class FinetuneConfig(AppConfig):
    """Guided fine-tuning: adapters the user fits to their own annotations.

    Needs ``"quantem.finetune"`` in ``INSTALLED_APPS``; without it the
    :class:`~quantem.finetune.models.Adapter` table does not exist and the job
    still runs, it just has nowhere to record the result (see
    :mod:`quantem.finetune.job`).
    """

    name = "quantem.finetune"
    label = "finetune"
    verbose_name = "Guided fine-tuning"
