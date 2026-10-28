"""Progress adapters for synchronous callers of the result materializer."""


class InlineReporter:
    def update(self, **kwargs):
        pass

    def log(self, *args):
        pass


class InlineCancellation:
    def check_cancelled(self):
        pass
