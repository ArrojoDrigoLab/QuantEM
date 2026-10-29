from quantem.assets.views import _deferred_upload_recovery_queryset


def test_deferred_upload_recovery_lock_does_not_require_distinct() -> None:
    queryset = _deferred_upload_recovery_queryset()

    assert queryset.query.select_for_update is True
    assert queryset.query.distinct is False
    assert "Exists" in repr(queryset.query.where)
