from anylabeling.services.batch_scope import (
    BatchScope,
    should_process_image,
)
from anylabeling.services.dataset_meta import DatasetMeta


def _meta(**kwargs):
    base = dict(
        image_path="x.jpg",
        exists=True,
        has_label_file=True,
        checked=False,
        verified_empty=False,
        shape_count=0,
    )
    base.update(kwargs)
    return DatasetMeta(**base)


def test_coerce_defaults_to_all():
    assert BatchScope.coerce("all") == BatchScope.ALL
    assert BatchScope.coerce("bogus") == BatchScope.ALL
    assert BatchScope.coerce(None) == BatchScope.ALL


def test_coerce_enum_and_case():
    assert BatchScope.coerce(BatchScope.UNANNOTATED_ONLY) == (
        BatchScope.UNANNOTATED_ONLY
    )
    assert BatchScope.coerce(BatchScope.UNCHECKED_ONLY) == (
        BatchScope.UNCHECKED_ONLY
    )
    assert BatchScope.coerce("ALL") == BatchScope.ALL
    assert BatchScope.coerce("  unannotated_only ") == (
        BatchScope.UNANNOTATED_ONLY
    )


def test_all_processes_everything():
    assert should_process_image(_meta(shape_count=5), "all") is True
    assert should_process_image(None, "all") is True


def test_unannotated_only():
    assert (
        should_process_image(_meta(shape_count=0), "unannotated_only") is True
    )
    assert (
        should_process_image(_meta(shape_count=2), "unannotated_only") is False
    )
    assert (
        should_process_image(
            _meta(shape_count=0, verified_empty=True), "unannotated_only"
        )
        is False
    )


def test_unchecked_only():
    assert (
        should_process_image(
            _meta(shape_count=1, checked=False), "unchecked_only"
        )
        is True
    )
    assert (
        should_process_image(
            _meta(shape_count=1, checked=True), "unchecked_only"
        )
        is False
    )
    assert should_process_image(_meta(shape_count=0), "unchecked_only") is True


def test_corrupt_and_missing_skipped_for_scoped():
    assert (
        should_process_image(_meta(corrupt=True), "unannotated_only") is False
    )
    assert should_process_image(_meta(exists=False), "unchecked_only") is False


def test_should_process_accepts_enum_scope():
    # Regression: coerce(enum) previously collapsed to ALL.
    assert (
        should_process_image(_meta(shape_count=2), BatchScope.UNANNOTATED_ONLY)
        is False
    )
    assert (
        should_process_image(_meta(shape_count=0), BatchScope.UNANNOTATED_ONLY)
        is True
    )
