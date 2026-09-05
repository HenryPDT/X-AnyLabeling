from anylabeling.services.dataset_meta import DatasetMeta
from anylabeling.views.labeling.utils.file_sort import (
    FileSortMode,
    sort_file_entries,
)


def _meta(**kwargs):
    base = dict(
        image_path="x.jpg",
        has_label_file=True,
        checked=False,
        verified_empty=False,
        shape_count=0,
    )
    base.update(kwargs)
    return DatasetMeta(**base)


def test_default_preserves_order():
    entries = [("b.jpg", _meta()), ("a.jpg", _meta())]
    out = sort_file_entries(entries, mode="default")
    assert [f for f, _ in out] == ["b.jpg", "a.jpg"]


def test_invalid_mode_falls_back_to_default():
    entries = [("b.jpg", _meta()), ("a.jpg", _meta())]
    out = sort_file_entries(entries, mode="nope")
    assert [f for f, _ in out] == ["b.jpg", "a.jpg"]


def test_unannotated_first():
    annotated = _meta(shape_count=2)
    empty = _meta(shape_count=0)
    verified = _meta(shape_count=0, verified_empty=True)
    entries = [
        ("annot.jpg", annotated),
        ("verified.jpg", verified),
        ("empty.jpg", empty),
    ]
    out = sort_file_entries(entries, mode="unannotated_first")
    assert [f for f, _ in out][0] == "empty.jpg"


def test_unchecked_first():
    unchecked = _meta(shape_count=1, checked=False)
    checked = _meta(shape_count=1, checked=True)
    empty = _meta(shape_count=0)
    entries = [
        ("checked.jpg", checked),
        ("unchecked.jpg", unchecked),
        ("empty.jpg", empty),
    ]
    out = sort_file_entries(entries, mode="unchecked_first")
    assert [f for f, _ in out] == [
        "empty.jpg",
        "unchecked.jpg",
        "checked.jpg",
    ]


def test_most_and_fewest_marks():
    a = _meta(shape_count=5)
    b = _meta(shape_count=1)
    c = _meta(shape_count=3)
    entries = [("a.jpg", a), ("b.jpg", b), ("c.jpg", c)]
    most = sort_file_entries(entries, mode="most_marks_first")
    assert [f for f, _ in most] == ["a.jpg", "c.jpg", "b.jpg"]
    fewest = sort_file_entries(entries, mode="fewest_marks_first")
    assert [f for f, _ in fewest] == ["b.jpg", "c.jpg", "a.jpg"]


def test_verified_empty_last():
    normal = _meta(shape_count=1)
    verified = _meta(shape_count=0, verified_empty=True)
    entries = [("v.jpg", verified), ("n.jpg", normal)]
    out = sort_file_entries(entries, mode="verified_empty_last")
    assert [f for f, _ in out] == ["n.jpg", "v.jpg"]


def test_file_sort_mode_values():
    assert "default" in FileSortMode.values()
    assert (
        FileSortMode.coerce("MOST_MARKS_FIRST")
        == FileSortMode.MOST_MARKS_FIRST
    )
    assert (
        FileSortMode.coerce("most_marks_first")
        == FileSortMode.MOST_MARKS_FIRST
    )
    assert FileSortMode.coerce("  Unannotated_First ") == (
        FileSortMode.UNANNOTATED_FIRST
    )
    assert FileSortMode.coerce(None) == FileSortMode.DEFAULT


def test_natural_sort_tiebreak():
    # Same rank must keep natsort discovery order, not lexicographic.
    same = _meta(shape_count=1)
    entries = [
        ("img10.jpg", same),
        ("img2.jpg", same),
        ("img1.jpg", same),
    ]
    out = sort_file_entries(entries, mode="most_marks_first")
    assert [f for f, _ in out] == ["img1.jpg", "img2.jpg", "img10.jpg"]


def test_sort_handles_none_and_corrupt():
    ok = _meta(shape_count=0)
    corrupt = _meta(shape_count=0, corrupt=True)
    entries = [("c.jpg", corrupt), ("n.jpg", None), ("o.jpg", ok)]
    out = sort_file_entries(entries, mode="unannotated_first")
    # None meta counts as unannotated; corrupt does not.
    assert [f for f, _ in out][0] == "n.jpg"
    assert [f for f, _ in out][1] == "o.jpg"
