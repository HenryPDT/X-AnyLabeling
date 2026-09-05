import os

import pytest
from PIL import Image
from PyQt6.QtWidgets import QApplication, QTableWidget

from anylabeling.views.labeling.widgets.review_table_base import (
    ThumbnailCache,
    crop_thumbnail_from_pil,
    export_rows_to_csv,
    get_label_color,
    setup_review_table,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_get_label_color_deterministic():
    c1 = get_label_color("cat")
    c2 = get_label_color("cat")
    c3 = get_label_color("dog")
    assert c1.getHsv() == c2.getHsv()
    assert c1.getHsv() != c3.getHsv()


def test_crop_thumbnail_from_pil(qapp):
    img = Image.new("RGB", (200, 200), color=(255, 0, 0))
    pix = crop_thumbnail_from_pil(img, [10, 10, 50, 50])
    assert pix is not None
    assert not pix.isNull()
    assert crop_thumbnail_from_pil(img, []) is None
    assert crop_thumbnail_from_pil(img, [5, 5, 5, 5]) is not None


def test_thumbnail_cache_eviction():
    cache = ThumbnailCache(max_entries=2)
    cache.put(("a", 0), object())
    cache.put(("b", 0), object())
    cache.put(("c", 0), object())
    assert len(cache) == 2
    assert ("a", 0) not in cache
    assert ("c", 0) in cache
    cache.clear()
    assert len(cache) == 0


def test_thumbnail_cache_does_not_store_none():
    cache = ThumbnailCache(max_entries=2)
    cache.put(("a", 0), None)
    assert len(cache) == 0
    assert ("a", 0) not in cache
    assert cache.get(("a", 0)) is None


def test_review_table_dialog_timer_connected(qapp):
    from anylabeling.views.labeling.widgets.review_table_base import (
        ReviewTableDialog,
    )

    dialog = ReviewTableDialog()
    # Timer must be wired (previously dead: singleShot with no receiver).
    receivers = dialog._thumb_timer.receivers(dialog._thumb_timer.timeout)
    assert receivers >= 1
    dialog.schedule_thumbnail_load()
    assert dialog._thumb_timer.isActive()
    dialog.close()


def test_setup_review_table(qapp):
    table = QTableWidget()
    setup_review_table(table, ["Thumb", "Class", "File"])
    assert table.columnCount() == 3
    assert (
        table.selectionBehavior() == QTableWidget.SelectionBehavior.SelectRows
    )


def test_export_rows_to_csv(tmp_path):
    out = os.path.join(str(tmp_path), "out.csv")
    ok = export_rows_to_csv(out, ["A", "B"], [["x", 1], ["y", None]])
    assert ok is True
    with open(out, encoding="utf-8") as f:
        text = f.read()
    assert "A,B" in text
    assert "x,1" in text
