import json
import os
import tempfile
from PIL import Image
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtWidgets import QApplication

from anylabeling.services.dataset_meta import VERIFIED_EMPTY_FIELD_PATTERN
from anylabeling.views.labeling.label_file import LabelFile
from anylabeling.views.labeling.label_widget import (
    _create_verified_background_icon,
)
from anylabeling.views.labeling.schema import (
    VERIFIED_EMPTY_FIELD,
    create_xlabel_template,
)
from anylabeling.views.labeling.widgets.canvas import Canvas


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_schema_verified_empty_field():
    assert VERIFIED_EMPTY_FIELD == "verified_empty"
    tpl = create_xlabel_template()
    assert "version" in tpl
    assert "shapes" in tpl


def test_verified_empty_pattern_regex():
    s_true = '{"version": "1.0", "verified_empty": true, "shapes": []}'
    match = VERIFIED_EMPTY_FIELD_PATTERN.search(s_true)
    assert match is not None
    assert match.group(1) == "true"

    s_false = '{"version": "1.0", "verified_empty": false, "shapes": []}'
    match_f = VERIFIED_EMPTY_FIELD_PATTERN.search(s_false)
    assert match_f is not None
    assert match_f.group(1) == "false"

    s_none = '{"version": "1.0", "shapes": []}'
    assert VERIFIED_EMPTY_FIELD_PATTERN.search(s_none) is None


def test_save_and_load_verified_empty_in_label_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "bg.jpg")
        im = Image.new("RGB", (64, 64), color=(30, 30, 30))
        im.save(img_path)

        lbl_path = os.path.join(tmp_dir, "bg.json")
        lf = LabelFile()
        lf.save(
            filename=lbl_path,
            shapes=[],
            image_path="bg.jpg",
            image_height=64,
            image_width=64,
            other_data={VERIFIED_EMPTY_FIELD: True, "checked": True},
        )

        assert os.path.isfile(lbl_path)
        with open(lbl_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data[VERIFIED_EMPTY_FIELD] is True
        assert data["checked"] is True
        assert len(data["shapes"]) == 0

        # Load back
        lf_loaded = LabelFile(filename=lbl_path, image_dir=tmp_dir)
        assert lf_loaded.other_data.get(VERIFIED_EMPTY_FIELD) is True


def test_create_verified_background_icon(qapp):
    icon = _create_verified_background_icon()
    assert icon is not None
    assert not icon.isNull()


def test_canvas_verified_background_paint(qapp):
    from PyQt6.QtWidgets import QWidget

    canvas = Canvas(parent=QWidget())
    pix = QPixmap(200, 200)
    pix.fill(Qt.GlobalColor.white)
    canvas.load_pixmap(pix)
    canvas.resize(400, 400)
    canvas.verified_empty = True
    assert len(canvas.shapes) == 0

    # Ensure paint does not throw error
    buffer_pixmap = QPixmap(400, 400)
    buffer_pixmap.fill(Qt.GlobalColor.black)
    painter = QPainter(buffer_pixmap)
    canvas._paint_verified_background_badge(painter)
    painter.end()
    assert not buffer_pixmap.isNull()


def test_clear_verified_empty_clears_state(qapp):
    from unittest.mock import MagicMock
    from anylabeling.views.labeling.label_widget import LabelingWidget

    widget = MagicMock(spec=LabelingWidget)
    widget.other_data = {VERIFIED_EMPTY_FIELD: True}
    widget.canvas = MagicMock()
    widget.canvas.verified_empty = True
    widget.canvas.shapes = []

    # Bind the real method to the mock
    LabelingWidget._clear_verified_empty(widget)

    assert widget.other_data[VERIFIED_EMPTY_FIELD] is False
    assert widget.canvas.verified_empty is False
    widget.canvas.update.assert_called_once()
    widget._update_current_file_checked_item.assert_called_once()
