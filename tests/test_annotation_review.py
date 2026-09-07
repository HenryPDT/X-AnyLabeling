import json
import os
import tempfile
from PIL import Image
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication, QMessageBox

from anylabeling.views.labeling.widgets.annotation_review_dialog import (
    AnnotationReviewDialog,
    ReviewItem,
    collect_dataset_review_items,
    collect_review_items_for_image,
    crop_thumbnail_from_pil,
    extract_thumbnail_pixmap,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_review_item_properties():
    item_normal = ReviewItem(
        image_path="/path/test.jpg",
        label_file="/path/test.json",
        shape_index=0,
        label="dog",
        shape_type="rectangle",
        points=[[10, 10], [50, 50]],
        xyxy=[10, 10, 50, 50],
        width=40,
        height=40,
        area=1600,
        aspect_ratio=1.0,
        overlap_iou=0.0,
    )
    assert item_normal.extreme_aspect_ratio_score == 1.0

    # Very wide box: 100x10 -> aspect ratio 10.0
    item_wide = ReviewItem(
        image_path="/path/test.jpg",
        label_file="/path/test.json",
        shape_index=1,
        label="strip",
        shape_type="rectangle",
        points=[[0, 0], [100, 10]],
        xyxy=[0, 0, 100, 10],
        width=100,
        height=10,
        area=1000,
        aspect_ratio=10.0,
        overlap_iou=0.5,
    )
    assert item_wide.extreme_aspect_ratio_score == 10.0

    # Very tall/narrow box: 10x100 -> aspect ratio 0.1
    item_tall = ReviewItem(
        image_path="/path/test.jpg",
        label_file="/path/test.json",
        shape_index=2,
        label="pole",
        shape_type="rectangle",
        points=[[0, 0], [10, 100]],
        xyxy=[0, 0, 10, 100],
        width=10,
        height=100,
        area=1000,
        aspect_ratio=0.1,
        overlap_iou=0.1,
    )
    assert abs(item_tall.extreme_aspect_ratio_score - 10.0) < 1e-4


def test_collect_review_items_and_overlap_computation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "test.png")
        # Create small test image
        im = Image.new("RGB", (200, 200), color=(100, 150, 200))
        im.save(img_path)

        lbl_path = os.path.join(tmp_dir, "test.json")
        lbl_data = {
            "imagePath": "test.png",
            "shapes": [
                {
                    "label": "car",
                    "shape_type": "rectangle",
                    "points": [[10, 10], [50, 50]],
                },
                {
                    "label": "car_dup",
                    "shape_type": "rectangle",
                    "points": [
                        [10, 10],
                        [50, 50],
                    ],  # Exact overlap (IoU = 1.0)
                },
                {
                    "label": "person",
                    "shape_type": "rectangle",
                    "points": [[100, 100], [110, 150]],  # Disjoint
                },
            ],
        }
        with open(lbl_path, "w", encoding="utf-8") as f:
            json.dump(lbl_data, f)

        items = collect_review_items_for_image(img_path)
        assert len(items) == 3

        # First two should have max overlap IoU of 1.0
        assert items[0].overlap_iou == 1.0
        assert items[1].overlap_iou == 1.0
        # Third shape is disjoint
        assert items[2].overlap_iou == 0.0

        # Dataset collector
        all_dataset_items = collect_dataset_review_items([img_path])
        assert len(all_dataset_items) == 3


def test_crop_thumbnail_from_pil(qapp):
    im = Image.new("RGB", (100, 100), color=(0, 255, 0))
    # Normal crop
    pix = crop_thumbnail_from_pil(im, [10, 10, 60, 60], max_size=32)
    assert pix is not None
    assert not pix.isNull()
    assert pix.width() <= 32
    assert pix.height() <= 32

    # Degenerate crops: point fallback returns context window, inverted is None
    point_pix = crop_thumbnail_from_pil(im, [50, 50, 50, 50])
    assert point_pix is not None and not point_pix.isNull()
    assert crop_thumbnail_from_pil(im, [60, 60, 40, 40]) is None


def test_extract_thumbnail_pixmap(qapp):
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "sample.png")
        im = Image.new("RGB", (100, 100), color=(255, 0, 0))
        im.save(img_path)

        # Valid box crop
        pix = extract_thumbnail_pixmap(img_path, [10, 10, 50, 50], max_size=40)
        assert pix is not None
        assert not pix.isNull()
        assert pix.width() <= 40
        assert pix.height() <= 40

        # Point box returns fallback context window (not None)
        pix_point = extract_thumbnail_pixmap(img_path, [100, 100, 100, 100])
        assert pix_point is not None and not pix_point.isNull()

        # Missing file
        pix_missing = extract_thumbnail_pixmap(
            os.path.join(tmp_dir, "missing.png"), [0, 0, 10, 10]
        )
        assert pix_missing is None


def test_matches_filters():
    item = ReviewItem(
        image_path="/path/dog_photo.jpg",
        label_file="/path/dog_photo.json",
        shape_index=0,
        label="dog",
        shape_type="rectangle",
        points=[[10, 10], [50, 50]],
        xyxy=[10, 10, 50, 50],
        width=40,
        height=40,
        area=1600,
        aspect_ratio=1.0,
        overlap_iou=0.0,
    )
    # When sel_class is None (all classes)
    assert AnnotationReviewDialog._matches_filters(None, item, None, "", "")
    assert AnnotationReviewDialog._matches_filters(
        None, item, "dog", "rectangle", "photo"
    )
    # Class mismatch
    assert not AnnotationReviewDialog._matches_filters(
        None, item, "cat", "", ""
    )
    # Shape type mismatch
    assert not AnnotationReviewDialog._matches_filters(
        None, item, None, "polygon", ""
    )
    # Search text mismatch
    assert not AnnotationReviewDialog._matches_filters(
        None, item, None, "", "cat"
    )


def test_lazy_thumbnail_loading_and_debouncing(qapp):
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "test_item.png")
        im = Image.new("RGB", (200, 200), color=(50, 100, 150))
        im.save(img_path)

        dialog = AnnotationReviewDialog(parent=None)
        # Verify dialog initialized empty with inactive timers
        assert dialog.table.rowCount() == 0
        assert not dialog.search_timer.isActive()

        # Supply mock items
        items = [
            ReviewItem(
                image_path=img_path,
                label_file="",
                shape_index=i,
                label=f"obj_{i}",
                shape_type="rectangle",
                points=[[10, 10], [50, 50]],
                xyxy=[10, 10, 50, 50],
                width=40,
                height=40,
                area=1600,
                aspect_ratio=1.0,
                overlap_iou=0.0,
            )
            for i in range(25)
        ]
        dialog.all_items = items
        dialog.filtered_items = list(items)

        # Populating table must be fast: items created with no icons yet
        dialog._populate_table()
        assert dialog.table.rowCount() == 25
        assert (img_path, 0) not in dialog.thumbnail_cache
        assert dialog.table.item(0, 0).icon().isNull()

        # Loading visible thumbnails loads the visible batch into cache and table
        dialog._load_visible_thumbnails()
        assert (img_path, 0) in dialog.thumbnail_cache
        assert dialog.thumbnail_cache[(img_path, 0)] is not None
        assert not dialog.table.item(0, 0).icon().isNull()

        # Search typing triggers the search timer debounce
        dialog.search_input.setText("obj_1")
        assert dialog.search_timer.isActive()

        dialog.close()


def test_apply_shape_modifications_descending_order():
    """Verify reverse-index deletion prevents index-drift bugs and preserves survivor indices."""
    from anylabeling.services.annotation_diagnostics import (
        apply_shape_modifications,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        lbl_path = os.path.join(tmp_dir, "test_drift.json")
        orig_data = {
            "imagePath": "test_drift.jpg",
            "shapes": [
                {"label": "shape0", "points": [[0, 0], [10, 10]]},
                {"label": "shape1", "points": [[10, 10], [20, 20]]},
                {"label": "shape2", "points": [[20, 20], [30, 30]]},
                {"label": "shape3", "points": [[30, 30], [40, 40]]},
                {"label": "shape4", "points": [[40, 40], [50, 50]]},
            ],
        }
        with open(lbl_path, "w", encoding="utf-8") as f:
            json.dump(orig_data, f, indent=2)

        # Reclassify 0 to "first", 4 to "last"
        # Delete indices 1 and 3 (in any order in the set)
        ok = apply_shape_modifications(
            label_file=lbl_path,
            deletions={3, 1},
            reclasses={0: "first", 4: "last"},
        )
        assert ok is True

        # Check modified file
        with open(lbl_path, "r", encoding="utf-8") as f:
            res = json.load(f)

        shapes = res["shapes"]
        assert len(shapes) == 3
        # Original 0 is reclassified to "first"
        assert shapes[0]["label"] == "first"
        assert shapes[0]["points"] == [[0, 0], [10, 10]]
        # Original 2 is untouched
        assert shapes[1]["label"] == "shape2"
        assert shapes[1]["points"] == [[20, 20], [30, 30]]
        # Original 4 is reclassified to "last" and survived
        assert shapes[2]["label"] == "last"
        assert shapes[2]["points"] == [[40, 40], [50, 50]]

        # No backup directories are created.
        assert [
            d
            for d in os.listdir(tmp_dir)
            if d.startswith(".backup_")
            and os.path.isdir(os.path.join(tmp_dir, d))
        ] == []


def test_apply_shape_modifications_edge_cases():
    """Verify robust error handling for missing files, invalid JSON, and out-of-range indices."""
    from anylabeling.services.annotation_diagnostics import (
        apply_shape_modifications,
    )

    assert (
        apply_shape_modifications("/nonexistent/file.json", {0}, {}) is False
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        lbl_path = os.path.join(tmp_dir, "test_corrupt.json")
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("NOT_JSON")
        assert apply_shape_modifications(lbl_path, {0}, {}) is False

        # Out-of-bounds indices should be gracefully ignored without crashing
        lbl_valid = os.path.join(tmp_dir, "test_valid.json")
        with open(lbl_valid, "w", encoding="utf-8") as f:
            json.dump(
                {"shapes": [{"label": "dog", "points": [[0, 0], [1, 1]]}]},
                f,
            )
        assert (
            apply_shape_modifications(
                lbl_valid, deletions={99}, reclasses={100: "cat"}
            )
            is True
        )
        with open(lbl_valid, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["shapes"]) == 1
        assert data["shapes"][0]["label"] == "dog"


def test_review_gallery_staging_and_undo(qapp):
    """Verify staging deletions, reclassifications, banner status, and undo stack."""
    dialog = AnnotationReviewDialog(parent=None)
    items = [
        ReviewItem(
            image_path="/path/test1.jpg",
            label_file="/path/test1.json",
            shape_index=0,
            label="cat",
            shape_type="rectangle",
            points=[[0, 0], [10, 10]],
            xyxy=[0, 0, 10, 10],
            width=10,
            height=10,
            area=100,
            aspect_ratio=1.0,
            overlap_iou=0.0,
        ),
        ReviewItem(
            image_path="/path/test1.jpg",
            label_file="/path/test1.json",
            shape_index=1,
            label="dog",
            shape_type="rectangle",
            points=[[10, 10], [20, 20]],
            xyxy=[10, 10, 20, 20],
            width=10,
            height=10,
            area=100,
            aspect_ratio=1.0,
            overlap_iou=0.0,
        ),
    ]
    dialog.all_items = items
    dialog.filtered_items = list(items)
    dialog._populate_table()

    assert not dialog.has_staged_changes()
    assert dialog.banner_staged.isHidden()

    # Stage row 0 for deletion
    dialog.toggle_stage_delete_selected_rows([0])
    assert dialog.has_staged_changes()
    assert ("/path/test1.jpg", 0) in dialog.staged_deletions
    assert not dialog.banner_staged.isHidden()
    assert "[Pending Delete]" in dialog.table.item(0, 1).text()
    assert dialog.table.item(0, 1).font().strikeOut()

    # Stage row 1 for reclassification to "bird"
    dialog.stage_reclass_selected_rows([1], "bird")
    assert ("/path/test1.jpg", 1) in dialog.staged_reclasses
    assert dialog.staged_reclasses[("/path/test1.jpg", 1)] == "bird"
    assert "dog → bird  [Staged]" in dialog.table.item(1, 1).text()

    # Undo reclassification
    dialog.undo_last_staged_action()
    assert ("/path/test1.jpg", 1) not in dialog.staged_reclasses
    assert dialog.table.item(1, 1).text() == "dog"

    # Undo deletion
    dialog.undo_last_staged_action()
    assert not dialog.has_staged_changes()
    assert dialog.banner_staged.isHidden()
    assert "[Pending Delete]" not in dialog.table.item(0, 1).text()

    dialog.close()


def test_review_gallery_column_sorting(qapp):
    """Verify interactive column sorting across Class, Type, Area, Aspect Ratio, IoU, and Image."""
    dialog = AnnotationReviewDialog(parent=None)
    items = [
        ReviewItem(
            image_path="/dir/b_img.jpg",
            label_file="",
            shape_index=0,
            label="zebra",
            shape_type="polygon",
            points=[],
            xyxy=[0, 0, 10, 10],
            width=10,
            height=10,
            area=100,
            aspect_ratio=1.0,
            overlap_iou=0.1,
        ),
        ReviewItem(
            image_path="/dir/a_img.jpg",
            label_file="",
            shape_index=1,
            label="apple",
            shape_type="rectangle",
            points=[],
            xyxy=[0, 0, 50, 20],
            width=50,
            height=20,
            area=1000,
            aspect_ratio=2.5,
            overlap_iou=0.9,
        ),
    ]
    dialog.all_items = items
    dialog.filtered_items = list(items)

    # Sort col 1 (Class): Ascending
    sorted_asc = dialog._sort_items_by_column(items, 1, ascending=True)
    assert sorted_asc[0].label == "apple"
    assert sorted_asc[1].label == "zebra"

    # Sort col 1 (Class): Descending
    sorted_desc = dialog._sort_items_by_column(items, 1, ascending=False)
    assert sorted_desc[0].label == "zebra"
    assert sorted_desc[1].label == "apple"

    # Sort col 3 (Area): Ascending
    sorted_area_asc = dialog._sort_items_by_column(items, 3, ascending=True)
    assert sorted_area_asc[0].area == 100
    assert sorted_area_asc[1].area == 1000

    # Sort col 2 (Type): Ascending -> polygon, rectangle
    sorted_type_asc = dialog._sort_items_by_column(items, 2, ascending=True)
    assert sorted_type_asc[0].shape_type == "polygon"
    assert sorted_type_asc[1].shape_type == "rectangle"

    # Sort col 4 (Aspect Ratio): Ascending -> 1.0, 2.5
    sorted_ar_asc = dialog._sort_items_by_column(items, 4, ascending=True)
    assert sorted_ar_asc[0].aspect_ratio == 1.0
    assert sorted_ar_asc[1].aspect_ratio == 2.5

    # Sort col 5 (Max Overlap IoU): Descending
    sorted_iou_desc = dialog._sort_items_by_column(items, 5, ascending=False)
    assert sorted_iou_desc[0].overlap_iou == 0.9
    assert sorted_iou_desc[1].overlap_iou == 0.1

    # Sort col 6 (Image Filename): Ascending
    sorted_img_asc = dialog._sort_items_by_column(items, 6, ascending=True)
    assert "a_img" in sorted_img_asc[0].image_path
    assert "b_img" in sorted_img_asc[1].image_path

    # Test header clicking simulation
    dialog.on_header_clicked(1)  # Class Ascending
    assert dialog._sort_col == 1
    assert dialog._sort_ascending is True
    assert dialog.filtered_items[0].label == "apple"

    dialog.on_header_clicked(1)  # Class Descending
    assert dialog._sort_ascending is False
    assert dialog.filtered_items[0].label == "zebra"

    dialog.close()


def test_review_gallery_keyboard_focus_guard(qapp):
    """Verify typing inside QLineEdit does not trigger Delete or reclass shortcuts."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeyEvent

    dialog = AnnotationReviewDialog(parent=None)
    items = [
        ReviewItem(
            image_path="/path/test.jpg",
            label_file="",
            shape_index=0,
            label="cat",
            shape_type="rectangle",
            points=[],
            xyxy=[0, 0, 10, 10],
            width=10,
            height=10,
            area=100,
            aspect_ratio=1.0,
            overlap_iou=0.0,
        )
    ]
    dialog.all_items = items
    dialog.filtered_items = list(items)
    dialog._populate_table()
    dialog.table.selectRow(0)

    # Focus is on search input
    dialog.search_input.setFocus()
    assert dialog.focusWidget() == dialog.search_input

    # Send Delete key while focus is on search_input
    del_event = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_Delete,
        Qt.KeyboardModifier.NoModifier,
    )
    dialog.keyPressEvent(del_event)
    # Row must NOT be staged for deletion!
    assert ("/path/test.jpg", 0) not in dialog.staged_deletions

    # Focus is on table
    dialog.table.setFocus()
    assert dialog.focusWidget() == dialog.table
    dialog.keyPressEvent(del_event)
    # Row MUST be staged for deletion!
    assert ("/path/test.jpg", 0) in dialog.staged_deletions

    # Toggle again to unstage before closing to prevent unsaved changes dialog prompt
    dialog.keyPressEvent(del_event)
    assert not dialog.has_staged_changes()
    dialog.close()


def test_review_gallery_batch_commit(qapp, monkeypatch):
    """Verify batch commit applies modifications to disk, clears staging, and creates no backup."""
    from PyQt6.QtWidgets import QMessageBox

    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "test_commit.jpg")
        Image.new("RGB", (100, 100)).save(img_path)
        lbl_path = os.path.join(tmp_dir, "test_commit.json")
        data = {
            "imagePath": "test_commit.jpg",
            "shapes": [
                {
                    "label": "cat",
                    "shape_type": "rectangle",
                    "points": [[0, 0], [10, 10]],
                },
                {
                    "label": "dog",
                    "shape_type": "rectangle",
                    "points": [[20, 20], [30, 30]],
                },
            ],
        }
        with open(lbl_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        dialog = AnnotationReviewDialog(parent=None)
        items = [
            ReviewItem(
                image_path=img_path,
                label_file=lbl_path,
                shape_index=0,
                label="cat",
                shape_type="rectangle",
                points=[[0, 0], [10, 10]],
                xyxy=[0, 0, 10, 10],
                width=10,
                height=10,
                area=100,
                aspect_ratio=1.0,
                overlap_iou=0.0,
            ),
            ReviewItem(
                image_path=img_path,
                label_file=lbl_path,
                shape_index=1,
                label="dog",
                shape_type="rectangle",
                points=[[20, 20], [30, 30]],
                xyxy=[20, 20, 30, 30],
                width=10,
                height=10,
                area=100,
                aspect_ratio=1.0,
                overlap_iou=0.0,
            ),
        ]
        dialog.all_items = items
        dialog.filtered_items = list(items)
        dialog._populate_table()

        # Stage row 0 for deletion, row 1 for reclass to "wolf"
        dialog.toggle_stage_delete_selected_rows([0])
        dialog.stage_reclass_selected_rows([1], "wolf")
        assert dialog.has_staged_changes()

        # Mock QMessageBox.question and information
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
        )
        monkeypatch.setattr(
            QMessageBox, "information", lambda *args, **kwargs: None
        )

        dialog.apply_staged_changes()

        assert not dialog.has_staged_changes()
        with open(lbl_path, "r", encoding="utf-8") as f:
            saved = json.load(f)

        # Only 1 shape should remain, with label "wolf"
        assert len(saved["shapes"]) == 1
        assert saved["shapes"][0]["label"] == "wolf"
        assert saved["shapes"][0]["points"] == [[20, 20], [30, 30]]

        # Confirmed UI writes must not litter .backup_* directories.
        assert [
            d for d in os.listdir(tmp_dir) if d.startswith(".backup_")
        ] == []

        dialog.close()


def test_review_gallery_quick_digit_reclass_and_numpad(qapp):
    dialog = AnnotationReviewDialog(parent=None)
    items = [
        ReviewItem(
            image_path="/path/test1.jpg",
            label_file="/path/test1.json",
            shape_index=0,
            label="cat",
            shape_type="rectangle",
            points=[[0, 0], [10, 10]],
            xyxy=[0, 0, 10, 10],
            width=10,
            height=10,
            area=100,
            aspect_ratio=1.0,
            overlap_iou=0.0,
        ),
        ReviewItem(
            image_path="/path/test1.jpg",
            label_file="/path/test1.json",
            shape_index=1,
            label="dog",
            shape_type="rectangle",
            points=[[20, 20], [30, 30]],
            xyxy=[20, 20, 30, 30],
            width=10,
            height=10,
            area=100,
            aspect_ratio=1.0,
            overlap_iou=0.0,
        ),
    ]
    dialog.all_items = items
    dialog.filtered_items = list(items)
    dialog.combo_class.clear()
    dialog.combo_class.addItem("All Classes")
    dialog.combo_class.addItem("cat")  # Key 1
    dialog.combo_class.addItem("dog")  # Key 2
    dialog.combo_class.addItem("car")  # Key 3
    dialog._populate_table()

    # 1. No selection -> press key 1
    dialog.table.clearSelection()
    dialog.table.setCurrentCell(-1, -1)
    ev1 = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_1,
        Qt.KeyboardModifier.NoModifier,
        "1",
    )
    dialog.keyPressEvent(ev1)
    assert "Select a row first" in dialog.lbl_count.text()
    assert not dialog.has_staged_changes()

    # 2. Select row 0 (cat) and press key 2 (dog)
    dialog.table.selectRow(0)
    ev2 = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_2,
        Qt.KeyboardModifier.NoModifier,
        "2",
    )
    dialog.keyPressEvent(ev2)
    assert ("/path/test1.jpg", 0) in dialog.staged_reclasses
    assert dialog.staged_reclasses[("/path/test1.jpg", 0)] == "dog"

    # 3. Test NumPad key: Press Key_3 with KeypadModifier on row 0 to reclass to "car"
    numpad3 = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_3,
        Qt.KeyboardModifier.KeypadModifier,
        "3",
    )
    dialog.keyPressEvent(numpad3)
    assert dialog.staged_reclasses[("/path/test1.jpg", 0)] == "car"

    # 4. Test unassigned digit (e.g. 9)
    ev9 = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_9,
        Qt.KeyboardModifier.NoModifier,
        "9",
    )
    dialog.keyPressEvent(ev9)
    assert "unassigned" in dialog.lbl_count.text().lower()

    # Discard before closing
    dialog.staged_reclasses.clear()
    dialog.close()


def test_review_gallery_instant_row_appearance_update(qapp, monkeypatch):
    dialog = AnnotationReviewDialog(parent=None)
    items = [
        ReviewItem(
            image_path="/path/test1.jpg",
            label_file="/path/test1.json",
            shape_index=0,
            label="cat",
            shape_type="rectangle",
            points=[[0, 0], [10, 10]],
            xyxy=[0, 0, 10, 10],
            width=10,
            height=10,
            area=100,
            aspect_ratio=1.0,
            overlap_iou=0.0,
        ),
    ]
    dialog.all_items = items
    dialog.filtered_items = list(items)
    dialog._populate_table()

    # Spy on _populate_table to ensure it is NOT called during row staging
    populate_called = False

    def spy_populate():
        nonlocal populate_called
        populate_called = True

    monkeypatch.setattr(dialog, "_populate_table", spy_populate)

    # 1. Stage delete row 0
    dialog.toggle_stage_delete_selected_rows([0])
    assert not populate_called, "_populate_table was unexpectedly called!"
    item_cls = dialog.table.item(0, 1)
    assert "[Pending Delete]" in item_cls.text()
    assert item_cls.font().strikeOut() is True

    # 2. Reclass row 0
    dialog.stage_reclass_selected_rows([0], "dog")
    assert not populate_called, "_populate_table was unexpectedly called!"
    assert "cat → dog  [Staged]" in item_cls.text()
    assert item_cls.font().strikeOut() is False

    dialog.staged_deletions.clear()
    dialog.staged_reclasses.clear()
    dialog.close()


def test_review_gallery_multi_row_immediate_delete(qapp, monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "batch_del.jpg")
        Image.new("RGB", (50, 50)).save(img_path)
        lbl_path = os.path.join(tmp_dir, "batch_del.json")
        data = {
            "imagePath": "batch_del.jpg",
            "shapes": [
                {
                    "label": "a",
                    "shape_type": "rectangle",
                    "points": [[0, 0], [5, 5]],
                },
                {
                    "label": "b",
                    "shape_type": "rectangle",
                    "points": [[5, 5], [10, 10]],
                },
                {
                    "label": "c",
                    "shape_type": "rectangle",
                    "points": [[10, 10], [15, 15]],
                },
            ],
        }
        with open(lbl_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        dialog = AnnotationReviewDialog(parent=None)
        items = [
            ReviewItem(
                image_path=img_path,
                label_file=lbl_path,
                shape_index=0,
                label="a",
                shape_type="rectangle",
                points=[[0, 0], [5, 5]],
                xyxy=[0, 0, 5, 5],
                width=5,
                height=5,
                area=25,
                aspect_ratio=1.0,
                overlap_iou=0.0,
            ),
            ReviewItem(
                image_path=img_path,
                label_file=lbl_path,
                shape_index=1,
                label="b",
                shape_type="rectangle",
                points=[[5, 5], [10, 10]],
                xyxy=[5, 5, 10, 10],
                width=5,
                height=5,
                area=25,
                aspect_ratio=1.0,
                overlap_iou=0.0,
            ),
            ReviewItem(
                image_path=img_path,
                label_file=lbl_path,
                shape_index=2,
                label="c",
                shape_type="rectangle",
                points=[[10, 10], [15, 15]],
                xyxy=[10, 10, 15, 15],
                width=5,
                height=5,
                area=25,
                aspect_ratio=1.0,
                overlap_iou=0.0,
            ),
        ]
        dialog.all_items = list(items)
        dialog.filtered_items = list(items)
        dialog._populate_table()

        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
        )

        dialog.delete_selected_shapes_immediately([0, 1])

        with open(lbl_path, "r", encoding="utf-8") as f:
            updated = json.load(f)

        # Only shape "c" should remain
        assert len(updated["shapes"]) == 1
        assert updated["shapes"][0]["label"] == "c"

        dialog.close()


def test_review_gallery_instant_discard_and_escape(qapp, monkeypatch):
    dialog = AnnotationReviewDialog(parent=None)
    items = [
        ReviewItem(
            image_path="/path/test1.jpg",
            label_file="/path/test1.json",
            shape_index=0,
            label="cat",
            shape_type="rectangle",
            points=[[0, 0], [10, 10]],
            xyxy=[0, 0, 10, 10],
            width=10,
            height=10,
            area=100,
            aspect_ratio=1.0,
            overlap_iou=0.0,
        ),
        ReviewItem(
            image_path="/path/test1.jpg",
            label_file="/path/test1.json",
            shape_index=1,
            label="dog",
            shape_type="rectangle",
            points=[[20, 20], [30, 30]],
            xyxy=[20, 20, 30, 30],
            width=10,
            height=10,
            area=100,
            aspect_ratio=1.0,
            overlap_iou=0.0,
        ),
    ]
    dialog.all_items = items
    dialog.filtered_items = list(items)
    dialog._populate_table()

    # 1. Stage row 0 for deletion and row 1 for reclass
    dialog.toggle_stage_delete_selected_rows([0])
    dialog.stage_reclass_selected_rows([1], "car")
    assert dialog.has_staged_changes()
    assert not dialog.banner_staged.isHidden()
    assert "Pending Delete" in dialog.table.item(0, 1).text()
    assert "Staged" in dialog.table.item(1, 1).text()

    # 2. Call discard_staged_changes directly -> instant in-place revert
    dialog.discard_staged_changes()
    assert not dialog.has_staged_changes()
    assert dialog.banner_staged.isHidden()
    assert dialog.table.item(0, 1).text() == "cat"
    assert not dialog.table.item(0, 1).font().strikeOut()
    assert dialog.table.item(1, 1).text() == "dog"

    # 3. Stage again and test Escape key shortcut through eventFilter with confirmation
    dialog.toggle_stage_delete_selected_rows([0])
    assert dialog.has_staged_changes()

    esc_ev = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_Escape,
        Qt.KeyboardModifier.NoModifier,
    )
    # If user cancels/declines confirmation, changes stay staged
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.No,
    )
    handled = dialog.eventFilter(dialog.table.viewport(), esc_ev)
    assert handled is True
    assert dialog.has_staged_changes()

    # If user confirms discard, changes are reverted
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )
    handled = dialog.eventFilter(dialog.table.viewport(), esc_ev)
    assert handled is True
    assert not dialog.has_staged_changes()
    assert dialog.table.item(0, 1).text() == "cat"

    dialog.close()


def _make_review_item(image_path, label_file, shape_index, label):
    return ReviewItem(
        image_path=image_path,
        label_file=label_file,
        shape_index=shape_index,
        label=label,
        shape_type="rectangle",
        points=[[0, 0], [10, 10]],
        xyxy=[0, 0, 10, 10],
        width=10,
        height=10,
        area=100,
        aspect_ratio=1.0,
        overlap_iou=0.0,
    )


def test_review_gallery_apply_preserves_file_order(qapp, monkeypatch):
    """Verify post-Apply row order matches a full Reload (no jump to bottom)."""
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        QMessageBox, "information", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)

    with tempfile.TemporaryDirectory() as tmp_dir:
        names = []
        for i, name in enumerate(["f_a", "f_b", "f_c", "f_d"]):
            img_path = os.path.join(tmp_dir, name + ".jpg")
            Image.new("RGB", (40, 40)).save(img_path)
            names.append(img_path)
            with open(os.path.join(tmp_dir, name + ".json"), "w") as f:
                json.dump(
                    {
                        "imagePath": name + ".jpg",
                        "shapes": [
                            {
                                "label": f"L{i}0",
                                "shape_type": "rectangle",
                                "points": [[0, 0], [5, 5]],
                            },
                            {
                                "label": f"L{i}1",
                                "shape_type": "rectangle",
                                "points": [[5, 5], [10, 10]],
                            },
                        ],
                    },
                    f,
                )

        dialog = AnnotationReviewDialog(parent=None)
        dialog.get_image_file_list = lambda: names
        dialog.get_output_dir = lambda: tmp_dir
        dialog.reload_gallery(silent=True)

        def snap():
            return [
                (it.image_path, it.shape_index, it.label)
                for it in dialog.filtered_items
            ]

        # Delete f_a idx1, reclass f_c idx0.
        dialog.toggle_stage_delete_selected_rows([1])
        dialog.stage_reclass_selected_rows([4], "ZED")
        dialog.apply_staged_changes()
        after = snap()

        dialog.reload_gallery(silent=True)
        assert after == snap()
        assert after[0] == (names[0], 0, "L00")
        assert (names[2], 0, "ZED") in after

        dialog.close()


def test_review_gallery_close_event_matrix(qapp, monkeypatch):
    """Verify closeEvent Yes/No/Cancel handling of staged changes."""
    from unittest.mock import patch
    from PyQt6.QtGui import QCloseEvent
    import anylabeling.views.labeling.widgets.annotation_review_dialog as mod

    dialog = AnnotationReviewDialog(parent=None)
    dialog.all_items = [
        _make_review_item("/p/a.jpg", "/p/a.json", 0, "cat"),
        _make_review_item("/p/a.jpg", "/p/a.json", 1, "dog"),
    ]
    dialog.filtered_items = list(dialog.all_items)
    dialog._populate_table()

    def close_with(answer):
        dialog.toggle_stage_delete_selected_rows([0])

        class FakeBox:
            Icon = QMessageBox.Icon
            StandardButton = QMessageBox.StandardButton

            def __init__(self, *a, **k):
                pass

            def setWindowModality(self, *a):
                pass

            def exec(self):
                return answer

        with patch.object(mod, "QMessageBox", FakeBox):
            ev = QCloseEvent()
            dialog.closeEvent(ev)
            return ev.isAccepted()

    # No: closes and discards staging.
    assert close_with(QMessageBox.StandardButton.No) is True
    assert not dialog.has_staged_changes()

    # Cancel: stays open, staging kept.
    assert close_with(QMessageBox.StandardButton.Cancel) is False
    assert dialog.has_staged_changes()

    dialog.discard_staged_changes()
    dialog.close()


def test_review_no_current_row_fallback(qapp):
    """_get_selected_rows must not fall back to currentRow (no surprise Del)."""
    dialog = AnnotationReviewDialog(parent=None)
    dialog.all_items = [
        _make_review_item("/p/a.jpg", "/p/a.json", 0, "cat"),
        _make_review_item("/p/a.jpg", "/p/a.json", 1, "dog"),
    ]
    dialog.filtered_items = list(dialog.all_items)
    dialog._populate_table()
    dialog.table.setCurrentCell(0, 1)
    dialog.table.clearSelection()
    assert dialog.table.currentRow() == 0
    assert dialog._get_selected_rows() == []
    dialog.close()


def test_review_confirm_discard_yes_no(qapp, monkeypatch):
    """Banner/Esc discard must confirm; No keeps staging."""
    dialog = AnnotationReviewDialog(parent=None)
    dialog.all_items = [
        _make_review_item("/p/a.jpg", "/p/a.json", 0, "cat"),
    ]
    dialog.filtered_items = list(dialog.all_items)
    dialog._populate_table()
    dialog.toggle_stage_delete_selected_rows([0])
    assert dialog.has_staged_changes()
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.No,
    )
    dialog.confirm_discard_staged_changes()
    assert dialog.has_staged_changes()
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )
    dialog.confirm_discard_staged_changes()
    assert not dialog.has_staged_changes()
    dialog.close()


def test_review_resolve_label_file_uses_index(qapp):
    """Bulk path must resolve via prebuilt index (O(1)), not O(N) scan."""
    dialog = AnnotationReviewDialog(parent=None)
    dialog.all_items = [
        _make_review_item("/p/a.jpg", "/custom/a.json", 0, "cat"),
    ]
    idx = dialog._build_item_index()
    assert dialog._resolve_label_file("/p/a.jpg", 0, idx) == "/custom/a.json"
    dialog.close()


def test_review_staged_class_text_i18n(qapp):
    """Staging badges must go through tr() and shared helper."""
    dialog = AnnotationReviewDialog(parent=None)
    assert "[Pending Delete]" in dialog._staged_class_text("cat", True, None)
    assert "[Staged]" in dialog._staged_class_text("cat", False, "dog")
    assert dialog._staged_class_text("cat", False, None) == "cat"
    dialog.close()


def test_review_decide_iou_reuse(qapp, tmp_path):
    """Dense files (>300 shapes) reuse stored IoU; normal files recompute exactly."""
    dialog = AnnotationReviewDialog(parent=None)
    img_dense = str(tmp_path / "dense.jpg")
    lbl_dense = str(tmp_path / "dense.json")
    img_normal = str(tmp_path / "normal.jpg")
    lbl_normal = str(tmp_path / "normal.json")

    # Create dense file with 301 shapes
    shapes_dense = [
        {
            "label": "cat",
            "shape_type": "rectangle",
            "points": [[0, 0], [10, 10]],
        }
        for _ in range(301)
    ]
    with open(lbl_dense, "w", encoding="utf-8") as f:
        json.dump({"shapes": shapes_dense}, f)

    # Create normal file with 2 shapes
    shapes_normal = [
        {
            "label": "dog",
            "shape_type": "rectangle",
            "points": [[0, 0], [10, 10]],
        }
        for _ in range(2)
    ]
    with open(lbl_normal, "w", encoding="utf-8") as f:
        json.dump({"shapes": shapes_normal}, f)

    dense_item = _make_review_item(img_dense, lbl_dense, 0, "cat")
    dense_item.overlap_iou = 0.85
    dialog.all_items = [dense_item]

    # Dense image triggers reuse
    reuse_map = dialog._decide_iou_reuse({img_dense}, output_dir=str(tmp_path))
    assert reuse_map is not None
    assert len(reuse_map) > 0

    # Normal image recomputes exactly (returns None)
    assert (
        dialog._decide_iou_reuse({img_normal}, output_dir=str(tmp_path))
        is None
    )
    dialog.close()


def test_review_patch_canvas_expected_old_mismatch_and_bounds(qapp):
    """Canvas patch must skip label mismatch and invalid indices."""
    from types import SimpleNamespace

    dialog = AnnotationReviewDialog(parent=None)
    mock_shape = SimpleNamespace(label="dog", shape_type="rectangle")
    mock_canvas = SimpleNamespace(shapes=[mock_shape])
    mock_widget = SimpleNamespace(
        filename="/p/test.jpg",
        canvas=mock_canvas,
        label_list=None,
    )
    dialog._label_widget = mock_widget

    # Mismatch: expected "cat", actual "dog" -> skip
    dialog._patch_canvas_relabels(
        {("/p/test.jpg", 0): "car"},
        expected_old={("/p/test.jpg", 0): "cat"},
    )
    assert mock_shape.label == "dog"

    # Out of range index -> skip without error
    dialog._patch_canvas_relabels(
        {("/p/test.jpg", 99): "car"},
    )
    assert mock_shape.label == "dog"

    # Matching expected label -> relabeled
    dialog._patch_canvas_relabels(
        {("/p/test.jpg", 0): "car"},
        expected_old={("/p/test.jpg", 0): "dog"},
    )
    assert mock_shape.label == "car"
    dialog.close()


def test_review_abort_if_main_dirty(qapp, monkeypatch):
    """Abort when current file in main widget is dirty."""
    from types import SimpleNamespace

    dialog = AnnotationReviewDialog(parent=None)
    mock_widget = SimpleNamespace(
        filename="/p/test.jpg",
        dirty=True,
    )
    dialog._label_widget = mock_widget
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    # Touched includes dirty file -> aborts
    assert dialog._abort_if_main_dirty({"/p/test.jpg"}) is True

    # Touched does not include dirty file -> does not abort
    assert dialog._abort_if_main_dirty({"/p/other.jpg"}) is False

    mock_widget.dirty = False
    assert dialog._abort_if_main_dirty({"/p/test.jpg"}) is False
    dialog.close()


def test_review_close_yes_failure_keeps_open(qapp, monkeypatch):
    """When apply fails on Close with Yes, close event must be ignored."""
    from PyQt6.QtGui import QCloseEvent
    import anylabeling.views.labeling.widgets.annotation_review_dialog as mod

    dialog = AnnotationReviewDialog(parent=None)
    dialog.all_items = [_make_review_item("/p/a.jpg", "/p/a.json", 0, "cat")]
    dialog.filtered_items = list(dialog.all_items)
    dialog._populate_table()
    dialog.toggle_stage_delete_selected_rows([0])

    class FakeBox:
        Icon = QMessageBox.Icon
        StandardButton = QMessageBox.StandardButton

        def __init__(self, *a, **k):
            pass

        def setWindowModality(self, *a):
            pass

        def exec(self):
            return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(mod, "QMessageBox", FakeBox)
    # Simulate apply failure (leaves changes staged)
    monkeypatch.setattr(dialog, "apply_staged_changes", lambda *a, **k: None)

    ev = QCloseEvent()
    dialog.closeEvent(ev)
    assert ev.isAccepted() is False
    assert dialog.has_staged_changes()
    dialog.discard_staged_changes()
    dialog.close()


def test_review_undo_prev_reclasses_and_was_deleted(qapp):
    """Undo restores previous reclassifications and previous delete states."""
    dialog = AnnotationReviewDialog(parent=None)
    dialog.all_items = [
        _make_review_item("/p/a.jpg", "/p/a.json", 0, "cat"),
        _make_review_item("/p/a.jpg", "/p/a.json", 1, "dog"),
    ]
    dialog.filtered_items = list(dialog.all_items)
    dialog._populate_table()

    # 1. Stage reclass row 0 as "car"
    dialog.stage_reclass_selected_rows([0], "car")
    assert dialog.staged_reclasses.get(("/p/a.jpg", 0)) == "car"

    # 2. Stage deletion on row 0 (overrides reclass, records prev_reclass)
    dialog.toggle_stage_delete_selected_rows([0])
    assert ("/p/a.jpg", 0) in dialog.staged_deletions
    assert ("/p/a.jpg", 0) not in dialog.staged_reclasses

    # 3. Undo deletion -> must restore the staged reclass "car"
    dialog.undo_last_staged_action()
    assert ("/p/a.jpg", 0) not in dialog.staged_deletions
    assert dialog.staged_reclasses.get(("/p/a.jpg", 0)) == "car"

    # 4. Stage delete on row 1, then stage reclass on row 1
    dialog.toggle_stage_delete_selected_rows([1])
    assert ("/p/a.jpg", 1) in dialog.staged_deletions
    dialog.stage_reclass_selected_rows([1], "bird")
    assert ("/p/a.jpg", 1) in dialog.staged_reclasses

    # 5. Undo reclass -> must restore to staged deletion
    dialog.undo_last_staged_action()
    assert ("/p/a.jpg", 1) in dialog.staged_deletions

    dialog.discard_staged_changes()
    dialog.close()


def test_review_drop_staging_for_images(qapp):
    """_drop_staging_for_images drops staging and filters undo stack for affected files."""
    dialog = AnnotationReviewDialog(parent=None)
    dialog.all_items = [
        _make_review_item("/p/a.jpg", "/p/a.json", 0, "cat"),
        _make_review_item("/p/b.jpg", "/p/b.json", 0, "dog"),
    ]
    dialog.filtered_items = list(dialog.all_items)
    dialog._populate_table()

    dialog.toggle_stage_delete_selected_rows([0])
    dialog.stage_reclass_selected_rows([1], "wolf")
    assert ("/p/a.jpg", 0) in dialog.staged_deletions
    assert ("/p/b.jpg", 0) in dialog.staged_reclasses

    # Drop staging for /p/a.jpg only
    dialog._drop_staging_for_images({"/p/a.jpg"})
    assert ("/p/a.jpg", 0) not in dialog.staged_deletions
    assert ("/p/b.jpg", 0) in dialog.staged_reclasses

    dialog.discard_staged_changes()
    dialog.close()


def test_review_accept_reject_routes_to_close(qapp):
    """accept() and reject() must route through close()."""
    dialog = AnnotationReviewDialog(parent=None)
    closed = []
    dialog.close = lambda: closed.append(True)

    dialog.accept()
    assert len(closed) == 1
    dialog.reject()
    assert len(closed) == 2
