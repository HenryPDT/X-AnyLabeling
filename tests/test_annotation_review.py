import json
import os
import tempfile
from PIL import Image
import pytest
from PyQt6.QtWidgets import QApplication

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
