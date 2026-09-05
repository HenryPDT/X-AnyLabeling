import json

from anylabeling.services.dataset_stats_core import (
    compute_full_dataset_stats,
)


def _write_image_and_label(tmp_path, name, shapes, verified_empty=False):
    img = str(tmp_path / f"{name}.jpg")
    with open(img, "wb") as f:
        f.write(b"fake")
    label = str(tmp_path / f"{name}.json")
    data = {
        "version": "1.0",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": f"{name}.jpg",
        "imageData": None,
        "imageHeight": 200,
        "imageWidth": 200,
    }
    if verified_empty:
        data["verified_empty"] = True
    with open(label, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return img


def test_compute_full_stats(tmp_path):
    img1 = _write_image_and_label(
        tmp_path,
        "a",
        [
            {
                "label": "cat",
                "shape_type": "rectangle",
                "points": [[0, 0], [10, 20]],
            },
            {
                "label": "cat",
                "shape_type": "rectangle",
                "points": [[0, 0], [30, 40]],
            },
        ],
    )
    img2 = _write_image_and_label(tmp_path, "b", [], verified_empty=True)
    img3 = str(tmp_path / "c.jpg")
    with open(img3, "wb") as f:
        f.write(b"fake")

    stats = compute_full_dataset_stats([img1, img2, img3])
    assert stats.total_images == 3
    assert stats.total_annotations == 2
    assert stats.verified_empty_images == 1
    assert stats.empty_images == 1
    assert "cat" in stats.per_class
    cat = stats.per_class["cat"]
    assert cat.count == 2
    assert cat.images == 1
    assert cat.mean_width > 0
    assert cat.stddev_width >= 0
    assert abs(cat.marks_per_image - 2.0) < 1e-6


def test_tiny_boxes_counted(tmp_path):
    img = _write_image_and_label(
        tmp_path,
        "t",
        [
            {
                "label": "dot",
                "shape_type": "rectangle",
                "points": [[0, 0], [5, 5]],
            },
        ],
    )
    stats = compute_full_dataset_stats([img], tiny_box_px=32.0)
    assert stats.tiny_boxes_count == 1


def test_invalid_shapes_not_counted(tmp_path):
    img = _write_image_and_label(
        tmp_path,
        "bad",
        [
            {"label": "cat", "shape_type": "rectangle", "points": []},
            "not-a-dict",
            {"label": "cat", "shape_type": "rectangle"},
        ],
    )
    stats = compute_full_dataset_stats([img])
    assert stats.total_annotations == 0
    assert stats.per_class == {}
    assert stats.empty_images == 1


def test_zero_area_shapes_skipped(tmp_path):
    img = _write_image_and_label(
        tmp_path,
        "zero",
        [
            {
                "label": "dot",
                "shape_type": "rectangle",
                "points": [[5, 5], [5, 5]],
            },
        ],
    )
    stats = compute_full_dataset_stats([img])
    assert stats.total_annotations == 0
    assert "dot" not in stats.per_class


def test_per_class_tiny_count(tmp_path):
    img = _write_image_and_label(
        tmp_path,
        "m",
        [
            {
                "label": "mix",
                "shape_type": "rectangle",
                "points": [[0, 0], [5, 5]],
            },
            {
                "label": "mix",
                "shape_type": "rectangle",
                "points": [[0, 0], [100, 100]],
            },
        ],
    )
    stats = compute_full_dataset_stats([img], tiny_box_px=32.0)
    assert stats.per_class["mix"].count == 2
    assert stats.per_class["mix"].tiny_count == 1


def test_stats_dialog_smoke(tmp_path):
    import os as _os

    _os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    from anylabeling.views.labeling.widgets.dataset_stats_dialog import (
        DatasetStatsDialog,
    )

    img = _write_image_and_label(
        tmp_path,
        "s",
        [
            {
                "label": "cat",
                "shape_type": "rectangle",
                "points": [[0, 0], [10, 10]],
            },
        ],
    )

    class FakeWidget:
        image_list = [img]
        output_dir = str(tmp_path)
        filename = img

    dialog = DatasetStatsDialog(parent=None)
    dialog._label_widget = FakeWidget()
    dialog.reload_stats(silent=True)
    assert dialog.stats is not None
    assert dialog.table.rowCount() == 1
    dialog.close()


def test_stats_dialog_non_silent_progress_path(tmp_path):
    """Regression: non-silent reload with >20 images must not crash.

    Covers QProgressDialog creation (WindowModality enum, not bool).
    """
    import os as _os

    _os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    from anylabeling.views.labeling.widgets.dataset_stats_dialog import (
        DatasetStatsDialog,
    )

    images = []
    for idx in range(25):
        images.append(
            _write_image_and_label(
                tmp_path,
                f"p{idx}",
                [
                    {
                        "label": "cat",
                        "shape_type": "rectangle",
                        "points": [[0, 0], [10, 10]],
                    },
                ],
            )
        )

    class FakeWidget:
        image_list = images
        output_dir = str(tmp_path)
        filename = images[0]

    dialog = DatasetStatsDialog(parent=None)
    dialog._label_widget = FakeWidget()
    # Must not raise TypeError from setWindowModality.
    dialog.reload_stats(silent=False)
    assert dialog.stats is not None
    assert dialog.stats.total_images == 25
    assert dialog.table.rowCount() >= 1
    dialog.close()
