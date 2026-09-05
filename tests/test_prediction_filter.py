import pytest

from anylabeling.services.auto_labeling.prediction_filter import (
    filter_duplicate_shapes,
    filter_prediction_sizes,
)
from anylabeling.views.labeling.shape import Shape
from PyQt6.QtCore import QPointF


def _make_shape(label, points, shape_type="rectangle"):
    s = Shape(label=label, shape_type=shape_type)
    for x, y in points:
        s.add_point(QPointF(float(x), float(y)))
    return s


def test_filter_prediction_sizes_basic():
    # 1000 x 1000 image
    img_w, img_h = 1000, 1000

    normal_shape = {
        "shape_type": "rectangle",
        "points": [[10, 10], [50, 50]],
        "label": "cat",
    }
    micro_shape = {
        "shape_type": "rectangle",
        "points": [[10, 10], [12, 12]],
        "label": "noise",
    }
    giant_shape = {
        "shape_type": "rectangle",
        "points": [[0, 0], [995, 995]],
        "label": "background_fp",
    }
    point_shape = {
        "shape_type": "point",
        "points": [[10, 10]],
        "label": "point_marker",
    }

    shapes = [normal_shape, micro_shape, giant_shape, point_shape]
    filtered = filter_prediction_sizes(
        shapes,
        img_width=img_w,
        img_height=img_h,
        min_size_px=5.0,
        max_percent=0.98,
    )

    labels = [s["label"] for s in filtered]
    assert "cat" in labels
    assert "point_marker" in labels
    assert "noise" not in labels
    assert "background_fp" not in labels


def test_filter_prediction_sizes_with_shape_objects():
    img_w, img_h = 800, 600

    valid_rect = _make_shape("person", [[100, 100], [200, 300]], "rectangle")
    micro_rect = _make_shape("noise", [[50, 50], [52, 52]], "rectangle")
    valid_polygon = _make_shape(
        "car", [[10, 10], [60, 10], [60, 60], [10, 60]], "polygon"
    )

    filtered = filter_prediction_sizes(
        [valid_rect, micro_rect, valid_polygon],
        img_width=img_w,
        img_height=img_h,
        min_size_px=5.0,
    )
    assert len(filtered) == 2
    assert filtered[0].label == "person"
    assert filtered[1].label == "car"


def test_filter_duplicate_shapes_against_existing():
    existing = [
        {
            "label": "dog",
            "shape_type": "rectangle",
            "points": [[50, 50], [150, 150]],
        }
    ]

    new_exact_dup = {
        "label": "dog",
        "shape_type": "rectangle",
        "points": [[50, 50], [150, 150]],
    }
    new_distinct = {
        "label": "cat",
        "shape_type": "rectangle",
        "points": [[300, 300], [400, 400]],
    }
    new_nested = {
        "label": "dog",
        "shape_type": "rectangle",
        "points": [[52, 52], [148, 148]],  # 96% containment
    }

    # All incoming
    incoming = [new_exact_dup, new_distinct, new_nested]
    filtered = filter_duplicate_shapes(
        incoming, existing, iou_threshold=0.85, containment_threshold=0.90
    )

    assert len(filtered) == 1
    assert filtered[0]["label"] == "cat"


def test_filter_duplicate_shapes_same_label_only():
    existing = [
        {
            "label": "person",
            "shape_type": "rectangle",
            "points": [[10, 10], [100, 100]],
        }
    ]

    # Same box but different label
    incoming_diff_label = {
        "label": "pedestrian",
        "shape_type": "rectangle",
        "points": [[10, 10], [100, 100]],
    }
    # Same box with same label
    incoming_same_label = {
        "label": "person",
        "shape_type": "rectangle",
        "points": [[10, 10], [100, 100]],
    }

    # When same_label_only=True: different label survives
    res1 = filter_duplicate_shapes(
        [incoming_diff_label, incoming_same_label],
        existing,
        iou_threshold=0.85,
        same_label_only=True,
    )
    assert len(res1) == 1
    assert res1[0]["label"] == "pedestrian"

    # When same_label_only=False (class agnostic): both overlapping are suppressed
    res2 = filter_duplicate_shapes(
        [incoming_diff_label, incoming_same_label],
        existing,
        iou_threshold=0.85,
        same_label_only=False,
    )
    assert len(res2) == 0


def test_filter_empty_inputs():
    assert filter_prediction_sizes([], 100, 100) == []
    assert filter_duplicate_shapes([], []) == []
    assert filter_duplicate_shapes([{"label": "a"}], []) == [{"label": "a"}]
