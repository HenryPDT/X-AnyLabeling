from PyQt6 import QtCore
import numpy as np
import pytest

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.shape_geometry import (
    apply_class_agnostic_shape_nms,
    apply_same_label_containment_nms,
    box_area,
    box_containment_ratio,
    box_intersection,
    box_iou,
    box_overlap_metrics,
    clamp_shape_to_image_bounds,
    clamp_shapes_to_image_bounds,
    detect_duplicate_shapes,
    get_shape_label,
    get_shape_score,
    get_shape_type,
    is_duplicate_box_geometry,
    shape_to_xyxy,
)


def test_shape_to_xyxy_various_inputs():
    # 1. Test from Shape object
    shape = Shape()
    shape.add_point(QtCore.QPointF(10.0, 20.0))
    shape.add_point(QtCore.QPointF(50.0, 80.0))
    assert shape_to_xyxy(shape) == [10.0, 20.0, 50.0, 80.0]

    # 2. Test from dict with points key
    dict_shape = {"points": [[15.0, 25.0], [55.0, 85.0]]}
    assert shape_to_xyxy(dict_shape) == [15.0, 25.0, 55.0, 85.0]

    # 3. Test from coordinate list or numpy array
    coords = [[0.0, 5.0], [100.0, 200.0], [50.0, 50.0]]
    assert shape_to_xyxy(coords) == [0.0, 5.0, 100.0, 200.0]
    assert shape_to_xyxy(np.array(coords)) == [0.0, 5.0, 100.0, 200.0]

    # 4. Test empty and invalid inputs
    assert shape_to_xyxy(None) is None
    assert shape_to_xyxy([]) is None
    empty_shape = Shape()
    assert shape_to_xyxy(empty_shape) is None
    assert shape_to_xyxy({"points": []}) is None


def test_box_area_and_intersection():
    box_a = [0.0, 0.0, 10.0, 20.0]
    assert box_area(box_a) == 200.0
    assert box_area(None) == 0.0
    assert box_area([10.0, 20.0, 5.0, 5.0]) == 0.0  # Inverted box

    box_b = [5.0, 10.0, 15.0, 30.0]
    # Intersection is [5, 10, 10, 20], width=5, height=10 -> area=50
    assert box_intersection(box_a, box_b) == 50.0

    disjoint_box = [30.0, 30.0, 40.0, 40.0]
    assert box_intersection(box_a, disjoint_box) == 0.0


def test_box_iou_and_containment():
    box_a = [0.0, 0.0, 10.0, 10.0]  # area = 100
    box_b = [0.0, 0.0, 10.0, 10.0]  # identical box
    assert box_iou(box_a, box_b) == 1.0
    assert box_containment_ratio(box_a, box_b) == 1.0

    inner_box = [2.0, 2.0, 6.0, 6.0]  # area = 16, entirely inside box_a
    assert box_containment_ratio(inner_box, box_a) == 1.0
    assert box_containment_ratio(box_a, inner_box) == 0.16
    assert box_iou(box_a, inner_box) == 16.0 / 100.0

    disjoint_box = [20.0, 20.0, 30.0, 30.0]
    assert box_iou(box_a, disjoint_box) == 0.0
    assert box_containment_ratio(box_a, disjoint_box) == 0.0


def test_box_overlap_metrics():
    box_a = [0.0, 0.0, 100.0, 100.0]
    box_b = [10.0, 10.0, 90.0, 90.0]  # 80x80 inside 100x100

    metrics = box_overlap_metrics(box_a, box_b)
    assert metrics["containment_b_in_a"] == 1.0
    assert metrics["containment"] == 1.0
    assert metrics["area_ratio"] == 6400.0 / 10000.0
    assert metrics["center_distance"] == 0.0
    assert metrics["relative_center_distance"] == 0.0


def test_is_duplicate_box_geometry():
    box_a = [10.0, 10.0, 100.0, 100.0]
    box_b = [11.0, 10.0, 100.0, 101.0]  # Near exact overlap
    is_dup, match_type, metrics = is_duplicate_box_geometry(
        box_a, box_b, iou_threshold=0.85
    )
    assert is_dup is True
    assert match_type == "iou"
    assert metrics["iou"] > 0.90

    disjoint = [200.0, 200.0, 300.0, 300.0]
    is_dup, _, _ = is_duplicate_box_geometry(box_a, disjoint)
    assert is_dup is False


def test_detect_duplicate_shapes():
    shape1 = Shape(label="car")
    for pt in [(10, 10), (50, 10), (50, 40), (10, 40)]:
        shape1.add_point(QtCore.QPointF(*pt))

    shape2 = Shape(label="car")
    for pt in [(11, 10), (50, 10), (50, 41), (11, 41)]:
        shape2.add_point(QtCore.QPointF(*pt))

    shape3 = Shape(label="person")
    for pt in [(80, 80), (100, 80), (100, 120), (80, 120)]:
        shape3.add_point(QtCore.QPointF(*pt))

    duplicates = detect_duplicate_shapes(
        [shape1, shape2, shape3], iou_threshold=0.85
    )
    assert len(duplicates) == 1
    assert duplicates[0]["index_a"] == 0
    assert duplicates[0]["index_b"] == 1
    assert duplicates[0]["match_type"] == "iou"


def test_shape_attribute_accessors():
    shape = Shape(label="dog", shape_type="rectangle")
    shape.score = 0.95
    assert get_shape_label(shape) == "dog"
    assert get_shape_score(shape) == 0.95
    assert get_shape_type(shape) == "rectangle"

    dict_shape = {"label": "cat", "score": 0.88, "shape_type": "polygon"}
    assert get_shape_label(dict_shape) == "cat"
    assert get_shape_score(dict_shape) == 0.88
    assert get_shape_type(dict_shape) == "polygon"

    assert get_shape_label(None) is None
    assert get_shape_score(None) is None
    assert get_shape_type(None) == "polygon"


def test_clamp_shape_to_image_bounds():
    # 1. Rectangle Shape extending beyond image bounds (2-pt stays 2-pt)
    shape_rect = Shape(shape_type="rectangle")
    shape_rect.add_point(QtCore.QPointF(-10.0, -5.0))
    shape_rect.add_point(QtCore.QPointF(120.0, 95.0))
    clamped_rect = clamp_shape_to_image_bounds(shape_rect, 100.0, 100.0)
    assert clamped_rect is not None
    assert len(clamped_rect.points) == 2
    pts = [(p.x(), p.y()) for p in clamped_rect.points]
    assert pts == [(0.0, 0.0), (100.0, 95.0)]

    # 2. Rectangle entirely outside image bounds returns None
    shape_outside = Shape(shape_type="rectangle")
    shape_outside.add_point(QtCore.QPointF(-50.0, -50.0))
    shape_outside.add_point(QtCore.QPointF(-10.0, -10.0))
    assert clamp_shape_to_image_bounds(shape_outside, 100.0, 100.0) is None

    # 3. Dict polygon with points extending beyond boundaries
    poly_dict = {
        "shape_type": "polygon",
        "points": [[-10.0, 20.0], [50.0, -15.0], [110.0, 20.0], [50.0, 120.0]],
    }
    clamped_poly = clamp_shape_to_image_bounds(poly_dict, 100.0, 100.0)
    assert clamped_poly is not None
    assert clamped_poly["points"] == [
        [0.0, 20.0],
        [50.0, 0.0],
        [100.0, 20.0],
        [50.0, 100.0],
    ]

    # 4. Polygon entirely outside bounds degenerates to < 3 points -> None
    poly_outside = {
        "shape_type": "polygon",
        "points": [[-30.0, -30.0], [-20.0, -10.0], [-10.0, -25.0]],
    }
    # All points clamp to (0, 0), deduped to 1 point -> returns None
    assert clamp_shape_to_image_bounds(poly_outside, 100.0, 100.0) is None

    # 5. Point shape clamping
    pt_dict = {"shape_type": "point", "points": [[-5.0, 150.0]]}
    clamped_pt = clamp_shape_to_image_bounds(pt_dict, 100.0, 100.0)
    assert clamped_pt["points"] == [[0.0, 100.0]]

    # 6. Non-positive image dimensions should be a passthrough
    assert clamp_shape_to_image_bounds(shape_rect, 0, 0) == shape_rect


def test_clamp_shapes_to_image_bounds():
    shapes = [
        {"shape_type": "point", "points": [[50.0, 50.0]]},  # inside
        {
            "shape_type": "rectangle",
            "points": [[-10.0, 0.0], [200.0, 100.0]],
        },  # clamped
        {
            "shape_type": "rectangle",
            "points": [[-50.0, -50.0], [-10.0, -10.0]],
        },  # degenerate (dropped)
    ]
    clamped = clamp_shapes_to_image_bounds(shapes, 100.0, 100.0)
    assert len(clamped) == 2
    assert clamped[0]["points"] == [[50.0, 50.0]]
    assert clamped[1]["points"] == [[0.0, 0.0], [100.0, 100.0]]

    # Empty list and invalid dimensions
    assert clamp_shapes_to_image_bounds([], 100.0, 100.0) == []
    assert clamp_shapes_to_image_bounds(shapes, 0, 0) == shapes
