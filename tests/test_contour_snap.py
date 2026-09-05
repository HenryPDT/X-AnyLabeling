"""Unit tests for contour and edge snapping utility functions."""

import numpy as np
import pytest
from PyQt6 import QtCore

from anylabeling.views.labeling.utils.contour_snap import (
    snap_bbox_to_contour,
    snap_obb_to_contour,
)


def test_snap_bbox_to_contour_bright_object():
    # 200x200 black background with a white rectangle at [50, 40, 150, 120]
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[40:120, 50:150] = 255

    # User draws a loose box slightly larger than the white rectangle
    x1, y1, x2, y2 = 43.0, 34.0, 157.0, 126.0
    sx1, sy1, sx2, sy2 = snap_bbox_to_contour(img, x1, y1, x2, y2, padding=15)

    # Snapped box should tightly enclose [50, 40, 150, 120]
    assert abs(sx1 - 50.0) <= 1.0
    assert abs(sy1 - 40.0) <= 1.0
    assert abs(sx2 - 150.0) <= 1.0
    assert abs(sy2 - 120.0) <= 1.0


def test_snap_bbox_to_contour_dark_object_light_background():
    # 200x200 white background with a black rectangle at [30, 30, 90, 90]
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    img[30:90, 30:90] = 0

    # User draws a loose box
    x1, y1, x2, y2 = 25.0, 24.0, 96.0, 94.0
    sx1, sy1, sx2, sy2 = snap_bbox_to_contour(img, x1, y1, x2, y2, padding=12)

    assert abs(sx1 - 30.0) <= 1.0
    assert abs(sy1 - 30.0) <= 1.0
    assert abs(sx2 - 90.0) <= 1.0
    assert abs(sy2 - 90.0) <= 1.0


def test_snap_bbox_fallback_on_blank_image():
    # All black image - no edge or contour to snap to
    img = np.zeros((150, 150, 3), dtype=np.uint8)
    orig_box = (20.0, 25.0, 80.0, 75.0)

    snapped = snap_bbox_to_contour(img, *orig_box)
    assert snapped == orig_box


def test_snap_bbox_degenerate_and_boundary_cases():
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Degenerate zero-width or near-zero-width box
    assert snap_bbox_to_contour(img, 10.0, 10.0, 10.5, 50.0) == (
        10.0,
        10.0,
        10.5,
        50.0,
    )

    # Inverted min/max coordinates
    img[20:80, 20:80] = 255
    sx1, sy1, sx2, sy2 = snap_bbox_to_contour(img, 85.0, 85.0, 15.0, 15.0)
    assert abs(sx1 - 20.0) <= 1.0
    assert abs(sx2 - 80.0) <= 1.0


def test_snap_obb_to_contour():
    # 200x200 black image with a white rectangle at [40, 40, 120, 100]
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[40:100, 40:120] = 255

    rough_points = [
        QtCore.QPointF(35.0, 35.0),
        QtCore.QPointF(125.0, 35.0),
        QtCore.QPointF(125.0, 105.0),
        QtCore.QPointF(35.0, 105.0),
    ]

    snapped = snap_obb_to_contour(img, rough_points, padding=15)
    assert len(snapped) == 4

    xs = [p[0] for p in snapped]
    ys = [p[1] for p in snapped]
    assert abs(min(xs) - 40.0) <= 1.5
    assert abs(max(xs) - 120.0) <= 1.5
    assert abs(min(ys) - 40.0) <= 1.5
    assert abs(max(ys) - 100.0) <= 1.5


def test_snap_obb_fallback_on_blank():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    raw_points = [(10.0, 10.0), (50.0, 10.0), (50.0, 40.0), (10.0, 40.0)]

    snapped = snap_obb_to_contour(img, raw_points)
    assert snapped == raw_points


def test_canvas_contour_snap_integration():
    from PyQt6 import QtGui, QtWidgets
    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.widgets.canvas import Canvas

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    canvas = Canvas(parent=QtWidgets.QWidget())

    # Create 200x200 image with white rectangle at [40, 40, 120, 100]
    img = QtGui.QImage(200, 200, QtGui.QImage.Format.Format_RGB888)
    img.fill(QtGui.QColor(0, 0, 0))
    painter = QtGui.QPainter(img)
    painter.fillRect(40, 40, 80, 60, QtGui.QColor(255, 255, 255))
    painter.end()

    canvas.load_pixmap(QtGui.QPixmap.fromImage(img))

    # Test disabled snap
    canvas.contour_snap_enabled = False
    shape = Shape(shape_type="rectangle")
    shape.points = [
        QtCore.QPointF(35.0, 35.0),
        QtCore.QPointF(125.0, 35.0),
        QtCore.QPointF(125.0, 105.0),
        QtCore.QPointF(35.0, 105.0),
    ]
    canvas.current = shape
    canvas.finalise()
    assert len(canvas.shapes) == 1
    assert abs(canvas.shapes[0].points[0].x() - 35.0) < 0.1

    # Test enabled snap in finalise
    canvas.shapes.clear()
    canvas.contour_snap_enabled = True
    shape2 = Shape(shape_type="rectangle")
    shape2.points = [
        QtCore.QPointF(35.0, 35.0),
        QtCore.QPointF(125.0, 35.0),
        QtCore.QPointF(125.0, 105.0),
        QtCore.QPointF(35.0, 105.0),
    ]
    canvas.current = shape2
    canvas.finalise()
    assert len(canvas.shapes) == 1
    # Snapped to [40, 40, 120, 100]
    assert abs(canvas.shapes[0].points[0].x() - 40.0) <= 1.0
    assert abs(canvas.shapes[0].points[2].x() - 120.0) <= 1.0

    # Test snap_selected_shapes
    canvas.shapes.clear()
    loose_shape = Shape(shape_type="rectangle")
    loose_shape.points = [
        QtCore.QPointF(36.0, 36.0),
        QtCore.QPointF(124.0, 36.0),
        QtCore.QPointF(124.0, 104.0),
        QtCore.QPointF(36.0, 104.0),
    ]
    canvas.shapes.append(loose_shape)
    canvas.selected_shapes = [loose_shape]
    modified = canvas.snap_selected_shapes()
    assert modified == 1
    assert abs(loose_shape.points[0].x() - 40.0) <= 1.0

    # Test snap_selected_shapes with 2-point rectangle
    canvas.shapes.clear()
    loose_2pt = Shape(shape_type="rectangle")
    loose_2pt.points = [
        QtCore.QPointF(36.0, 36.0),
        QtCore.QPointF(124.0, 104.0),
    ]
    canvas.shapes.append(loose_2pt)
    canvas.selected_shapes = [loose_2pt]
    modified_2pt = canvas.snap_selected_shapes()
    assert modified_2pt == 1
    assert len(loose_2pt.points) == 2
    assert abs(loose_2pt.points[0].x() - 40.0) <= 1.0
    assert abs(loose_2pt.points[1].x() - 120.0) <= 1.0
