"""Polygon -> quadrilateral shape conversion (e.g. SAM masks to quads)."""

import math

from anylabeling.views.labeling.utils.shape import (
    CONVERSION_MODE_MAP,
    CONVERSION_TARGETS,
    _apply_shape_conversion,
    convert_single_shape,
)


def _data(shapes):
    return {"shapes": shapes}


def _poly(points, **kw):
    shape = {
        "label": "plate",
        "points": points,
        "shape_type": "polygon",
        "group_id": None,
        "difficult": False,
        "flags": {},
    }
    shape.update(kw)
    return shape


def test_targets_include_polygon_to_quadrilateral():
    assert "quadrilateral" in CONVERSION_TARGETS["polygon"]
    assert CONVERSION_MODE_MAP[("polygon", "quadrilateral")] == (
        "polygon_to_quadrilateral"
    )


def test_four_point_polygon_keeps_vertices():
    pts = [[20.0, 20.0], [180.0, 20.0], [180.0, 80.0], [20.0, 80.0]]
    data = _data([_poly([p[:] for p in pts])])
    _apply_shape_conversion(data, "polygon_to_quadrilateral", {})
    (shape,) = data["shapes"]
    assert shape["shape_type"] == "quadrilateral"
    assert shape["points"] == pts


def test_closed_sam_style_polygon_drops_duplicate():
    pts = [[20.0, 20.0], [180.0, 20.0], [180.0, 80.0], [20.0, 80.0]]
    data = _data([_poly(pts + [pts[0][:]])])
    _apply_shape_conversion(data, "polygon_to_quadrilateral", {})
    (shape,) = data["shapes"]
    assert shape["shape_type"] == "quadrilateral"
    assert len(shape["points"]) == 4


def test_many_point_quadish_polygon_fits_four_corners():
    # Rectangle with subdivided edges (typical SAM mask contour).
    pts = []
    for x in (20, 60, 100, 140, 180):
        pts.append([float(x), 20.0])
    for y in (35, 50, 65, 80):
        pts.append([180.0, float(y)])
    for x in (140, 100, 60, 20):
        pts.append([float(x), 80.0])
    for y in (65, 50, 35):
        pts.append([20.0, float(y)])
    assert len(pts) > 4
    data = _data([_poly(pts)])
    _apply_shape_conversion(data, "polygon_to_quadrilateral", {})
    (shape,) = data["shapes"]
    assert shape["shape_type"] == "quadrilateral"
    assert len(shape["points"]) == 4
    corners = {(20.0, 20.0), (180.0, 20.0), (180.0, 80.0), (20.0, 80.0)}
    for x, y in shape["points"]:
        assert min(math.hypot(x - cx, y - cy) for cx, cy in corners) < 2.0


def test_degenerate_polygon_skipped():
    data = _data([_poly([[0.0, 0.0], [1.0, 1.0]])])
    _apply_shape_conversion(data, "polygon_to_quadrilateral", {})
    assert data["shapes"][0]["shape_type"] == "polygon"


def test_other_and_locked_shapes_untouched():
    rect = {
        "label": "plate",
        "points": [[0.0, 0.0], [10.0, 10.0]],
        "shape_type": "rectangle",
        "group_id": None,
        "difficult": False,
        "flags": {},
    }
    locked = _poly(
        [[20.0, 20.0], [180.0, 20.0], [180.0, 80.0], [20.0, 80.0]],
        locked=True,
    )
    data = _data([rect, locked])
    _apply_shape_conversion(data, "polygon_to_quadrilateral", {})
    assert data["shapes"][0]["shape_type"] == "rectangle"
    assert data["shapes"][1]["shape_type"] == "polygon"


def test_convert_single_shape_polygon_to_quadrilateral():
    pts = [[20.0, 20.0], [180.0, 20.0], [180.0, 80.0], [20.0, 80.0]]
    result = convert_single_shape("polygon", pts, "quadrilateral")
    assert result is not None
    new_type, new_points, direction = result
    assert new_type == "quadrilateral"
    assert new_points == pts
    assert direction is None


def test_convert_single_shape_closed_polygon():
    pts = [[20.0, 20.0], [180.0, 20.0], [180.0, 80.0], [20.0, 80.0]]
    result = convert_single_shape("polygon", pts + [pts[0]], "quadrilateral")
    assert result is not None
    assert result[0] == "quadrilateral"
    assert len(result[1]) == 4


def test_convert_single_shape_rejects_unsupported_and_degenerate():
    assert (
        convert_single_shape("polygon", [[0.0, 0.0]], "quadrilateral")
        is None
    )
    assert (
        convert_single_shape(
            "polygon",
            [[20.0, 20.0], [180.0, 20.0]],
            "quadrilateral",
        )
        is None
    )
    assert (
        convert_single_shape(
            "line",
            [[0.0, 0.0], [1.0, 1.0]],
            "quadrilateral",
        )
        is None
    )


def test_convert_single_shape_rotation_reports_direction():
    pts = [[20.0, 20.0], [180.0, 20.0], [180.0, 80.0], [20.0, 80.0]]
    new_type, _, direction = convert_single_shape(
        "polygon", pts, "rotation"
    )
    assert new_type == "rotation"
    assert direction is not None


def _rounded_rect_points(x0, y0, x1, y1, radius, steps=4):
    """Rectangle with rounded corners (e.g. blurred plate mask)."""
    import math

    pts = []
    for cx, cy, start in (
        (x1 - radius, y0 + radius, -90.0),
        (x1 - radius, y1 - radius, 0.0),
        (x0 + radius, y1 - radius, 90.0),
        (x0 + radius, y0 + radius, 180.0),
    ):
        for i in range(steps + 1):
            angle = math.radians(start + 90.0 * i / steps)
            pts.append(
                [cx + radius * math.cos(angle), cy + radius * math.sin(angle)]
            )
    return pts


def test_noisy_rounded_mask_holds_true_extents():
    # Heavily rounded corners: pure approx fitting insets the corners,
    # the IoU-scored fit must prefer the stable (min-area style) box.
    pts = _rounded_rect_points(20.0, 20.0, 180.0, 80.0, 18.0)
    assert len(pts) > 4
    data = _data([_poly(pts)])
    _apply_shape_conversion(data, "polygon_to_quadrilateral", {})
    (shape,) = data["shapes"]
    assert shape["shape_type"] == "quadrilateral"
    xs = [p[0] for p in shape["points"]]
    ys = [p[1] for p in shape["points"]]
    assert abs(min(xs) - 20.0) < 4.0
    assert abs(max(xs) - 180.0) < 4.0
    assert abs(min(ys) - 20.0) < 4.0
    assert abs(max(ys) - 80.0) < 4.0


def test_best_iou_candidate_wins():
    import numpy as np

    from anylabeling.views.labeling.utils.shape import _quad_mask_iou

    # Concave bite: no single simplification is obviously right, so the
    # winner must be at least as good as the min-area box by mask IoU.
    pts = np.array(
        [
            [20.0, 20.0],
            [100.0, 20.0],
            [100.0, 35.0],
            [140.0, 35.0],
            [140.0, 20.0],
            [180.0, 20.0],
            [180.0, 80.0],
            [20.0, 80.0],
        ]
    )
    data = _data([_poly(pts.tolist())])
    _apply_shape_conversion(data, "polygon_to_quadrilateral", {})
    (shape,) = data["shapes"]
    assert shape["shape_type"] == "quadrilateral"
    import cv2

    box = cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32)))
    result_iou = _quad_mask_iou(pts, np.asarray(shape["points"]))
    box_iou = _quad_mask_iou(pts, box)
    assert result_iou is not None and box_iou is not None
    assert result_iou >= box_iou - 1e-9


def _noisy_perspective_contour(truth, radius=8.0, step=2.0, noise=1.2, seed=0):
    """Skewed quad edges with rounded corners + noise (SAM-like mask)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    truth = np.asarray(truth, dtype=float)
    pts = []
    n = len(truth)
    for i in range(n):
        cur, nxt, nn = truth[i], truth[(i + 1) % n], truth[(i + 2) % n]
        edge = nxt - cur
        edge_len = float(np.linalg.norm(edge))
        unit = edge / edge_len
        start, end = cur + unit * radius, nxt - unit * radius
        for t in np.linspace(0.0, 1.0, max(2, int(edge_len / step)), endpoint=False):
            pts.append(start + (end - start) * t)
        edge2 = nn - nxt
        unit2 = edge2 / float(np.linalg.norm(edge2))
        ctrl_end = nxt + unit2 * radius
        for t in np.linspace(0.0, 1.0, 6, endpoint=False):
            pts.append(
                (1 - t) ** 2 * end + 2 * (1 - t) * t * nxt + t**2 * ctrl_end
            )
    return np.asarray(pts) + rng.normal(0.0, noise, (len(pts), 2))


def _cyclic_max_error(quad, truth):
    import numpy as np

    quad = np.asarray(quad, dtype=float)
    best = float("inf")
    for rev in (False, True):
        ref = truth[::-1] if rev else truth
        for shift in range(4):
            err = max(
                float(np.linalg.norm(quad[(shift + k) % 4] - ref[k]))
                for k in range(4)
            )
            best = min(best, err)
    return best


def test_line_refinement_beats_minrect_on_perspective_plate():
    import cv2
    import numpy as np

    # Strongly skewed quad: the min-area box is structurally wrong here
    # (off by ~9px), line-intersection refinement must recover the true
    # corners from the rounded, noisy contour.
    truth = np.array(
        [[30.0, 40.0], [170.0, 15.0], [190.0, 80.0], [40.0, 100.0]]
    )
    for seed in (0, 1, 2):
        pts = _noisy_perspective_contour(truth, seed=seed)
        data = _data([_poly(pts.tolist())])
        _apply_shape_conversion(data, "polygon_to_quadrilateral", {})
        (shape,) = data["shapes"]
        assert shape["shape_type"] == "quadrilateral"
        refined_err = _cyclic_max_error(shape["points"], truth)
        box = cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32)))
        box_err = _cyclic_max_error(box, truth)
        assert refined_err < 3.0
        assert refined_err < box_err


def test_get_label_file_list_walks_scene_subdirs(tmp_path):
    from types import SimpleNamespace

    from anylabeling.views.labeling.label_widget import LabelingWidget

    (tmp_path / "scene_001").mkdir()
    (tmp_path / "scene_002").mkdir()
    (tmp_path / "scene_001" / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "scene_002" / "b.json").write_text("{}", encoding="utf-8")
    (tmp_path / "scene_002" / "b.jpg").write_text("x", encoding="utf-8")

    stub = SimpleNamespace(
        image_list=["x"],
        filename=str(tmp_path / "scene_001" / "a.jpg"),
        output_dir=None,
        last_open_dir=str(tmp_path),
    )
    assert sorted(LabelingWidget.get_label_file_list(stub)) == sorted(
        [
            str(tmp_path / "scene_001" / "a.json"),
            str(tmp_path / "scene_002" / "b.json"),
        ]
    )

    stub_current_dir_only = SimpleNamespace(
        image_list=["x"],
        filename=str(tmp_path / "scene_001" / "a.jpg"),
        output_dir=None,
        last_open_dir=None,
    )
    assert LabelingWidget.get_label_file_list(stub_current_dir_only) == [
        str(tmp_path / "scene_001" / "a.json")
    ]
