import json
import math
import uuid

import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QProgressDialog

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import get_bounding_boxes
from anylabeling.views.labeling.widgets.polygon_sides_dialog import (
    PolygonSidesDialog,
)
from anylabeling.views.labeling.widgets.popup import Popup
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import *
from anylabeling.services.auto_labeling.utils import calculate_rotation_theta

CONVERSION_TARGETS = {
    "polygon": ["rectangle", "rotation", "quadrilateral"],
    "rectangle": ["rotation", "polygon", "circle", "quadrilateral"],
    "rotation": ["rectangle", "quadrilateral", "polygon", "circle"],
    "line": ["linestrip"],
    "circle": ["rectangle", "rotation", "quadrilateral", "polygon"],
    "quadrilateral": ["polygon"],
}

CONVERSION_MODE_MAP = {
    (source_type, target_type): f"{source_type}_to_{target_type}"
    for source_type, target_types in CONVERSION_TARGETS.items()
    for target_type in target_types
}

LEGACY_MODE_MAP = {
    "hbb_to_obb": "rectangle_to_rotation",
    "obb_to_hbb": "rotation_to_rectangle",
    "polygon_to_hbb": "polygon_to_rectangle",
    "polygon_to_obb": "polygon_to_rotation",
}


def _normalize_mode(mode):
    return LEGACY_MODE_MAP.get(mode, mode)


def _to_axis_aligned_box(points):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return None
    xmin = int(np.min(points[:, 0]))
    ymin = int(np.min(points[:, 1]))
    xmax = int(np.max(points[:, 0]))
    ymax = int(np.max(points[:, 1]))
    return [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ]


def _circle_center_radius(points):
    points = np.asarray(points, dtype=np.float32)
    if len(points) != 2:
        return None, None, None
    center_x, center_y = points[0]
    edge_x, edge_y = points[1]
    radius = math.sqrt((edge_x - center_x) ** 2 + (edge_y - center_y) ** 2)
    if radius <= 0:
        return None, None, None
    return float(center_x), float(center_y), float(radius)


def _circle_points(center_x, center_y, radius):
    return [[center_x, center_y], [center_x + radius, center_y]]


def _rotation_center_inscribed_radius(points):
    points = np.asarray(points, dtype=np.float32)
    if len(points) != 4:
        return None, None, None
    center_x = float(np.mean(points[:, 0]))
    center_y = float(np.mean(points[:, 1]))
    side_lengths = []
    for i in range(4):
        side_lengths.append(
            float(np.linalg.norm(points[(i + 1) % 4] - points[i]))
        )
    radius = min(side_lengths) / 2.0 if side_lengths else 0.0
    if radius <= 0:
        return None, None, None
    return center_x, center_y, radius


def _quad_mask_iou(contour, quad):
    """IoU between the source contour mask and a candidate quad mask.

    Rasterized on the tight bounding crop of both. Returns None when the
    crop is degenerate or unreasonably large (caller falls back to an
    area-ratio score instead).
    """
    pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    cand = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    lo = np.floor(np.min(np.vstack([pts, cand]), axis=0)).astype(int)
    hi = np.ceil(np.max(np.vstack([pts, cand]), axis=0)).astype(int)
    w, h = int(hi[0] - lo[0] + 1), int(hi[1] - lo[1] + 1)
    if w <= 0 or h <= 0 or w * h > 4_000_000:
        return None
    mask_contour = np.zeros((h, w), dtype=np.uint8)
    mask_quad = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_contour, [(pts - lo).astype(np.int32)], 1)
    cv2.fillPoly(mask_quad, [(cand - lo).astype(np.int32)], 1)
    inter = np.logical_and(mask_contour, mask_quad).sum()
    union = np.logical_or(mask_contour, mask_quad).sum()
    if union == 0:
        return None
    return float(inter) / float(union)


def _ransac_quad_candidate(contour, tol, seed=0):
    """Fit a quad by sequential RANSAC line fitting.

    Each of the 4 plate edges is found by sampling point pairs and
    keeping the line with the most inliers, so spikes/notches (outliers)
    cannot drag corners the way vertex-based simplification can. Lines
    are ordered by angle around the centroid, intersected consecutively,
    and rejected unless convex.

    Returns 4 corners, or None when no trustworthy quad is found.
    """
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 16:
        return None
    rng = np.random.default_rng(seed)
    remaining = np.ones(n, dtype=bool)
    lines = []
    for _ in range(4):
        idx = np.nonzero(remaining)[0]
        if len(idx) < 8:
            return None
        cand = pts[idx]
        best_inliers = None
        for _ in range(120):
            a, b = rng.integers(0, len(cand), size=2)
            if a == b:
                continue
            d = cand[b] - cand[a]
            length = float(np.linalg.norm(d))
            if length <= 1e-9:
                continue
            unit = d / length
            dist = np.abs((cand - cand[a]) @ np.array([-unit[1], unit[0]]))
            inliers = np.nonzero(dist <= tol)[0]
            if best_inliers is None or len(inliers) > len(best_inliers):
                best_inliers = inliers
        if best_inliers is None or len(best_inliers) < max(
            8, int(0.05 * n)
        ):
            return None
        vx, vy, x0, y0 = cv2.fitLine(
            cand[best_inliers].astype(np.float32),
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01,
        ).flatten()
        norm = math.hypot(float(vx), float(vy))
        if norm <= 0:
            return None
        lines.append(
            (
                np.array([float(x0), float(y0)]),
                np.array([float(vx) / norm, float(vy) / norm]),
            )
        )
        remaining[idx[best_inliers]] = False
    centroid = np.mean(pts, axis=0)
    order = sorted(
        range(4),
        key=lambda i: math.atan2(
            lines[i][0][1] - centroid[1], lines[i][0][0] - centroid[0]
        ),
    )
    ordered = [lines[i] for i in order]
    corners = []
    for j in range(4):
        (p1, d1), (p2, d2) = ordered[(j - 1) % 4], ordered[j]
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-9:
            return None
        w = p2 - p1
        t = (w[0] * d2[1] - w[1] * d2[0]) / denom
        corners.append(p1 + t * d1)
    corners = np.asarray(corners)
    signs = []
    for i in range(4):
        e1 = corners[(i + 1) % 4] - corners[i]
        e2 = corners[(i + 2) % 4] - corners[(i + 1) % 4]
        signs.append(e1[0] * e2[1] - e1[1] * e2[0])
    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        return None
    return corners.tolist()


def _refine_quad_corners(contour, quad, min_edge_points=6, iou_tolerance=0.05):
    """Refine quad corners by intersecting robustly fitted edge lines.

    The true corner of a blurred/rounded plate mask is usually not among
    the contour vertices at all, so no simplification can recover it.
    Instead each initial edge claims the contour points along its
    straight middle section (corners excluded, where rounding lives),
    fits a Huber-robust line, and adjacent lines are intersected. Sides
    with too few supporting points keep their initial edge.

    Returns the refined 4 corners, or None when the refinement is not
    trustworthy (degenerate lines, non-convex result, implausible area,
    or mask IoU well below the initial quad).
    """
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    corners = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 8 or len(corners) != 4:
        return None
    peri = float(cv2.arcLength(pts.astype(np.float32), True))
    if peri <= 0:
        return None
    tol_px = max(2.0, 0.01 * peri)
    lines = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        edge = b - a
        edge_len = float(np.linalg.norm(edge))
        if edge_len <= 0:
            return None
        # All quantities below are plain pixels.
        unit = edge / edge_len
        rel = pts - a
        proj = rel[:, 0] * unit[0] + rel[:, 1] * unit[1]
        dist = np.abs(rel[:, 0] * unit[1] - rel[:, 1] * unit[0])
        side_pts = pts[
            (dist <= tol_px)
            & (proj >= 0.05 * edge_len)
            & (proj <= 0.95 * edge_len)
        ]
        if len(side_pts) < min_edge_points:
            lines.append((a, unit))
            continue
        vx, vy, x0, y0 = cv2.fitLine(
            side_pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
        ).flatten()
        norm = math.hypot(float(vx), float(vy))
        if norm <= 0:
            lines.append((a, unit))
            continue
        lines.append(
            (
                np.array([float(x0), float(y0)]),
                np.array([float(vx) / norm, float(vy) / norm]),
            )
        )
    refined = []
    for j in range(4):
        (p1, d1), (p2, d2) = lines[(j - 1) % 4], lines[j]
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-9:
            return None
        w = p2 - p1
        t = (w[0] * d2[1] - w[1] * d2[0]) / denom
        refined.append(p1 + t * d1)
    refined = np.asarray(refined)
    signs = []
    for i in range(4):
        e1 = refined[(i + 1) % 4] - refined[i]
        e2 = refined[(i + 2) % 4] - refined[(i + 1) % 4]
        signs.append(e1[0] * e2[1] - e1[1] * e2[0])
    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        return None
    quad_area = abs(float(cv2.contourArea(refined.astype(np.float32))))
    contour_area = abs(float(cv2.contourArea(pts.astype(np.float32))))
    if contour_area <= 0 or not (
        0.5 * contour_area <= quad_area <= 2.0 * contour_area
    ):
        return None
    initial_iou = _quad_mask_iou(pts, corners)
    refined_iou = _quad_mask_iou(pts, refined)
    if (
        initial_iou is not None
        and refined_iou is not None
        and refined_iou < initial_iou - iou_tolerance
    ):
        return None
    return refined.tolist()


def _polygon_to_quad_points(points):
    """Fit quadrilateral corners to polygon points.

    Four-point polygons keep their vertices (a closing duplicate is
    dropped first, e.g. SAM-style closed contours). Otherwise every
    approxPolyDP simplification that yields 4 corners, plus the
    min-area-rectangle box, are scored by mask IoU against the source
    contour and the best-scoring quad wins. The winner is then refined
    by intersecting Huber-robust edge lines, which recovers true
    corners that a blurred/rounded mask rounded off (see
    :func:`_refine_quad_corners`).

    Returns a list of 4 points, or None if the polygon is degenerate.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    # Drop closing duplicate (e.g. SAM-style closed polygons).
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) == 4:
        return pts.tolist()
    if len(pts) < 3:
        return None
    peri = float(cv2.arcLength(pts, True))
    contour_area = abs(float(cv2.contourArea(pts)))
    if peri <= 0 or contour_area <= 0:
        return None
    candidates = []
    for factor in (0.005, 0.01, 0.02, 0.04, 0.08, 0.12):
        approx = cv2.approxPolyDP(pts, factor * peri, True).reshape(-1, 2)
        if len(approx) == 4 and not any(
            np.allclose(approx, c) for c in candidates
        ):
            candidates.append(np.asarray(approx, dtype=np.float32))
    box = np.asarray(
        cv2.boxPoints(cv2.minAreaRect(pts)), dtype=np.float32
    )
    if not any(np.allclose(box, c) for c in candidates):
        candidates.append(box)
    if not candidates:
        return box.tolist()

    def _area_score(candidate):
        area = abs(float(cv2.contourArea(candidate)))
        if contour_area <= 0:
            return -1.0
        return -abs(1.0 - area / contour_area)

    best, best_key = None, None
    for candidate in candidates:
        iou = _quad_mask_iou(pts, candidate)
        key = (iou if iou is not None else -1.0, _area_score(candidate))
        if best_key is None or key > best_key:
            best, best_key = candidate, key
    if best_key is not None and best_key[0] < 0.98:
        # Cheap candidates disagree with the mask: spend RANSAC effort on
        # an outlier-robust line-based quad (spikes/notches cannot drag
        # its corners the way vertex simplification can).
        ransac = _ransac_quad_candidate(pts, max(2.0, 0.01 * peri))
        if ransac is not None:
            iou = _quad_mask_iou(pts, np.asarray(ransac, dtype=np.float32))
            key = (iou if iou is not None else -1.0, _area_score(
                np.asarray(ransac, dtype=np.float32)
            ))
            if key > best_key:
                best, best_key = (
                    np.asarray(ransac, dtype=np.float32),
                    key,
                )
    refined = _refine_quad_corners(pts, best)
    if refined is not None:
        return np.asarray(refined, dtype=np.float32).tolist()
    return np.asarray(best, dtype=np.float32).tolist()


def _apply_shape_conversion(data, mode, params):
    normalized_mode = _normalize_mode(mode)
    for j in range(len(data["shapes"])):
        shape = data["shapes"][j]
        if shape.get("locked", False):
            continue
        shape_type = shape.get("shape_type")
        points = shape.get("points", [])

        if (
            normalized_mode == "rectangle_to_rotation"
            and shape_type == "rectangle"
        ):
            points = _to_axis_aligned_box(points)
            if points is None:
                continue
            shape["shape_type"] = "rotation"
            shape["points"] = points
            shape["direction"] = 0

        elif (
            normalized_mode == "rotation_to_rectangle"
            and shape_type == "rotation"
        ):
            points = _to_axis_aligned_box(points)
            if points is None:
                continue
            shape.pop("direction", None)
            shape["shape_type"] = "rectangle"
            shape["points"] = points

        elif (
            normalized_mode == "polygon_to_rectangle"
            and shape_type == "polygon"
        ):
            if len(points) < 3:
                continue
            points = _to_axis_aligned_box(points)
            if points is None:
                continue
            shape["shape_type"] = "rectangle"
            shape["points"] = points

        elif (
            normalized_mode == "polygon_to_rotation"
            and shape_type == "polygon"
        ):
            points = np.asarray(points)
            if len(points) < 3:
                continue
            contours = points.reshape((-1, 1, 2)).astype(np.float32)
            _, rotation_box = get_bounding_boxes(contours)
            shape["shape_type"] = "rotation"
            shape["points"] = rotation_box.tolist()
            shape["direction"] = calculate_rotation_theta(rotation_box)

        elif normalized_mode == "circle_to_polygon" and shape_type == "circle":
            center_x, center_y, radius = _circle_center_radius(points)
            if radius is None:
                continue
            num_sides = params.get("num_sides", 32)
            polygon_points = []
            for i in range(num_sides):
                angle = 2 * math.pi * i / num_sides
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                polygon_points.append([x, y])
            shape["shape_type"] = "polygon"
            shape["points"] = polygon_points
            shape.pop("direction", None)

        elif (
            normalized_mode == "rectangle_to_polygon"
            and shape_type == "rectangle"
        ):
            points = _to_axis_aligned_box(points)
            if points is None:
                continue
            shape["shape_type"] = "polygon"
            shape["points"] = points
            shape.pop("direction", None)

        elif (
            normalized_mode == "rotation_to_polygon"
            and shape_type == "rotation"
        ):
            points = np.asarray(points).tolist()
            if len(points) != 4:
                continue
            shape["shape_type"] = "polygon"
            shape["points"] = points
            shape.pop("direction", None)

        elif (
            normalized_mode == "rectangle_to_quadrilateral"
            and shape_type == "rectangle"
        ):
            points = _to_axis_aligned_box(points)
            if points is None:
                continue
            shape["shape_type"] = "quadrilateral"
            shape["points"] = points
            shape.pop("direction", None)

        elif (
            normalized_mode == "rotation_to_quadrilateral"
            and shape_type == "rotation"
        ):
            points = np.asarray(points).tolist()
            if len(points) != 4:
                continue
            shape["shape_type"] = "quadrilateral"
            shape["points"] = points
            shape.pop("direction", None)

        elif (
            normalized_mode == "quadrilateral_to_polygon"
            and shape_type == "quadrilateral"
        ):
            points = np.asarray(points).tolist()
            if len(points) != 4:
                continue
            shape["shape_type"] = "polygon"
            shape["points"] = points
            shape.pop("direction", None)

        elif (
            normalized_mode == "polygon_to_quadrilateral"
            and shape_type == "polygon"
        ):
            quad = _polygon_to_quad_points(points)
            if quad is None:
                continue
            shape["shape_type"] = "quadrilateral"
            shape["points"] = quad
            shape.pop("direction", None)

        elif normalized_mode == "line_to_linestrip" and shape_type == "line":
            points = np.asarray(points).tolist()
            if len(points) < 2:
                continue
            shape["shape_type"] = "linestrip"
            shape["points"] = points
            shape.pop("direction", None)

        elif (
            normalized_mode == "rectangle_to_circle"
            and shape_type == "rectangle"
        ):
            points = _to_axis_aligned_box(points)
            if points is None:
                continue
            width = abs(points[1][0] - points[0][0])
            height = abs(points[2][1] - points[1][1])
            radius = min(width, height) / 2.0
            if radius <= 0:
                continue
            center_x = (points[0][0] + points[2][0]) / 2.0
            center_y = (points[0][1] + points[2][1]) / 2.0
            shape["shape_type"] = "circle"
            shape["points"] = _circle_points(center_x, center_y, radius)
            shape.pop("direction", None)

        elif (
            normalized_mode == "rotation_to_circle"
            and shape_type == "rotation"
        ):
            center_x, center_y, radius = _rotation_center_inscribed_radius(
                points
            )
            if radius is None:
                continue
            shape["shape_type"] = "circle"
            shape["points"] = _circle_points(center_x, center_y, radius)
            shape.pop("direction", None)

        elif (
            normalized_mode == "circle_to_rectangle" and shape_type == "circle"
        ):
            center_x, center_y, radius = _circle_center_radius(points)
            if radius is None:
                continue
            shape["shape_type"] = "rectangle"
            shape["points"] = [
                [int(center_x - radius), int(center_y - radius)],
                [int(center_x + radius), int(center_y - radius)],
                [int(center_x + radius), int(center_y + radius)],
                [int(center_x - radius), int(center_y + radius)],
            ]
            shape.pop("direction", None)

        elif (
            normalized_mode == "circle_to_rotation" and shape_type == "circle"
        ):
            center_x, center_y, radius = _circle_center_radius(points)
            if radius is None:
                continue
            shape["shape_type"] = "rotation"
            shape["points"] = [
                [int(center_x - radius), int(center_y - radius)],
                [int(center_x + radius), int(center_y - radius)],
                [int(center_x + radius), int(center_y + radius)],
                [int(center_x - radius), int(center_y + radius)],
            ]
            shape["direction"] = 0

        elif (
            normalized_mode == "circle_to_quadrilateral"
            and shape_type == "circle"
        ):
            center_x, center_y, radius = _circle_center_radius(points)
            if radius is None:
                continue
            shape["shape_type"] = "quadrilateral"
            shape["points"] = [
                [int(center_x - radius), int(center_y - radius)],
                [int(center_x + radius), int(center_y - radius)],
                [int(center_x + radius), int(center_y + radius)],
                [int(center_x - radius), int(center_y + radius)],
            ]
            shape.pop("direction", None)


def open_shape_converter(self):
    from anylabeling.views.labeling.widgets.shape_converter_dialog import (
        ShapeConverterDialog,
    )

    dialog = ShapeConverterDialog(self)
    dialog.exec()


def convert_single_shape(shape_type, points, target_type, params=None):
    """Convert one shape's geometry to another shape type.

    Shares the batch conversion logic in :func:`_apply_shape_conversion`.

    Args:
        shape_type: Source shape type (e.g. ``"polygon"``).
        points: List of ``[x, y]`` coordinates.
        target_type: Destination shape type (e.g. ``"quadrilateral"``).
        params: Optional conversion params (see :func:`get_conversion_params`).

    Returns:
        ``(new_type, new_points, direction)`` on success, or None when the
        conversion is unsupported or the geometry is degenerate.
        ``direction`` is only meaningful for ``rotation`` targets.
    """
    if target_type not in CONVERSION_TARGETS.get(shape_type, []):
        return None
    shape = {
        "shape_type": shape_type,
        "points": [list(p) for p in points],
    }
    _apply_shape_conversion(
        {"shapes": [shape]},
        f"{shape_type}_to_{target_type}",
        params or {},
    )
    if shape["shape_type"] != target_type:
        return None
    return shape["shape_type"], shape["points"], shape.get("direction")


def get_conversion_params(self, mode: str):
    """Get parameters required for specific conversion modes.

    Args:
        mode (str): The conversion mode

    Returns:
        dict: Parameters dictionary, or None if user cancelled
    """
    if mode == "circle_to_polygon":
        dialog = PolygonSidesDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return {"num_sides": dialog.get_value()}
        else:
            return None

    return {}


def shape_conversion(self, mode):
    label_file_list = self.get_label_file_list()
    if len(label_file_list) == 0:
        return

    params = get_conversion_params(self, mode)
    if params is None:
        return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be changed"))
    response.setInformativeText(
        self.tr("Are you sure you want to perform this conversion?")
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Cancel
        | QtWidgets.QMessageBox.StandardButton.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Converting..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())
    progress_dialog.show()

    try:
        for i, label_file in enumerate(label_file_list):
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            _apply_shape_conversion(data, mode, params)

            with open(label_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            progress_dialog.setValue(i)
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        popup = Popup(
            self.tr("Conversion completed successfully!"),
            self,
            msec=1000,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

        self.load_file(self.filename)

    except Exception as e:
        logger.error(f"Error occurred while converting shapes: {e}")
        popup = Popup(
            self.tr("Error occurred while converting shapes!"),
            self,
            msec=1000,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def polygons_to_mask(img_shape, polygons, shape_type=None):
    logger.warning(
        "The 'polygons_to_mask' function is deprecated, "
        "use 'shape_to_mask' instead."
    )
    return shape_to_mask(img_shape, points=polygons, shape_type=shape_type)


def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "rotation":
        assert len(xy) == 4, "Shape of shape_type=rotation must have 4 points"
        draw.polygon(xy=xy, outline=1, fill=1)
    elif shape_type == "quadrilateral":
        assert len(xy) == 4, (
            "Shape of shape_type=quadrilateral must have 4 points"
        )
        draw.polygon(xy=xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


def masks_to_bboxes(masks):
    if masks.ndim != 3:
        raise ValueError(f"masks.ndim must be 3, but it is {masks.ndim}")
    if masks.dtype != bool:
        raise ValueError(
            f"masks.dtype must be bool type, but it is {masks.dtype}"
        )
    bboxes = []
    for mask in masks:
        where = np.argwhere(mask)
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
        bboxes.append((y1, x1, y2, x2))
    bboxes = np.asarray(bboxes, dtype=np.float32)
    return bboxes


def rectangle_from_diagonal(diagonal_vertices):
    """
    Generate rectangle vertices from diagonal vertices.

    Parameters:
    - diagonal_vertices (list of lists):
        List containing two points representing the diagonal vertices.

    Returns:
    - list of lists:
        List containing four points representing the rectangle's four corners.
        [tl -> tr -> br -> bl]
    """
    x1, y1 = diagonal_vertices[0]
    x2, y2 = diagonal_vertices[1]

    # Creating the four-point representation
    rectangle_vertices = [
        [x1, y1],  # Top-left
        [x2, y1],  # Top-right
        [x2, y2],  # Bottom-right
        [x1, y2],  # Bottom-left
    ]

    return rectangle_vertices
