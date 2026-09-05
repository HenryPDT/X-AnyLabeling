"""Consolidated geometry and overlap utilities for shapes and bounding boxes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def shape_to_xyxy(shape_or_points: Any) -> Optional[List[float]]:
    """Return axis-aligned [x1, y1, x2, y2] from a Shape, point list, or dict.

    Args:
        shape_or_points: A Shape object, dict with 'points' key,
            numpy array of shape (N, 2), or list of coordinates/points.

    Returns:
        List of [x1, y1, x2, y2] as floats, or None if invalid/empty.
    """
    if shape_or_points is None:
        return None

    points = None
    if hasattr(shape_or_points, "points"):
        points = shape_or_points.points
    elif isinstance(shape_or_points, dict) and "points" in shape_or_points:
        points = shape_or_points["points"]
    elif isinstance(shape_or_points, (list, tuple, np.ndarray)):
        points = shape_or_points

    if points is None or len(points) == 0:
        return None

    xs: List[float] = []
    ys: List[float] = []

    for pt in points:
        if pt is None:
            continue
        if hasattr(pt, "x") and hasattr(pt, "y"):
            try:
                x_val = float(pt.x())
                y_val = float(pt.y())
            except Exception:
                continue
        elif isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2:
            try:
                x_val = float(pt[0])
                y_val = float(pt[1])
            except (ValueError, TypeError):
                continue
        else:
            continue

        if math.isfinite(x_val) and math.isfinite(y_val):
            xs.append(x_val)
            ys.append(y_val)

    if not xs or not ys:
        return None

    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def box_area(box: Optional[List[float]]) -> float:
    """Calculate the area of an axis-aligned bounding box [x1, y1, x2, y2]."""
    if box is None or len(box) < 4:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def box_intersection(
    box_a: Optional[List[float]], box_b: Optional[List[float]]
) -> float:
    """Calculate the intersection area between two boxes."""
    if box_a is None or box_b is None or len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(
    box_a: Optional[List[float]], box_b: Optional[List[float]]
) -> float:
    """Calculate Intersection over Union (IoU) between two boxes."""
    inter = box_intersection(box_a, box_b)
    if inter <= 0.0:
        return 0.0
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    union = area_a + area_b - inter
    return inter / union if union > 1e-12 else 0.0


def box_containment_ratio(
    inner_box: Optional[List[float]], outer_box: Optional[List[float]]
) -> float:
    """Fraction of inner_box area that lies inside outer_box."""
    area = box_area(inner_box)
    if area <= 0.0:
        return 0.0
    return box_intersection(inner_box, outer_box) / area


def box_overlap_metrics(
    box_a: Optional[List[float]], box_b: Optional[List[float]]
) -> Dict[str, float]:
    """Calculate comprehensive overlap and proximity metrics between two boxes.

    Returns dict containing:
        - iou: Intersection over union
        - containment: Max containment ratio between the two boxes
        - containment_a_in_b: Fraction of box_a inside box_b
        - containment_b_in_a: Fraction of box_b inside box_a
        - area_ratio: Ratio of smaller area over larger area
        - center_distance: Euclidean distance between box centroids
        - relative_center_distance: Distance normalized by average diagonal
    """
    empty_metrics = {
        "iou": 0.0,
        "containment": 0.0,
        "containment_a_in_b": 0.0,
        "containment_b_in_a": 0.0,
        "area_ratio": 0.0,
        "center_distance": 9999.0,
        "relative_center_distance": 9999.0,
    }
    if box_a is None or box_b is None or len(box_a) < 4 or len(box_b) < 4:
        return empty_metrics

    area_a = box_area(box_a)
    area_b = box_area(box_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return empty_metrics

    inter = box_intersection(box_a, box_b)
    union = area_a + area_b - inter
    iou = inter / union if union > 1e-12 else 0.0

    cont_a_in_b = inter / area_a
    cont_b_in_a = inter / area_b
    containment = max(cont_a_in_b, cont_b_in_a)

    smaller_area = min(area_a, area_b)
    larger_area = max(area_a, area_b)
    area_ratio = smaller_area / larger_area if larger_area > 0.0 else 0.0

    center_a = (
        (float(box_a[0]) + float(box_a[2])) / 2.0,
        (float(box_a[1]) + float(box_a[3])) / 2.0,
    )
    center_b = (
        (float(box_b[0]) + float(box_b[2])) / 2.0,
        (float(box_b[1]) + float(box_b[3])) / 2.0,
    )
    center_dist = math.hypot(
        center_a[0] - center_b[0], center_a[1] - center_b[1]
    )

    diag_a = math.hypot(
        float(box_a[2]) - float(box_a[0]), float(box_a[3]) - float(box_a[1])
    )
    diag_b = math.hypot(
        float(box_b[2]) - float(box_b[0]), float(box_b[3]) - float(box_b[1])
    )
    avg_diag = max(1e-6, (diag_a + diag_b) / 2.0)
    rel_center_dist = center_dist / avg_diag

    return {
        "iou": float(iou),
        "containment": float(containment),
        "containment_a_in_b": float(cont_a_in_b),
        "containment_b_in_a": float(cont_b_in_a),
        "area_ratio": float(area_ratio),
        "center_distance": float(center_dist),
        "relative_center_distance": float(rel_center_dist),
    }


def is_duplicate_box_geometry(
    box_a: List[float],
    box_b: List[float],
    iou_threshold: float = 0.85,
    containment_threshold: float = 0.90,
) -> Tuple[bool, str, Dict[str, float]]:
    """Determine whether two boxes represent duplicate annotations.

    Returns:
        (is_duplicate, match_type, metrics)
    """
    metrics = box_overlap_metrics(box_a, box_b)
    iou = metrics["iou"]
    containment = metrics["containment"]
    area_ratio = metrics["area_ratio"]
    rel_center = metrics["relative_center_distance"]

    if iou >= float(iou_threshold):
        return True, "iou", metrics
    if (
        containment >= float(containment_threshold)
        and area_ratio >= 0.20
        and rel_center <= 0.40
    ):
        return True, "containment", metrics
    return False, "", metrics


def get_shape_label(shape: Any) -> Optional[str]:
    """Extract label string from a Shape object or dictionary."""
    if hasattr(shape, "label"):
        return getattr(shape, "label")
    if isinstance(shape, dict):
        return shape.get("label")
    return None


def get_shape_score(shape: Any) -> Optional[float]:
    """Extract score float from a Shape object or dictionary."""
    val = (
        getattr(shape, "score", None)
        if hasattr(shape, "score")
        else shape.get("score")
        if isinstance(shape, dict)
        else None
    )
    return float(val) if val is not None else None


def get_shape_type(shape: Any, default: str = "polygon") -> str:
    """Extract shape_type string from a Shape object or dictionary."""
    stype = (
        getattr(shape, "shape_type", None)
        if hasattr(shape, "shape_type")
        else shape.get("shape_type")
        if isinstance(shape, dict)
        else None
    )
    return str(stype or default)


def detect_duplicate_shapes(
    shapes: List[Any],
    iou_threshold: float = 0.85,
    containment_threshold: float = 0.90,
    same_label_only: bool = False,
) -> List[Dict[str, Any]]:
    """Detect all pairs of duplicate or heavily overlapping shapes in an image.

    Args:
        shapes: List of Shape objects or shape dicts.
        iou_threshold: Minimum IoU to consider duplicate.
        containment_threshold: Minimum containment ratio to consider duplicate.
        same_label_only: If True, only flag duplicates with identical labels.

    Returns:
        List of duplicate candidate records.
    """
    if not shapes or len(shapes) < 2:
        return []

    duplicates = []
    n = len(shapes)
    boxes = [shape_to_xyxy(s) for s in shapes]

    for i in range(n):
        box_i = boxes[i]
        if box_i is None:
            continue
        label_i = get_shape_label(shapes[i])

        for j in range(i + 1, n):
            box_j = boxes[j]
            if box_j is None:
                continue
            label_j = get_shape_label(shapes[j])

            if same_label_only and label_i != label_j:
                continue

            is_dup, match_type, metrics = is_duplicate_box_geometry(
                box_i,
                box_j,
                iou_threshold=iou_threshold,
                containment_threshold=containment_threshold,
            )
            if is_dup:
                duplicates.append(
                    {
                        "index_a": i,
                        "index_b": j,
                        "shape_a": shapes[i],
                        "shape_b": shapes[j],
                        "label_a": label_i,
                        "label_b": label_j,
                        "match_type": match_type,
                        "metrics": metrics,
                    }
                )

    return duplicates


def apply_class_agnostic_shape_nms(
    shapes: List[Any],
    iou_threshold: Optional[float],
    group_by_shape_type: bool = True,
    containment_threshold: Optional[float] = None,
    containment_keep: str = "score",
) -> List[Any]:
    """Class-agnostic IoU NMS and containment suppression for shapes."""
    if not shapes:
        return shapes

    kept = list(shapes)
    if iou_threshold is not None and iou_threshold > 0:
        if group_by_shape_type:
            grouped: Dict[Any, List[Any]] = {}
            for shape in kept:
                stype = get_shape_type(shape)
                grouped.setdefault(stype, []).append(shape)
            next_kept = []
            for group_shapes in grouped.values():
                next_kept.extend(
                    _apply_class_agnostic_shape_nms_group(
                        group_shapes, iou_threshold
                    )
                )
            kept = next_kept
        else:
            kept = _apply_class_agnostic_shape_nms_group(kept, iou_threshold)

    if containment_threshold is not None and containment_threshold > 0:
        if group_by_shape_type:
            grouped = {}
            for shape in kept:
                stype = get_shape_type(shape)
                grouped.setdefault(stype, []).append(shape)
            next_kept = []
            for group_shapes in grouped.values():
                next_kept.extend(
                    apply_same_label_containment_nms(
                        group_shapes,
                        containment_threshold,
                        keep_mode=containment_keep,
                    )
                )
            kept = next_kept
        else:
            kept = apply_same_label_containment_nms(
                kept,
                containment_threshold,
                keep_mode=containment_keep,
            )
    return kept


def apply_same_label_containment_nms(
    shapes: List[Any],
    containment_threshold: Optional[float],
    keep_mode: str = "score",
) -> List[Any]:
    """Drop same-label nested shapes using score or area priority."""
    if (
        not shapes
        or containment_threshold is None
        or containment_threshold <= 0
    ):
        return list(shapes)

    keep_mode = (keep_mode or "score").lower()
    if keep_mode not in {"score", "area"}:
        keep_mode = "score"

    boxes = []
    scores = []
    areas = []
    labels = []
    valid_shapes = []
    passthrough_shapes = []
    for shape in shapes:
        box = shape_to_xyxy(shape)
        if box is None:
            passthrough_shapes.append(shape)
            continue
        boxes.append(box)
        score_val = get_shape_score(shape)
        scores.append(score_val if score_val is not None else 0.0)
        areas.append(box_area(box))
        labels.append(get_shape_label(shape))
        valid_shapes.append(shape)

    if not valid_shapes:
        return list(shapes)

    priority = areas if keep_mode == "area" else scores
    order = np.argsort(-np.asarray(priority, dtype=np.float32), kind="stable")
    keep: List[int] = []
    for index in order:
        candidate_box = boxes[int(index)]
        candidate_label = labels[int(index)]
        suppressed = False
        for kept_index in keep:
            if labels[kept_index] != candidate_label:
                continue
            kept_box = boxes[kept_index]
            candidate_in_kept = box_containment_ratio(candidate_box, kept_box)
            kept_in_candidate = box_containment_ratio(kept_box, candidate_box)
            if (
                max(candidate_in_kept, kept_in_candidate)
                > containment_threshold
            ):
                suppressed = True
                break
        if not suppressed:
            keep.append(int(index))

    keep_set = set(keep)
    result = [
        shape for index, shape in enumerate(valid_shapes) if index in keep_set
    ]
    result.extend(passthrough_shapes)
    return result


def _apply_class_agnostic_shape_nms_group(
    shapes: List[Any], iou_threshold: float
) -> List[Any]:
    from anylabeling.services.auto_labeling.utils.box import numpy_nms

    boxes = []
    scores = []
    valid_shapes = []
    passthrough_shapes = []
    for shape in shapes:
        box = shape_to_xyxy(shape)
        if box is None:
            passthrough_shapes.append(shape)
            continue
        boxes.append(box)
        score_val = get_shape_score(shape)
        scores.append(score_val if score_val is not None else 0.0)
        valid_shapes.append(shape)

    if not valid_shapes:
        return list(shapes)

    keep = numpy_nms(
        np.asarray(boxes, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        iou_threshold,
    )
    result = [valid_shapes[int(index)] for index in keep]
    result.extend(passthrough_shapes)
    return result


def _clamp_rectangle_points(
    pts: List[Tuple[float, float]], w: float, h: float
) -> Optional[List[Tuple[float, float]]]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x = max(0.0, min(w, min(xs)))
    max_x = max(0.0, min(w, max(xs)))
    min_y = max(0.0, min(h, min(ys)))
    max_y = max(0.0, min(h, max(ys)))
    if max_x <= min_x or max_y <= min_y:
        return None
    return [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]


def _clamp_polygon_points(
    pts: List[Tuple[float, float]], w: float, h: float
) -> Optional[List[Tuple[float, float]]]:
    clamped_tuples = [
        (max(0.0, min(w, p[0])), max(0.0, min(h, p[1]))) for p in pts
    ]
    dedup_tuples: List[Tuple[float, float]] = []
    for pt in clamped_tuples:
        if not dedup_tuples or pt != dedup_tuples[-1]:
            dedup_tuples.append(pt)
    if len(dedup_tuples) > 1 and dedup_tuples[0] == dedup_tuples[-1]:
        dedup_tuples.pop()
    if len(dedup_tuples) < 3:
        return None
    return dedup_tuples


def clamp_shape_to_image_bounds(
    shape: Any, img_width: float, img_height: float
) -> Optional[Any]:
    """Clamp shape coordinates to remain within [0, img_width] and [0, img_height].

    Args:
        shape: Shape object or dictionary.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        The shape with clamped points, or None if the shape degenerates/lies entirely outside.
    """
    if img_width <= 0 or img_height <= 0:
        return shape

    stype = get_shape_type(shape)
    is_obj = hasattr(shape, "points")
    raw_points = shape.points if is_obj else shape.get("points", [])
    if not raw_points:
        return shape

    pts: List[Tuple[float, float]] = []
    for p in raw_points:
        if hasattr(p, "x") and hasattr(p, "y"):
            pts.append((float(p.x()), float(p.y())))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append((float(p[0]), float(p[1])))

    if not pts:
        return shape

    w = float(img_width)
    h = float(img_height)

    if stype == "rectangle":
        clamped_4 = _clamp_rectangle_points(pts, w, h)
        if clamped_4 is None:
            return None
        # Preserve 2-pt representation to avoid JSON churn
        if len(pts) == 2:
            clamped_tuples = [clamped_4[0], clamped_4[2]]
        else:
            clamped_tuples = clamped_4
    elif stype == "point":
        cx = max(0.0, min(w, pts[0][0]))
        cy = max(0.0, min(h, pts[0][1]))
        clamped_tuples = [(cx, cy)]
    elif stype == "rotation":
        direction = (
            getattr(shape, "direction", 0)
            if is_obj
            else shape.get("direction", 0)
        )
        if direction == 0:
            clamped_tuples = _clamp_rectangle_points(pts, w, h)
        else:
            clamped_tuples = [
                (max(0.0, min(w, p[0])), max(0.0, min(h, p[1]))) for p in pts
            ]
    elif stype == "polygon":
        clamped_tuples = _clamp_polygon_points(pts, w, h)
    else:
        clamped_tuples = [
            (max(0.0, min(w, p[0])), max(0.0, min(h, p[1]))) for p in pts
        ]

    if clamped_tuples is None:
        return None

    if is_obj:
        from PyQt6 import QtCore

        shape.points = [QtCore.QPointF(x, y) for x, y in clamped_tuples]
    else:
        shape["points"] = [[x, y] for x, y in clamped_tuples]
    return shape


def clamp_shapes_to_image_bounds(
    shapes: List[Any], img_width: float, img_height: float
) -> List[Any]:
    """Clamp a list of shapes to image boundaries, discarding degenerates."""
    if not shapes or img_width <= 0 or img_height <= 0:
        return list(shapes) if shapes else []
    clamped: List[Any] = []
    for s in shapes:
        res = clamp_shape_to_image_bounds(s, img_width, img_height)
        if res is not None:
            clamped.append(res)
    return clamped
