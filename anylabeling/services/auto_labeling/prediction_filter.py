from __future__ import annotations

from typing import Any, List

from anylabeling.views.labeling.utils.shape_geometry import (
    get_shape_label,
    get_shape_type,
    is_duplicate_box_geometry,
    shape_to_xyxy,
)


def filter_prediction_sizes(
    shapes: List[Any],
    img_width: float,
    img_height: float,
    min_size_px: float = 5.0,
    max_percent: float = 0.98,
) -> List[Any]:
    """Filter out prediction shapes that are micro-noise or span almost the whole image.

    Args:
        shapes: List of Shape objects or shape dictionaries.
        img_width: Image width in pixels.
        img_height: Image height in pixels.
        min_size_px: Minimum width and height for a bounding box in pixels.
        max_percent: Maximum allowed box area as a fraction of total image area (0.0 - 1.0).

    Returns:
        List of surviving prediction shapes.
    """
    if not shapes:
        return []

    try:
        min_size_px = max(0.0, float(min_size_px))
    except (TypeError, ValueError):
        min_size_px = 5.0
    try:
        max_percent = float(max_percent)
        if not 0.0 < max_percent <= 1.0:
            max_percent = 0.98
    except (TypeError, ValueError):
        max_percent = 0.98

    img_area = max(0.0, float(img_width) * float(img_height))
    max_area = img_area * float(max_percent) if img_area > 0 else float("inf")
    filtered: List[Any] = []

    for shape in shapes:
        shape_type = str(get_shape_type(shape)).lower()

        # Exempt point shapes from min size checks
        if shape_type == "point":
            filtered.append(shape)
            continue

        box = shape_to_xyxy(shape)
        if box is None:
            filtered.append(shape)
            continue

        w = max(0.0, box[2] - box[0])
        h = max(0.0, box[3] - box[1])
        area = w * h

        # Lines: length gate, not width/height gate
        if shape_type in ("line", "linestrip"):
            import math as _math

            if _math.hypot(w, h) < min_size_px:
                continue
            if img_area > 0 and area > max_area:
                continue
            filtered.append(shape)
            continue

        # Discard micro-noise
        if w < min_size_px or h < min_size_px:
            continue

        # Discard predictions that span essentially the entire image
        if img_area > 0 and area > max_area:
            continue

        filtered.append(shape)

    return filtered


def filter_duplicate_shapes(
    new_shapes: List[Any],
    existing_shapes: List[Any],
    iou_threshold: float = 0.85,
    containment_threshold: float = 0.90,
    same_label_only: bool = False,
) -> List[Any]:
    """Filter out new predictions that heavily overlap with existing annotations.

    Args:
        new_shapes: List of newly predicted Shape objects or dicts.
        existing_shapes: List of already present Shape objects or dicts on the canvas.
        iou_threshold: IoU cutoff above which an incoming prediction is considered duplicate.
        containment_threshold: Containment cutoff above which an incoming prediction is considered duplicate.
        same_label_only: If True, only drops duplicates when class labels match.

    Returns:
        List of non-duplicate new shapes.
    """
    if not new_shapes:
        return []
    if not existing_shapes:
        return list(new_shapes)

    existing_boxes = [shape_to_xyxy(s) for s in existing_shapes]
    kept_shapes: List[Any] = []

    for new_shape in new_shapes:
        new_box = shape_to_xyxy(new_shape)
        if new_box is None:
            kept_shapes.append(new_shape)
            continue

        new_label = get_shape_label(new_shape)
        is_dup = False

        for idx, ex_box in enumerate(existing_boxes):
            if ex_box is None:
                continue

            if same_label_only:
                ex_label = get_shape_label(existing_shapes[idx])
                if new_label != ex_label:
                    continue

            dup_flag, _, _ = is_duplicate_box_geometry(
                new_box,
                ex_box,
                iou_threshold=iou_threshold,
                containment_threshold=containment_threshold,
            )
            if dup_flag:
                is_dup = True
                break

        if not is_dup:
            kept_shapes.append(new_shape)

    return kept_shapes
