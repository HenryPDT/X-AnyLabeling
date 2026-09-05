"""Contour and edge snapping utilities for manual annotation drawing.

Snaps user-drawn rough bounding boxes and oriented bounding boxes (OBBs)
to prominent high-contrast contours using OpenCV edge and contour analysis.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from anylabeling.views.labeling.utils.shape_geometry import box_iou


def _extract_roi(
    image_np: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    padding: int = 15,
) -> Tuple[np.ndarray, int, int]:
    """Crop region of interest around bounding box with padding.

    Returns (roi_image, roi_x1, roi_y1).
    """
    img_h, img_w = image_np.shape[:2]
    roi_x1 = max(0, int(round(min(x1, x2) - padding)))
    roi_y1 = max(0, int(round(min(y1, y2) - padding)))
    roi_x2 = min(img_w, int(round(max(x1, x2) + padding)))
    roi_y2 = min(img_h, int(round(max(y1, y2) + padding)))

    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return np.empty((0, 0, 3), dtype=image_np.dtype), roi_x1, roi_y1

    roi = image_np[roi_y1:roi_y2, roi_x1:roi_x2]
    return roi, roi_x1, roi_y1


def _to_grayscale(roi: np.ndarray) -> np.ndarray:
    """Convert ROI image to single-channel grayscale uint8 array."""
    if roi.ndim == 2:
        return roi
    if roi.shape[2] == 4:
        return cv2.cvtColor(roi, cv2.COLOR_RGBA2GRAY)
    if roi.shape[2] == 3:
        return cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def _find_candidate_contours(roi: np.ndarray) -> List[np.ndarray]:
    """Extract candidate external contours from ROI using multiple complementary detectors."""
    if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
        return []

    gray = _to_grayscale(roi)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # If ROI is flat without contrast, no meaningful contours exist
    if int(blurred.max()) - int(blurred.min()) < 10:
        return []

    all_contours: List[np.ndarray] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 1. Canny edge detector with median-based thresholds + closing
    median_val = float(np.median(blurred))
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(blurred, lower, upper)
    edges_closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    contours_edges, _ = cv2.findContours(
        edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    all_contours.extend(contours_edges)

    # 2. Otsu thresholding (foreground on dark or light background)
    _, binary_otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours_otsu, _ = cv2.findContours(
        binary_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    all_contours.extend(contours_otsu)

    # 3. Inverted Otsu for light backgrounds
    binary_inv = cv2.bitwise_not(binary_otsu)
    contours_inv, _ = cv2.findContours(
        binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    all_contours.extend(contours_inv)

    return all_contours


def _score_contour(
    contour: np.ndarray,
    target_roi_box: Tuple[float, float, float, float],
    min_contour_area: float,
    roi_shape: Optional[Tuple[int, int]] = None,
) -> float:
    """Score a candidate contour against target box in ROI coordinate frame."""
    area = cv2.contourArea(contour)
    if area < min_contour_area:
        return 0.0

    bx, by, bw, bh = cv2.boundingRect(contour)

    # Reject contours that simply wrap the entire cropped ROI frame
    if roi_shape is not None:
        roi_h, roi_w = roi_shape
        if bw >= roi_w - 2 and bh >= roi_h - 2:
            return 0.0

    cand_box = (float(bx), float(by), float(bx + bw), float(by + bh))
    cand_area = float(bw * bh)
    target_w = target_roi_box[2] - target_roi_box[0]
    target_h = target_roi_box[3] - target_roi_box[1]
    target_area = max(1.0, target_w * target_h)

    # Guard against extreme size explosions or contractions
    area_ratio = cand_area / target_area
    if area_ratio > 3.0 or area_ratio < 0.15:
        return 0.0

    iou = box_iou(cand_box, target_roi_box)
    return float(iou)


def snap_bbox_to_contour(
    image_np: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    padding: int = 15,
    min_contour_area: float = 25.0,
    min_iou: float = 0.35,
) -> Tuple[float, float, float, float]:
    """Snap bounding box coordinates [x1, y1, x2, y2] to the closest prominent contour.

    If a high-confidence contour is detected, returns snapped [sx1, sy1, sx2, sy2].
    Otherwise, safely returns original [x1, y1, x2, y2].
    """
    orig_x1, orig_x2 = min(x1, x2), max(x1, x2)
    orig_y1, orig_y2 = min(y1, y2), max(y1, y2)

    # Reject degenerate boxes
    if orig_x2 - orig_x1 < 2 or orig_y2 - orig_y1 < 2:
        return orig_x1, orig_y1, orig_x2, orig_y2

    roi, roi_x1, roi_y1 = _extract_roi(
        image_np, orig_x1, orig_y1, orig_x2, orig_y2, padding=padding
    )
    if roi.size == 0:
        return orig_x1, orig_y1, orig_x2, orig_y2

    target_roi_box = (
        orig_x1 - roi_x1,
        orig_y1 - roi_y1,
        orig_x2 - roi_x1,
        orig_y2 - roi_y1,
    )

    contours = _find_candidate_contours(roi)
    best_cnt = None
    best_score = -1.0

    for cnt in contours:
        score = _score_contour(
            cnt, target_roi_box, min_contour_area, roi_shape=roi.shape[:2]
        )
        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None or best_score < min_iou:
        return orig_x1, orig_y1, orig_x2, orig_y2

    bx, by, bw, bh = cv2.boundingRect(best_cnt)
    img_h, img_w = image_np.shape[:2]
    snapped_x1 = max(0.0, min(float(img_w), float(bx + roi_x1)))
    snapped_y1 = max(0.0, min(float(img_h), float(by + roi_y1)))
    snapped_x2 = max(0.0, min(float(img_w), float(bx + bw + roi_x1)))
    snapped_y2 = max(0.0, min(float(img_h), float(by + bh + roi_y1)))

    return snapped_x1, snapped_y1, snapped_x2, snapped_y2


def snap_obb_to_contour(
    image_np: np.ndarray,
    points: Sequence[Any],
    padding: int = 15,
    min_contour_area: float = 25.0,
    min_iou: float = 0.35,
) -> List[Tuple[float, float]]:
    """Snap 4-point oriented bounding box (OBB) to prominent contour via minAreaRect.

    Accepts sequence of (x, y) tuples or objects with .x() and .y() methods.
    Returns list of 4 (x, y) tuples.
    """
    raw_pts: List[Tuple[float, float]] = []
    for p in points:
        if hasattr(p, "x") and hasattr(p, "y"):
            raw_pts.append((float(p.x()), float(p.y())))
        else:
            raw_pts.append((float(p[0]), float(p[1])))

    if len(raw_pts) < 4:
        return raw_pts

    xs = [pt[0] for pt in raw_pts]
    ys = [pt[1] for pt in raw_pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    if x2 - x1 < 2 or y2 - y1 < 2:
        return raw_pts

    roi, roi_x1, roi_y1 = _extract_roi(
        image_np, x1, y1, x2, y2, padding=padding
    )
    if roi.size == 0:
        return raw_pts

    target_roi_box = (x1 - roi_x1, y1 - roi_y1, x2 - roi_x1, y2 - roi_y1)
    contours = _find_candidate_contours(roi)

    best_cnt = None
    best_score = -1.0

    for cnt in contours:
        score = _score_contour(
            cnt, target_roi_box, min_contour_area, roi_shape=roi.shape[:2]
        )
        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None or best_score < min_iou:
        return raw_pts

    rect = cv2.minAreaRect(best_cnt)
    box = cv2.boxPoints(rect)  # 4x2 float array

    img_h, img_w = image_np.shape[:2]
    snapped: List[Tuple[float, float]] = []
    for pt in box:
        sx = max(0.0, min(float(img_w), float(pt[0] + roi_x1)))
        sy = max(0.0, min(float(img_h), float(pt[1] + roi_y1)))
        snapped.append((sx, sy))

    return snapped
