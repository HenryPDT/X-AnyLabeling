"""Pure dataset statistics computation (Qt-free).

Computes per-class size/count aggregates used by the stats dashboard.
All functions are deterministic and have no I/O except optional caller-
provided shape lists, so they are trivially unit-testable.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from anylabeling.views.labeling.utils.shape_geometry import shape_to_xyxy


@dataclass
class ClassSizeStats:
    """Aggregated size stats for a single class label."""

    label: str = ""
    count: int = 0
    images: int = 0
    min_width: float = 0.0
    max_width: float = 0.0
    mean_width: float = 0.0
    stddev_width: float = 0.0
    min_height: float = 0.0
    max_height: float = 0.0
    mean_height: float = 0.0
    stddev_height: float = 0.0
    min_area: float = 0.0
    max_area: float = 0.0
    mean_area: float = 0.0
    stddev_area: float = 0.0
    marks_per_image: float = 0.0
    tiny_count: int = 0


@dataclass
class DatasetStats:
    """Dataset-wide aggregates."""

    total_images: int = 0
    total_annotations: int = 0
    empty_images: int = 0
    verified_empty_images: int = 0
    tiny_boxes_count: int = 0
    per_class: Dict[str, ClassSizeStats] = field(default_factory=dict)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _stddev(values: List[float], mean_val: Optional[float] = None) -> float:
    if len(values) < 2:
        return 0.0
    mean = mean_val if mean_val is not None else _mean(values)
    var = sum((v - mean) ** 2 for v in values) / float(len(values))
    return math.sqrt(max(0.0, var))


def compute_class_size_stats(
    label: str,
    widths: List[float],
    heights: List[float],
    areas: List[float],
    image_count: int,
    tiny_count: int = 0,
) -> ClassSizeStats:
    """Build ClassSizeStats from parallel width/height/area samples."""
    stats = ClassSizeStats(label=label, count=len(widths))
    stats.images = int(image_count)
    stats.tiny_count = int(tiny_count)
    if widths:
        stats.min_width = float(min(widths))
        stats.max_width = float(max(widths))
        stats.mean_width = _mean(widths)
        stats.stddev_width = _stddev(widths, stats.mean_width)
    if heights:
        stats.min_height = float(min(heights))
        stats.max_height = float(max(heights))
        stats.mean_height = _mean(heights)
        stats.stddev_height = _stddev(heights, stats.mean_height)
    if areas:
        stats.min_area = float(min(areas))
        stats.max_area = float(max(areas))
        stats.mean_area = _mean(areas)
        stats.stddev_area = _stddev(areas, stats.mean_area)
    if image_count > 0:
        stats.marks_per_image = float(len(widths)) / float(image_count)
    return stats


def box_size(box: List[float]) -> Optional[Tuple[float, float, float]]:
    """Return true (width, height, area) or None if degenerate.

    Unlike the review gallery (which floors to 1px for thumbnails),
    stats use true geometry so zero-area shapes never distort means.
    """
    if box is None or len(box) < 4:
        return None
    width = float(box[2] - box[0])
    height = float(box[3] - box[1])
    if width <= 0 or height <= 0:
        return None
    return (width, height, width * height)


def collect_shape_sizes(
    shapes: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[float]]]:
    """Group width/height/area samples per class label.

    Returns mapping label -> {"widths": [...], "heights": [...],
    "areas": [...], "images": set-count handled by caller}.
    Invalid and degenerate shapes are skipped (same rule as review gallery).
    """
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "unknown")).strip() or "unknown"
        box = shape_to_xyxy(shape)
        sized = box_size(box) if box is not None else None
        if sized is None:
            continue
        width, height, area = sized
        entry = grouped.setdefault(
            label, {"widths": [], "heights": [], "areas": []}
        )
        entry["widths"].append(width)
        entry["heights"].append(height)
        entry["areas"].append(area)
    return grouped


def compute_dataset_stats_from_metas(
    metas: List[Any],
    # Kept for API compat; per-box geometry is unavailable from metas
    # alone (see compute_full_dataset_stats).
    tiny_box_px: float = 32.0,
) -> DatasetStats:
    """Compute stats from DatasetMeta-like objects (duck-typed).

    Expects attributes: shape_count, verified_empty. Used for fast
    file-list stats without re-reading full shape geometry.

    Missing label files count as empty. Per-class geometry and
    tiny-box counts require full shape reads; use
    :func:`compute_full_dataset_stats` when geometry available.
    """
    stats = DatasetStats(total_images=len(metas))
    for meta in metas:
        shape_count = int(getattr(meta, "shape_count", 0) or 0)
        verified = bool(getattr(meta, "verified_empty", False))
        stats.total_annotations += shape_count
        if shape_count == 0:
            if verified:
                stats.verified_empty_images += 1
            else:
                stats.empty_images += 1
    # Per-class marks-per-image needs image counts; caller fills
    # per_class via compute_class_size_stats when geometry available.
    return stats


def _load_shapes(
    label_file: str,
) -> Tuple[Optional[List[Any]], bool]:
    """Load shape list from a label file.

    Returns ``(shapes, verified_empty)`` where ``shapes`` is None for
    missing/corrupt files, ``[]`` for files with no shapes, and the
    shape list otherwise.
    """
    if not os.path.isfile(label_file):
        return None, False
    try:
        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, False
    if not isinstance(data, dict):
        return None, False
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        return None, False
    if len(shapes) == 0:
        return [], data.get("verified_empty") is True
    return shapes, False


def compute_full_dataset_stats(
    image_paths: List[str],
    output_dir: Optional[str] = None,
    tiny_box_px: float = 32.0,
    progress_callback=None,
    cancel_callback=None,
) -> DatasetStats:
    """Compute full geometry stats by reading annotation JSON files.

    Qt-free; corrupt/missing files are counted as empty, never raise.
    """
    from anylabeling.services.dataset_meta import get_label_file_path

    stats = DatasetStats(total_images=len(image_paths))
    widths_by_label: Dict[str, List[float]] = {}
    heights_by_label: Dict[str, List[float]] = {}
    areas_by_label: Dict[str, List[float]] = {}
    images_by_label: Dict[str, set] = {}
    tiny_acc: Dict[str, int] = {}
    total = len(image_paths)
    for idx, image_path in enumerate(image_paths):
        if cancel_callback is not None:
            try:
                if cancel_callback():
                    break
            except Exception:
                pass
        if progress_callback is not None:
            try:
                progress_callback(idx + 1, total, os.path.basename(image_path))
            except Exception:
                pass
        label_file = get_label_file_path(image_path, output_dir=output_dir)
        shapes, verified = _load_shapes(label_file)
        if shapes is None:
            stats.empty_images += 1
            continue
        if len(shapes) == 0:
            if verified:
                stats.verified_empty_images += 1
            else:
                stats.empty_images += 1
            continue
        seen_labels: set = set()
        tiny_by_label: Dict[str, int] = {}
        valid_in_image = 0
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            label = str(shape.get("label", "unknown")).strip() or "unknown"
            box = shape_to_xyxy(shape)
            sized = box_size(box) if box is not None else None
            if sized is None:
                continue
            width, height, area = sized
            valid_in_image += 1
            stats.total_annotations += 1
            widths_by_label.setdefault(label, []).append(width)
            heights_by_label.setdefault(label, []).append(height)
            areas_by_label.setdefault(label, []).append(area)
            seen_labels.add(label)
            if max(width, height) < tiny_box_px:
                stats.tiny_boxes_count += 1
                tiny_by_label[label] = tiny_by_label.get(label, 0) + 1
        if valid_in_image == 0:
            # Annotation file held no usable geometry: treat as empty
            # so the image stays visible in health stats.
            stats.empty_images += 1
            continue
        for label in seen_labels:
            images_by_label.setdefault(label, set()).add(image_path)
        for label, tiny in tiny_by_label.items():
            tiny_acc[label] = tiny_acc.get(label, 0) + tiny
    for label, widths in widths_by_label.items():
        heights = heights_by_label.get(label, [])
        areas = areas_by_label.get(label, [])
        image_count = len(images_by_label.get(label, set()))
        stats.per_class[label] = compute_class_size_stats(
            label,
            widths,
            heights,
            areas,
            image_count,
            tiny_count=tiny_acc.get(label, 0),
        )
    return stats
