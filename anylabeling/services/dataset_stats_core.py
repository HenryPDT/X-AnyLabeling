"""Pure dataset statistics computation (Qt-free).

Computes per-class size/count aggregates used by the stats dashboard.
All functions are deterministic and have no I/O except optional caller-
provided shape lists, so they are trivially unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
) -> ClassSizeStats:
    """Build ClassSizeStats from parallel width/height/area samples."""
    stats = ClassSizeStats(label=label, count=len(widths))
    stats.images = int(image_count)
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


def collect_shape_sizes(
    shapes: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[float]]]:
    """Group width/height/area samples per class label.

    Returns mapping label -> {"widths": [...], "heights": [...],
    "areas": [...], "images": set-count handled by caller}.
    Invalid shapes are skipped (same rule as review gallery).
    """
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "unknown")).strip() or "unknown"
        box = shape_to_xyxy(shape)
        if box is None:
            continue
        width = max(1.0, float(box[2] - box[0]))
        height = max(1.0, float(box[3] - box[1]))
        area = width * height
        entry = grouped.setdefault(
            label, {"widths": [], "heights": [], "areas": []}
        )
        entry["widths"].append(width)
        entry["heights"].append(height)
        entry["areas"].append(area)
    return grouped


def compute_dataset_stats_from_metas(
    metas: List[Any],
    tiny_box_px: float = 32.0,  # noqa: ARG001 - kept for API compat; per-box
    # geometry is unavailable from metas alone (see compute_full_dataset_stats)
) -> DatasetStats:
    """Compute stats from DatasetMeta-like objects (duck-typed).

    Expects attributes: shape_count, verified_empty, has_label_file,
    corrupt, class_counts. Used for fast file-list stats without
    re-reading full shape geometry.

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
