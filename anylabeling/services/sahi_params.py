"""SAHI tiled-inference parameters (Qt-free, additive).

Single source of truth for slice/overlap defaults, clamping, and
SAHI model-type detection so backends, dialogs, and configs agree.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from anylabeling.services.model_type import normalize_model_type


SAHI_DEFAULT_SLICE_HEIGHT = 512
SAHI_DEFAULT_SLICE_WIDTH = 512
SAHI_DEFAULT_OVERLAP_RATIO = 0.2

SAHI_MIN_SLICE = 256
SAHI_MAX_SLICE = 2048
SAHI_MIN_OVERLAP = 0.0
SAHI_MAX_OVERLAP = 0.5

SAHI_MODEL_TYPES = frozenset(
    {
        "yolov5_sahi",
        "yolov8_sahi",
        "yolo11_sahi",
        "yolo26_sahi",
    }
)

_SAHI_NORMALIZED_TYPES = frozenset(
    "".join(ch for ch in t.lower() if ch.isalnum()) for t in SAHI_MODEL_TYPES
)


def is_sahi_model_type(model_type: Any) -> bool:
    """Return True for SAHI tiled-inference model types."""
    # Normalized set drops underscores: yolov5sahi, ...
    return normalize_model_type(model_type) in _SAHI_NORMALIZED_TYPES


def _to_slice(value: Any, default: int) -> int:
    """Coerce to int slice size with fallback for bad/non-finite input."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    try:
        ivalue = int(number)
    except (OverflowError, ValueError):
        return default
    return max(SAHI_MIN_SLICE, min(SAHI_MAX_SLICE, ivalue))


def _to_overlap(value: Any, default: float) -> float:
    """Coerce to overlap ratio with fallback for bad/non-finite input."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(SAHI_MIN_OVERLAP, min(SAHI_MAX_OVERLAP, number))


def clamp_sahi_params(
    slice_height: Any,
    slice_width: Any,
    overlap_ratio: Any,
) -> Tuple[int, int, float]:
    """Clamp raw SAHI params to valid ranges with safe fallbacks."""
    slice_h = _to_slice(slice_height, SAHI_DEFAULT_SLICE_HEIGHT)
    slice_w = _to_slice(slice_width, SAHI_DEFAULT_SLICE_WIDTH)
    overlap = _to_overlap(overlap_ratio, SAHI_DEFAULT_OVERLAP_RATIO)
    return slice_h, slice_w, overlap


def resolve_sahi_params(
    config: Dict[str, Any] | None,
    overrides: Dict[str, Any] | None = None,
) -> Tuple[int, int, float]:
    """Resolve effective SAHI params from config + optional overrides.

    Accepts both model keys (``slice_height`` ...) and app-config keys
    (``sahi_slice_height`` ...); overrides take precedence.
    """
    cfg = config or {}
    ovr = overrides or {}

    def _pick(*keys: str, default):
        for source in (ovr, cfg):
            for key in keys:
                if key in source:
                    return source[key]
        return default

    slice_h = _pick(
        "slice_height",
        "sahi_slice_height",
        default=SAHI_DEFAULT_SLICE_HEIGHT,
    )
    slice_w = _pick(
        "slice_width",
        "sahi_slice_width",
        default=SAHI_DEFAULT_SLICE_WIDTH,
    )
    overlap = _pick(
        "overlap_ratio",
        "overlap_height_ratio",
        "sahi_overlap_ratio",
        default=SAHI_DEFAULT_OVERLAP_RATIO,
    )
    return clamp_sahi_params(slice_h, slice_w, overlap)


def apply_sahi_params_to(
    obj: Any,
    slice_height: Any,
    slice_width: Any,
    overlap_ratio: Any,
) -> Tuple[int, int, float]:
    """Clamp and assign SAHI slice/overlap attrs on a model object.

    Shared by all ``*_sahi`` backends so runtime tuning stays identical.
    Returns the applied ``(h, w, overlap)``.
    """
    slice_h, slice_w, overlap = clamp_sahi_params(
        slice_height, slice_width, overlap_ratio
    )
    obj.slice_height = slice_h
    obj.slice_width = slice_w
    obj.overlap_height_ratio = overlap
    obj.overlap_width_ratio = overlap
    return slice_h, slice_w, overlap
