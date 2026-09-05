"""Dataset-level metadata helpers for prioritization and batch workflows.

This module provides a single Qt-free source of truth for per-image
annotation state (checked / verified-empty / shape counts). It mirrors
the fast regex logic used by the labeling widget without importing Qt,
so file-sort, batch-scope, stats, split, and active-learning features
share one implementation instead of duplicating JSON parsing.

No behavior change to existing dialogs: this module is additive only.

NOTE: Despite the historical "cache" wording, this module is
stateless (no shared dict/lru). ``DatasetMeta.mtime`` is informational
only (image mtime at collection time) to aid future caching keyed by
``(abspath, mtime_ns, size)``. Callers needing batch performance should
use :func:`get_dataset_meta_list` once instead of per-file calls.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


CHECKED_FIELD_PATTERN = re.compile(r'"checked"\s*:\s*(true|false)')
VERIFIED_EMPTY_FIELD_PATTERN = re.compile(
    r'"verified_empty"\s*:\s*(true|false)'
)


@dataclass
class DatasetMeta:
    """Lightweight per-image annotation state."""

    image_path: str
    label_file: Optional[str] = None
    exists: bool = False
    has_label_file: bool = False
    corrupt: bool = False
    checked: bool = False
    verified_empty: bool = False
    shape_count: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    mtime: float = 0.0


def get_label_file_path(
    image_path: str, output_dir: Optional[str] = None
) -> str:
    """Resolve expected JSON label path for an image path."""
    json_name = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    if output_dir:
        return os.path.join(output_dir, json_name)
    return os.path.splitext(image_path)[0] + ".json"


def is_label_file_checked_fast(label_file: str) -> bool:
    """Return True if label JSON has `"checked": true` (streaming scan)."""
    if not os.path.isfile(label_file):
        return False
    try:
        buffer = ""
        with open(label_file, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                buffer = buffer[-32:] + chunk
                match = CHECKED_FIELD_PATTERN.search(buffer)
                if match:
                    return match.group(1) == "true"
    except Exception:
        return False
    return False


def is_label_file_verified_empty_fast(label_file: str) -> bool:
    """Return True if label JSON has `"verified_empty": true`.

    Uses head+tail scan to avoid reading multi-MB base64 imageData.
    Mirrors LabelingWidget._label_file_verified_empty logic.

    Binary-safe: seeks by byte offset and decodes with
    ``errors="ignore"`` so a mid-character split cannot raise, and keeps
    a 32B overlap so a token split across the head/tail gap is found.
    """
    if not os.path.isfile(label_file):
        return False
    try:
        size = os.path.getsize(label_file)
        with open(label_file, "rb") as f:
            if size <= 256 * 1024:
                text = f.read().decode("utf-8", errors="ignore")
                match = VERIFIED_EMPTY_FIELD_PATTERN.search(text)
                return bool(match) and match.group(1) == "true"
            head = f.read(32768).decode("utf-8", errors="ignore")
            match = VERIFIED_EMPTY_FIELD_PATTERN.search(head)
            if match:
                return match.group(1) == "true"
            try:
                f.seek(max(0, size - 131072))
                tail = f.read().decode("utf-8", errors="ignore")
            except Exception:
                tail = ""
            # Prepend overlap: token may straddle head/tail gap.
            tail = head[-32:] + tail
            match = VERIFIED_EMPTY_FIELD_PATTERN.search(tail)
            return bool(match) and match.group(1) == "true"
    except Exception:
        return False
    return False


def _load_label_json(label_file: str):
    """Load label JSON; return (data, shapes) or (None, None) if corrupt."""
    try:
        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None
    if not isinstance(data, dict):
        return None, None
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        return None, None
    return data, shapes


def _count_labels(shapes: list) -> Dict[str, int]:
    """Count non-blank labels in a shape list."""
    counts: Dict[str, int] = {}
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = shape.get("label", "")
        label_str = str(label).strip() if label else ""
        if not label_str:
            continue
        counts[label_str] = counts.get(label_str, 0) + 1
    return counts


def get_dataset_meta(
    image_path: str, output_dir: Optional[str] = None
) -> DatasetMeta:
    """Collect metadata for a single image without requiring Qt.

    Small files (<=256k) are parsed once and flags derived from JSON
    directly, skipping the regex pre-scans. Large files keep the
    head+tail fast scans first (to avoid reading multi-MB imageData
    twice), with parsed JSON taking precedence when available.
    """
    label_file = get_label_file_path(image_path, output_dir=output_dir)
    meta = DatasetMeta(image_path=image_path, label_file=label_file)
    try:
        meta.mtime = os.path.getmtime(image_path)
    except Exception:
        meta.mtime = 0.0
    meta.exists = os.path.isfile(image_path)
    if not meta.exists:
        return meta
    if not os.path.isfile(label_file):
        return meta
    meta.has_label_file = True
    try:
        label_size = os.path.getsize(label_file)
    except Exception:
        label_size = 0
    if label_size <= 256 * 1024:
        # Single parse path: no redundant regex scans.
        data, shapes = _load_label_json(label_file)
        if data is None:
            meta.corrupt = True
            return meta
        meta.checked = data.get("checked") is True
        if not shapes:
            meta.verified_empty = data.get("verified_empty") is True
            return meta
        meta.verified_empty = False
        meta.shape_count = len(shapes)
        meta.class_counts = _count_labels(shapes)
        return meta
    meta.checked = is_label_file_checked_fast(label_file)
    meta.verified_empty = is_label_file_verified_empty_fast(label_file)
    data, shapes = _load_label_json(label_file)
    if data is None:
        meta.corrupt = True
        return meta
    # verified_empty only counts when there are truly zero shapes
    if not shapes:
        if data.get("verified_empty") is True:
            meta.verified_empty = True
        return meta
    # Non-empty file cannot be a verified background
    if meta.verified_empty and len(shapes) > 0:
        meta.verified_empty = False
    meta.shape_count = len(shapes)
    meta.class_counts = _count_labels(shapes)
    # Trust parsed JSON over regex for checked flag when available
    try:
        if "checked" in data:
            meta.checked = data.get("checked") is True
    except Exception:
        pass
    return meta


def get_dataset_meta_list(
    image_paths: List[str],
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> List[DatasetMeta]:
    """Collect metadata for a list of images with optional progress."""
    metas: List[DatasetMeta] = []
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
        metas.append(get_dataset_meta(image_path, output_dir=output_dir))
    return metas
