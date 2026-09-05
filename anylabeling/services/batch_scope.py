"""Batch scope filtering (Qt-free, additive).

Determines whether an image should be processed during batch
auto-labeling based on existing annotation state. Uses DatasetMeta
so file-sort, stats, and batch share one definition of
"annotated / checked / empty".
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class BatchScope(str, Enum):
    """Which images to include in batch auto-labeling."""

    ALL = "all"
    UNANNOTATED_ONLY = "unannotated_only"
    UNCHECKED_ONLY = "unchecked_only"

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]

    @classmethod
    def coerce(cls, value: Any) -> "BatchScope":
        try:
            if isinstance(value, cls):
                return value
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.ALL


def should_process_image(meta: Any, scope: Any) -> bool:
    """Return True if image with given meta should be batch-processed."""
    batch_scope = BatchScope.coerce(scope)
    if batch_scope == BatchScope.ALL:
        return True
    if meta is None:
        return True
    if getattr(meta, "corrupt", False):
        return False
    if not getattr(meta, "exists", True):
        return False
    shape_count = int(getattr(meta, "shape_count", 0) or 0)
    checked = bool(getattr(meta, "checked", False))
    verified = bool(getattr(meta, "verified_empty", False))

    if batch_scope == BatchScope.UNANNOTATED_ONLY:
        # Skip anything with shapes or verified-background flag.
        if shape_count > 0 or verified:
            return False
        return True
    if batch_scope == BatchScope.UNCHECKED_ONLY:
        # Skip checked images; unannotated and unchecked both process.
        if checked:
            return False
        return True
    return True
