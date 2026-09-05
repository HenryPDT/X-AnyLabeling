"""Smart file-list sorting (Qt-free, additive).

Sorts file entries using DatasetMeta without changing default
natsort discovery order unless the user explicitly selects a mode.
All functions are pure and deterministic for unit testing.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Tuple

from natsort import natsort_key


class FileSortMode(str, Enum):
    """Available file-list orderings."""

    DEFAULT = "default"
    UNANNOTATED_FIRST = "unannotated_first"
    UNCHECKED_FIRST = "unchecked_first"
    MOST_MARKS_FIRST = "most_marks_first"
    FEWEST_MARKS_FIRST = "fewest_marks_first"
    VERIFIED_EMPTY_LAST = "verified_empty_last"

    @classmethod
    def values(cls) -> List[str]:
        return [m.value for m in cls]

    @classmethod
    def coerce(cls, value: Any) -> "FileSortMode":
        try:
            if isinstance(value, cls):
                return value
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.DEFAULT


def _is_unannotated(meta: Any) -> bool:
    if meta is None:
        return True
    if getattr(meta, "corrupt", False):
        return False
    if getattr(meta, "shape_count", 0) > 0:
        return False
    if getattr(meta, "verified_empty", False):
        return False
    return True


def sort_file_entries(
    entries: List[Tuple[str, Any]],
    mode: Any = FileSortMode.DEFAULT,
) -> List[Tuple[str, Any]]:
    """Sort (filename, meta) entries stably.

    DEFAULT preserves input order (existing natsort behavior).
    All other modes are stable sorts with natural filename tiebreak
    (same ``natsort`` order as discovery) so results are deterministic
    across runs.
    """
    sort_mode = FileSortMode.coerce(mode)
    if sort_mode == FileSortMode.DEFAULT:
        return list(entries)

    def _key(entry: Tuple[str, Any]):
        filename, meta = entry
        shape_count = int(getattr(meta, "shape_count", 0) or 0)
        checked = bool(getattr(meta, "checked", False))
        verified = bool(getattr(meta, "verified_empty", False))
        unannotated = _is_unannotated(meta)
        natural = natsort_key(filename)

        if sort_mode == FileSortMode.UNANNOTATED_FIRST:
            return (0 if unannotated else 1, natural)
        if sort_mode == FileSortMode.UNCHECKED_FIRST:
            # Unannotated first, then unchecked, then checked;
            # verified-empty backgrounds sort with checked (done).
            if unannotated:
                rank = 0
            elif not checked:
                rank = 1
            else:
                rank = 2
            return (rank, natural)
        if sort_mode == FileSortMode.MOST_MARKS_FIRST:
            return (-shape_count, natural)
        if sort_mode == FileSortMode.FEWEST_MARKS_FIRST:
            return (shape_count, natural)
        if sort_mode == FileSortMode.VERIFIED_EMPTY_LAST:
            return (1 if verified else 0, natural)
        return (natural,)

    return sorted(entries, key=_key)
