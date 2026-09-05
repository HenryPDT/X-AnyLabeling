"""Shared review-table building blocks (additive, no behavior change).

Provides a single implementation for thumbnail caching, label colors,
table setup, and CSV export used by the review gallery, diagnostics
dialog, and future stats / validation / active-learning dialogs.

Existing dialogs are intentionally left untouched in the foundation
commit; new dialogs should subclass or compose these helpers so a
future consolidation does not require a patchwork refactor.
"""

from __future__ import annotations

import csv
import os
import zlib
from typing import Dict, List, Optional, Tuple

from PIL import Image
from PyQt6 import QtCore
from PyQt6.QtGui import QColor, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import QDialog, QHeaderView, QTableWidget

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.style import get_dialog_style
from anylabeling.views.labeling.utils.theme import get_theme


MAX_THUMBNAIL_CACHE_ENTRIES = 1000


def get_label_color(label: str) -> QColor:
    """Deterministic pastel color for a label string."""
    label_id = zlib.crc32(label.encode("utf-8"))
    hue = label_id % 360
    saturation = 120 + ((label_id >> 9) % 60)
    value = 210 + ((label_id >> 16) % 35)
    return QColor.fromHsv(hue, saturation, value)


def crop_thumbnail_from_pil(
    pil_img: Image.Image, xyxy: List[float], max_size: int = 56
) -> Optional[QPixmap]:
    """Crop an object thumbnail from an opened PIL image."""
    if not xyxy or len(xyxy) < 4:
        return None
    width = float(xyxy[2]) - float(xyxy[0])
    height = float(xyxy[3]) - float(xyxy[1])
    if abs(width) < 1.0 and abs(height) < 1.0:
        cx = (float(xyxy[0]) + float(xyxy[2])) / 2.0
        cy = (float(xyxy[1]) + float(xyxy[3])) / 2.0
        half = 28.0
        xyxy = [cx - half, cy - half, cx + half, cy + half]
    elif width <= 0 or height <= 0:
        return None
    try:
        img_w, img_h = pil_img.size
        x1, y1, x2, y2 = xyxy
        pad_x = max(2.0, (x2 - x1) * 0.05)
        pad_y = max(2.0, (y2 - y1) * 0.05)
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(img_w, int(x2 + pad_x))
        cy2 = min(img_h, int(y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        crop = pil_img.crop((cx1, cy1, cx2, cy2))
        crop.thumbnail((max_size, max_size))
        if crop.mode != "RGBA":
            crop = crop.convert("RGBA")
        data = crop.tobytes("raw", "RGBA")
        qimg = QImage(
            data,
            crop.size[0],
            crop.size[1],
            crop.size[0] * 4,
            QImage.Format.Format_RGBA8888,
        ).copy()
        return QPixmap.fromImage(qimg)
    except Exception as exc:
        logger.warning(f"Failed to crop thumbnail: {exc}")
        return None


def extract_thumbnail_pixmap(
    image_path: str, xyxy: List[float], max_size: int = 56
) -> Optional[QPixmap]:
    """Load image from disk and extract an object thumbnail."""
    if not os.path.isfile(image_path):
        return None
    try:
        with Image.open(image_path) as pil_img:
            return crop_thumbnail_from_pil(pil_img, xyxy, max_size=max_size)
    except Exception as exc:
        logger.warning(f"Failed to extract thumbnail: {exc}")
        return None


class ThumbnailCache:
    """Bounded FIFO cache for QPixmap thumbnails.

    ``None`` pixmaps (load failures) are never cached so a transient
    miss can be retried; use ``key in cache`` to distinguish miss
    from cached value.
    """

    def __init__(self, max_entries: int = MAX_THUMBNAIL_CACHE_ENTRIES):
        self._store: Dict[Tuple[str, int], Optional[QPixmap]] = {}
        self._max_entries = int(max_entries)

    def get(self, key: Tuple[str, int]) -> Optional[QPixmap]:
        return self._store.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def put(self, key: Tuple[str, int], pix: Optional[QPixmap]) -> None:
        if pix is None:
            # Do not cache failures; allow retry on next request.
            self._store.pop(key, None)
            return
        if key not in self._store and len(self._store) >= self._max_entries:
            try:
                oldest = next(iter(self._store))
                self._store.pop(oldest, None)
            except StopIteration:
                pass
        self._store[key] = pix

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


def get_review_dialog_style() -> str:
    """Shared QSS for review-style dialogs."""
    theme = get_theme()
    return (
        get_dialog_style()
        + f"""
        .secondary-button {{
            background-color: {theme["surface"]};
            color: {theme["text"]};
            border: 1px solid {theme["border_light"]};
            border-radius: 8px;
            font-weight: 500;
            min-width: 90px;
            height: 32px;
            padding: 0 12px;
        }}
        .secondary-button:hover {{
            background-color: {theme["surface_hover"]};
        }}
        .primary-button {{
            background-color: {theme["primary"]};
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            min-width: 90px;
            height: 32px;
            padding: 0 12px;
        }}
        .primary-button:hover {{
            background-color: {theme["primary_hover"]};
        }}
        QLineEdit, QComboBox {{
            background-color: {theme["background_secondary"]};
            color: {theme["text"]};
            border: 1px solid {theme["border_light"]};
            border-radius: 6px;
            padding: 4px 8px;
            height: 28px;
        }}
    """
    )


def setup_review_table(
    table: QTableWidget,
    headers: List[str],
    stretch_last: bool = True,
) -> None:
    """Apply shared table defaults (selection, headers, sorting)."""
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    if stretch_last and len(headers) > 0:
        header.setStretchLastSection(True)
    table.setIconSize(QtCore.QSize(56, 56))


def set_thumbnail_cell(
    table: QTableWidget,
    row: int,
    col: int,
    pix: Optional[QPixmap],
    cache: Optional[ThumbnailCache] = None,
    cache_key: Optional[Tuple[str, int]] = None,
) -> None:
    """Set a thumbnail icon cell, consulting cache when provided.

    Write-through: a non-null ``pix`` with ``cache_key`` is stored in
    ``cache`` so later lookups hit.
    """
    from PyQt6.QtWidgets import QTableWidgetItem

    item = table.item(row, col)
    if item is None:
        item = QTableWidgetItem("")
        table.setItem(row, col, item)
    resolved = pix
    if resolved is None and cache is not None and cache_key is not None:
        resolved = cache.get(cache_key)
    elif resolved is not None and cache is not None and cache_key is not None:
        cache.put(cache_key, resolved)
    if resolved is not None and not resolved.isNull():
        item.setIcon(QIcon(resolved))


def export_rows_to_csv(
    file_path: str, headers: List[str], rows: List[List[object]]
) -> bool:
    """Write review rows to CSV. Returns True on success."""
    try:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(["" if v is None else str(v) for v in row])
        return True
    except Exception as exc:
        logger.warning(f"Failed to export CSV {file_path}: {exc}")
        return False


class ReviewTableDialog(QDialog):
    """Optional base dialog with shared style + thumbnail cache."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thumbnail_cache = ThumbnailCache()
        self._thumb_timer = QtCore.QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.setInterval(60)
        self._thumb_timer.timeout.connect(self._on_thumb_timeout)

    def _on_thumb_timeout(self) -> None:
        """Subclasses override to lazily load visible thumbnails."""

    def schedule_thumbnail_load(self) -> None:
        if not self._thumb_timer.isActive():
            self._thumb_timer.start()

    def put_thumbnail(
        self, key: Tuple[str, int], pix: Optional[QPixmap]
    ) -> None:
        self.thumbnail_cache.put(key, pix)
