from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import zlib

from PIL import Image
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from anylabeling.services.annotation_diagnostics import get_label_file_path
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.shape_geometry import (
    box_iou,
    shape_to_xyxy,
)
from anylabeling.views.labeling.utils.style import (
    get_dialog_style,
    get_progress_dialog_style,
)
from anylabeling.views.labeling.utils.theme import get_theme


@dataclass
class ReviewItem:
    """Represents a single annotation instance for visual review."""

    image_path: str
    label_file: str
    shape_index: int
    label: str
    shape_type: str
    points: List[Any]
    xyxy: List[float]
    width: float
    height: float
    area: float
    aspect_ratio: float
    overlap_iou: float

    @property
    def extreme_aspect_ratio_score(self) -> float:
        """Score how extreme the aspect ratio is (both thin-tall and flat-wide)."""
        ar = max(0.001, self.aspect_ratio)
        return max(ar, 1.0 / ar)


def crop_thumbnail_from_pil(
    pil_img: Image.Image, xyxy: List[float], max_size: int = 56
) -> Optional[QPixmap]:
    """Extract and scale an object crop from an opened PIL image as a QPixmap."""
    if not xyxy or len(xyxy) < 4:
        return None
    w0 = float(xyxy[2]) - float(xyxy[0])
    h0 = float(xyxy[3]) - float(xyxy[1])
    # Point fallback: near-zero area -> 56px window around point
    if abs(w0) < 1.0 and abs(h0) < 1.0:
        cx = (float(xyxy[0]) + float(xyxy[2])) / 2.0
        cy = (float(xyxy[1]) + float(xyxy[3])) / 2.0
        half = 28.0
        xyxy = [cx - half, cy - half, cx + half, cy + half]
    elif w0 <= 0 or h0 <= 0:
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
        logger.warning(f"Failed to crop thumbnail from PIL image: {exc}")
        return None


def extract_thumbnail_pixmap(
    image_path: str, xyxy: List[float], max_size: int = 56
) -> Optional[QPixmap]:
    """Extract and scale an object crop from an image file as a QPixmap."""
    if not os.path.isfile(image_path):
        return None
    try:
        with Image.open(image_path) as pil_img:
            return crop_thumbnail_from_pil(pil_img, xyxy, max_size=max_size)
    except Exception as exc:
        logger.warning(f"Failed to extract thumbnail for {image_path}: {exc}")
        return None


def _get_label_color(label: str) -> QColor:
    """Generate deterministic pastel color for a label."""
    label_id = zlib.crc32(label.encode("utf-8"))
    hue = label_id % 360
    saturation = 120 + ((label_id >> 9) % 60)
    value = 210 + ((label_id >> 16) % 35)
    return QColor.fromHsv(hue, saturation, value)


def collect_review_items_for_image(
    image_path: str,
    output_dir: Optional[str] = None,
) -> List[ReviewItem]:
    """Parse annotation shapes and compute overlap and bounding metrics for one image."""
    label_file = get_label_file_path(image_path, output_dir=output_dir)
    if not os.path.isfile(label_file):
        return []

    try:
        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(f"Error reading {label_file}: {exc}")
        return []

    shapes = data.get("shapes", [])
    if not shapes:
        return []

    boxes: List[Optional[List[float]]] = []
    for s in shapes:
        boxes.append(shape_to_xyxy(s))

    items: List[ReviewItem] = []
    for idx, s in enumerate(shapes):
        box = boxes[idx]
        if box is None:
            continue

        w = max(1.0, float(box[2] - box[0]))
        h = max(1.0, float(box[3] - box[1]))
        ar = w / h
        area = w * h

        # Calculate max IoU with any other shape on the same image
        max_iou = 0.0
        for other_idx, other_box in enumerate(boxes):
            if other_idx == idx or other_box is None:
                continue
            curr_iou = box_iou(box, other_box)
            if curr_iou > max_iou:
                max_iou = curr_iou

        lbl = str(s.get("label", "unknown")).strip()
        stype = str(s.get("shape_type", "polygon")).strip()
        pts = s.get("points", [])

        items.append(
            ReviewItem(
                image_path=image_path,
                label_file=label_file,
                shape_index=idx,
                label=lbl,
                shape_type=stype,
                points=pts,
                xyxy=box,
                width=w,
                height=h,
                area=area,
                aspect_ratio=ar,
                overlap_iou=max_iou,
            )
        )

    return items


def collect_dataset_review_items(
    image_paths: List[str],
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> List[ReviewItem]:
    """Scan all project images and extract review items."""
    all_items: List[ReviewItem] = []
    total = len(image_paths)

    for i, img_path in enumerate(image_paths):
        if cancel_callback and cancel_callback():
            break
        if progress_callback:
            progress_callback(
                i + 1, total, f"Parsing: {os.path.basename(img_path)}"
            )
        img_items = collect_review_items_for_image(
            img_path, output_dir=output_dir
        )
        all_items.extend(img_items)

    return all_items


def _get_review_style() -> str:
    """Returns the QSS stylesheet for the Review Gallery."""
    t = get_theme()
    return (
        get_dialog_style()
        + f"""
        .secondary-button {{
            background-color: {t["surface"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 8px;
            font-weight: 500;
            min-width: 90px;
            height: 32px;
            padding: 0 12px;
        }}
        .secondary-button:hover {{
            background-color: {t["surface_hover"]};
        }}
        .primary-button {{
            background-color: {t["primary"]};
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            min-width: 90px;
            height: 32px;
            padding: 0 12px;
        }}
        .primary-button:hover {{
            background-color: {t["primary_hover"]};
        }}
        QLineEdit, QComboBox {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 6px;
            padding: 4px 8px;
            height: 28px;
        }}
    """
    )


class AnnotationReviewDialog(QDialog):
    """Modeless gallery dialog for visually reviewing objects with outlier sorting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label_widget = parent
        self.all_items: List[ReviewItem] = []
        self.filtered_items: List[ReviewItem] = []
        self.thumbnail_cache: Dict[Tuple[str, int], Optional[QPixmap]] = {}

        self._thumb_timer = QtCore.QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.setInterval(60)
        self._thumb_timer.timeout.connect(self._load_visible_thumbnails)

        self.search_timer = QtCore.QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(200)
        self.search_timer.timeout.connect(self.apply_filters)

        self.init_ui()
        self.reload_gallery(silent=True)

    def init_ui(self) -> None:
        self.setWindowTitle(self.tr("Visual Annotation Review Gallery"))
        self.resize(1100, 720)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Filter and Sorting Controls
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(8)

        self.combo_class = QComboBox(self)
        self.combo_class.addItem(self.tr("All Classes"))
        self.combo_class.currentIndexChanged.connect(self.apply_filters)
        ctrl_layout.addWidget(self.combo_class)

        self.combo_shape_type = QComboBox(self)
        shape_types = [
            (self.tr("All Shape Types"), ""),
            (self.tr("rectangle"), "rectangle"),
            (self.tr("polygon"), "polygon"),
            (self.tr("rotation"), "rotation"),
            (self.tr("point"), "point"),
        ]
        for display_name, key in shape_types:
            self.combo_shape_type.addItem(display_name, userData=key)
        self.combo_shape_type.currentIndexChanged.connect(self.apply_filters)
        ctrl_layout.addWidget(self.combo_shape_type)

        self.combo_sort = QComboBox(self)
        self.combo_sort.addItems(
            [
                self.tr("Sort: Default (Dataset Order)"),
                self.tr("Sort: Overlap IoU (High to Low)"),
                self.tr("Sort: Aspect Ratio Outliers"),
                self.tr("Sort: Area (Smallest First)"),
                self.tr("Sort: Area (Largest First)"),
            ]
        )
        self.combo_sort.currentIndexChanged.connect(self.apply_filters)
        ctrl_layout.addWidget(self.combo_sort)

        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText(
            self.tr("Search by class or image filename...")
        )
        self.search_input.textChanged.connect(self.search_timer.start)
        self.search_input.returnPressed.connect(self.apply_filters)
        ctrl_layout.addWidget(self.search_input, stretch=1)

        self.lbl_count = QLabel(self)
        ctrl_layout.addWidget(self.lbl_count)

        main_layout.addLayout(ctrl_layout)

        # 2. Main Gallery Table
        self.table = QTableWidget(self)
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("Thumbnail"),
                self.tr("Class"),
                self.tr("Type"),
                self.tr("Dimensions (W x H)"),
                self.tr("Aspect Ratio"),
                self.tr("Max Overlap (IoU)"),
                self.tr("Image File"),
            ]
        )
        self.table.setIconSize(QtCore.QSize(56, 56))
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.verticalHeader().setDefaultSectionSize(62)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.ResizeMode.Stretch
        )

        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.verticalScrollBar().valueChanged.connect(
            self._schedule_load_visible_thumbnails
        )
        main_layout.addWidget(self.table, stretch=1)

        # 3. Bottom Actions Row
        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)

        self.btn_reload = QPushButton(self.tr("Reload Gallery"), self)
        self.btn_reload.setProperty("class", "primary-button")
        self.btn_reload.clicked.connect(
            lambda: self.reload_gallery(silent=False)
        )
        action_layout.addWidget(self.btn_reload)

        self.btn_export = QPushButton(self.tr("Export CSV..."), self)
        self.btn_export.setProperty("class", "secondary-button")
        self.btn_export.clicked.connect(self.export_csv)
        action_layout.addWidget(self.btn_export)

        action_layout.addStretch(1)

        self.btn_close = QPushButton(self.tr("Close"), self)
        self.btn_close.setProperty("class", "secondary-button")
        self.btn_close.clicked.connect(self.accept)
        action_layout.addWidget(self.btn_close)

        main_layout.addLayout(action_layout)
        self.setStyleSheet(_get_review_style())

    def get_image_file_list(self) -> List[str]:
        if not self._label_widget or not hasattr(
            self._label_widget, "file_list_widget"
        ):
            return []
        images = []
        count = self._label_widget.file_list_widget.count()
        for idx in range(count):
            images.append(self._label_widget.file_list_widget.item(idx).text())
        return images

    def get_output_dir(self) -> Optional[str]:
        if self._label_widget and hasattr(self._label_widget, "output_dir"):
            return self._label_widget.output_dir
        return None

    def reload_gallery(self, silent: bool = True) -> None:
        """Scan dataset and load review items."""
        image_paths = self.get_image_file_list()
        if not image_paths:
            if not silent:
                QMessageBox.information(
                    self,
                    self.tr("No Images"),
                    self.tr("No image files loaded in project."),
                )
            return

        self.thumbnail_cache.clear()

        progress = QProgressDialog(
            self.tr("Extracting object annotations for review..."),
            self.tr("Cancel"),
            0,
            len(image_paths),
            self,
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle(self.tr("Gallery Loading"))
        progress.setStyleSheet(
            get_progress_dialog_style(color="#1d1d1f", height=20)
        )

        def progress_cb(cur: int, tot: int, msg: str) -> None:
            progress.setValue(cur)
            progress.setLabelText(msg)
            QtWidgets.QApplication.processEvents()

        def cancel_cb() -> bool:
            return progress.wasCanceled()

        try:
            self.all_items = collect_dataset_review_items(
                image_paths=image_paths,
                output_dir=self.get_output_dir(),
                progress_callback=progress_cb,
                cancel_callback=cancel_cb,
            )
        finally:
            progress.close()
        QtWidgets.QApplication.processEvents()

        if image_paths and not self.all_items:
            self.lbl_count.setText(
                self.tr(
                    "No annotations found — check JSON sidecars or Upload YOLO first"
                )
            )

        # Update class combo items
        classes = sorted(
            list({item.label for item in self.all_items if item.label})
        )
        current_cls = self.combo_class.currentText()
        self.combo_class.blockSignals(True)
        self.combo_class.clear()
        self.combo_class.addItem(self.tr("All Classes"))
        for c in classes:
            self.combo_class.addItem(c)
        idx = self.combo_class.findText(current_cls)
        if idx >= 0:
            self.combo_class.setCurrentIndex(idx)
        self.combo_class.blockSignals(False)

        self.apply_filters()

    def _matches_filters(
        self,
        item: ReviewItem,
        sel_class: Optional[str],
        sel_type: str,
        search_txt: str,
    ) -> bool:
        if sel_class is not None and item.label != sel_class:
            return False
        if sel_type and item.shape_type.lower() != sel_type.lower():
            return False
        if search_txt:
            base_img = os.path.basename(item.image_path).lower()
            if (
                search_txt not in item.label.lower()
                and search_txt not in base_img
            ):
                return False
        return True

    def _sort_items(
        self, items: List[ReviewItem], sort_mode: int
    ) -> List[ReviewItem]:
        if sort_mode == 1:
            return sorted(items, key=lambda x: x.overlap_iou, reverse=True)
        if sort_mode == 2:
            return sorted(
                items, key=lambda x: x.extreme_aspect_ratio_score, reverse=True
            )
        if sort_mode == 3:
            return sorted(items, key=lambda x: x.area)
        if sort_mode == 4:
            return sorted(items, key=lambda x: x.area, reverse=True)
        return items

    def apply_filters(self) -> None:
        """Filter and sort items based on current control selections."""
        if self.search_timer.isActive():
            self.search_timer.stop()

        sel_class = (
            self.combo_class.currentText()
            if self.combo_class.currentIndex() > 0
            else None
        )
        sel_type = self.combo_shape_type.currentData() or ""
        sort_mode = self.combo_sort.currentIndex()
        search_txt = self.search_input.text().strip().lower()

        filtered = [
            it
            for it in self.all_items
            if self._matches_filters(it, sel_class, sel_type, search_txt)
        ]
        self.filtered_items = self._sort_items(filtered, sort_mode)

        self.lbl_count.setText(
            self.tr("Showing %d of %d objects")
            % (len(self.filtered_items), len(self.all_items))
        )
        self._populate_table()

    def _populate_table(self) -> None:
        """Render rows into QTableWidget."""
        self.table.setRowCount(len(self.filtered_items))
        t = get_theme()

        for row_idx, item in enumerate(self.filtered_items):
            # 0. Thumbnail (placeholder or cached)
            cache_key = (item.image_path, item.shape_index)
            thumb_item = QTableWidgetItem()
            if cache_key in self.thumbnail_cache:
                pix = self.thumbnail_cache[cache_key]
                if pix and not pix.isNull():
                    thumb_item.setIcon(QIcon(pix))
            self.table.setItem(row_idx, 0, thumb_item)

            # 1. Class with color
            cls_item = QTableWidgetItem(item.label)
            cls_color = _get_label_color(item.label)
            cls_item.setForeground(cls_color)
            font = cls_item.font()
            font.setBold(True)
            cls_item.setFont(font)
            self.table.setItem(row_idx, 1, cls_item)

            # 2. Shape Type
            self.table.setItem(row_idx, 2, QTableWidgetItem(item.shape_type))

            # 3. Dimensions
            dim_str = f"{int(item.width)} x {int(item.height)}"
            self.table.setItem(row_idx, 3, QTableWidgetItem(dim_str))

            # 4. Aspect Ratio
            ar_str = f"{item.aspect_ratio:.2f}"
            ar_item = QTableWidgetItem(ar_str)
            if item.extreme_aspect_ratio_score >= 5.0:
                ar_item.setForeground(QColor(t["warning"]))
            self.table.setItem(row_idx, 4, ar_item)

            # 5. Overlap IoU
            iou_str = f"{item.overlap_iou:.2f}"
            iou_item = QTableWidgetItem(iou_str)
            if item.overlap_iou >= 0.80:
                iou_item.setForeground(QColor(t["error"]))
                iou_font = iou_item.font()
                iou_font.setBold(True)
                iou_item.setFont(iou_font)
            elif item.overlap_iou >= 0.50:
                iou_item.setForeground(QColor(t["warning"]))
            self.table.setItem(row_idx, 5, iou_item)

            # 6. Image Filename
            img_item = QTableWidgetItem(os.path.basename(item.image_path))
            self.table.setItem(row_idx, 6, img_item)

        QtCore.QTimer.singleShot(0, self._load_visible_thumbnails)

    def _schedule_load_visible_thumbnails(self) -> None:
        self._thumb_timer.start()

    def _get_visible_row_range(self) -> Tuple[int, int]:
        total = len(self.filtered_items)
        if total == 0:
            return 0, 0
        viewport = self.table.viewport()
        vp_height = viewport.height() if viewport else 0
        top_row = self.table.rowAt(0)
        if top_row < 0:
            top_row = 0
        bottom_row = self.table.rowAt(vp_height - 1) if vp_height > 0 else -1
        if bottom_row < 0:
            bottom_row = min(top_row + 20, total - 1)

        start_row = max(0, top_row - 5)
        end_row = min(total, bottom_row + 15)
        return start_row, end_row

    def _cache_thumbnail(self, key: Tuple[str, int], pix) -> None:
        """Bounded cache with FIFO eviction to avoid unbounded growth."""
        if (
            key not in self.thumbnail_cache
            and len(self.thumbnail_cache) >= 1000
        ):
            # Evict oldest
            try:
                oldest = next(iter(self.thumbnail_cache))
                self.thumbnail_cache.pop(oldest, None)
            except StopIteration:
                pass
        self.thumbnail_cache[key] = pix

    def _load_thumbnails_for_image(
        self, img_path: str, r_indices: List[int]
    ) -> None:
        """Load and cache thumbnails for rows associated with a single image."""
        if not os.path.isfile(img_path):
            for r in r_indices:
                if r < len(self.filtered_items):
                    it = self.filtered_items[r]
                    self._cache_thumbnail(
                        (it.image_path, it.shape_index), None
                    )
            return

        try:
            with Image.open(img_path) as pil_img:
                # Downsample huge images before cropping for speed
                try:
                    pil_img.draft("RGB", (1024, 1024))
                except Exception:
                    pass
                for r in r_indices:
                    if r >= len(self.filtered_items):
                        continue
                    it = self.filtered_items[r]
                    if it.image_path != img_path:
                        continue
                    cache_key = (it.image_path, it.shape_index)
                    pix = crop_thumbnail_from_pil(pil_img, it.xyxy)
                    self._cache_thumbnail(cache_key, pix)
                    if r < self.table.rowCount():
                        thumb_item = self.table.item(r, 0)
                        if thumb_item is not None and pix and not pix.isNull():
                            thumb_item.setIcon(QIcon(pix))
        except Exception as exc:
            logger.warning(
                f"Failed to load image for thumbnails {img_path}: {exc}"
            )
            for r in r_indices:
                if r < len(self.filtered_items):
                    it = self.filtered_items[r]
                    self._cache_thumbnail(
                        (it.image_path, it.shape_index), None
                    )

    def _load_visible_thumbnails(self) -> None:
        start_row, end_row = self._get_visible_row_range()
        if start_row >= end_row:
            return

        rows_to_load = []
        for r in range(start_row, end_row):
            if r >= len(self.filtered_items):
                break
            item = self.filtered_items[r]
            cache_key = (item.image_path, item.shape_index)
            if cache_key not in self.thumbnail_cache:
                rows_to_load.append(r)

        if not rows_to_load:
            return

        by_image: Dict[str, List[int]] = {}
        for r in rows_to_load:
            it = self.filtered_items[r]
            by_image.setdefault(it.image_path, []).append(r)

        for img_path, r_indices in by_image.items():
            self._load_thumbnails_for_image(img_path, r_indices)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._load_visible_thumbnails)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._schedule_load_visible_thumbnails()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._thumb_timer.stop()
        self.search_timer.stop()
        try:
            self.thumbnail_cache.clear()
        except Exception:
            pass
        super().closeEvent(event)

    def on_cell_double_clicked(self, row: int, _col: int) -> None:
        """Double click row to navigate to annotation in main window."""
        if (
            not self._label_widget
            or row < 0
            or row >= len(self.filtered_items)
            or not hasattr(self._label_widget, "load_file")
        ):
            return

        item = self.filtered_items[row]
        if not os.path.isfile(item.image_path):
            return

        self._label_widget.load_file(item.image_path)
        if hasattr(self._label_widget, "canvas"):
            shapes = getattr(self._label_widget.canvas, "shapes", [])
            if 0 <= item.shape_index < len(shapes):
                self._label_widget.canvas.select_shapes(
                    [shapes[item.shape_index]]
                )

    def show_context_menu(self, pos: QtCore.QPoint) -> None:
        """Display right-click context menu for table row."""
        row = self.table.rowAt(pos.y())
        if row < 0 or row >= len(self.filtered_items):
            return

        item = self.filtered_items[row]
        menu = QMenu(self)

        act_jump = menu.addAction(self.tr("Jump to Annotation"))
        act_delete = menu.addAction(self.tr("Delete Shape"))

        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen == act_jump:
            self.on_cell_double_clicked(row, 0)
        elif chosen == act_delete:
            self.delete_shape(item)

    def delete_shape(self, item: ReviewItem) -> None:
        """Remove a shape directly from its annotation JSON (with backup)."""
        ans = QMessageBox.question(
            self,
            self.tr("Delete Shape"),
            self.tr(
                "Are you sure you want to delete shape #%d ('%s') from '%s'?\n"
                "A backup (.backup_*) will be created."
            )
            % (
                item.shape_index + 1,
                item.label,
                os.path.basename(item.image_path),
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        try:
            import shutil
            from datetime import datetime

            if not os.path.isfile(item.label_file):
                raise FileNotFoundError(item.label_file)
            with open(item.label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("JSON root is not an object")
            shapes = data.get("shapes", [])
            if not isinstance(shapes, list):
                raise ValueError("'shapes' is not a list")
            if not (0 <= item.shape_index < len(shapes)):
                raise IndexError("shape index out of range (stale view?)")
            # Stale-index guard: verify label matches
            existing = shapes[item.shape_index]
            if isinstance(existing, dict) and isinstance(item.label, str):
                if str(existing.get("label", "")) != item.label:
                    raise ValueError(
                        f"Label mismatch (expected '{item.label}', "
                        f"found '{existing.get('label')}') — gallery is stale, reload first"
                    )
            # Backup
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            bdir = os.path.join(
                os.path.dirname(item.label_file), f".backup_{ts}"
            )
            os.makedirs(bdir, exist_ok=True)
            shutil.copy2(
                item.label_file,
                os.path.join(bdir, os.path.basename(item.label_file)),
            )
            shapes.pop(item.shape_index)
            data["shapes"] = shapes
            tmp_path = item.label_file + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, item.label_file)
        except Exception as exc:
            QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr("Failed to delete shape: %s") % str(exc),
            )
            return

        # Evict thumbnails for this image (indices shifted)
        try:
            for k in [
                k for k in self.thumbnail_cache if k[0] == item.image_path
            ]:
                self.thumbnail_cache.pop(k, None)
        except Exception:
            pass

        if self._label_widget and hasattr(self._label_widget, "filename"):
            if self._label_widget.filename == item.image_path:
                try:
                    if hasattr(self._label_widget, "may_continue"):
                        if not self._label_widget.may_continue():
                            pass
                        else:
                            self._label_widget.load_file(
                                self._label_widget.filename
                            )
                    else:
                        self._label_widget.load_file(
                            self._label_widget.filename
                        )
                except Exception:
                    pass

        self.reload_gallery(silent=True)

    def export_csv(self) -> None:
        """Export current filtered objects list to CSV."""
        if not self.filtered_items:
            QMessageBox.information(
                self, self.tr("Empty"), self.tr("No items to export.")
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Export Review Items to CSV"),
            "annotation_review.csv",
            "CSV Files (*.csv)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Image Path",
                        "Shape Index",
                        "Class",
                        "Shape Type",
                        "Width",
                        "Height",
                        "Area",
                        "Aspect Ratio",
                        "Max IoU Overlap",
                    ]
                )
                for it in self.filtered_items:
                    writer.writerow(
                        [
                            it.image_path,
                            it.shape_index + 1,
                            it.label,
                            it.shape_type,
                            f"{it.width:.1f}",
                            f"{it.height:.1f}",
                            f"{it.area:.1f}",
                            f"{it.aspect_ratio:.2f}",
                            f"{it.overlap_iou:.2f}",
                        ]
                    )
            QMessageBox.information(
                self,
                self.tr("Export Succeeded"),
                self.tr("Saved %d items to %s")
                % (len(self.filtered_items), file_path),
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                self.tr("Export Failed"),
                self.tr("Failed to export: %s") % str(exc),
            )
