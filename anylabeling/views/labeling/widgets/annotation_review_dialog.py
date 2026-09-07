from __future__ import annotations

import csv
from dataclasses import dataclass
import faulthandler
import json
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import zlib

from PIL import Image
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
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

from anylabeling.services.annotation_diagnostics import (
    apply_shape_modifications,
    get_label_file_path,
)
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


class _WatchdogGuard:
    """Reentrant watchdog using faulthandler to avoid nesting cancellation leaks."""

    _depth = 0

    def __enter__(self):
        if _WatchdogGuard._depth == 0:
            try:
                faulthandler.dump_traceback_later(25, exit=False)
            except Exception:
                pass
        _WatchdogGuard._depth += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _WatchdogGuard._depth = max(0, _WatchdogGuard._depth - 1)
        if _WatchdogGuard._depth == 0:
            try:
                faulthandler.cancel_dump_traceback_later()
            except Exception:
                pass


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


def _review_metrics_key(
    image_path: Any, shape_type: Any, points: Any
) -> Optional[Tuple[str, str, Tuple[Tuple[float, float], ...]]]:
    """Hashable key to carry IoU across refreshes (label-free, per-image)."""
    try:
        return (
            str(image_path or ""),
            str(shape_type or ""),
            tuple((float(p[0]), float(p[1])) for p in (points or [])),
        )
    except Exception:
        return None


def collect_review_items_for_image(
    image_path: str,
    output_dir: Optional[str] = None,
    reuse_iou: Optional[Dict[Tuple, float]] = None,
) -> List[ReviewItem]:
    """Parse annotation shapes and compute overlap and bounding metrics for one image.

    When ``reuse_iou`` maps geometry keys to previously computed max-IoU,
    the O(S^2) pairwise pass is skipped (matched shapes reuse their IoU,
    others default to 0.0). Used by incremental refresh so a dense file
    can't freeze the GUI with no progress shown.
    """
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
        if reuse_iou is not None:
            key = _review_metrics_key(
                image_path, s.get("shape_type"), s.get("points")
            )
            if key is not None:
                max_iou = float(reuse_iou.get(key, 0.0))
        else:
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
        try:
            img_items = collect_review_items_for_image(
                img_path, output_dir=output_dir
            )
        except Exception as exc:
            # One corrupt file must not kill the whole gallery load.
            logger.warning(f"Skipping {img_path} during review scan: {exc}")
            continue
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
        .staged-banner {{
            background-color: {t["surface"]};
            border: 1px solid {t["primary"]};
            border-radius: 8px;
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

        # Staging engine and column sorting state
        self.staged_deletions: Set[Tuple[str, int]] = set()
        self.staged_reclasses: Dict[Tuple[str, int], str] = {}
        self._undo_stack: List[Dict[str, Any]] = []
        self._sort_col: Optional[int] = None
        self._sort_ascending: bool = True

        self._thumb_timer = QtCore.QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.setInterval(60)
        self._thumb_timer.timeout.connect(self._load_visible_thumbnails)
        # Generation counter + reentrancy flag for thumbnail loads: every
        # populate/refresh bumps the generation so overlapping kicks abort
        # instead of piling image decodes onto the GUI thread.
        self._thumb_gen = 0
        self._thumb_busy = False
        self._populating = False
        self._gallery_generation = 0

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
        # Thumbnail column is Fixed: icons are injected asynchronously AFTER
        # populate, and ResizeToContents would re-measure every row on each
        # setIcon() (thumbnail storm -> apparent hang on large galleries).
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed
        )
        self.table.setColumnWidth(0, 64)
        # Text columns are Interactive with widths fitted once per populate
        # (see _populate_table tail): ResizeToContents re-measures all rows
        # on every text/icon change, which lags then freezes at 5k+ rows.
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Interactive
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Interactive
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Interactive
        )
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Interactive
        )
        self.table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Interactive
        )
        self.table.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.ResizeMode.Stretch
        )

        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.horizontalHeader().sectionClicked.connect(
            self.on_header_clicked
        )
        self.table.horizontalHeader().setToolTip(
            self.tr("Click a column header to sort; click again to reverse")
        )
        self.table.setToolTip(
            self.tr(
                "Del: stage deletion • 1-9/0: reclassify • Enter: apply • Esc: discard • Ctrl+Z: undo"
            )
        )
        self.table.verticalScrollBar().valueChanged.connect(
            self._schedule_load_visible_thumbnails
        )
        self.table.installEventFilter(self)
        self.table.viewport().installEventFilter(self)
        main_layout.addWidget(self.table, stretch=1)

        # 2.5 Staged Changes Action Banner
        self.banner_staged = QFrame(self)
        self.banner_staged.setProperty("class", "staged-banner")
        banner_layout = QHBoxLayout(self.banner_staged)
        banner_layout.setContentsMargins(12, 6, 12, 6)
        banner_layout.setSpacing(10)

        self.lbl_staged_status = QLabel(self)
        self.lbl_staged_status.setStyleSheet("font-weight: 600;")
        banner_layout.addWidget(self.lbl_staged_status, stretch=1)

        self.btn_apply_staged = QPushButton(
            self.tr("Apply Changes (Enter)"), self
        )
        self.btn_apply_staged.setProperty("class", "primary-button")
        self.btn_apply_staged.clicked.connect(self.apply_staged_changes)
        banner_layout.addWidget(self.btn_apply_staged)

        self.btn_discard_staged = QPushButton(
            self.tr("Discard All (Esc)"), self
        )
        self.btn_discard_staged.setProperty("class", "secondary-button")
        self.btn_discard_staged.clicked.connect(
            self.confirm_discard_staged_changes
        )
        banner_layout.addWidget(self.btn_discard_staged)

        self.banner_staged.hide()
        main_layout.addWidget(self.banner_staged)

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
        self.btn_close.clicked.connect(self.close)
        action_layout.addWidget(self.btn_close)

        main_layout.addLayout(action_layout)
        self.setStyleSheet(_get_review_style())
        self.table.setFocus()

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

        # Pump events periodically (not per file): per-file pumping lets
        # timers/thumbnail loads reenter 1600x and stalls the scan itself.
        pump_every = 25
        pump_state = {"n": 0}

        def progress_cb(cur: int, tot: int, msg: str) -> None:
            if self._gallery_generation != gen:
                return
            progress.setValue(cur)
            progress.setLabelText(msg)
            pump_state["n"] += 1
            if pump_state["n"] % pump_every == 0 or cur >= tot:
                QtWidgets.QApplication.processEvents()

        def cancel_cb() -> bool:
            return self._gallery_generation != gen or progress.wasCanceled()

        self._gallery_generation += 1
        gen = self._gallery_generation
        t0 = time.perf_counter()
        with _WatchdogGuard():
            try:
                items = collect_dataset_review_items(
                    image_paths=image_paths,
                    output_dir=self.get_output_dir(),
                    progress_callback=progress_cb,
                    cancel_callback=cancel_cb,
                )
                if self._gallery_generation != gen:
                    return
                self.all_items = items
            finally:
                progress.close()
            logger.info(
                f"Gallery scanned {len(image_paths)} images in "
                f"{time.perf_counter() - t0:.1f}s"
            )
            QtWidgets.QApplication.processEvents()
            if self._gallery_generation != gen:
                return

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

            logger.info(
                f"Gallery rebuilding table for {len(self.all_items)} objects..."
            )
            self.apply_filters()
            logger.info(
                "Gallery reloaded: %d images, %d objects in %.1fs",
                len(image_paths),
                len(self.all_items),
                time.perf_counter() - t0,
            )

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

    def on_header_clicked(self, col: int) -> None:
        """Sort table rows by clicked column header in a type-safe manner."""
        if col == 0:
            # Thumbnail column has no meaningful sort order.
            return
        if self._sort_col == col:
            self._sort_ascending = not self._sort_ascending
        else:
            self._sort_col = col
            self._sort_ascending = True

        header = self.table.horizontalHeader()
        header.setSortIndicatorShown(True)
        order = (
            Qt.SortOrder.AscendingOrder
            if self._sort_ascending
            else Qt.SortOrder.DescendingOrder
        )
        header.setSortIndicator(col, order)

        # Reset combo_sort to default so user knows column sorting is active
        self.combo_sort.blockSignals(True)
        self.combo_sort.setCurrentIndex(0)
        self.combo_sort.blockSignals(False)

        self.filtered_items = self._sort_items_by_column(
            self.filtered_items, self._sort_col, self._sort_ascending
        )
        self._populate_table()

    def _sort_items_by_column(
        self, items: List[ReviewItem], col: int, ascending: bool
    ) -> List[ReviewItem]:
        rev = not ascending
        if col == 1:
            return sorted(
                items, key=lambda x: (x.label or "").lower(), reverse=rev
            )
        if col == 2:
            return sorted(
                items, key=lambda x: (x.shape_type or "").lower(), reverse=rev
            )
        if col == 3:
            return sorted(
                items,
                key=lambda x: (x.width * x.height, x.width, x.height),
                reverse=rev,
            )
        if col == 4:
            return sorted(items, key=lambda x: x.aspect_ratio, reverse=rev)
        if col == 5:
            return sorted(items, key=lambda x: x.overlap_iou, reverse=rev)
        if col == 6:
            return sorted(
                items,
                key=lambda x: os.path.basename(x.image_path).lower(),
                reverse=rev,
            )
        return items

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

        if sort_mode > 0:
            self._sort_col = None
            self.table.horizontalHeader().setSortIndicatorShown(False)
            self.filtered_items = self._sort_items(filtered, sort_mode)
        elif self._sort_col is not None:
            self.filtered_items = self._sort_items_by_column(
                filtered, self._sort_col, self._sort_ascending
            )
        else:
            self.filtered_items = filtered

        self.lbl_count.setText(
            self.tr("Showing %d of %d objects")
            % (len(self.filtered_items), len(self.all_items))
        )
        self._populate_table()

    @staticmethod
    def _mark_staged_deleted(
        cell: QTableWidgetItem, theme: Dict[str, str]
    ) -> None:
        """Apply pending-delete styling (strikeout + secondary color)."""
        f = cell.font()
        f.setStrikeOut(True)
        cell.setFont(f)
        cell.setForeground(QColor(theme["text_secondary"]))

    def _staged_class_text(
        self, label: str, is_staged_del: bool, staged_new_lbl: Optional[str]
    ) -> str:
        """Translatable class-cell text with staging badge."""
        if is_staged_del:
            return self.tr("%s  [Pending Delete]") % label
        if staged_new_lbl:
            return self.tr("%s → %s  [Staged]") % (label, staged_new_lbl)
        return label

    @staticmethod
    def _iou_foreground(
        overlap_iou: float, is_staged_del: bool, theme: Dict[str, str]
    ) -> Optional[QColor]:
        """Shared IoU color so build + in-place update cannot diverge."""
        if is_staged_del:
            return QColor(theme["text_secondary"])
        if overlap_iou >= 0.80:
            return QColor(theme["error"])
        if overlap_iou >= 0.50:
            return QColor(theme["warning"])
        return None

    @staticmethod
    def _ar_foreground(
        extreme_score: float, is_staged_del: bool, theme: Dict[str, str]
    ) -> Optional[QColor]:
        if is_staged_del:
            return QColor(theme["text_secondary"])
        if extreme_score >= 5.0:
            return QColor(theme["warning"])
        return None

    def _build_table_row_items(
        self,
        item: ReviewItem,
        is_staged_del: bool,
        staged_new_lbl: Optional[str],
        theme: Dict[str, str],
    ) -> List[QTableWidgetItem]:
        """Build the 7 display cells for one gallery row."""
        # 0. Thumbnail placeholder; icons are injected async after
        # populate (see _load_visible_thumbnails).
        cells = [QTableWidgetItem()]

        # 1. Class with color and staging badge
        cls_text = self._staged_class_text(
            item.label, is_staged_del, staged_new_lbl
        )
        cls_item = QTableWidgetItem(cls_text)
        if is_staged_del:
            cls_item.setForeground(QColor(theme["error"]))
        elif staged_new_lbl:
            cls_item.setForeground(QColor(theme["warning"]))
        else:
            cls_item.setForeground(_get_label_color(item.label))
        font = cls_item.font()
        font.setBold(True)
        if is_staged_del:
            font.setStrikeOut(True)
        cls_item.setFont(font)
        cells.append(cls_item)

        # 2. Shape Type / 3. Dimensions / 6. Image Filename (plain or staged)
        cells.append(QTableWidgetItem(item.shape_type))
        cells.append(
            QTableWidgetItem(f"{int(item.width)} x {int(item.height)}")
        )

        # 4. Aspect Ratio
        ar_item = QTableWidgetItem(f"{item.aspect_ratio:.2f}")
        ar_fg = self._ar_foreground(
            item.extreme_aspect_ratio_score, is_staged_del, theme
        )
        if ar_fg is not None:
            ar_item.setForeground(ar_fg)
        cells.append(ar_item)

        # 5. Overlap IoU
        iou_item = QTableWidgetItem(f"{item.overlap_iou:.2f}")
        iou_fg = self._iou_foreground(item.overlap_iou, is_staged_del, theme)
        if iou_fg is not None:
            iou_item.setForeground(iou_fg)
        if not is_staged_del and item.overlap_iou >= 0.80:
            iou_font = iou_item.font()
            iou_font.setBold(True)
            iou_item.setFont(iou_font)
        cells.append(iou_item)

        cells.append(QTableWidgetItem(os.path.basename(item.image_path)))

        if is_staged_del:
            for cell in cells[2:]:
                self._mark_staged_deleted(cell, theme)
        return cells

    def _populate_table(self) -> None:
        """Render rows into QTableWidget with visual staging indicators."""
        t_pop = time.perf_counter()
        self.table.setUpdatesEnabled(False)
        self._populating = True
        blocker = (
            QtCore.QSignalBlocker(self.table.selectionModel())
            if self.table.selectionModel()
            else None
        )
        try:
            # Bulk-drop rows first so the loop below always inserts into an
            # empty table. Replacing cells one by one in a full table stalls
            # inside setItem on reload (observed 25s+ watchdog freeze with
            # identical data that populated fine into an empty table).
            self.table.setRowCount(0)
            self.table.setRowCount(len(self.filtered_items))
            t = get_theme()

            for row_idx, item in enumerate(self.filtered_items):
                k = (item.image_path, item.shape_index)
                for col, cell in enumerate(
                    self._build_table_row_items(
                        item,
                        k in self.staged_deletions,
                        self.staged_reclasses.get(k),
                        t,
                    )
                ):
                    self.table.setItem(row_idx, col, cell)
        finally:
            self.table.setUpdatesEnabled(True)
            self._populating = False
            try:
                if blocker is not None:
                    del blocker
            except Exception:
                pass

        if len(self.filtered_items) > 0 and self.table.currentRow() < 0:
            self.table.selectRow(0)

        self._update_staged_banner()
        # Fit text columns once per rebuild (they are Interactive, so no
        # per-keystroke remeasure later). Skip Fixed thumbnail col 0 and
        # Stretch col 6.
        try:
            for _c in range(1, 6):
                self.table.resizeColumnToContents(_c)
        except Exception:
            pass
        dt_pop = time.perf_counter() - t_pop
        if dt_pop > 3.0:
            logger.warning(
                f"Slow table populate: {len(self.filtered_items)} rows "
                f"in {dt_pop:.1f}s"
            )
        self._kick_thumbnails()

    def _update_row_appearance(self, row_idx: int, theme=None) -> None:
        """Update visual styling and staging markers for a single row in place without table rebuild."""
        if row_idx < 0 or row_idx >= len(self.filtered_items):
            return
        if row_idx >= self.table.rowCount():
            return

        item = self.filtered_items[row_idx]
        k = (item.image_path, item.shape_index)
        is_staged_del = k in self.staged_deletions
        staged_new_lbl = self.staged_reclasses.get(k)
        t = theme or get_theme()

        # 1. Class
        cls_item = self.table.item(row_idx, 1)
        if cls_item:
            cls_item.setText(
                self._staged_class_text(
                    item.label, is_staged_del, staged_new_lbl
                )
            )

            if is_staged_del:
                cls_item.setForeground(QColor(t["error"]))
            elif staged_new_lbl:
                cls_item.setForeground(QColor(t["warning"]))
            else:
                cls_color = _get_label_color(item.label)
                cls_item.setForeground(cls_color)

            f = cls_item.font()
            f.setBold(True)
            f.setStrikeOut(is_staged_del)
            cls_item.setFont(f)

        # 2. Shape Type
        type_item = self.table.item(row_idx, 2)
        if type_item:
            f = type_item.font()
            f.setStrikeOut(is_staged_del)
            type_item.setFont(f)
            type_item.setForeground(
                QColor(t["text_secondary"])
                if is_staged_del
                else QColor(t["text"])
            )

        # 3. Dimensions
        dim_item = self.table.item(row_idx, 3)
        if dim_item:
            f = dim_item.font()
            f.setStrikeOut(is_staged_del)
            dim_item.setFont(f)
            dim_item.setForeground(
                QColor(t["text_secondary"])
                if is_staged_del
                else QColor(t["text"])
            )

        # 4. Aspect Ratio
        ar_item = self.table.item(row_idx, 4)
        if ar_item:
            f = ar_item.font()
            f.setStrikeOut(is_staged_del)
            ar_item.setFont(f)
            ar_fg = self._ar_foreground(
                item.extreme_aspect_ratio_score, is_staged_del, t
            )
            ar_item.setForeground(
                ar_fg if ar_fg is not None else QColor(t["text"])
            )

        # 5. Overlap IoU
        iou_item = self.table.item(row_idx, 5)
        if iou_item:
            f = iou_item.font()
            f.setStrikeOut(is_staged_del)
            f.setBold(not is_staged_del and item.overlap_iou >= 0.80)
            iou_item.setFont(f)
            iou_fg = self._iou_foreground(item.overlap_iou, is_staged_del, t)
            iou_item.setForeground(
                iou_fg if iou_fg is not None else QColor(t["text"])
            )

        # 6. Image Filename
        img_item = self.table.item(row_idx, 6)
        if img_item:
            f = img_item.font()
            f.setStrikeOut(is_staged_del)
            img_item.setFont(f)
            img_item.setForeground(
                QColor(t["text_secondary"])
                if is_staged_del
                else QColor(t["text"])
            )

    def _update_rows_appearance(self, rows: Iterable[int]) -> None:
        """Batch update appearance of specified row indices without full table re-render."""
        self.table.setUpdatesEnabled(False)
        try:
            theme = get_theme()
            for r in rows:
                self._update_row_appearance(r, theme=theme)
        finally:
            self.table.setUpdatesEnabled(True)
        self._update_staged_banner()

    def _schedule_load_visible_thumbnails(self) -> None:
        self._thumb_timer.start()

    def _kick_thumbnails(self, delay_ms: int = 250) -> None:
        """Schedule a visible-row thumbnail load bound to this generation."""
        self._thumb_gen += 1
        gen = self._thumb_gen
        QtCore.QTimer.singleShot(
            delay_ms,
            lambda: (
                self._load_visible_thumbnails()
                if gen == self._thumb_gen
                else None
            ),
        )

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

    # Images larger than this are never fully decoded for thumbnails:
    # PIL's draft() downscale only applies to JPEGs, so a huge PNG/TIFF
    # would otherwise load hundreds of MB on the GUI thread (freeze + OOM).
    _THUMB_MAX_PIXELS = 25_000_000
    _THUMB_MAX_DIM = 8000

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
            # Header-only probe (no pixel decode): skip giant images.
            try:
                with Image.open(img_path) as probe:
                    pw, ph = probe.size
                if (
                    pw * ph > self._THUMB_MAX_PIXELS
                    or max(pw, ph) > self._THUMB_MAX_DIM
                ):
                    logger.warning(
                        f"Skipping thumbnail for oversized image {img_path} "
                        f"({pw}x{ph}); showing placeholder."
                    )
                    for r in r_indices:
                        if r < len(self.filtered_items):
                            it = self.filtered_items[r]
                            self._cache_thumbnail(
                                (it.image_path, it.shape_index), None
                            )
                    return
            except Exception:
                pass
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
        # Guarded by generation + busy flag (see _schedule_...): overlapping
        # kicks during populate/refresh abort instead of piling up decodes.
        if self._thumb_busy or getattr(self, "_populating", False):
            return
        gen = self._thumb_gen
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
            else:
                # Inject already-decoded icons (populate no longer does).
                try:
                    pix = self.thumbnail_cache[cache_key]
                    if pix is not None and not pix.isNull():
                        if r < self.table.rowCount():
                            thumb_item = self.table.item(r, 0)
                            if (
                                thumb_item is not None
                                and thumb_item.icon().isNull()
                            ):
                                thumb_item.setIcon(QIcon(pix))
                except Exception:
                    pass

        if not rows_to_load:
            return

        by_image: Dict[str, List[int]] = {}
        for r in rows_to_load:
            it = self.filtered_items[r]
            by_image.setdefault(it.image_path, []).append(r)

        self._thumb_busy = True
        try:
            for img_path, r_indices in by_image.items():
                if gen != self._thumb_gen:
                    return
                self._load_thumbnails_for_image(img_path, r_indices)
        finally:
            self._thumb_busy = False

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._schedule_load_visible_thumbnails()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._schedule_load_visible_thumbnails()

    def has_staged_changes(self) -> bool:
        """Return True if any shapes are staged for deletion or reclassification."""
        return bool(self.staged_deletions or self.staged_reclasses)

    def _update_staged_banner(self) -> None:
        """Update floating action banner text and visibility."""
        if not self.has_staged_changes():
            self.banner_staged.hide()
            return

        del_cnt = len(self.staged_deletions)
        rec_cnt = len(self.staged_reclasses)
        parts = []
        if del_cnt > 0:
            parts.append(self.tr("%d deletion(s)") % del_cnt)
        if rec_cnt > 0:
            parts.append(self.tr("%d reclassification(s)") % rec_cnt)
        msg = self.tr("Staged for batch apply: %s") % ", ".join(parts)
        self.lbl_staged_status.setText(msg)
        self.banner_staged.show()

    def _get_selected_rows(self) -> List[int]:
        """Return sorted unique row indices currently selected in the table.

        Only real selections count: falling back to the current row would
        apply Delete/digits to row 0 after an explicit deselect.
        """
        rows = set()
        selection_model = self.table.selectionModel()
        if selection_model:
            for idx in selection_model.selectedRows():
                rows.add(idx.row())
        return sorted([r for r in rows if 0 <= r < len(self.filtered_items)])

    def _get_available_classes(self) -> List[str]:
        """Return unique ordered list of available class labels."""
        cls_list: List[str] = []
        if self._label_widget and hasattr(
            self._label_widget, "get_project_classes"
        ):
            try:
                for c in self._label_widget.get_project_classes():
                    if c and c not in cls_list:
                        cls_list.append(c)
            except Exception:
                pass
        for i in range(1, self.combo_class.count()):
            c = self.combo_class.itemText(i)
            if c and c not in cls_list:
                cls_list.append(c)
        return cls_list

    def _get_class_for_digit(self, digit: int) -> Optional[str]:
        """Resolve digit (1-9, 0) to project class label string."""
        if self._label_widget and hasattr(
            self._label_widget, "_get_target_label_for_digit"
        ):
            try:
                target = self._label_widget._get_target_label_for_digit(digit)
                if target:
                    return target
            except Exception:
                pass
        classes = self._get_available_classes()
        target_index = 9 if digit == 0 else digit - 1
        if 0 <= target_index < len(classes):
            return classes[target_index]
        return None

    def _build_item_index(
        self,
    ) -> Dict[Tuple[str, int], ReviewItem]:
        """Build O(1) lookup for (image_path, shape_index) -> ReviewItem."""
        # filtered_items always shares objects with all_items, so one pass.
        return {(it.image_path, it.shape_index): it for it in self.all_items}

    def _resolve_label_file(
        self,
        img_path: str,
        shape_idx: int,
        item_index: Optional[Dict[Tuple[str, int], ReviewItem]] = None,
    ) -> str:
        """Prefer stored ReviewItem.label_file over recomputed path."""
        if item_index is not None:
            it = item_index.get((img_path, shape_idx))
            if it is not None and it.label_file:
                return it.label_file
            return get_label_file_path(
                img_path, output_dir=self.get_output_dir()
            )
        for it in self.all_items:
            if it.image_path == img_path and it.shape_index == shape_idx:
                if it.label_file:
                    return it.label_file
                break
        return get_label_file_path(img_path, output_dir=self.get_output_dir())

    # Above this size, touched-file refresh reuses stored IoU instead of
    # the O(S^2) pairwise pass (a dense file would otherwise freeze the GUI
    # with no progress shown). At/below it, refresh recomputes exactly, so
    # the table after Apply matches a full Reload.
    _REFRESH_EXACT_IOU_MAX_SHAPES = 300

    def _decide_iou_reuse(
        self, images: Set[str], output_dir: Optional[str]
    ) -> Optional[Dict[Tuple, float]]:
        """Return stored-IoU map, or None when every file recomputes exactly.

        Carrying over stored max-IoU lets dense touched files skip the
        O(S^2) pairwise pass (it would freeze the GUI with no progress
        shown). Normal-size files recompute exactly, so Apply matches
        Reload. Deletions can only lower true IoU, so reused values on
        dense files stay conservative; the next full reload recomputes.
        """
        for img in images:
            try:
                lf = get_label_file_path(img, output_dir=output_dir)
                with open(lf, "r", encoding="utf-8") as f:
                    n_shapes = len(json.load(f).get("shapes", []) or [])
                if n_shapes > self._REFRESH_EXACT_IOU_MAX_SHAPES:
                    break
            except Exception:
                continue
        else:
            return None
        reuse: Dict[Tuple, float] = {}
        for it in self.all_items:
            if it.image_path in images:
                key = _review_metrics_key(
                    it.image_path, it.shape_type, it.points
                )
                if key is not None:
                    reuse.setdefault(key, float(it.overlap_iou or 0.0))
        return reuse

    def _splice_refreshed_items(
        self, images: Set[str], fresh: List[ReviewItem]
    ) -> None:
        """Replace touched images' items, keeping original file order.

        Plain `keep + fresh` would move touched images to the end of the
        gallery until the next full Reload.
        """
        by_img: Dict[str, List[ReviewItem]] = {}
        for it in self.all_items:
            if it.image_path not in images:
                by_img.setdefault(it.image_path, []).append(it)
        for it in fresh:
            by_img.setdefault(it.image_path, []).append(it)
        seen_imgs: Set[str] = set()
        ordered: List[ReviewItem] = []
        for it in self.all_items:
            if it.image_path not in seen_imgs:
                seen_imgs.add(it.image_path)
                ordered.extend(by_img.pop(it.image_path, []))
        # Any brand-new images (not seen before) go at the end.
        for remaining in by_img.values():
            ordered.extend(remaining)
        self.all_items = ordered

    def refresh_items_for_images(self, images: Set[str]) -> None:
        """Incrementally re-parse only modified images (no full rescan).

        Full reload_gallery() re-reads every JSON + rebuilds the whole
        table, which freezes the UI on large datasets. After Apply/Delete
        only a handful of files changed, so patch all_items in place and
        re-apply filters. Falls back to full reload if anything looks off.
        """
        if not images:
            self.apply_filters()
            return
        t0 = time.perf_counter()
        with _WatchdogGuard():
            try:
                output_dir = self.get_output_dir()
                reuse = self._decide_iou_reuse(images, output_dir)
                fresh: List[ReviewItem] = []
                for img in images:
                    try:
                        fresh.extend(
                            collect_review_items_for_image(
                                img, output_dir=output_dir, reuse_iou=reuse
                            )
                        )
                    except Exception as exc:
                        logger.warning(f"Refresh failed for {img}: {exc}")
                        # Fall back to full reload on unexpected error.
                        self.reload_gallery(silent=True)
                        return
                self._splice_refreshed_items(images, fresh)
                for k in list(self.thumbnail_cache.keys()):
                    if k[0] in images:
                        self.thumbnail_cache.pop(k, None)
                self._rebuild_class_combo()
                # Preserve scroll/current-row across the rebuild: clearing rows
                # resets the scrollbar to top, which reads as "the table looks
                # different" after every Apply.
                vbar = self.table.verticalScrollBar()
                saved_scroll = vbar.value() if vbar is not None else 0
                saved_current = self.table.currentRow()
                self.apply_filters()
                try:
                    if vbar is not None:
                        vbar.setValue(saved_scroll)
                    if saved_current >= 0 and self.table.rowCount() > 0:
                        self.table.setCurrentCell(
                            min(saved_current, self.table.rowCount() - 1), 1
                        )
                except Exception:
                    pass
                logger.info(
                    "Gallery refreshed: %d touched images, %d objects in %.1fs",
                    len(images),
                    len(self.all_items),
                    time.perf_counter() - t0,
                )
            except Exception as exc:
                logger.warning(
                    f"Incremental refresh failed, full reload: {exc}"
                )
                try:
                    self.reload_gallery(silent=True)
                except Exception:
                    pass

    def _patch_canvas_relabels(
        self,
        relabels: Dict[Tuple[str, int], str],
        expected_old: Optional[Dict[Tuple[str, int], str]] = None,
    ) -> None:
        """Update open canvas shape labels in place (no image reload).

        Shapes whose in-memory label no longer matches ``expected_old``
        are skipped instead of relabeling the wrong shape. Locked shapes
        are patched too: disk already changed, so the canvas must match it.
        The dirty flag is left untouched (canvas now matches disk).
        """
        w = self._label_widget
        if not w or not getattr(w, "filename", None) or not relabels:
            return
        # No prompt here: the pre-write dirty guard already aborted on
        # clobber risk. Disk is the new truth; just patch in-memory shapes.
        try:
            cur = w.filename
            canvas_shapes = list(
                getattr(getattr(w, "canvas", None), "shapes", []) or []
            )
            label_list = getattr(w, "label_list", None)
            changed = False
            for (img, idx), new_lbl in relabels.items():
                if img != cur or not (0 <= idx < len(canvas_shapes)):
                    continue
                shape = canvas_shapes[idx]
                try:
                    if (
                        expected_old is not None
                        and (img, idx) in expected_old
                        and str(getattr(shape, "label", ""))
                        != expected_old[(img, idx)]
                    ):
                        continue
                    shape.label = new_lbl
                    if hasattr(w, "_update_shape_color"):
                        w._update_shape_color(shape)
                    if label_list is not None and hasattr(
                        label_list, "find_item_by_shape"
                    ):
                        item = label_list.find_item_by_shape(shape)
                        if item is not None:
                            try:
                                from anylabeling.views.labeling.label_widget import (
                                    _format_label_list_text,
                                )

                                item.setText(
                                    _format_label_list_text(
                                        shape.label,
                                        getattr(shape, "group_id", None),
                                    )
                                )
                            except Exception:
                                try:
                                    item.setText(str(new_lbl))
                                except Exception:
                                    pass
                    changed = True
                except Exception:
                    continue
            if changed:
                try:
                    w.canvas.update()
                except Exception:
                    pass
        except Exception:
            pass

    def _rebuild_class_combo(self) -> None:
        """Refresh class combo from current all_items (no refilter)."""
        try:
            classes = sorted(
                list({it.label for it in self.all_items if it.label})
            )
            cur = self.combo_class.currentText()
            self.combo_class.blockSignals(True)
            self.combo_class.clear()
            self.combo_class.addItem(self.tr("All Classes"))
            for c in classes:
                self.combo_class.addItem(c)
            idx = self.combo_class.findText(cur)
            if idx >= 0:
                self.combo_class.setCurrentIndex(idx)
            self.combo_class.blockSignals(False)
        except Exception:
            try:
                self.combo_class.blockSignals(False)
            except Exception:
                pass

    def _patch_canvas_deletions(
        self,
        deleted_by_file: Dict[str, List[int]],
        expected_old: Optional[Dict[int, str]] = None,
    ) -> None:
        """Remove deleted shapes from the open canvas in place (no reload).

        An index is only removed when the in-memory shape's label matches
        ``expected_old`` (when provided) — never delete the wrong shape on
        a stale canvas. Out-of-range indices are skipped.
        """
        w = self._label_widget
        if not w or not getattr(w, "filename", None) or not deleted_by_file:
            return
        # No prompt: the pre-write dirty guard already aborted on clobber
        # risk. Disk is the new truth; just sync in-memory shapes.
        try:
            cur = w.filename
            indices = deleted_by_file.get(cur)
            if not indices:
                return
            canvas = getattr(w, "canvas", None)
            if canvas is None:
                return
            shapes = getattr(canvas, "shapes", None)
            if not isinstance(shapes, list):
                return
            victims = []
            for idx in sorted(set(indices), reverse=True):
                if not (0 <= idx < len(shapes)):
                    continue
                if (
                    expected_old is not None
                    and idx in expected_old
                    and str(getattr(shapes[idx], "label", ""))
                    != expected_old[idx]
                ):
                    continue
                victims.append(shapes[idx])
                del shapes[idx]
            if not victims:
                return
            victim_ids = {id(s) for s in victims}
            try:
                canvas.selected_shapes = [
                    s
                    for s in getattr(canvas, "selected_shapes", [])
                    if id(s) not in victim_ids
                ]
            except Exception:
                pass
            try:
                if hasattr(w, "remove_labels"):
                    w.remove_labels(victims)
            except Exception:
                pass
            try:
                if hasattr(canvas, "store_shapes"):
                    canvas.store_shapes()
            except Exception:
                pass
            try:
                canvas.update()
            except Exception:
                pass
        except Exception:
            pass

    def _refresh_count_label(self) -> None:
        """Restore the count label, or show a staged summary if staged."""
        if self.has_staged_changes():
            del_cnt = len(self.staged_deletions)
            rec_cnt = len(self.staged_reclasses)
            parts = []
            if del_cnt:
                parts.append(self.tr("%d staged for deletion") % del_cnt)
            if rec_cnt:
                parts.append(self.tr("%d staged as new class") % rec_cnt)
            self.lbl_count.setText("; ".join(parts))
        else:
            self.lbl_count.setText(
                self.tr("Showing %d of %d objects")
                % (len(self.filtered_items), len(self.all_items))
            )

    def toggle_stage_delete_selected_rows(
        self, rows: Optional[List[int]] = None
    ) -> None:
        """Toggle staged deletion status for selected table rows."""
        if rows is None:
            rows = self._get_selected_rows()
        if not rows:
            return

        valid_rows = [r for r in rows if 0 <= r < len(self.filtered_items)]
        if not valid_rows:
            return

        items = [self.filtered_items[r] for r in valid_rows]
        keys = [(it.image_path, it.shape_index) for it in items]
        all_staged = all(k in self.staged_deletions for k in keys)

        if all_staged:
            removed = list(keys)
            for k in removed:
                self.staged_deletions.discard(k)
            self._undo_stack.append(
                {
                    "type": "batch_delete",
                    "added": [],
                    "removed": removed,
                    "prev_reclasses": {},
                }
            )
        else:
            added = [k for k in keys if k not in self.staged_deletions]
            # Staging delete clears any staged reclass for the same shape
            # (a shape cannot be both deleted and reclassified).
            prev_reclasses = {}
            for k in added:
                if k in self.staged_reclasses:
                    prev_reclasses[k] = self.staged_reclasses.pop(k)
                self.staged_deletions.add(k)
            self._undo_stack.append(
                {
                    "type": "batch_delete",
                    "added": added,
                    "removed": [],
                    "prev_reclasses": prev_reclasses,
                }
            )

        self._update_rows_appearance(valid_rows)
        if all_staged:
            self.lbl_count.setText(
                self.tr("Unstaged deletion for %d shape(s)") % len(removed)
            )
        else:
            self.lbl_count.setText(
                self.tr("Staged %d shape(s) for deletion (Pending)")
                % len(added)
            )

    def stage_reclass_selected_rows(
        self, rows: Optional[List[int]] = None, new_label: str = ""
    ) -> None:
        """Stage reclassification for selected table rows."""
        if rows is None:
            rows = self._get_selected_rows()
        if not rows or not new_label:
            return

        valid_rows = [r for r in rows if 0 <= r < len(self.filtered_items)]
        if not valid_rows:
            return

        changes = []
        for r in valid_rows:
            it = self.filtered_items[r]
            k = (it.image_path, it.shape_index)
            was_deleted = k in self.staged_deletions
            if it.label == new_label:
                # Reclassifying to the current label only clears staging.
                if k in self.staged_reclasses:
                    prev = self.staged_reclasses.pop(k)
                    if was_deleted:
                        self.staged_deletions.remove(k)
                    changes.append(
                        {
                            "target": k,
                            "prev": prev,
                            "new": None,
                            "was_deleted": was_deleted,
                        }
                    )
                continue

            prev = self.staged_reclasses.get(k)
            # Skip no-op re-staging of the identical label.
            if prev == new_label and not was_deleted:
                continue
            self.staged_reclasses[k] = new_label
            # If item was staged for deletion, reclassification unmarks deletion
            if was_deleted:
                self.staged_deletions.remove(k)
            changes.append(
                {
                    "target": k,
                    "prev": prev,
                    "new": new_label,
                    "was_deleted": was_deleted,
                }
            )

        if not changes:
            return
        self._undo_stack.append({"type": "batch_reclass", "changes": changes})

        self._update_rows_appearance(valid_rows)
        self.lbl_count.setText(
            self.tr("Staged %d shape(s) as '%s'") % (len(changes), new_label)
        )

    def undo_last_staged_action(self) -> None:
        """Undo the most recent staging action (batch-aware)."""
        if not self._undo_stack:
            return
        action = self._undo_stack.pop()
        act_type = action.get("type")

        affected_keys = set()
        if act_type == "batch_delete":
            for k in action.get("added", []):
                self.staged_deletions.discard(k)
                affected_keys.add(k)
            for k in action.get("removed", []):
                self.staged_deletions.add(k)
                affected_keys.add(k)
            for k, prev_lbl in (action.get("prev_reclasses") or {}).items():
                self.staged_reclasses[k] = prev_lbl
                affected_keys.add(k)
        elif act_type == "batch_reclass":
            for ch in action.get("changes", []):
                k = ch.get("target")
                if not k:
                    continue
                prev = ch.get("prev")
                if prev is None:
                    self.staged_reclasses.pop(k, None)
                else:
                    self.staged_reclasses[k] = prev
                if ch.get("was_deleted"):
                    self.staged_deletions.add(k)
                affected_keys.add(k)
        else:
            self._update_staged_banner()
            return

        affected_rows = [
            r
            for r, it in enumerate(self.filtered_items)
            if (it.image_path, it.shape_index) in affected_keys
        ]
        if affected_rows:
            self._update_rows_appearance(affected_rows)
        else:
            self._update_staged_banner()
        self._refresh_count_label()

    def _drop_staging_for_images(self, images: Set[str]) -> None:
        """Drop staged keys for images whose indices shifted after a write."""
        if not images or not self.has_staged_changes():
            return
        self.staged_deletions = {
            k for k in self.staged_deletions if k[0] not in images
        }
        self.staged_reclasses = {
            k: v
            for k, v in self.staged_reclasses.items()
            if k[0] not in images
        }
        # Filter (not drop) batch entries touching dropped images so undo
        # survives for keys that are still staged.
        kept = []
        for entry in self._undo_stack:
            if entry.get("type") == "batch_delete":
                added = [
                    k for k in entry.get("added", []) if k[0] not in images
                ]
                removed = [
                    k for k in entry.get("removed", []) if k[0] not in images
                ]
                prev_re = {
                    k: v
                    for k, v in (entry.get("prev_reclasses") or {}).items()
                    if k[0] not in images
                }
                if added or removed or prev_re:
                    kept.append(
                        {
                            "type": "batch_delete",
                            "added": added,
                            "removed": removed,
                            "prev_reclasses": prev_re,
                        }
                    )
            elif entry.get("type") == "batch_reclass":
                changes = [
                    c
                    for c in entry.get("changes", [])
                    if not (
                        isinstance(c.get("target"), (list, tuple))
                        and c["target"][0] in images
                    )
                ]
                if changes:
                    kept.append({"type": "batch_reclass", "changes": changes})
            else:
                kept.append(entry)
        self._undo_stack = kept
        self._update_staged_banner()

    def discard_staged_changes(self) -> None:
        """Discard all staged changes instantly in place without full table rebuild."""
        if not self.has_staged_changes():
            return

        staged_keys = set(self.staged_deletions) | set(
            self.staged_reclasses.keys()
        )
        self.staged_deletions.clear()
        self.staged_reclasses.clear()
        self._undo_stack.clear()

        affected_rows = [
            i
            for i, it in enumerate(self.filtered_items)
            if (it.image_path, it.shape_index) in staged_keys
        ]
        if affected_rows:
            # _update_rows_appearance already refreshes the banner.
            self._update_rows_appearance(affected_rows)
        else:
            self._update_staged_banner()
        self._refresh_count_label()

    def confirm_discard_staged_changes(self) -> None:
        """Confirm-then-discard for banner/Esc destructive action."""
        if not self.has_staged_changes():
            return
        ans = QMessageBox.question(
            self,
            self.tr("Discard Staged Changes?"),
            self.tr("Discard %d staged change(s) without applying?")
            % (len(self.staged_deletions) + len(self.staged_reclasses)),
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
        )
        if ans == QMessageBox.StandardButton.Yes:
            self.discard_staged_changes()

    def _abort_if_main_dirty(self, touched: Set[str]) -> bool:
        """Abort a disk write that would clobber unsaved main-window edits.

        Returns True when the caller must abort. The warning is parented to
        this gallery dialog (always on top) — deliberately NOT
        main_window.may_continue(), whose modal prompt opens behind the
        modeless gallery and looks like a total freeze.
        """
        try:
            w = self._label_widget
            if (
                w is not None
                and getattr(w, "filename", None) in touched
                and getattr(w, "dirty", False)
            ):
                QMessageBox.warning(
                    self,
                    self.tr("Unsaved Changes"),
                    self.tr(
                        "The current file has unsaved changes in the main "
                        "window. Save or discard them there first, then retry."
                    ),
                )
                return True
        except Exception:
            pass
        return False

    def _group_staged_modifications(
        self, item_index: Dict[Tuple[str, int], ReviewItem]
    ) -> Dict[str, Dict[str, Any]]:
        """Group staged deletions/reclasses by label file for one write each."""
        output_dir = self.get_output_dir()

        def _lf_for(img_path: str, shape_idx: int) -> str:
            it = item_index.get((img_path, shape_idx))
            if it is not None and it.label_file:
                return it.label_file
            return get_label_file_path(img_path, output_dir=output_dir)

        def _lbl_for(img_path: str, shape_idx: int) -> Optional[str]:
            it = item_index.get((img_path, shape_idx))
            return it.label if it is not None else None

        mod_by_file: Dict[str, Dict[str, Any]] = {}

        def _entry(lbl_file: str) -> Dict[str, Any]:
            return mod_by_file.setdefault(
                lbl_file,
                {
                    "deletions": set(),
                    "reclasses": {},
                    "img_paths": set(),
                    "expected": {},
                    "keys": set(),
                },
            )

        for img_path, shape_idx in self.staged_deletions:
            entry = _entry(_lf_for(img_path, shape_idx))
            entry["deletions"].add(shape_idx)
            entry["img_paths"].add(img_path)
            entry["keys"].add((img_path, shape_idx))
            exp = _lbl_for(img_path, shape_idx)
            if exp:
                entry["expected"][shape_idx] = exp

        for (img_path, shape_idx), new_label in self.staged_reclasses.items():
            if (img_path, shape_idx) in self.staged_deletions:
                # Defensive: both-staged should not happen (toggle clears
                # reclass, reclass clears delete), deletion wins.
                continue
            entry = _entry(_lf_for(img_path, shape_idx))
            entry["reclasses"][shape_idx] = new_label
            entry["img_paths"].add(img_path)
            entry["keys"].add((img_path, shape_idx))
            exp = _lbl_for(img_path, shape_idx)
            if exp:
                entry["expected"].setdefault(shape_idx, exp)
        return mod_by_file

    def _commit_modification_groups(
        self, mod_by_file: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Write each label file once; collect per-file success/failure."""
        success_count = 0
        failed_files: List[str] = []
        modified_images: Set[str] = set()
        succeeded_keys: Set[Tuple[str, int]] = set()
        succeeded_relabels: Dict[Tuple[str, int], str] = {}
        has_deletion_success = False
        for lbl_file, ops in mod_by_file.items():
            if not os.path.isfile(lbl_file):
                failed_files.append(lbl_file)
                continue
            ok = apply_shape_modifications(
                label_file=lbl_file,
                deletions=ops["deletions"],
                reclasses=ops["reclasses"],
                expected_labels=ops["expected"] or None,
            )
            if ok:
                success_count += 1
                modified_images.update(ops["img_paths"])
                succeeded_keys.update(ops["keys"])
                if ops["deletions"]:
                    has_deletion_success = True
                for r_idx, new_lbl in ops["reclasses"].items():
                    for img in ops["img_paths"]:
                        if (img, r_idx) in ops["keys"]:
                            succeeded_relabels[(img, r_idx)] = new_lbl
            else:
                failed_files.append(lbl_file)
                logger.error(f"Failed to apply modifications to {lbl_file}")
        return {
            "success_count": success_count,
            "failed_files": failed_files,
            "modified_images": modified_images,
            "succeeded_keys": succeeded_keys,
            "succeeded_relabels": succeeded_relabels,
            "has_deletion_success": has_deletion_success,
        }

    def _prune_undo_for_keys(
        self, succeeded_keys: Set[Tuple[str, int]]
    ) -> None:
        """Drop succeeded keys from undo entries; keep retryable ones."""
        if not succeeded_keys:
            return
        self.staged_deletions -= succeeded_keys
        for k in list(self.staged_reclasses.keys()):
            if k in succeeded_keys:
                del self.staged_reclasses[k]
        kept = []
        for entry in self._undo_stack:
            if entry.get("type") == "batch_delete":
                added = [
                    k
                    for k in entry.get("added", [])
                    if k not in succeeded_keys
                ]
                removed = [
                    k
                    for k in entry.get("removed", [])
                    if k not in succeeded_keys
                ]
                prev_re = {
                    k: v
                    for k, v in (entry.get("prev_reclasses") or {}).items()
                    if k not in succeeded_keys
                }
                if added or removed or prev_re:
                    kept.append(
                        {
                            "type": "batch_delete",
                            "added": added,
                            "removed": removed,
                            "prev_reclasses": prev_re,
                        }
                    )
            elif entry.get("type") == "batch_reclass":
                changes = [
                    c
                    for c in entry.get("changes", [])
                    if c.get("target") not in succeeded_keys
                ]
                if changes:
                    kept.append({"type": "batch_reclass", "changes": changes})
            else:
                kept.append(entry)
        self._undo_stack = kept

    def _sync_canvas_after_apply(
        self,
        item_index: Dict[Tuple[str, int], ReviewItem],
        modified_images: Set[str],
        succeeded_keys: Set[Tuple[str, int]],
        succeeded_relabels: Dict[Tuple[str, int], str],
        has_deletion_success: bool,
    ) -> None:
        """Patch the open canvas in place (no full image reload).

        load_file() decodes the image, rebuilds pixmaps/brightness cache
        and can appear as a hang on large images. Disk is the new truth
        (the pre-write dirty guard already aborted on clobber risk).
        """
        if not (
            self._label_widget and hasattr(self._label_widget, "filename")
        ):
            return
        if self._label_widget.filename not in modified_images:
            return
        # Pre-write labels for stale-guarded canvas patching.
        pre_labels = {k: it.label for k, it in item_index.items() if it.label}
        # Relabels first: they use pre-delete indices, which the
        # deletion patch below shifts.
        if succeeded_relabels:
            self._patch_canvas_relabels(succeeded_relabels, pre_labels)
        if has_deletion_success:
            dels_for_canvas: Dict[str, List[int]] = {}
            expected_canvas: Dict[int, str] = {}
            for img, idx in succeeded_keys:
                if (
                    img == self._label_widget.filename
                    and (
                        img,
                        idx,
                    )
                    not in succeeded_relabels
                ):
                    dels_for_canvas.setdefault(img, []).append(idx)
                    if (img, idx) in pre_labels:
                        expected_canvas[idx] = pre_labels[(img, idx)]
            if dels_for_canvas:
                self._patch_canvas_deletions(dels_for_canvas, expected_canvas)

    def _report_apply_result(
        self, success_count: int, failed_files: List[str], quiet: bool
    ) -> None:
        if failed_files:
            QMessageBox.warning(
                self,
                self.tr("Partial Success"),
                self.tr(
                    "Updated %d file(s); %d file(s) failed and were kept staged for retry: %s."
                )
                % (
                    success_count,
                    len(failed_files),
                    ", ".join(os.path.basename(p) for p in failed_files[:5]),
                ),
            )
        elif not quiet:
            QMessageBox.information(
                self,
                self.tr("Changes Applied"),
                self.tr("Successfully updated %d annotation file(s).")
                % success_count,
            )

    def apply_staged_changes(
        self, skip_confirm: bool = False, quiet: bool = False
    ) -> None:
        """Commit all staged deletions and reclassifications atomically to disk.

        When ``quiet`` is set (closeEvent path), result boxes are shown only
        for failures so no modal stacks on top of the closing prompt.
        """
        if not self.has_staged_changes():
            return

        total_dels = len(self.staged_deletions)
        total_reclasses = len(self.staged_reclasses)

        if not skip_confirm:
            ans = QMessageBox.question(
                self,
                self.tr("Apply Changes"),
                self.tr(
                    "Apply %d deletion(s) and %d reclassification(s) to dataset files?"
                )
                % (total_dels, total_reclasses),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return

        # Pre-write guard: gallery writes JSON directly. If the currently
        # open file has unsaved in-memory edits and is about to be
        # overwritten, abort with a gallery-parented warning instead of
        # silently clobbering it (and instead of a main-window modal that
        # hides behind this dialog and mimics a freeze).
        touched_pre = {img for img, _ in self.staged_deletions} | {
            img for img, _ in self.staged_reclasses.keys()
        }
        if self._abort_if_main_dirty(touched_pre):
            return

        # Group modifications by label file path, preferring the stored
        # ReviewItem.label_file so custom output_dir mappings keep working.
        # Build one O(1) index instead of O(N) scans per staged key.
        item_index = self._build_item_index()
        result = self._commit_modification_groups(
            self._group_staged_modifications(item_index)
        )
        success_count = result["success_count"]
        failed_files = result["failed_files"]
        modified_images = result["modified_images"]
        succeeded_keys = result["succeeded_keys"]
        succeeded_relabels = result["succeeded_relabels"]
        has_deletion_success = result["has_deletion_success"]

        # Only clear staging that actually succeeded; keep failed for retry.
        # Undo entries are filtered (not dropped) so retryable keys keep
        # their undo history.
        self._prune_undo_for_keys(succeeded_keys)

        # Refresh main canvas without full load_file() image decode.
        # (refresh_items_for_images() below evicts thumbnails itself.)
        self._sync_canvas_after_apply(
            item_index,
            modified_images,
            succeeded_keys,
            succeeded_relabels,
            has_deletion_success,
        )

        self._report_apply_result(success_count, failed_files, quiet)
        # Refresh gallery from disk for touched files only (no full rescan:
        # full reload_gallery() re-reads every JSON, which is what used to
        # freeze the UI). touched-file re-parse is strictly cheaper than one
        # initial gallery open. Canvas was already patched above, so no
        # image reload happens here.
        if modified_images:
            self.refresh_items_for_images(modified_images)
        elif not self.has_staged_changes():
            self.apply_filters()

    def _handle_shortcut_key(self, event: QtGui.QKeyEvent) -> bool:
        """Handle keyboard shortcuts for review table and viewport."""
        if event.isAutoRepeat():
            return False
        fw = self.focusWidget()
        if isinstance(
            fw,
            (
                QLineEdit,
                QtWidgets.QTextEdit,
                QtWidgets.QPlainTextEdit,
                QComboBox,
                QtWidgets.QAbstractSpinBox,
            ),
        ):
            return False
        key = event.key()
        modifiers = event.modifiers()

        # Escape: confirm discard when staged, else close dialog
        if key == Qt.Key.Key_Escape:
            if self.has_staged_changes():
                self.confirm_discard_staged_changes()
                return True
            self.close()
            return True

        # Table-modifying shortcuts only fire from the table itself, never from
        # focused buttons (Close/Reload) or other controls.
        if fw not in (self.table, self.table.viewport()):
            return False

        # Ctrl+Z (plain, no Shift/Alt): Undo last staging operation.
        # Ctrl+Shift+Z is redo — must not trigger undo.
        if key == Qt.Key.Key_Z and (
            modifiers & Qt.KeyboardModifier.ControlModifier
        ):
            if modifiers & (
                Qt.KeyboardModifier.ShiftModifier
                | Qt.KeyboardModifier.AltModifier
            ):
                return False
            if not self._undo_stack:
                return False
            self.undo_last_staged_action()
            return True

        # Enter / Return: Apply staged changes
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if modifiers & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.AltModifier
                | Qt.KeyboardModifier.MetaModifier
            ):
                return False
            if self.has_staged_changes():
                self.apply_staged_changes()
                return True

        # Delete / Backspace: Toggle staged deletion on selected rows
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if modifiers & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.AltModifier
                | Qt.KeyboardModifier.MetaModifier
            ):
                return False
            rows = self._get_selected_rows()
            if rows:
                self.toggle_stage_delete_selected_rows(rows)
                return True
            return False

        # Digit keys 1-9, 0: Quick class reclassification
        digit_handled = self._handle_digit_key(event, key, modifiers)
        if digit_handled is not None:
            return digit_handled

        return False

    def _handle_digit_key(
        self, event: QtGui.QKeyEvent, key: int, modifiers: Qt.KeyboardModifier
    ) -> Optional[bool]:
        """Handle digit keys 1-9, 0 for quick reclassification.

        Returns True/False when handled, None when the key is not a digit.
        """
        digit = None
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            digit = key - Qt.Key.Key_0
        elif key == Qt.Key.Key_0:
            digit = 0
        else:
            txt = event.text().strip()
            if len(txt) == 1 and txt.isdigit():
                digit = int(txt)
        if digit is None:
            return None
        if modifiers & (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.MetaModifier
        ):
            return False
        rows = self._get_selected_rows()
        if not rows:
            self.lbl_count.setText(
                self.tr("Select a row first to reclassify with digits 1-9")
            )
            return True
        label = self._get_class_for_digit(digit)
        if label:
            self.stage_reclass_selected_rows(rows, label)
            return True
        avail = self._get_available_classes()
        if avail:
            preview = ", ".join(
                f"{i + 1}:{c}" for i, c in enumerate(avail[:9])
            )
            self.lbl_count.setText(
                self.tr("Key '%d' unassigned. Available: %s")
                % (digit, preview)
            )
        else:
            self.lbl_count.setText(
                self.tr("Key '%d' unassigned — no classes detected.") % digit
            )
        return True

    def eventFilter(
        self, watched: QtCore.QObject, event: QtCore.QEvent
    ) -> bool:
        """Intercept key events from table and viewport before QAbstractItemView consumes them."""
        if event.type() == QtCore.QEvent.Type.KeyPress and isinstance(
            event, QtGui.QKeyEvent
        ):
            if self._handle_shortcut_key(event):
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle keyboard shortcuts with input focus protection."""
        if self._handle_shortcut_key(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def reject(self) -> None:
        """Route reject/close requests through closeEvent."""
        # NOTE: the dialog result stays Rejected by design; this dialog is
        # used modelessly (show()) and no caller reads exec() results.
        self.close()

    def accept(self) -> None:
        """Route accept through close() so cleanup + staged guard always run."""
        # NOTE: see reject() — result intentionally stays Rejected.
        self.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Prompt to apply or discard uncommitted staged changes before closing."""
        if self.has_staged_changes():
            box = QMessageBox(
                QMessageBox.Icon.Question,
                self.tr("Unsaved Changes"),
                self.tr(
                    "You have %d pending staged change(s). Do you want to apply them before closing?"
                )
                % (len(self.staged_deletions) + len(self.staged_reclasses)),
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                self,
            )
            box.setWindowModality(Qt.WindowModality.WindowModal)
            ans = box.exec()
            if ans == QMessageBox.StandardButton.Yes:
                # Already confirmed via this prompt; skip second confirm and
                # result box (failures still warn, keeping the dialog open).
                self.apply_staged_changes(skip_confirm=True, quiet=True)
                if self.has_staged_changes():
                    event.ignore()
                    return
            elif ans == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            else:
                # "No" discards uncommitted staging with the dialog.
                self.staged_deletions.clear()
                self.staged_reclasses.clear()
                self._undo_stack.clear()

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

        selected_rows = self._get_selected_rows()
        if row not in selected_rows:
            selected_rows = [row]

        item = self.filtered_items[row]
        menu = QMenu(self)

        act_jump = menu.addAction(self.tr("Jump to Annotation"))
        menu.addSeparator()

        count_str = (
            f" ({len(selected_rows)} shapes)" if len(selected_rows) > 1 else ""
        )
        all_del = all(
            (
                self.filtered_items[r].image_path,
                self.filtered_items[r].shape_index,
            )
            in self.staged_deletions
            for r in selected_rows
        )
        act_stage_del = menu.addAction(
            self.tr("Unstage Deletion%s") % count_str
            if all_del
            else self.tr("Stage for Deletion (Del)%s") % count_str
        )

        reclass_menu = menu.addMenu(
            self.tr("Stage Reclassification%s") % count_str
        )
        classes = self._get_available_classes()
        if not classes:
            reclass_menu.setEnabled(False)
        selected_snapshot = tuple(selected_rows)
        for c in classes:
            act_c = reclass_menu.addAction(c)
            act_c.triggered.connect(
                lambda checked=False, target_cls=c, target_rows=selected_snapshot: (
                    self.stage_reclass_selected_rows(
                        list(target_rows), target_cls
                    )
                )
            )

        menu.addSeparator()
        if len(selected_rows) == 1:
            act_delete = menu.addAction(
                self.tr("Delete Shape Immediately (Permanent)")
            )
        else:
            act_delete = menu.addAction(
                self.tr("Delete %d Shapes Immediately (Permanent)")
                % len(selected_rows)
            )

        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen == act_jump:
            self.on_cell_double_clicked(row, 0)
        elif chosen == act_stage_del:
            self.toggle_stage_delete_selected_rows(selected_rows)
        elif chosen == act_delete:
            if len(selected_rows) == 1:
                self.delete_shape(item)
            else:
                self.delete_selected_shapes_immediately(selected_rows)

    def delete_selected_shapes_immediately(self, rows: List[int]) -> None:
        """Remove multiple shapes immediately from annotation files."""
        valid_rows = [r for r in rows if 0 <= r < len(self.filtered_items)]
        if not valid_rows:
            return
        items = [self.filtered_items[r] for r in valid_rows]
        if self._abort_if_main_dirty({it.image_path for it in items}):
            return
        ans = QMessageBox.question(
            self,
            self.tr("Delete Shapes"),
            self.tr("Are you sure you want to permanently delete %d shape(s)?")
            % len(items),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        by_file: Dict[str, Set[int]] = {}
        expected_by_file: Dict[str, Dict[int, str]] = {}
        imgs_by_file: Dict[str, Set[str]] = {}
        keys_by_file: Dict[str, Set[Tuple[str, int]]] = {}
        item_index = self._build_item_index()
        for it in items:
            lf = it.label_file or self._resolve_label_file(
                it.image_path, it.shape_index, item_index
            )
            if lf and os.path.isfile(lf):
                by_file.setdefault(lf, set()).add(it.shape_index)
                imgs_by_file.setdefault(lf, set()).add(it.image_path)
                keys_by_file.setdefault(lf, set()).add(
                    (it.image_path, it.shape_index)
                )
                if it.label:
                    expected_by_file.setdefault(lf, {})[it.shape_index] = (
                        it.label
                    )

        succeeded_images: Set[str] = set()
        succeeded_keys: Set[Tuple[str, int]] = set()
        all_ok = True
        for lf, indices in by_file.items():
            ok = apply_shape_modifications(
                label_file=lf,
                deletions=indices,
                reclasses={},
                expected_labels=expected_by_file.get(lf) or None,
            )
            if ok:
                succeeded_images.update(imgs_by_file.get(lf, set()))
                succeeded_keys.update(keys_by_file.get(lf, set()))
            else:
                all_ok = False

        if not all_ok:
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Some shapes could not be deleted from files."),
            )
            if not succeeded_keys:
                return

        # Staged indices for these images are now stale; drop them.
        self._drop_staging_for_images(succeeded_images)
        if self._label_widget and getattr(
            self._label_widget, "filename", None
        ):
            dels_for_canvas: Dict[str, List[int]] = {}
            expected_canvas: Dict[int, str] = {}
            labels_by_key = {
                (it.image_path, it.shape_index): it.label for it in items
            }
            for img, idx in succeeded_keys:
                if img == self._label_widget.filename:
                    dels_for_canvas.setdefault(img, []).append(idx)
                    if (img, idx) in labels_by_key and labels_by_key[
                        (img, idx)
                    ]:
                        expected_canvas[idx] = labels_by_key[(img, idx)]
            if dels_for_canvas:
                self._patch_canvas_deletions(dels_for_canvas, expected_canvas)
        self.refresh_items_for_images(succeeded_images)

    def delete_shape(self, item: ReviewItem) -> None:
        """Remove a shape immediately from its annotation JSON."""
        if self._abort_if_main_dirty({item.image_path}):
            return
        ans = QMessageBox.question(
            self,
            self.tr("Delete Shape"),
            self.tr(
                "Are you sure you want to delete shape #%d ('%s') from '%s'?"
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

        label_file = item.label_file or self._resolve_label_file(
            item.image_path, item.shape_index
        )
        ok = apply_shape_modifications(
            label_file=label_file,
            deletions={item.shape_index},
            reclasses={},
            expected_labels={item.shape_index: item.label}
            if item.label
            else None,
        )
        if not ok:
            QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr("Failed to delete shape from file."),
            )
            return

        self._drop_staging_for_images({item.image_path})
        if (
            self._label_widget
            and getattr(self._label_widget, "filename", None)
            == item.image_path
        ):
            self._patch_canvas_deletions(
                {item.image_path: [item.shape_index]},
                {item.shape_index: item.label} if item.label else None,
            )
        self.refresh_items_for_images({item.image_path})

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
