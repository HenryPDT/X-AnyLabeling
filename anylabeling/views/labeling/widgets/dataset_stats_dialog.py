"""Dataset statistics dashboard (additive).

Per-class count/size aggregates with mean/stddev, marks-per-image,
and tiny-box warnings. Uses ReviewTableBase shared style/table and
dataset_stats_core pure computation so logic stays Qt-free.
"""

from __future__ import annotations

import os
from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from anylabeling.services.dataset_stats_core import (
    DatasetStats,
    compute_full_dataset_stats,
)
from anylabeling.views.labeling.utils.style import get_progress_dialog_style
from anylabeling.views.labeling.widgets.review_table_base import (
    export_rows_to_csv,
    get_review_dialog_style,
    setup_review_table,
)


HEADERS = [
    "Class",
    "Count",
    "Images",
    "Marks/Img",
    "Mean W",
    "Std W",
    "Mean H",
    "Std H",
    "Mean Area",
    "Tiny Warn",
]


class DatasetStatsDialog(QDialog):
    """Modeless dashboard for dataset size/count health."""

    TINY_WARN_RATE = 0.1

    def __init__(self, parent=None, tiny_box_px: float = 32.0):
        super().__init__(parent)
        self._label_widget = parent
        self.stats: Optional[DatasetStats] = None
        self.tiny_box_px = float(tiny_box_px)
        self.init_ui()
        self.reload_stats(silent=False)

    def init_ui(self) -> None:
        self.setWindowTitle(self.tr("Dataset Statistics"))
        self.resize(980, 640)
        self.setStyleSheet(get_review_dialog_style())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.summary_label = QLabel("", self)
        layout.addWidget(self.summary_label)

        self.table = QTableWidget(self)
        setup_review_table(self.table, [self.tr(h) for h in HEADERS])
        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_refresh = QPushButton(self.tr("Refresh"), self)
        self.btn_refresh.setProperty("class", "secondary-button")
        self.btn_refresh.clicked.connect(lambda: self.reload_stats())
        btn_row.addWidget(self.btn_refresh)
        self.btn_export = QPushButton(self.tr("Export CSV"), self)
        self.btn_export.setProperty("class", "primary-button")
        self.btn_export.clicked.connect(self.on_export_csv)
        btn_row.addWidget(self.btn_export)
        layout.addLayout(btn_row)

    def _collect_image_paths(self) -> List[str]:
        widget = self._label_widget
        if widget is None:
            return []
        try:
            image_list = list(getattr(widget, "image_list", []) or [])
        except Exception:
            image_list = []
        if image_list:
            return image_list
        filename = getattr(widget, "filename", None)
        if filename:
            return [str(filename)]
        return []

    def reload_stats(self, silent: bool = False) -> None:
        image_paths = self._collect_image_paths()
        if not image_paths and not silent:
            self.summary_label.setText(self.tr("No images loaded."))
            return
        output_dir = getattr(self._label_widget, "output_dir", None)
        progress = None
        if not silent and len(image_paths) > 20:
            progress = QProgressDialog(
                self.tr("Computing statistics..."),
                self.tr("Cancel"),
                0,
                len(image_paths),
                self,
            )
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setStyleSheet(get_progress_dialog_style())
            progress.show()

            def _progress_cb(done, total, _msg):
                if progress.wasCanceled():
                    return
                progress.setValue(done)

            def _cancel_cb():
                return progress.wasCanceled()

        else:

            def _progress_cb(_a, _b, _c):
                return

            def _cancel_cb():
                return False

        try:
            self.stats = compute_full_dataset_stats(
                image_paths,
                output_dir=output_dir,
                tiny_box_px=self.tiny_box_px,
                progress_callback=_progress_cb,
                cancel_callback=_cancel_cb,
            )
        finally:
            if progress is not None:
                progress.close()
        self._render()

    def _render(self) -> None:
        if self.stats is None:
            return
        stats = self.stats
        avg_marks = 0.0
        denom = max(1, stats.total_images)
        avg_marks = float(stats.total_annotations) / float(denom)
        tiny_px = (
            int(self.tiny_box_px)
            if float(self.tiny_box_px).is_integer()
            else self.tiny_box_px
        )
        self.summary_label.setText(
            self.tr(
                "Images: {a} | Annotations: {b} | Avg marks/img: {c:.2f} | "
                "Empty: {d} | Verified BG: {e} | Tiny (<{g}px): {f}"
            ).format(
                a=stats.total_images,
                b=stats.total_annotations,
                c=avg_marks,
                d=stats.empty_images,
                e=stats.verified_empty_images,
                f=stats.tiny_boxes_count,
                g=tiny_px,
            )
        )
        rows = sorted(
            stats.per_class.values(),
            key=lambda s: (-s.count, s.label),
        )
        self.table.setRowCount(len(rows))
        for row_idx, item in enumerate(rows):
            tiny_rate = (
                float(item.tiny_count) / float(item.count)
                if item.count > 0
                else 0.0
            )
            values = [
                item.label,
                str(item.count),
                str(item.images),
                f"{item.marks_per_image:.2f}",
                f"{item.mean_width:.1f}",
                f"{item.stddev_width:.1f}",
                f"{item.mean_height:.1f}",
                f"{item.stddev_height:.1f}",
                f"{item.mean_area:.0f}",
                self.tr("YES") if tiny_rate > self.TINY_WARN_RATE else "",
            ]
            for col_idx, value in enumerate(values):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(value))

    def on_export_csv(self) -> None:
        if not self.stats or not self.stats.per_class:
            return
        default = os.path.join(os.path.expanduser("~"), "dataset_stats.csv")
        file_path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export Statistics CSV"), default, "CSV (*.csv)"
        )
        if not file_path:
            return
        rows = []
        for item in sorted(
            self.stats.per_class.values(), key=lambda s: (-s.count, s.label)
        ):
            tiny_rate = (
                float(item.tiny_count) / float(item.count)
                if item.count > 0
                else 0.0
            )
            rows.append(
                [
                    item.label,
                    item.count,
                    item.images,
                    f"{item.marks_per_image:.2f}",
                    f"{item.mean_width:.1f}",
                    f"{item.stddev_width:.1f}",
                    f"{item.mean_height:.1f}",
                    f"{item.stddev_height:.1f}",
                    f"{item.mean_area:.0f}",
                    "YES" if tiny_rate > self.TINY_WARN_RATE else "",
                ]
            )
        export_rows_to_csv(file_path, HEADERS, rows)
