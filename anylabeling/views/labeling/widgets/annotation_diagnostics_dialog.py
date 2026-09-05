from __future__ import annotations

import os
from typing import List, Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from anylabeling.services.annotation_diagnostics import (
    AnnotationDiagnosticsScanner,
    ScanIssue,
    ScanReport,
    clamp_out_of_bounds_shapes,
    deduplicate_dataset_shapes,
    export_report_to_file,
    move_empty_images,
    purge_micro_shapes,
)
from anylabeling.views.labeling.utils.style import (
    get_dialog_style,
    get_progress_dialog_style,
)
from anylabeling.views.labeling.utils.theme import get_theme


def _get_diagnostics_style() -> str:
    """Returns the combined stylesheet for the Diagnostics dialog."""
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
        .secondary-button:pressed {{
            background-color: {t["surface_pressed"]};
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
        .primary-button:pressed {{
            background-color: {t["primary_pressed"]};
        }}
        .metric-card {{
            background-color: {t["surface"]};
            border: 1px solid {t["border"]};
            border-radius: 8px;
            padding: 8px 12px;
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


class MetricCard(QFrame):
    """Displays a KPI metric card with title and count."""

    def __init__(
        self,
        title: str,
        initial_value: int | str = 0,
        accent_color: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setProperty("class", "metric-card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        t = get_theme()
        val_color = accent_color or t["text"]

        self.val_label = QLabel(str(initial_value))
        self.val_label.setStyleSheet(
            f"font-size: 20px; font-weight: bold; color: {val_color};"
        )
        self.val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            f"font-size: 11px; font-weight: 500; color: {t['text_secondary']};"
        )
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.val_label)
        layout.addWidget(self.title_label)

    def set_value(self, value: int | str) -> None:
        self.val_label.setText(str(value))


class AnnotationDiagnosticsDialog(QDialog):
    """Dialog for scanning dataset annotation health, viewing issues, and batch repairing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label_widget = parent
        self.scanner = AnnotationDiagnosticsScanner()
        self.report = ScanReport()
        self.filtered_issues: List[ScanIssue] = []

        self.init_ui()
        self.run_scan(silent=False)

    def init_ui(self) -> None:
        self.setWindowTitle(
            self.tr("Dataset Annotation Diagnostics & Health Scanner")
        )
        self.resize(1020, 680)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Top metric cards
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(10)
        t = get_theme()

        self.card_images = MetricCard(self.tr("Total Images"), 0, parent=self)
        self.card_annotations = MetricCard(
            self.tr("Annotations"), 0, parent=self
        )
        self.card_duplicates = MetricCard(
            self.tr("Duplicate Shapes"),
            0,
            accent_color=t["warning"],
            parent=self,
        )
        self.card_corrupt = MetricCard(
            self.tr("Corrupt / Bounds"),
            0,
            accent_color=t["error"],
            parent=self,
        )
        self.card_micro = MetricCard(
            self.tr("Micro-Noise"), 0, accent_color=t["warning"], parent=self
        )
        self.card_dup_images = MetricCard(
            self.tr("Duplicate Files"),
            0,
            accent_color=t["highlight"],
            parent=self,
        )

        for card in [
            self.card_images,
            self.card_annotations,
            self.card_duplicates,
            self.card_corrupt,
            self.card_micro,
            self.card_dup_images,
        ]:
            cards_layout.addWidget(card)

        main_layout.addLayout(cards_layout)

        # 2. Filter & Search Row
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(8)

        self.combo_severity = QComboBox(self)
        self.combo_severity.addItems(
            [
                self.tr("All Severities"),
                self.tr("Errors"),
                self.tr("Warnings"),
                self.tr("Info"),
            ]
        )
        self.combo_severity.currentIndexChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.combo_severity)

        self.combo_type = QComboBox(self)
        issue_types = [
            (self.tr("All Issue Types"), ""),
            (self.tr("Duplicate Shape"), "duplicate_shape"),
            (self.tr("Degenerate Geometry"), "degenerate_geometry"),
            (self.tr("Micro-Noise"), "micro_noise"),
            (self.tr("Duplicate Image"), "duplicate_image"),
            (self.tr("Out of Bounds"), "out_of_bounds"),
            (self.tr("Missing Label"), "missing_label"),
            (self.tr("Missing Image"), "missing_image"),
        ]
        for display_name, key in issue_types:
            self.combo_type.addItem(display_name, userData=key)
        self.combo_type.currentIndexChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.combo_type)

        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText(
            self.tr("Filter by filename, label, or details...")
        )
        self.search_input.textChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.search_input, stretch=1)

        self.lbl_count = QLabel(self)
        self.lbl_count.setStyleSheet(f"color: {t['text_secondary']};")
        filter_layout.addWidget(self.lbl_count)

        main_layout.addLayout(filter_layout)

        # 3. Table of issues
        self.table = QTableWidget(self)
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("Severity"),
                self.tr("Issue Type"),
                self.tr("Image File"),
                self.tr("Shape #"),
                self.tr("Label"),
                self.tr("Details"),
                self.tr("Suggestion"),
            ]
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
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
            5, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)
        main_layout.addWidget(self.table, stretch=1)

        # 4. Actions Row
        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)

        self.btn_scan = QPushButton(self.tr("Re-Scan"), self)
        self.btn_scan.setProperty("class", "primary-button")
        self.btn_scan.clicked.connect(lambda: self.run_scan(silent=False))
        action_layout.addWidget(self.btn_scan)

        self.btn_dedup = QPushButton(self.tr("Auto-Deduplicate"), self)
        self.btn_dedup.setProperty("class", "secondary-button")
        self.btn_dedup.clicked.connect(self.on_auto_deduplicate)
        action_layout.addWidget(self.btn_dedup)

        self.btn_purge = QPushButton(self.tr("Purge Micro-Noise"), self)
        self.btn_purge.setProperty("class", "secondary-button")
        self.btn_purge.clicked.connect(self.on_purge_micro_noise)
        action_layout.addWidget(self.btn_purge)

        self.btn_clamp = QPushButton(self.tr("Clamp Out-of-Bounds"), self)
        self.btn_clamp.setProperty("class", "secondary-button")
        self.btn_clamp.clicked.connect(self.on_clamp_out_of_bounds)
        action_layout.addWidget(self.btn_clamp)

        self.btn_move_empty = QPushButton(
            self.tr("Move Empty Images..."), self
        )
        self.btn_move_empty.setProperty("class", "secondary-button")
        self.btn_move_empty.clicked.connect(self.on_move_empty_images)
        action_layout.addWidget(self.btn_move_empty)

        action_layout.addStretch(1)

        self.btn_export = QPushButton(self.tr("Export Report..."), self)
        self.btn_export.setProperty("class", "secondary-button")
        self.btn_export.clicked.connect(self.on_export_report)
        action_layout.addWidget(self.btn_export)

        self.btn_close = QPushButton(self.tr("Close"), self)
        self.btn_close.setProperty("class", "secondary-button")
        self.btn_close.clicked.connect(self.accept)
        action_layout.addWidget(self.btn_close)

        main_layout.addLayout(action_layout)
        self.setStyleSheet(_get_diagnostics_style())

    def get_image_file_list(self) -> List[str]:
        """Fetch current image paths from parent file list."""
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

    def run_scan(self, silent: bool = True) -> None:
        """Execute full dataset diagnostic scan."""
        image_paths = self.get_image_file_list()
        if not image_paths:
            if not silent:
                QMessageBox.information(
                    self,
                    self.tr("No Images"),
                    self.tr("No image files loaded in project."),
                )
            return

        progress = QProgressDialog(
            self.tr("Analyzing dataset annotations..."),
            self.tr("Cancel"),
            0,
            len(image_paths),
            self,
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle(self.tr("Diagnostics Progress"))
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
            self.report = self.scanner.scan(
                image_paths=image_paths,
                output_dir=self.get_output_dir(),
                progress_callback=progress_cb,
                cancel_callback=cancel_cb,
            )
        finally:
            progress.close()

        # Update KPI cards
        self.card_images.set_value(self.report.total_images)
        self.card_annotations.set_value(self.report.total_annotations)
        self.card_duplicates.set_value(self.report.duplicate_shapes_count)
        self.card_corrupt.set_value(
            self.report.corrupt_shapes_count + self.report.out_of_bounds_count
        )
        self.card_micro.set_value(self.report.micro_shapes_count)
        self.card_dup_images.set_value(self.report.duplicate_images_count)

        self.apply_filter()

    def _matches_filter(
        self, issue: ScanIssue, sev_sel: str, type_sel: str, search_txt: str
    ) -> bool:
        """Check if an issue matches the selected filter criteria."""
        if sev_sel == "Errors" and issue.severity != "error":
            return False
        if sev_sel == "Warnings" and issue.severity != "warning":
            return False
        if sev_sel == "Info" and issue.severity != "info":
            return False

        if type_sel and issue.issue_type != type_sel:
            return False

        if search_txt:
            tokens = [
                os.path.basename(issue.image_path).lower(),
                issue.details.lower(),
                issue.suggestion.lower(),
            ]
            if isinstance(issue.shape_data, dict):
                try:
                    tokens.append(
                        str(issue.shape_data.get("label", "")).lower()
                    )
                except Exception:
                    pass
            if not any(search_txt in tok for tok in tokens):
                return False

        return True

    def apply_filter(self) -> None:
        """Filter issues by severity, type, and search keyword."""
        sev_idx = self.combo_severity.currentIndex()
        sev_sel = ["All", "Errors", "Warnings", "Info"][sev_idx]
        type_sel = self.combo_type.currentData() or ""
        search_txt = self.search_input.text().strip().lower()

        self.filtered_issues = [
            iss
            for iss in self.report.issues
            if self._matches_filter(iss, sev_sel, type_sel, search_txt)
        ]

        self.lbl_count.setText(
            self.tr("Showing %d of %d issues")
            % (len(self.filtered_issues), len(self.report.issues))
        )
        self._populate_table()

    def _populate_table(self) -> None:
        """Populate QTableWidget with filtered issues."""
        self.table.setRowCount(len(self.filtered_issues))
        t = get_theme()

        for row_idx, issue in enumerate(self.filtered_issues):
            # Severity item
            sev_item = QTableWidgetItem(issue.severity.upper())
            if issue.severity == "error":
                sev_item.setForeground(QColor(t["error"]))
            elif issue.severity == "warning":
                sev_item.setForeground(QColor(t["warning"]))
            else:
                sev_item.setForeground(QColor(t["text_secondary"]))
            self.table.setItem(row_idx, 0, sev_item)

            # Issue type
            display_type = issue.issue_type.replace("_", " ").title()
            self.table.setItem(row_idx, 1, QTableWidgetItem(display_type))

            # Image filename
            base_img = os.path.basename(issue.image_path)
            self.table.setItem(row_idx, 2, QTableWidgetItem(base_img))

            # Shape index
            shape_idx_str = (
                str(issue.shape_index + 1)
                if issue.shape_index is not None
                else "-"
            )
            self.table.setItem(row_idx, 3, QTableWidgetItem(shape_idx_str))

            # Class label
            if isinstance(issue.shape_data, dict):
                try:
                    lbl_str = str(issue.shape_data.get("label", "-"))
                except Exception:
                    lbl_str = "-"
            else:
                lbl_str = "-"
            self.table.setItem(row_idx, 4, QTableWidgetItem(lbl_str))

            # Details & Suggestion
            self.table.setItem(row_idx, 5, QTableWidgetItem(issue.details))
            self.table.setItem(row_idx, 6, QTableWidgetItem(issue.suggestion))

    def _safe_reload_current_file(self) -> None:
        """Reload current file only if clean, else skip to avoid data loss."""
        w = self._label_widget
        if not w or not getattr(w, "filename", None):
            return
        try:
            if hasattr(w, "may_continue") and not w.may_continue():
                return
        except Exception:
            return
        try:
            w.load_file(w.filename)
        except Exception:
            pass

    def on_cell_double_clicked(self, row: int, _col: int) -> None:
        """Double click row to navigate to image and highlight shape."""
        if (
            not self._label_widget
            or row < 0
            or row >= len(self.filtered_issues)
            or not hasattr(self._label_widget, "load_file")
        ):
            return

        issue = self.filtered_issues[row]
        if not os.path.isfile(issue.image_path):
            return

        # Revalidate stale index: match label/points if possible
        self._label_widget.load_file(issue.image_path)
        if issue.shape_index is not None and hasattr(
            self._label_widget, "canvas"
        ):
            shapes = getattr(self._label_widget.canvas, "shapes", [])
            if 0 <= issue.shape_index < len(shapes):
                candidate = shapes[issue.shape_index]
                try:
                    if isinstance(issue.shape_data, dict):
                        exp_label = issue.shape_data.get("label")
                        cand_label = getattr(candidate, "label", None)
                        if exp_label is not None and cand_label != exp_label:
                            # Label mismatch — likely stale, still select by index
                            # but do not fail; user can verify visually.
                            pass
                except Exception:
                    pass
                self._label_widget.canvas.select_shapes(
                    [shapes[issue.shape_index]]
                )

    def on_auto_deduplicate(self) -> None:
        """Batch remove duplicate shapes across dataset."""
        if self.report.duplicate_shapes_count == 0:
            QMessageBox.information(
                self,
                self.tr("No Duplicates"),
                self.tr("No duplicate shapes detected in current dataset."),
            )
            return

        ans = QMessageBox.question(
            self,
            self.tr("Auto-Deduplicate Shapes"),
            self.tr(
                "Remove same-label duplicates only (cross-class overlaps are kept for review)?\n"
                "A timestamped backup (.backup_*) will be created before modifying any files."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        image_paths = self.get_image_file_list()
        removed = deduplicate_dataset_shapes(
            image_paths=image_paths,
            output_dir=self.get_output_dir(),
            iou_threshold=self.scanner.iou_duplicate_threshold,
            containment_threshold=self.scanner.containment_duplicate_threshold,
            backup=True,
            same_label_only=True,
        )
        QMessageBox.information(
            self,
            self.tr("Deduplication Complete"),
            self.tr("Successfully removed %d duplicate shape(s).") % removed,
        )

        self._safe_reload_current_file()
        self.run_scan(silent=True)

    def on_purge_micro_noise(self) -> None:
        """Batch purge micro-noise shapes."""
        if self.report.micro_shapes_count == 0:
            QMessageBox.information(
                self,
                self.tr("No Micro-Noise"),
                self.tr("No micro-noise shapes detected in current dataset."),
            )
            return

        ans = QMessageBox.question(
            self,
            self.tr("Purge Micro-Noise"),
            self.tr(
                "Are you sure you want to purge all micro-noise shapes (< %.1fpx or < %.1fpx²)?\n"
                "A timestamped backup (.backup_*) will be created before modifying any files."
            )
            % (self.scanner.min_size_px, self.scanner.min_area_px),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        image_paths = self.get_image_file_list()
        purged = purge_micro_shapes(
            image_paths=image_paths,
            output_dir=self.get_output_dir(),
            min_size_px=self.scanner.min_size_px,
            min_area_px=self.scanner.min_area_px,
            backup=True,
        )
        QMessageBox.information(
            self,
            self.tr("Purge Complete"),
            self.tr("Successfully purged %d micro-noise shape(s).") % purged,
        )

        self._safe_reload_current_file()
        self.run_scan(silent=True)

    def on_clamp_out_of_bounds(self) -> None:
        """Batch repair to clamp all out-of-bounds shapes to image boundaries."""
        if self.report.out_of_bounds_count == 0:
            QMessageBox.information(
                self,
                self.tr("No Out-of-Bounds Shapes"),
                self.tr(
                    "No out-of-bounds shapes detected in current dataset."
                ),
            )
            return

        ans = QMessageBox.question(
            self,
            self.tr("Clamp Out-of-Bounds Shapes"),
            self.tr(
                "Are you sure you want to clamp all shapes extending outside image boundaries to image dimensions?\n"
                "A timestamped backup (.backup_*) will be created. Fully-outside shapes will be deleted."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        image_paths = self.get_image_file_list()
        clamped, deleted = clamp_out_of_bounds_shapes(
            image_paths=image_paths,
            output_dir=self.get_output_dir(),
            backup=True,
        )
        QMessageBox.information(
            self,
            self.tr("Clamp Complete"),
            self.tr("Clamped %d shape(s), deleted %d fully-outside shape(s).")
            % (clamped, deleted),
        )

        self._safe_reload_current_file()
        self.run_scan(silent=True)

    def on_move_empty_images(self) -> None:
        """Move 0-annotation images to destination directory."""
        if self.report.empty_images == 0:
            QMessageBox.information(
                self,
                self.tr("No Empty Images"),
                self.tr(
                    "No unannotated/empty images detected in current dataset."
                ),
            )
            return

        dest_dir = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Destination Directory for Empty Images"),
            "",
        )
        if not dest_dir:
            return

        image_paths = self.get_image_file_list()
        moved = move_empty_images(
            image_paths=image_paths,
            destination_dir=dest_dir,
            output_dir=self.get_output_dir(),
        )
        QMessageBox.information(
            self,
            self.tr("Move Complete"),
            self.tr(
                "Successfully moved %d empty image(s) to:\n%s\n\n"
                "Please reopen the image folder to refresh the file list."
            )
            % (moved, dest_dir),
        )

        self.run_scan(silent=True)

    def on_export_report(self) -> None:
        """Export report to JSON or CSV."""
        if not self.report.issues:
            QMessageBox.information(
                self,
                self.tr("No Issues"),
                self.tr("No issues to export."),
            )
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            self.tr("Export Diagnostics Report"),
            "diagnostics_report.json",
            "JSON Report (*.json);;CSV Report (*.csv)",
        )
        if not file_path:
            return

        export_fmt = (
            "csv"
            if "csv" in selected_filter.lower() or file_path.endswith(".csv")
            else "json"
        )
        try:
            export_report_to_file(
                self.report, file_path, export_format=export_fmt
            )
            QMessageBox.information(
                self,
                self.tr("Export Succeeded"),
                self.tr("Report exported successfully to:\n%s") % file_path,
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                self.tr("Export Failed"),
                self.tr("Failed to export report: %s") % str(exc),
            )
