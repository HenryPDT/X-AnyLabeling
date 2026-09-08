from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets
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
    QMenu,
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
    apply_shape_modifications,
    clamp_out_of_bounds_shapes,
    deduplicate_dataset_shapes,
    delete_duplicate_images,
    export_report_to_file,
    get_label_file_path,
    move_empty_images,
    purge_micro_shapes,
)
from anylabeling.services.dataset_files import soft_delete_dataset_images
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

        self.value = initial_value
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
        self.value = value
        self.val_label.setText(str(value))


class AnnotationDiagnosticsDialog(QDialog):
    """Dialog for scanning dataset annotation health, viewing issues, and batch repairing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label_widget = parent
        self.scanner = AnnotationDiagnosticsScanner()
        self.report = ScanReport()
        self.filtered_issues: List[ScanIssue] = []
        self._sort_col: Optional[int] = None
        self._sort_ascending: bool = True

        self.init_ui()
        self.run_scan(silent=True)

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
        # Debounce typing like the review gallery (was per-keystroke freeze).
        self.search_timer = QtCore.QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(200)
        self.search_timer.timeout.connect(self.apply_filter)
        self.search_input.textChanged.connect(self.search_timer.start)
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
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
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
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.horizontalHeader().sectionClicked.connect(
            self.on_header_clicked
        )
        self.table.horizontalHeader().setToolTip(
            self.tr("Click a column header to sort; click again to reverse")
        )
        self.table.setToolTip(
            self.tr("Right-click for Jump / Gallery / Delete (Del)")
        )
        self.table.installEventFilter(self)
        self.table.viewport().installEventFilter(self)
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

    def get_dataset_root(self) -> Optional[str]:
        if self._label_widget and getattr(
            self._label_widget, "last_open_dir", None
        ):
            return self._label_widget.last_open_dir
        return None

    def _refresh_main_window_file_list(self, deleted_paths: Set[str]) -> None:
        """Reload sidebar/file list after image files are removed from disk."""
        w = self._label_widget
        if not w:
            return
        fallback = None
        if deleted_paths:
            fallback = os.path.dirname(next(iter(deleted_paths)))
        try:
            w.reload_after_image_paths_removed(
                deleted_paths, fallback_dir=fallback
            )
        except Exception:
            pass

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

    def on_header_clicked(self, col: int) -> None:
        """Sort issues table by clicked column header."""
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

        self.filtered_issues = self._sort_issues_by_column(
            self.filtered_issues, self._sort_col, self._sort_ascending
        )
        self._populate_table()

    def _sort_issues_by_column(
        self, issues: List[ScanIssue], col: int, ascending: bool
    ) -> List[ScanIssue]:
        rev = not ascending
        if col == 0:
            rank = {"error": 0, "warning": 1, "info": 2}
            return sorted(
                issues,
                key=lambda x: rank.get((x.severity or "").lower(), 99),
                reverse=rev,
            )
        if col == 1:
            return sorted(
                issues,
                key=lambda x: (x.issue_type or "").lower(),
                reverse=rev,
            )
        if col == 2:
            return sorted(
                issues,
                key=lambda x: os.path.basename(x.image_path or "").lower(),
                reverse=rev,
            )
        if col == 3:
            return sorted(
                issues,
                key=lambda x: (
                    x.shape_index if x.shape_index is not None else -1
                ),
                reverse=rev,
            )
        if col == 4:

            def _get_lbl(x: ScanIssue) -> str:
                if isinstance(x.shape_data, dict):
                    return str(x.shape_data.get("label", "")).lower()
                return ""

            return sorted(issues, key=_get_lbl, reverse=rev)
        if col == 5:
            return sorted(
                issues, key=lambda x: (x.details or "").lower(), reverse=rev
            )
        if col == 6:
            return sorted(
                issues, key=lambda x: (x.suggestion or "").lower(), reverse=rev
            )
        return issues

    def apply_filter(self) -> None:
        """Filter issues by severity, type, and search keyword."""
        if hasattr(self, "search_timer") and self.search_timer.isActive():
            self.search_timer.stop()
        sev_idx = self.combo_severity.currentIndex()
        sev_sel = ["All", "Errors", "Warnings", "Info"][sev_idx]
        type_sel = self.combo_type.currentData() or ""
        search_txt = self.search_input.text().strip().lower()

        filtered = [
            iss
            for iss in self.report.issues
            if self._matches_filter(iss, sev_sel, type_sel, search_txt)
        ]

        if self._sort_col is not None:
            self.filtered_issues = self._sort_issues_by_column(
                filtered, self._sort_col, self._sort_ascending
            )
        else:
            self.filtered_issues = filtered

        self.lbl_count.setText(
            self.tr("Showing %d of %d issues")
            % (len(self.filtered_issues), len(self.report.issues))
        )
        self._populate_table()

    def _populate_table(self) -> None:
        """Populate QTableWidget with filtered issues."""
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(len(self.filtered_issues))
            t = get_theme()

            for row_idx, issue in enumerate(self.filtered_issues):
                # Severity item
                sev = (issue.severity or "info").lower()
                sev_item = QTableWidgetItem(sev.upper())
                if sev == "error":
                    sev_item.setForeground(QColor(t["error"]))
                elif sev == "warning":
                    sev_item.setForeground(QColor(t["warning"]))
                else:
                    sev_item.setForeground(QColor(t["text_secondary"]))
                self.table.setItem(row_idx, 0, sev_item)

                # Issue type
                display_type = (
                    (issue.issue_type or "").replace("_", " ").title()
                )
                self.table.setItem(row_idx, 1, QTableWidgetItem(display_type))

                # Image filename
                base_img = os.path.basename(issue.image_path or "")
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
                self.table.setItem(
                    row_idx, 5, QTableWidgetItem(issue.details or "")
                )
                self.table.setItem(
                    row_idx, 6, QTableWidgetItem(issue.suggestion or "")
                )
        finally:
            self.table.setUpdatesEnabled(True)

        if len(self.filtered_issues) > 0 and self.table.currentRow() < 0:
            self.table.selectRow(0)

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

    def _abort_if_main_dirty(self, touched: Set[str]) -> bool:
        """Abort a disk write that would clobber unsaved main-window edits."""
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

        # Do not clobber unsaved main-window edits with a navigation load.
        try:
            w = self._label_widget
            if (
                getattr(w, "filename", None) != issue.image_path
                and getattr(w, "dirty", False)
                and hasattr(w, "may_continue")
                and not w.may_continue()
            ):
                return
        except Exception:
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

    def _get_selected_rows(self) -> List[int]:
        """Return sorted unique row indices currently selected in the table.

        Only real selections count: falling back to the current row would
        delete row 0 after an explicit deselect.
        """
        rows = set()
        selection_model = self.table.selectionModel()
        if selection_model:
            for idx in selection_model.selectedRows():
                rows.add(idx.row())
            if not rows:
                for idx in selection_model.selectedIndexes():
                    rows.add(idx.row())
        return sorted([r for r in rows if 0 <= r < len(self.filtered_issues)])

    def _get_selected_issues(self) -> List[ScanIssue]:
        """Return list of ScanIssue objects corresponding to selected table rows."""
        rows = self._get_selected_rows()
        return [self.filtered_issues[r] for r in rows]

    def show_context_menu(self, pos: QtCore.QPoint) -> None:
        """Display right-click context menu for table row."""
        row = self.table.rowAt(pos.y())
        if row < 0 or row >= len(self.filtered_issues):
            return

        selected_rows = self._get_selected_rows()
        if row not in selected_rows:
            selected_rows = [row]

        selected_issues = [self.filtered_issues[r] for r in selected_rows]
        deletable_issues = [
            iss for iss in selected_issues if iss.shape_index is not None
        ]
        image_deletable_issues = [
            iss
            for iss in selected_issues
            if iss.issue_type == "duplicate_image"
        ]
        unique_image_paths: List[str] = []
        seen_images: Set[str] = set()
        for iss in image_deletable_issues:
            if iss.image_path and iss.image_path not in seen_images:
                seen_images.add(iss.image_path)
                unique_image_paths.append(iss.image_path)

        menu = QMenu(self)
        act_jump = menu.addAction(self.tr("Jump to Annotation"))
        act_gallery = menu.addAction(self.tr("Review in Gallery..."))

        act_delete = None
        act_delete_image = None
        if deletable_issues or unique_image_paths:
            menu.addSeparator()
        if deletable_issues:
            if len(deletable_issues) == 1:
                act_delete = menu.addAction(self.tr("Delete Shape (Del)"))
            else:
                act_delete = menu.addAction(
                    self.tr("Delete %d Selected Shapes (Del)")
                    % len(deletable_issues)
                )
        if unique_image_paths:
            if len(unique_image_paths) == 1:
                act_delete_image = menu.addAction(self.tr("Delete Image File"))
            else:
                act_delete_image = menu.addAction(
                    self.tr("Delete %d Selected Images")
                    % len(unique_image_paths)
                )

        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen == act_jump:
            self.on_cell_double_clicked(row, 0)
        elif chosen == act_gallery:
            self.open_review_gallery_for_issue(self.filtered_issues[row])
        elif act_delete and chosen == act_delete:
            self.delete_shapes_for_issues(deletable_issues)
        elif act_delete_image and chosen == act_delete_image:
            self.delete_images_for_issues(image_deletable_issues)

    def delete_shape_for_issue(self, issue: ScanIssue) -> None:
        """Remove a single shape associated with this diagnostic issue."""
        self.delete_shapes_for_issues([issue])

    def _confirm_shape_deletion(self, deletable: List[ScanIssue]) -> bool:
        """Confirm deletion; returns True when the user accepts."""
        if len(deletable) == 1:
            iss = deletable[0]
            msg = self.tr(
                "Are you sure you want to delete shape #%d from '%s'?"
            ) % (
                (iss.shape_index or 0) + 1,
                os.path.basename(iss.image_path or ""),
            )
        else:
            files_cnt = len({iss.image_path for iss in deletable})
            msg = self.tr(
                "Are you sure you want to delete %d shapes across %d file(s)?"
            ) % (len(deletable), files_cnt)

        ans = QMessageBox.question(
            self,
            self.tr("Delete Shape(s)"),
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return ans == QMessageBox.StandardButton.Yes

    def _group_deletable_by_file(
        self, deletable: List[ScanIssue]
    ) -> Tuple[
        Dict[str, Set[int]], Dict[str, Dict[int, str]], List[ScanIssue]
    ]:
        """Group deletable issues by label file with stale-guard labels."""
        by_file: Dict[str, Set[int]] = {}
        expected_by_file: Dict[str, Dict[int, str]] = {}
        skipped: List[ScanIssue] = []
        output_dir = self.get_output_dir()
        for iss in deletable:
            label_file = iss.label_file or get_label_file_path(
                iss.image_path, output_dir=output_dir
            )
            if label_file and os.path.isfile(label_file):
                by_file.setdefault(label_file, set()).add(iss.shape_index)
                # Stale guard: verify on-disk label still matches scan time.
                if isinstance(iss.shape_data, dict):
                    exp = iss.shape_data.get("label")
                    if isinstance(exp, str) and exp:
                        expected_by_file.setdefault(label_file, {})[
                            iss.shape_index
                        ] = exp
            else:
                skipped.append(iss)
        return by_file, expected_by_file, skipped

    def _write_deletions(
        self,
        by_file: Dict[str, Set[int]],
        expected_by_file: Dict[str, Dict[int, str]],
    ) -> Tuple[Set[str], List[str]]:
        """Apply one batched delete per label file; return (succeeded, failed)."""
        succeeded: Set[str] = set()
        failed: List[str] = []
        for lf, del_indices in by_file.items():
            ok = apply_shape_modifications(
                label_file=lf,
                deletions=del_indices,
                reclasses={},
                expected_labels=expected_by_file.get(lf) or None,
            )
            if ok:
                succeeded.add(lf)
            else:
                failed.append(lf)
        return succeeded, failed

    def _report_deletion_outcome(
        self, succeeded: Set[str], failed: List[str], skipped: List[ScanIssue]
    ) -> bool:
        """Show result boxes; returns False when nothing was deleted."""
        if skipped and not succeeded and not failed:
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr(
                    "No matching annotation files found on disk; nothing was deleted."
                ),
            )
            return False

        if failed and not succeeded:
            QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr(
                    "Failed to delete shapes from file(s): %s. "
                    "Nothing was removed from the report (files may be stale or locked)."
                )
                % ", ".join(os.path.basename(p) for p in failed),
            )
            return False

        if failed or skipped:
            skipped_files = len({s.image_path for s in skipped})
            QMessageBox.warning(
                self,
                self.tr("Partial Success"),
                self.tr(
                    "Deleted shapes from %d file(s); %d file(s) failed and "
                    "%d issue(s) in %d file(s) were missing and kept in the report."
                )
                % (
                    len(succeeded),
                    len(failed),
                    len(skipped),
                    skipped_files,
                ),
            )
        return True

    def _remove_report_issues(
        self, by_file: Dict[str, Set[int]], succeeded: Set[str]
    ) -> List[ScanIssue]:
        """Drop removed issues from the report; shift survivor indices.

        Matches by (label_file, shape_index) key so sibling issues for the
        same shape (e.g. micro + duplicate) don't orphan. Returns all
        removed issues for KPI updates.
        """
        removed_keys = set()
        for lf in succeeded:
            for d in by_file[lf]:
                removed_keys.add((lf, d))
        output_dir = self.get_output_dir()
        lf_cache: Dict[int, Optional[str]] = {}

        def _lf_of(iss: ScanIssue) -> Optional[str]:
            k = id(iss)
            if k not in lf_cache:
                lf_cache[k] = iss.label_file or get_label_file_path(
                    iss.image_path, output_dir=output_dir
                )
            return lf_cache[k]

        removed_issues = [
            iss
            for iss in self.report.issues
            if (_lf_of(iss), iss.shape_index) in removed_keys
        ]
        removed_ids = {id(iss) for iss in removed_issues}
        self.report.issues = [
            iss for iss in self.report.issues if id(iss) not in removed_ids
        ]

        for lf in succeeded:
            sorted_dels = sorted(by_file[lf])
            for iss in self.report.issues:
                if iss.shape_index is None:
                    continue
                if _lf_of(iss) != lf:
                    continue
                shift = sum(1 for d in sorted_dels if d < iss.shape_index)
                if shift:
                    iss.shape_index -= shift
        return removed_issues

    def _refresh_issue_cards(
        self, unique_removed: int, removed_issues: List[ScanIssue]
    ) -> None:
        """Update KPI cards: total by unique shapes, per-type by issues."""
        self.report.total_annotations = max(
            0, self.report.total_annotations - unique_removed
        )
        for iss in removed_issues:
            if iss.issue_type == "duplicate_shape":
                self.report.duplicate_shapes_count = max(
                    0, self.report.duplicate_shapes_count - 1
                )
            elif iss.issue_type == "micro_noise":
                self.report.micro_shapes_count = max(
                    0, self.report.micro_shapes_count - 1
                )
            elif iss.issue_type == "out_of_bounds":
                self.report.out_of_bounds_count = max(
                    0, self.report.out_of_bounds_count - 1
                )
            elif iss.issue_type in ("degenerate_geometry", "missing_label"):
                self.report.corrupt_shapes_count = max(
                    0, self.report.corrupt_shapes_count - 1
                )

        self.card_annotations.set_value(self.report.total_annotations)
        self.card_duplicates.set_value(self.report.duplicate_shapes_count)
        self.card_corrupt.set_value(
            self.report.corrupt_shapes_count + self.report.out_of_bounds_count
        )
        self.card_micro.set_value(self.report.micro_shapes_count)

    def delete_shapes_for_issues(self, issues: List[ScanIssue]) -> None:
        """Remove shapes associated with diagnostic issues directly."""
        deletable = [iss for iss in issues if iss.shape_index is not None]
        if not deletable:
            QMessageBox.information(
                self,
                self.tr("Cannot Delete"),
                self.tr(
                    "The selected diagnostic issue(s) are not associated with specific shape indices."
                ),
            )
            return

        if not self._confirm_shape_deletion(deletable):
            return

        if self._abort_if_main_dirty({iss.image_path for iss in deletable}):
            return

        by_file, expected_by_file, skipped = self._group_deletable_by_file(
            deletable
        )
        succeeded, failed = self._write_deletions(by_file, expected_by_file)
        if not self._report_deletion_outcome(succeeded, failed, skipped):
            return

        removed_issues = self._remove_report_issues(by_file, succeeded)
        unique_removed = len(
            {(lf, d) for lf in succeeded for d in by_file[lf]}
        )
        self._refresh_issue_cards(unique_removed, removed_issues)

        self._safe_reload_current_file()
        self.apply_filter()

    def _confirm_image_deletion(self, image_paths: List[str]) -> bool:
        """Confirm image removal; returns True when accepted."""
        if len(image_paths) == 1:
            msg = self.tr(
                "Move '%s' to the _delete_ folder and permanently delete its "
                "annotation JSON file?"
            ) % os.path.basename(image_paths[0])
        else:
            msg = self.tr(
                "Move %d image(s) to the _delete_ folder and permanently "
                "delete their annotation JSON files?"
            ) % len(image_paths)

        ans = QMessageBox.question(
            self,
            self.tr("Delete Image File(s)"),
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return ans == QMessageBox.StandardButton.Yes

    def _remove_report_issues_for_images(
        self, deleted_paths: Set[str]
    ) -> None:
        """Drop all issues for deleted image paths and refresh KPI cards."""
        if not deleted_paths:
            return

        removed_issues = [
            iss
            for iss in self.report.issues
            if iss.image_path in deleted_paths
        ]
        removed_ids = {id(iss) for iss in removed_issues}
        self.report.issues = [
            iss for iss in self.report.issues if id(iss) not in removed_ids
        ]

        dup_images_removed = sum(
            1 for iss in removed_issues if iss.issue_type == "duplicate_image"
        )
        self.report.duplicate_images_count = max(
            0, self.report.duplicate_images_count - dup_images_removed
        )

        shape_issues_removed = [
            iss for iss in removed_issues if iss.shape_index is not None
        ]
        if shape_issues_removed:
            self._refresh_issue_cards(
                len(shape_issues_removed), shape_issues_removed
            )

        self.card_dup_images.set_value(self.report.duplicate_images_count)

    def delete_images_for_issues(self, issues: List[ScanIssue]) -> None:
        """Remove duplicate-image files (soft-delete) and their JSON labels."""
        image_paths: List[str] = []
        seen: Set[str] = set()
        for iss in issues:
            if iss.issue_type != "duplicate_image" or not iss.image_path:
                continue
            if iss.image_path not in seen:
                seen.add(iss.image_path)
                image_paths.append(iss.image_path)

        if not image_paths:
            QMessageBox.information(
                self,
                self.tr("Cannot Delete"),
                self.tr(
                    "The selected diagnostic issue(s) are not associated "
                    "with deletable duplicate image files."
                ),
            )
            return

        if not self._confirm_image_deletion(image_paths):
            return

        if self._abort_if_main_dirty(set(image_paths)):
            return

        deleted_count, failed = soft_delete_dataset_images(
            image_paths=image_paths,
            output_dir=self.get_output_dir(),
            dataset_root=self.get_dataset_root(),
        )
        if deleted_count == 0:
            QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr("Failed to delete image file(s): %s")
                % ", ".join(os.path.basename(p) for p in failed),
            )
            return

        deleted_paths = {
            path for path in image_paths if path not in set(failed)
        }
        self._remove_report_issues_for_images(deleted_paths)

        if failed:
            QMessageBox.warning(
                self,
                self.tr("Partial Success"),
                self.tr(
                    "Removed %d image(s); failed to remove %d image(s): %s"
                )
                % (
                    deleted_count,
                    len(failed),
                    ", ".join(os.path.basename(p) for p in failed),
                ),
            )
        else:
            QMessageBox.information(
                self,
                self.tr("Delete Complete"),
                self.tr(
                    "Successfully removed %d image(s) to the _delete_ folder."
                )
                % deleted_count,
            )

        self._refresh_main_window_file_list(deleted_paths)
        self.apply_filter()

    def _handle_shortcut_key(self, event: QtGui.QKeyEvent) -> bool:
        """Handle keyboard shortcuts in diagnostics table."""
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
        # Destructive shortcut: only fire from the table itself, never from
        # focused buttons (Close/Re-Scan) or other controls.
        if fw is not None and fw not in (
            self,
            self.table,
            self.table.viewport(),
        ):
            return False

        key = event.key()
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            issues = self._get_selected_issues()
            deletable = [iss for iss in issues if iss.shape_index is not None]
            if deletable:
                self.delete_shapes_for_issues(deletable)
                return True
            return False

        return False

    def eventFilter(
        self, watched: QtCore.QObject, event: QtCore.QEvent
    ) -> bool:
        """Intercept key events from table and viewport."""
        if event.type() == QtCore.QEvent.Type.KeyPress and isinstance(
            event, QtGui.QKeyEvent
        ):
            if self._handle_shortcut_key(event):
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle keyboard shortcuts in diagnostics table."""
        if self._handle_shortcut_key(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def open_review_gallery_for_issue(self, issue: ScanIssue) -> None:
        """Open AnnotationReviewDialog focused on this issue's image and shape."""
        from anylabeling.views.labeling.widgets.annotation_review_dialog import (
            AnnotationReviewDialog,
        )

        dialog = AnnotationReviewDialog(parent=self._label_widget)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        base_img = os.path.basename(issue.image_path)
        dialog.search_input.setText(base_img)
        dialog.apply_filters()
        if issue.shape_index is not None:
            for r_idx, item in enumerate(dialog.filtered_items):
                if (
                    item.image_path == issue.image_path
                    and item.shape_index == issue.shape_index
                ):
                    dialog.table.selectRow(r_idx)
                    break
        dialog.exec()

    def on_auto_deduplicate(self) -> None:
        """Batch remove duplicate shapes and duplicate image files."""
        shape_dups = self.report.duplicate_shapes_count
        image_dups = self.report.duplicate_images_count
        if shape_dups == 0 and image_dups == 0:
            QMessageBox.information(
                self,
                self.tr("No Duplicates"),
                self.tr(
                    "No duplicate shapes or duplicate image files detected "
                    "in current dataset."
                ),
            )
            return

        parts: List[str] = []
        if shape_dups > 0:
            parts.append(
                self.tr(
                    "Remove same-label duplicate shapes (cross-class overlaps "
                    "are kept for review)"
                )
            )
        if image_dups > 0:
            parts.append(
                self.tr(
                    "Move exact duplicate image files to the _delete_ folder "
                    "and permanently delete their JSON labels (first "
                    "occurrence is kept)"
                )
            )
        confirm_msg = self.tr(
            "Proceed with the following?\n\n- "
        ) + "\n- ".join(parts)

        ans = QMessageBox.question(
            self,
            self.tr("Auto-Deduplicate"),
            confirm_msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        image_paths = self.get_image_file_list()
        output_dir = self.get_output_dir()
        removed_shapes = 0
        removed_images = 0
        removed_image_paths: List[str] = []

        if shape_dups > 0:
            removed_shapes = deduplicate_dataset_shapes(
                image_paths=image_paths,
                output_dir=output_dir,
                iou_threshold=self.scanner.iou_duplicate_threshold,
                containment_threshold=self.scanner.containment_duplicate_threshold,
                same_label_only=True,
            )
        if image_dups > 0:
            removed_images, removed_image_paths = delete_duplicate_images(
                image_paths=image_paths,
                output_dir=output_dir,
                dataset_root=self.get_dataset_root(),
            )

        result_parts: List[str] = []
        if shape_dups > 0:
            result_parts.append(
                self.tr("Removed %d duplicate shape(s).") % removed_shapes
            )
        if image_dups > 0:
            result_parts.append(
                self.tr("Removed %d duplicate image file(s) to _delete_.")
                % removed_images
            )
        result_msg = "\n".join(result_parts)

        QMessageBox.information(
            self,
            self.tr("Deduplication Complete"),
            result_msg,
        )

        if removed_images > 0:
            self._refresh_main_window_file_list(set(removed_image_paths))
        else:
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
                "Are you sure you want to purge all micro-noise shapes (< %.1fpx or < %.1fpx²)?"
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
                "Fully-outside shapes will be deleted."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        image_paths = self.get_image_file_list()
        clamped, deleted = clamp_out_of_bounds_shapes(
            image_paths=image_paths,
            output_dir=self.get_output_dir(),
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
