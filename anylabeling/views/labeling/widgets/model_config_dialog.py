import os
import re
from typing import List, Optional

import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from anylabeling.config import get_work_directory
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_lineedit_style,
    get_normal_button_style,
    get_ok_btn_style,
)
from anylabeling.views.labeling.utils.theme import get_theme


def load_classes_from_file(file_path: Optional[str]) -> List[str]:
    """Read line-separated class names from a text file.

    Strips whitespace and ignores empty lines or '#' comments.
    """
    if not file_path or not os.path.isfile(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
    except Exception as exc:
        logger.warning(f"Failed to read classes from {file_path}: {exc}")
        return []


class ModelConfigDialog(QDialog):
    """Dialog to configure model weights and class file for custom models."""

    def __init__(
        self,
        parent=None,
        config_file: Optional[str] = None,
        model_path: Optional[str] = None,
        labelfile_path: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        super().__init__(parent)
        self.parent_widget = parent
        self.config_file = config_file
        if config_file and os.path.isfile(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                model_path = model_path or cfg.get("model_path")
                labelfile_path = labelfile_path or cfg.get("labelfile_path")
                display_name = display_name or cfg.get("display_name")
            except Exception as e:
                logger.warning(
                    f"Could not load config from {config_file}: {e}"
                )

        self.initial_model_path = model_path or ""
        self.initial_labelfile_path = labelfile_path or ""
        self.initial_display_name = display_name or ""

        self.setWindowTitle(self.tr("Configure Model"))
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        self.setMinimumWidth(540)
        self.setStyleSheet(get_dialog_style())

        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        form_layout.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        # 1. Model File Row
        model_container = QHBoxLayout()
        self.model_path_input = QLineEdit(self.initial_model_path)
        self.model_path_input.setStyleSheet(get_lineedit_style())
        self.model_path_input.setPlaceholderText(
            self.tr("Path to model weights file (*.onnx)")
        )
        self.model_browse_btn = QPushButton(self.tr("Browse..."))
        self.model_browse_btn.setStyleSheet(get_normal_button_style())
        self.model_browse_btn.clicked.connect(self.on_browse_model)
        model_container.addWidget(self.model_path_input)
        model_container.addWidget(self.model_browse_btn)
        form_layout.addRow(self.tr("Model File:"), model_container)

        # 2. Class File Row
        class_container = QHBoxLayout()
        self.class_path_input = QLineEdit(self.initial_labelfile_path)
        self.class_path_input.setStyleSheet(get_lineedit_style())
        self.class_path_input.setPlaceholderText(
            self.tr("Path to class file (classes.txt, labels.txt, *.names)")
        )
        self.class_path_input.textChanged.connect(self.update_classes_preview)
        self.class_browse_btn = QPushButton(self.tr("Browse..."))
        self.class_browse_btn.setStyleSheet(get_normal_button_style())
        self.class_browse_btn.clicked.connect(self.on_browse_classes)
        class_container.addWidget(self.class_path_input)
        class_container.addWidget(self.class_browse_btn)
        form_layout.addRow(self.tr("Class File:"), class_container)

        # 3. Class Preview Badge
        self.preview_label = QLabel()
        t = get_theme()
        self.preview_label.setStyleSheet(
            f"color: {t['text_secondary']}; font-size: 12px; margin-left: 2px;"
        )
        form_layout.addRow("", self.preview_label)

        # 4. Display Name Row
        self.display_name_input = QLineEdit(self.initial_display_name)
        self.display_name_input.setStyleSheet(get_lineedit_style())
        self.display_name_input.setPlaceholderText(
            self.tr("Display name for model dropdown")
        )
        form_layout.addRow(self.tr("Display Name:"), self.display_name_input)

        layout.addLayout(form_layout)

        # Action Buttons Row
        button_container = QHBoxLayout()
        button_container.addStretch()

        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(get_cancel_btn_style())
        self.cancel_button.clicked.connect(self.reject)
        button_container.addWidget(self.cancel_button)

        self.apply_button = QPushButton(self.tr("Apply & Reload"))
        self.apply_button.setStyleSheet(get_ok_btn_style())
        self.apply_button.clicked.connect(self.on_apply)
        button_container.addWidget(self.apply_button)

        layout.addLayout(button_container)

        if not self.initial_display_name and self.initial_model_path:
            self._set_default_display_name(self.initial_model_path)

        self.update_classes_preview()

    def _set_default_display_name(self, model_path: str) -> None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        cleaned = re.sub(r"[_\-]+", " ", base_name).strip()
        self.display_name_input.setText(cleaned.title())

    def on_browse_model(self) -> None:
        current_path = self.model_path_input.text().strip()
        start_dir = (
            os.path.dirname(current_path)
            if current_path and os.path.exists(os.path.dirname(current_path))
            else ""
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Model File"),
            start_dir,
            self.tr("ONNX Model (*.onnx);;All Files (*)"),
        )
        if file_path:
            self.model_path_input.setText(file_path)
            if not self.display_name_input.text().strip():
                self._set_default_display_name(file_path)

    def on_browse_classes(self) -> None:
        current_path = self.class_path_input.text().strip()
        start_dir = (
            os.path.dirname(current_path)
            if current_path and os.path.exists(os.path.dirname(current_path))
            else ""
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Class File"),
            start_dir,
            self.tr("Class Files (*.txt *.names);;All Files (*)"),
        )
        if file_path:
            self.class_path_input.setText(file_path)

    def update_classes_preview(self) -> None:
        class_file = self.class_path_input.text().strip()
        if not class_file:
            self.preview_label.setText(self.tr("No class file specified."))
            return
        if not os.path.isfile(class_file):
            self.preview_label.setText(self.tr("⚠️ File does not exist."))
            return

        classes = load_classes_from_file(class_file)
        if not classes:
            self.preview_label.setText(self.tr("⚠️ No classes found in file."))
            return

        count = len(classes)
        sample = ", ".join(classes[:4])
        if count > 4:
            sample += f", ... (+{count - 4} more)"
        self.preview_label.setText(self.tr(f"✓ {count} classes: {sample}"))

    def on_apply(self) -> None:
        model_path = os.path.abspath(self.model_path_input.text().strip())
        class_path = os.path.abspath(self.class_path_input.text().strip())
        display_name = self.display_name_input.text().strip()

        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(
                self,
                self.tr("Invalid Model"),
                self.tr(
                    "Please select a valid model weights file that exists."
                ),
            )
            return

        if not class_path or not os.path.isfile(class_path):
            QMessageBox.warning(
                self,
                self.tr("Invalid Class File"),
                self.tr("Please select a valid class file that exists."),
            )
            return

        classes = load_classes_from_file(class_path)
        if not classes:
            QMessageBox.warning(
                self,
                self.tr("Empty Class File"),
                self.tr(
                    "The selected class file does not contain any valid classes."
                ),
            )
            return

        if not display_name:
            self._set_default_display_name(model_path)
            display_name = self.display_name_input.text().strip()

        # Update or create config YAML
        config_file = self.config_file
        if config_file and os.path.isfile(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Failed to read existing config: {e}")
                config = {}
        else:
            # Create a new YAML config file for this model
            custom_dir = os.path.join(
                get_work_directory(), "xanylabeling_data", "custom_models"
            )
            os.makedirs(custom_dir, exist_ok=True)
            stem = re.sub(
                r"[^a-zA-Z0-9_.-]",
                "_",
                os.path.splitext(os.path.basename(model_path))[0],
            ).lower()
            config_file = os.path.join(custom_dir, f"{stem}.yaml")
            input_shape = [640, 640]
            try:
                import onnx

                onnx_model = onnx.load(model_path)
                in_shape = onnx_model.graph.input[0].type.tensor_type.shape
                dims = [d.dim_value for d in in_shape.dim]
                if len(dims) >= 4 and dims[2] > 0 and dims[3] > 0:
                    input_shape = [int(dims[2]), int(dims[3])]
            except Exception:
                pass
            config = {
                "type": "deepstream_det",
                "name": stem,
                "display_name": display_name,
                "model_path": model_path,
                "labelfile_path": class_path,
                "input_shape": input_shape,
                "conf_threshold": 0.25,
                "iou_threshold": 0.45,
                "containment_threshold": 0.0,
                "containment_keep": "area",
                "model_color_format": 1,
                "net_scale_factor": 1.0,
                "maintain_aspect_ratio": 1,
                "symmetric_padding": 0,
                "pad_value": 114,
                "cluster_mode": 2,
            }

        # Update user-facing fields while preserving all architecture fields
        config["model_path"] = model_path
        config["labelfile_path"] = class_path
        config["classes"] = classes
        config["display_name"] = display_name

        try:
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(config, f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Save Error"),
                self.tr(f"Failed to save model configuration: {e}"),
            )
            return

        self.config_file = config_file

        # Reload model via parent AutoLabelingWidget if present
        if self.parent_widget and hasattr(
            self.parent_widget, "load_custom_model_config"
        ):
            self.parent_widget.load_custom_model_config(config_file)

        self.accept()
