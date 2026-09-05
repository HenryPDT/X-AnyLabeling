"""Train/val split builder dialog (additive).

Deterministic seeded split with stratification preview. Writes
train.txt / val.txt / dataset.yaml without touching annotations.
"""

from __future__ import annotations

import os
from typing import List, Optional

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from anylabeling.views.labeling.widgets.review_table_base import (
    get_review_dialog_style,
)
from anylabeling.views.labeling.utils.split import (
    build_dataset_yaml,
    collect_labels_and_names,
    scene_key_for_image,
    stratified_split,
    write_list_file,
)


class DatasetSplitDialog(QDialog):
    """Build train/val lists with seed and YAML guard."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label_widget = parent
        self.ratio_spin: Optional[QDoubleSpinBox] = None
        self.seed_spin: Optional[QSpinBox] = None
        self.preview_label: Optional[QLabel] = None
        self.init_ui()
        self.update_preview()

    def init_ui(self) -> None:
        self.setWindowTitle(self.tr("Train/Val Split Builder"))
        self.setMinimumWidth(420)
        self.setStyleSheet(get_review_dialog_style())
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.ratio_spin = QDoubleSpinBox(self)
        self.ratio_spin.setRange(0.1, 0.9)
        self.ratio_spin.setSingleStep(0.05)
        self.ratio_spin.setDecimals(2)
        self.ratio_spin.setValue(0.8)
        self.ratio_spin.valueChanged.connect(lambda _v: self.update_preview())
        form.addRow(self.tr("Train ratio"), self.ratio_spin)
        self.seed_spin = QSpinBox(self)
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        self.seed_spin.valueChanged.connect(lambda _v: self.update_preview())
        form.addRow(self.tr("Seed"), self.seed_spin)
        layout.addLayout(form)
        self.preview_label = QLabel("", self)
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_build = QPushButton(self.tr("Build Split..."), self)
        self.btn_build.setProperty("class", "primary-button")
        self.btn_build.clicked.connect(self.on_build)
        btn_row.addWidget(self.btn_build)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        btn_row.addWidget(buttons)
        layout.addLayout(btn_row)

    def _collect_images(self) -> List[str]:
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

    def _collect_labels_and_names(self, images: List[str]):
        widget = self._label_widget
        output_dir = getattr(widget, "output_dir", None) if widget else None
        return collect_labels_and_names(images, output_dir=output_dir)

    def update_preview(self) -> None:
        images = self._collect_images()
        if not images:
            self.preview_label.setText(self.tr("No images loaded."))
            return
        labels, names = self._collect_labels_and_names(images)
        result = stratified_split(
            images,
            train_ratio=self.ratio_spin.value(),
            seed=self.seed_spin.value(),
            labels_by_image=labels,
        )
        scenes = {scene_key_for_image(img) for img in images}
        self.preview_label.setText(
            self.tr(
                "Images: {a} | Train: {b} | Val: {c} | "
                "Classes: {d} | Scenes: {e}"
            ).format(
                a=len(images),
                b=len(result.train),
                c=len(result.val),
                d=len(names),
                e=len(scenes),
            )
        )

    def on_build(self) -> None:
        images = self._collect_images()
        if not images:
            return
        labels, names = self._collect_labels_and_names(images)
        result = stratified_split(
            images,
            train_ratio=self.ratio_spin.value(),
            seed=self.seed_spin.value(),
            labels_by_image=labels,
        )
        dest_dir = QFileDialog.getExistingDirectory(
            self, self.tr("Choose output directory for split files")
        )
        if not dest_dir:
            return
        train_path = os.path.join(dest_dir, "train.txt")
        val_path = os.path.join(dest_dir, "val.txt")
        yaml_path = os.path.join(dest_dir, "dataset.yaml")
        existing = [
            p for p in (train_path, val_path, yaml_path) if os.path.exists(p)
        ]
        if existing:
            ans = QMessageBox.question(
                self,
                self.tr("Overwrite Split Files?"),
                self.tr(
                    "These files already exist and will be overwritten:\n{files}"
                ).format(files="\n".join(existing)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        write_list_file(train_path, result.train)
        write_list_file(val_path, result.val)
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(
                build_dataset_yaml(
                    train_path, val_path, names, project_root=dest_dir
                )
            )
        self.preview_label.setText(
            self.tr(
                "Wrote train.txt ({a}), val.txt ({b}), dataset.yaml"
            ).format(a=len(result.train), b=len(result.val))
        )
