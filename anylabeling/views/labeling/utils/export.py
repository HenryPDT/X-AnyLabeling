import json
import os
import os.path as osp
import pathlib
import shutil
import time

from PyQt6 import QtWidgets
from PyQt6.QtCore import QCoreApplication, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QProgressDialog,
)

from anylabeling.views.labeling.label_converter import (
    LabelConverter,
    PoseClassError,
    PoseGroupError,
)
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.split import (
    build_dataset_yaml,
    collect_labels_and_names,
    effective_images_for_export,
    stratified_split,
    write_list_file,
)
from anylabeling.views.labeling.widgets import Popup
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import *


# Train/val split option (export-time, cf. DarkMark ExportDialog):
# opt-in section inside YOLO/COCO export dialogs. Defaults preserve
# today's single-export behavior.
_SPLIT_MIN_RATIO = 0.5
_SPLIT_MAX_RATIO = 0.95
_SPLIT_DEFAULT_RATIO = 0.8
_SPLIT_DEFAULT_SEED = 42


def _load_split_prefs(widget):
    """Return persisted (enabled, ratio, seed) split prefs."""
    enabled, ratio, seed = False, _SPLIT_DEFAULT_RATIO, _SPLIT_DEFAULT_SEED
    try:
        settings = getattr(widget, "settings", None)
        if settings is None:
            return enabled, ratio, seed
        enabled = settings.value("export_split/enabled", False, type=bool)
        ratio = float(settings.value("export_split/train_ratio", ratio))
        seed = int(settings.value("export_split/seed", seed))
    except (TypeError, ValueError):
        pass
    return bool(enabled), ratio, seed


def _save_split_prefs(widget, enabled, ratio, seed):
    try:
        settings = getattr(widget, "settings", None)
        if settings is None:
            return
        settings.setValue("export_split/enabled", bool(enabled))
        settings.setValue("export_split/train_ratio", float(ratio))
        settings.setValue("export_split/seed", int(seed))
    except Exception as err:
        logger.warning(f"Could not persist split prefs: {err}")


def _create_split_section(
    dialog,
    widget,
    image_provider,
    output_dir_provider,
    skip_empty_provider=None,
):
    """Build the shared train/val split dialog section.

    Returns a dict with the widgets plus a ``refresh`` callable that
    recomputes the Train/Val preview. Callers wire ``refresh`` to
    their own option toggles (e.g. skip-empty changes the counts).
    """
    enable_checkbox = QtWidgets.QCheckBox(
        QCoreApplication.translate("LabelingWidget", "Enable train/val split?")
    )
    ratio_spin = QtWidgets.QDoubleSpinBox(dialog)
    ratio_spin.setRange(_SPLIT_MIN_RATIO, _SPLIT_MAX_RATIO)
    ratio_spin.setSingleStep(0.05)
    ratio_spin.setDecimals(2)
    seed_spin = QtWidgets.QSpinBox(dialog)
    seed_spin.setRange(0, 999999)
    preview_label = QtWidgets.QLabel("", dialog)
    preview_label.setWordWrap(True)
    preview_label.setStyleSheet("opacity: 0.7; font-size: 11px;")

    saved_enabled, saved_ratio, saved_seed = _load_split_prefs(widget)
    enable_checkbox.setChecked(saved_enabled)
    ratio_spin.setValue(
        max(_SPLIT_MIN_RATIO, min(_SPLIT_MAX_RATIO, saved_ratio))
    )
    seed_spin.setValue(max(0, min(999999, saved_seed)))

    def refresh():
        if not enable_checkbox.isChecked():
            preview_label.setText(
                QCoreApplication.translate(
                    "LabelingWidget", "Split disabled — single export."
                )
            )
            return
        try:
            images = list(image_provider() or [])
            output_dir = output_dir_provider()
            labels, _ = collect_labels_and_names(images, output_dir=output_dir)
            skip_empty = bool(
                skip_empty_provider() if skip_empty_provider else False
            )
            effective = effective_images_for_export(images, labels, skip_empty)
            result = stratified_split(
                effective,
                train_ratio=ratio_spin.value(),
                seed=seed_spin.value(),
                labels_by_image=labels,
            )
            preview_label.setText(
                QCoreApplication.translate(
                    "LabelingWidget", "→ Train: {train} | Val: {val}"
                ).format(train=len(result.train), val=len(result.val))
            )
        except Exception as err:
            logger.warning(f"Could not preview split: {err}")

    enable_checkbox.toggled.connect(lambda _c: refresh())
    ratio_spin.valueChanged.connect(lambda _v: refresh())
    seed_spin.valueChanged.connect(lambda _v: refresh())
    refresh()

    row = QtWidgets.QHBoxLayout()
    row.setSpacing(8)
    ratio_label = QtWidgets.QLabel(
        QCoreApplication.translate("LabelingWidget", "Train ratio")
    )
    row.addWidget(ratio_label)
    row.addWidget(ratio_spin)
    seed_label = QtWidgets.QLabel(
        QCoreApplication.translate("LabelingWidget", "Seed")
    )
    row.addWidget(seed_label)
    row.addWidget(seed_spin)
    row.addStretch(1)

    def sync_enabled(active):
        ratio_label.setEnabled(bool(active))
        ratio_spin.setEnabled(bool(active))
        seed_label.setEnabled(bool(active))
        seed_spin.setEnabled(bool(active))

    enable_checkbox.toggled.connect(sync_enabled)
    sync_enabled(saved_enabled)

    return {
        "enable": enable_checkbox,
        "ratio": ratio_spin,
        "seed": seed_spin,
        "preview": preview_label,
        "row": row,
        "refresh": refresh,
    }


def _coco_split_filename(mode, subset):
    stems = {
        "rectangle": "coco_detection",
        "polygon": "coco_instance_segmentation",
        "pose": "coco_keypoints",
    }
    return f"{stems.get(mode, 'coco')}_{subset}.json"


def _plan_export_split(
    image_list, output_dir, train_ratio, seed, skip_empty=False
):
    """Partition the exported image set; raise ValueError if unusable.

    Returns ``(split_result, class_names, split_info)`` where
    ``split_info`` is the reproducibility provenance recorded in
    dataset.yaml / COCO info.
    """
    labels_by_image, names = collect_labels_and_names(
        image_list, output_dir=output_dir
    )
    effective = effective_images_for_export(
        image_list, labels_by_image, skip_empty
    )
    if len(effective) < 2:
        raise ValueError(
            "Train/val split needs at least 2 exported images "
            f"(found {len(effective)})."
        )
    result = stratified_split(
        effective,
        train_ratio=train_ratio,
        seed=seed,
        labels_by_image=labels_by_image,
    )
    if not result.train or not result.val:
        raise ValueError(
            "Train/val split produced an empty subset "
            f"(train={len(result.train)}, val={len(result.val)}). "
            "Add more images or adjust the ratio."
        )
    split_info = {
        "enabled": True,
        "train_ratio": result.train_ratio,
        "seed": result.seed,
    }
    return result, names, split_info


def _make_split_path_toggler(path_edit, default_path):
    """Append '_split' to the export path while split is enabled."""
    state = {"adjusted": False, "previous": default_path}

    def on_toggled(checked):
        current = path_edit.text().rstrip("/\\")
        if checked:
            if not current.endswith("_split"):
                state["previous"] = path_edit.text()
                path_edit.setText(current + "_split")
                state["adjusted"] = True
        elif state["adjusted"]:
            path_edit.setText(state["previous"])
            state["adjusted"] = False

    return on_toggled


class ExportThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        converter,
        image_list,
        label_dir_path,
        save_path,
        mode,
        prefix=None,
        split_lists=None,
        split_info=None,
        save_images=False,
    ):
        super().__init__()
        self.converter = converter
        self.image_list = image_list
        self.label_dir_path = label_dir_path
        self.save_path = save_path
        self.mode = mode
        self.prefix = prefix
        # Export-time train/val split (COCO): ((train imgs, val imgs)).
        # When set, two JSONs land in <save_path>/annotations/ and (if
        # save_images) copies in <save_path>/{train,val}2017/.
        self.split_lists = split_lists
        self.split_info = split_info
        self.save_images = save_images

    def _copy_coco_images(self, images, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        for image_file in images:
            dest = osp.join(dest_dir, osp.basename(image_file))
            if osp.abspath(image_file) != osp.abspath(dest):
                shutil.copy(image_file, dest)

    def run(self):
        try:
            time.sleep(1)

            if self.mode == "vlm_r1_ovd":
                self.converter.custom_to_vlm_r1_ovd(
                    self.image_list,
                    self.label_dir_path,
                    self.save_path,
                    self.prefix,
                )
            elif self.mode == "mot":
                self.converter.custom_to_mot(
                    self.label_dir_path, self.save_path
                )
            elif self.mode == "mots":
                self.converter.custom_to_mots(
                    self.label_dir_path, self.save_path
                )
            elif self.mode == "odvg":
                self.converter.custom_to_odvg(
                    self.image_list, self.label_dir_path, self.save_path
                )
            elif self.split_lists is not None:
                train_images, val_images = self.split_lists
                annotations_dir = osp.join(self.save_path, "annotations")
                os.makedirs(annotations_dir, exist_ok=True)
                self.converter.custom_to_coco(
                    train_images,
                    self.label_dir_path,
                    annotations_dir,
                    self.mode,
                    output_filename=_coco_split_filename(self.mode, "train"),
                    split_info=self.split_info,
                )
                self.converter.custom_to_coco(
                    val_images,
                    self.label_dir_path,
                    annotations_dir,
                    self.mode,
                    output_filename=_coco_split_filename(self.mode, "val"),
                    split_info=self.split_info,
                )
                if self.save_images:
                    self._copy_coco_images(
                        train_images, osp.join(self.save_path, "train2017")
                    )
                    self._copy_coco_images(
                        val_images, osp.join(self.save_path, "val2017")
                    )
            else:
                self.converter.custom_to_coco(
                    self.image_list,
                    self.label_dir_path,
                    self.save_path,
                    self.mode,
                )
                if self.save_images:
                    self._copy_coco_images(
                        self.image_list, osp.join(self.save_path, "images")
                    )
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, str(e))


def _check_filename_exist(self):
    if not self.may_continue():
        return False

    if not self.filename:
        popup = Popup(
            self.tr("Please load an image folder before proceeding!"),
            self,
            icon=new_icon_path("warning", "svg"),
        )
        popup.show_popup(self, position="center")
        return False

    return True


def _show_yolo_export_error(parent, image_file, error):
    image_path = osp.abspath(image_file) if image_file else None
    message = (
        QCoreApplication.translate("LabelingWidget", "Failed on image: %s")
        % image_path
        if image_path
        else QCoreApplication.translate("LabelingWidget", "Export failed.")
    )
    if isinstance(error, PoseGroupError):
        message += "\n\n" + QCoreApplication.translate(
            "LabelingWidget",
            "Reason: Pose instance grouping is incomplete or mismatched.\n"
            "Please ensure that each instance has one bounding box and that "
            "its bounding box and keypoints use the same numeric group ID.",
        )
    elif isinstance(error, PoseClassError):
        message += "\n\n" + QCoreApplication.translate(
            "LabelingWidget",
            "Reason: The bounding box label is not defined in the pose "
            "configuration.\nPlease ensure that the bounding box label is "
            "listed under classes in the pose YAML file.",
        )
    elif str(error):
        message += "\n\n" + QCoreApplication.translate(
            "LabelingWidget", "Reason: %s"
        ) % str(error)

    msg_box = QtWidgets.QMessageBox(parent)
    msg_box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
    msg_box.setWindowTitle(
        QCoreApplication.translate("LabelingWidget", "Export Failed")
    )
    msg_box.setText(message)
    msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Ok)
    msg_box.setStyleSheet(get_msg_box_style())
    msg_box.exec()

    loaded_image_path = (
        osp.abspath(parent.filename) if parent.filename else None
    )
    if image_path and image_path != loaded_image_path:
        parent.load_file(image_file)


def _get_yolo_source_root(filename, last_open_dir):
    source_root = osp.dirname(osp.abspath(filename))
    if not last_open_dir:
        return source_root

    last_open_dir = osp.abspath(last_open_dir)
    try:
        if osp.commonpath((last_open_dir, source_root)) == last_open_dir:
            return last_open_dir
    except ValueError:
        pass
    return source_root


def _validate_yolo_export_path(source_root, save_path, allow_same_dir=False):
    if not save_path:
        raise ValueError("Please select an export root directory.")

    source_root = osp.realpath(source_root)
    save_path = osp.realpath(save_path)
    if osp.exists(save_path) and not osp.isdir(save_path):
        raise ValueError("The export root path must be a directory.")

    try:
        common_path = osp.commonpath((source_root, save_path))
    except ValueError:
        return
    if not allow_same_dir:
        if common_path == source_root:
            raise ValueError(
                "The export root directory cannot be the loaded image directory "
                "or one of its subdirectories."
            )
        if common_path == save_path:
            raise ValueError(
                "The export root directory cannot contain the loaded image "
                "directory."
            )


def _get_yolo_export_files(image_list, source_root, save_path, layout=None):
    """Map images to YOLO label/image destinations.

    Without ``layout`` this preserves ``relpath`` under ``save_path``
    (single export). With ``layout`` (``"train"``/``"val"``) it builds
    the YOLOv5 split layout: ``<save_path>/labels/<layout>/...`` for
    labels and ``<save_path>/images/<layout>/...`` for image copies.
    """
    export_files = []
    label_destinations = {}
    is_in_place = osp.realpath(save_path) == osp.realpath(source_root)
    for image_file in image_list:
        try:
            relative_image_path = osp.relpath(image_file, source_root)
        except ValueError:
            relative_image_path = osp.basename(image_file)
        if relative_image_path == osp.pardir or relative_image_path.startswith(
            osp.pardir + osp.sep
        ):
            relative_image_path = osp.basename(image_file)
        relative_label_path = osp.splitext(relative_image_path)[0] + ".txt"

        if is_in_place:
            dst_file = osp.splitext(image_file)[0] + ".txt"
            image_dst = image_file
        elif layout is not None:
            dst_file = osp.join(
                save_path, "labels", layout, relative_label_path
            )
            image_dst = osp.join(
                save_path, "images", layout, relative_image_path
            )
        else:
            dst_file = osp.join(save_path, relative_label_path)
            image_dst = osp.join(save_path, relative_image_path)

        destination_key = osp.normcase(osp.normpath(dst_file))
        if destination_key in label_destinations:
            raise ValueError(
                "Multiple images map to the same YOLO label file "
                f"'{relative_label_path}':\n"
                f"{label_destinations[destination_key]}\n{image_file}"
            )
        label_destinations[destination_key] = image_file
        export_files.append((image_file, dst_file, image_dst))
    return export_files


def _get_wpod_export_files(image_list, source_root, save_path, layout=None):
    """Map images to WPOD label/image destinations.

    Unlike YOLO, WPOD keeps images and labels side by side in the same
    folder. Without ``layout`` this preserves ``relpath`` under
    ``save_path`` (single export). With ``layout`` (``"train"``/``"val"``)
    it builds ``<save_path>/<layout>/...`` with ``*.jpg`` and ``*.txt``
    siblings, so camera/scene subdirectories survive as
    ``train/<scene>/`` and ``val/<scene>/``.
    """
    export_files = []
    label_destinations = {}
    is_in_place = osp.realpath(save_path) == osp.realpath(source_root)
    for image_file in image_list:
        try:
            relative_image_path = osp.relpath(image_file, source_root)
        except ValueError:
            relative_image_path = osp.basename(image_file)
        if relative_image_path == osp.pardir or relative_image_path.startswith(
            osp.pardir + osp.sep
        ):
            relative_image_path = osp.basename(image_file)
        relative_label_path = osp.splitext(relative_image_path)[0] + ".txt"

        if is_in_place:
            dst_file = osp.splitext(image_file)[0] + ".txt"
            image_dst = image_file
        elif layout is not None:
            dst_file = osp.join(save_path, layout, relative_label_path)
            image_dst = osp.join(save_path, layout, relative_image_path)
        else:
            dst_file = osp.join(save_path, relative_label_path)
            image_dst = osp.join(save_path, relative_image_path)

        destination_key = osp.normcase(osp.normpath(dst_file))
        if destination_key in label_destinations:
            raise ValueError(
                "Multiple images map to the same WPOD label file "
                f"'{relative_label_path}':\n"
                f"{label_destinations[destination_key]}\n{image_file}"
            )
        label_destinations[destination_key] = image_file
        export_files.append((image_file, dst_file, image_dst))
    return export_files


def export_yolo_annotation(self, mode):  # noqa: C901
    if not _check_filename_exist(self):
        return

    # Handle config/classes file selection based on mode
    if mode == "pose":
        filter = QCoreApplication.translate(
            "LabelingWidget", "Classes Files (*.yaml);;All Files (*)"
        )
        self.yaml_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            QCoreApplication.translate(
                "LabelingWidget",
                "Select a specific yolo-pose config file",
            ),
            "",
            filter,
        )
        if not self.yaml_file:
            return
        try:
            converter = LabelConverter(pose_cfg_file=self.yaml_file)
        except Exception as e:
            logger.error(f"Failed to load pose config: {self.yaml_file}: {e}")
            popup = Popup(
                QCoreApplication.translate(
                    "LabelingWidget", "Invalid pose config file:\n%s"
                )
                % str(e),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
            return

    elif mode in ["hbb", "obb", "seg"]:
        project_classes = (
            self.get_project_classes()
            if hasattr(self, "get_project_classes")
            else []
        )
        if not project_classes:
            filter = QCoreApplication.translate(
                "LabelingWidget", "Classes Files (*.txt);;All Files (*)"
            )
            self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                QCoreApplication.translate(
                    "LabelingWidget", "Select a specific classes file"
                ),
                "",
                filter,
            )
            if not self.classes_file:
                return
            converter = LabelConverter(classes_file=self.classes_file)
        else:
            converter = LabelConverter(classes=project_classes)

    source_root = _get_yolo_source_root(
        self.filename, getattr(self, "last_open_dir", None)
    )

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(
        QCoreApplication.translate("LabelingWidget", "Export options")
    )
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    # Show summary of classes being exported
    if (
        hasattr(converter, "classes")
        and isinstance(converter.classes, (list, tuple))
        and converter.classes
    ):
        classes_summary = ", ".join(
            f"[{idx}] {c}" for idx, c in enumerate(converter.classes[:6])
        )
        if len(converter.classes) > 6:
            classes_summary += f", ... ({len(converter.classes)} classes)"
        summary_label = QtWidgets.QLabel(
            QCoreApplication.translate("LabelingWidget", "Classes: %s")
            % classes_summary
        )
        summary_label.setStyleSheet("opacity: 0.7; font-size: 11px;")
        layout.addWidget(summary_label)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(
        QCoreApplication.translate("LabelingWidget", "Export path")
    )
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    default_labels_path = osp.realpath(osp.join(source_root, "..", "labels"))
    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(default_labels_path)
    path_edit.setPlaceholderText(
        QCoreApplication.translate("LabelingWidget", "Select Export Directory")
    )

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            QCoreApplication.translate(
                "LabelingWidget", "Select Export Directory"
            ),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(
        QCoreApplication.translate("LabelingWidget", "Browse")
    )
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    options_label = QtWidgets.QLabel(
        QCoreApplication.translate("LabelingWidget", "Export Options")
    )
    layout.addWidget(options_label)

    in_place_checkbox = QtWidgets.QCheckBox(
        QCoreApplication.translate(
            "LabelingWidget", "Export labels alongside images (in-place)?"
        )
    )
    in_place_checkbox.setChecked(False)
    layout.addWidget(in_place_checkbox)

    save_images_checkbox = QtWidgets.QCheckBox(
        QCoreApplication.translate("LabelingWidget", "Save with images?")
    )
    save_images_checkbox.setChecked(False)
    layout.addWidget(save_images_checkbox)

    skip_empty_files_checkbox = QtWidgets.QCheckBox(
        QCoreApplication.translate("LabelingWidget", "Skip empty labels?")
    )
    skip_empty_files_checkbox.setChecked(False)
    layout.addWidget(skip_empty_files_checkbox)

    split_section = _create_split_section(
        dialog,
        self,
        image_provider=lambda: (
            list(self.image_list) if self.image_list else [self.filename]
        ),
        output_dir_provider=lambda: getattr(self, "output_dir", None),
        skip_empty_provider=skip_empty_files_checkbox.isChecked,
    )
    layout.addWidget(split_section["enable"])
    layout.addLayout(split_section["row"])
    layout.addWidget(split_section["preview"])
    skip_empty_files_checkbox.toggled.connect(
        lambda _c: split_section["refresh"]()
    )

    in_place = False
    last_custom_path = default_labels_path
    split_path_toggler = _make_split_path_toggler(
        path_edit, default_labels_path
    )

    def on_in_place_toggled(checked):
        nonlocal in_place, last_custom_path
        in_place = checked
        if checked:
            last_custom_path = path_edit.text()
            path_edit.setText(source_root)
            path_edit.setEnabled(False)
            path_button.setEnabled(False)
            save_images_checkbox.setChecked(False)
            save_images_checkbox.setEnabled(False)
            split_section["enable"].setChecked(False)
            split_section["enable"].setEnabled(False)
        else:
            path_edit.setText(last_custom_path)
            path_edit.setEnabled(True)
            path_button.setEnabled(True)
            save_images_checkbox.setEnabled(True)
            split_section["enable"].setEnabled(True)

    in_place_checkbox.toggled.connect(on_in_place_toggled)

    def on_split_toggled(checked):
        if checked:
            in_place_checkbox.setChecked(False)
            in_place_checkbox.setEnabled(False)
        else:
            in_place_checkbox.setEnabled(True)
        split_path_toggler(checked)

    split_section["enable"].toggled.connect(on_split_toggled)
    if split_section["enable"].isChecked():
        # Persisted pref was on: apply split-mode UI state.
        on_split_toggled(True)
        split_section["refresh"]()

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(
        QCoreApplication.translate("LabelingWidget", "Cancel")
    )
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(
        QCoreApplication.translate("LabelingWidget", "OK")
    )
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    is_in_place = in_place or (
        osp.realpath(save_path) == osp.realpath(source_root)
    )
    save_images = save_images_checkbox.isChecked() and not is_in_place
    skip_empty_files = skip_empty_files_checkbox.isChecked()
    split_enabled = split_section["enable"].isChecked() and not is_in_place
    split_ratio = split_section["ratio"].value()
    split_seed = split_section["seed"].value()
    _save_split_prefs(self, split_enabled, split_ratio, split_seed)
    image_list = self.image_list if self.image_list else [self.filename]

    split_result = None
    split_names: list = []
    split_info = None
    if split_enabled:
        try:
            split_result, split_names, split_info = _plan_export_split(
                image_list,
                getattr(self, "output_dir", None),
                split_ratio,
                split_seed,
                skip_empty_files,
            )
        except ValueError as error:
            _show_yolo_export_error(self, None, error)
            return

    try:
        _validate_yolo_export_path(
            source_root, save_path, allow_same_dir=is_in_place
        )
        if split_result is not None:
            export_files = _get_yolo_export_files(
                split_result.train, source_root, save_path, layout="train"
            ) + _get_yolo_export_files(
                split_result.val, source_root, save_path, layout="val"
            )
        else:
            export_files = _get_yolo_export_files(
                image_list, source_root, save_path
            )
    except ValueError as error:
        _show_yolo_export_error(self, None, error)
        return

    def get_label_file(image_file):
        label_file_name = osp.splitext(osp.basename(image_file))[0] + ".json"
        label_dir = self.output_dir or osp.dirname(image_file)
        return osp.join(label_dir, label_file_name)

    obb_boundary_policy = "keep"
    if mode == "obb":
        out_of_bounds_count = 0
        for image_file in image_list:
            label_file = get_label_file(image_file)
            if not osp.exists(label_file):
                continue
            data = converter.read_json(label_file)
            image_width = data["imageWidth"]
            image_height = data["imageHeight"]
            for shape in data["shapes"]:
                points = shape["points"]
                if shape["shape_type"] != "rotation" or len(points) != 4:
                    continue
                if any(
                    point[0] < 0
                    or point[0] > image_width
                    or point[1] < 0
                    or point[1] > image_height
                    for point in points
                ):
                    out_of_bounds_count += 1

        if out_of_bounds_count:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg_box.setWindowTitle(
                QCoreApplication.translate(
                    "LabelingWidget", "Out-of-bounds OBBs"
                )
            )
            msg_box.setText(
                QCoreApplication.translate(
                    "LabelingWidget",
                    "Detected %d oriented bounding boxes with points outside "
                    "the image boundaries. Keep them?",
                )
                % out_of_bounds_count
            )
            msg_box.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No
                | QtWidgets.QMessageBox.StandardButton.Cancel
            )
            msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)
            msg_box.setStyleSheet(get_msg_box_style())
            response = QtWidgets.QMessageBox.StandardButton(msg_box.exec())
            if response == QtWidgets.QMessageBox.StandardButton.Cancel:
                return
            if response == QtWidgets.QMessageBox.StandardButton.No:
                obb_boundary_policy = "skip"

    if osp.exists(save_path) and not is_in_place:
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(
            QCoreApplication.translate(
                "LabelingWidget", "Output Directory Exists!"
            )
        )
        msg_box.setText(
            QCoreApplication.translate(
                "LabelingWidget",
                "Directory already exists. Choose an action:",
            )
        )
        msg_box.setInformativeText(
            QCoreApplication.translate(
                "LabelingWidget",
                "• Yes    - Merge with existing files\n"
                "• No     - Delete existing directory\n"
                "• Cancel - Abort export",
            )
        )

        msg_box.addButton(
            QCoreApplication.translate("LabelingWidget", "Yes"),
            QtWidgets.QMessageBox.ButtonRole.YesRole,
        )
        no_button = msg_box.addButton(
            QCoreApplication.translate("LabelingWidget", "No"),
            QtWidgets.QMessageBox.ButtonRole.NoRole,
        )
        cancel_button = msg_box.addButton(
            QCoreApplication.translate("LabelingWidget", "Cancel"),
            QtWidgets.QMessageBox.ButtonRole.RejectRole,
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == no_button:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        elif clicked_button == cancel_button:
            return
    elif not osp.exists(save_path):
        os.makedirs(save_path)

    progress_dialog = QProgressDialog(
        QCoreApplication.translate("LabelingWidget", "Exporting..."),
        QCoreApplication.translate("LabelingWidget", "Cancel"),
        0,
        len(export_files),
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(
        QCoreApplication.translate("LabelingWidget", "Progress")
    )
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    current_image_file = None
    skipped_empty: set = set()
    try:
        for i, (image_file, dst_file, image_dst) in enumerate(export_files):
            current_image_file = image_file
            src_file = get_label_file(image_file)
            os.makedirs(osp.dirname(dst_file), exist_ok=True)

            is_empty_file = converter.custom_to_yolo(
                src_file,
                dst_file,
                mode,
                skip_empty_files=skip_empty_files,
                obb_boundary_policy=obb_boundary_policy,
            )

            if save_images and not (skip_empty_files and is_empty_file):
                if osp.realpath(image_file) != osp.realpath(image_dst):
                    os.makedirs(osp.dirname(image_dst), exist_ok=True)
                    shutil.copy(image_file, image_dst)

            if skip_empty_files and is_empty_file and osp.exists(dst_file):
                os.remove(dst_file)
            if skip_empty_files and is_empty_file:
                skipped_empty.add(image_file)

            progress_dialog.setValue(i)
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

        # Auto-write classes.txt
        if (
            hasattr(converter, "classes")
            and isinstance(converter.classes, (list, tuple))
            and converter.classes
        ):
            classes_text = "\n".join(converter.classes) + "\n"
            try:
                with open(
                    osp.join(save_path, "classes.txt"), "w", encoding="utf-8"
                ) as f:
                    f.write(classes_text)
                if osp.basename(save_path) == "labels":
                    with open(
                        osp.join(osp.dirname(save_path), "classes.txt"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(classes_text)
            except Exception as err:
                logger.warning(f"Could not auto-write classes.txt: {err}")

        if split_result is not None:
            # YOLOv5 split layout: images/{train,val}, labels/{train,val},
            # train.txt/val.txt, dataset.yaml (+ split provenance).
            names = (
                list(converter.classes)
                if (
                    hasattr(converter, "classes")
                    and isinstance(converter.classes, (list, tuple))
                    and converter.classes
                )
                else split_names
            )
            image_dests = {
                src: image_dst for src, _label, image_dst in export_files
            }
            # Labels filtered out as empty mid-export must not linger in
            # the split lists (e.g. shapes outside converter.classes).
            final_train = [
                p for p in split_result.train if p not in skipped_empty
            ]
            final_val = [p for p in split_result.val if p not in skipped_empty]
            if save_images:
                train_images = [image_dests[p] for p in final_train]
                val_images = [image_dests[p] for p in final_val]
                train_ref = osp.join(save_path, "images", "train")
                val_ref = osp.join(save_path, "images", "val")
            else:
                train_images = list(final_train)
                val_images = list(final_val)
                train_ref = osp.join(save_path, "train.txt")
                val_ref = osp.join(save_path, "val.txt")
            write_list_file(
                osp.join(save_path, "train.txt"),
                train_images,
                relto=save_path if save_images else None,
            )
            write_list_file(
                osp.join(save_path, "val.txt"),
                val_images,
                relto=save_path if save_images else None,
            )
            try:
                with open(
                    osp.join(save_path, "dataset.yaml"), "w", encoding="utf-8"
                ) as f:
                    f.write(
                        build_dataset_yaml(
                            train_ref,
                            val_ref,
                            names,
                            project_root=save_path,
                            split_info=split_info,
                        )
                    )
            except Exception as err:
                logger.warning(f"Could not write dataset.yaml: {err}")

        current_image_file = None
        progress_dialog.close()
        template = QCoreApplication.translate(
            "LabelingWidget",
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s",
        )
        message_text = template % save_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        progress_dialog.close()
        failed_image_path = (
            osp.abspath(current_image_file) if current_image_file else None
        )
        if failed_image_path:
            logger.error(
                "Error occurred while exporting annotations for image:\n"
                f"{failed_image_path}\n{e}"
            )
        else:
            logger.error(f"Error occurred while exporting annotations: {e}")

        _show_yolo_export_error(self, current_image_file, e)


def export_voc_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(osp.dirname(self.filename), "..", "Annotations"))
    )
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    options_label = QtWidgets.QLabel(self.tr("Export Options"))
    layout.addWidget(options_label)

    save_images_checkbox = QtWidgets.QCheckBox(self.tr("Save with images?"))
    save_images_checkbox.setChecked(False)
    layout.addWidget(save_images_checkbox)

    skip_empty_files_checkbox = QtWidgets.QCheckBox(
        self.tr("Skip empty labels?")
    )
    skip_empty_files_checkbox.setChecked(False)
    layout.addWidget(skip_empty_files_checkbox)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_images = save_images_checkbox.isChecked()
    skip_empty_files = skip_empty_files_checkbox.isChecked()
    save_path = path_edit.text()

    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(self.tr("Directory already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Yes    - Merge with existing files\n"
                "• No     - Delete existing directory\n"
                "• Cancel - Abort export"
            )
        )

        msg_box.addButton(
            self.tr("Yes"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        no_button = msg_box.addButton(
            self.tr("No"), QtWidgets.QMessageBox.ButtonRole.NoRole
        )
        cancel_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == no_button:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        elif clicked_button == cancel_button:
            return
    else:
        os.makedirs(save_path)

    converter = LabelConverter()

    image_list = self.image_list if self.image_list else [self.filename]

    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        for i, image_file in enumerate(image_list):
            image_file_name = osp.basename(image_file)
            label_file_name = osp.splitext(image_file_name)[0] + ".json"
            dst_file_name = osp.splitext(image_file_name)[0] + ".xml"

            if self.output_dir:
                src_file = osp.join(self.output_dir, label_file_name)
            else:
                src_file = osp.join(osp.dirname(image_file), label_file_name)
            dst_file = osp.join(save_path, dst_file_name)

            is_empty_file = converter.custom_to_voc(
                image_file, src_file, dst_file, mode, skip_empty_files
            )

            if save_images and not (skip_empty_files and is_empty_file):
                image_dst = osp.join(save_path, image_file_name)
                shutil.copy(image_file, image_dst)

            if skip_empty_files and is_empty_file and osp.exists(dst_file):
                os.remove(dst_file)

            progress_dialog.setValue(i)
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = self.tr(
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % save_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = f"Error occurred while exporting annotations: {str(e)}"
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def export_coco_annotation(self, mode):  # noqa: C901
    if not _check_filename_exist(self):
        return

    if mode == "pose":
        filter = "Classes Files (*.yaml);;All Files (*)"
        self.yaml_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific coco-pose config file"),
            "",
            filter,
        )
        if not self.yaml_file:
            return
        try:
            converter = LabelConverter(pose_cfg_file=self.yaml_file)
        except Exception as e:
            logger.error(f"Failed to load pose config: {self.yaml_file}: {e}")
            popup = Popup(
                self.tr("Invalid pose config file:\n%s") % str(e),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
            return
    elif mode in ["rectangle", "polygon"]:
        project_classes = (
            self.get_project_classes()
            if hasattr(self, "get_project_classes")
            else []
        )
        if not project_classes:
            filter = "Classes Files (*.txt);;All Files (*)"
            self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("Select a specific classes file"),
                "",
                filter,
            )
            if not self.classes_file:
                return
            converter = LabelConverter(classes_file=self.classes_file)
        else:
            converter = LabelConverter(classes=project_classes)

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(label_dir_path, "..", "annotations"))
    )
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    options_label = QtWidgets.QLabel(self.tr("Export Options"))
    layout.addWidget(options_label)

    save_images_checkbox = QtWidgets.QCheckBox(self.tr("Save with images?"))
    save_images_checkbox.setChecked(False)
    layout.addWidget(save_images_checkbox)

    split_section = _create_split_section(
        dialog,
        self,
        image_provider=lambda: (
            list(self.image_list) if self.image_list else [self.filename]
        ),
        output_dir_provider=lambda: label_dir_path,
    )
    layout.addWidget(split_section["enable"])
    layout.addLayout(split_section["row"])
    layout.addWidget(split_section["preview"])

    default_annotations_path = path_edit.text()
    on_coco_split_toggled = _make_split_path_toggler(
        path_edit, default_annotations_path
    )

    split_section["enable"].toggled.connect(on_coco_split_toggled)
    if split_section["enable"].isChecked():
        on_coco_split_toggled(True)
        split_section["refresh"]()

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(self.tr("Directory already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Overwrite - Overwrite existing directory\n"
                "• Cancel - Abort export"
            )
        )

        msg_box.addButton(
            self.tr("Overwrite"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        cancel_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == cancel_button:
            return
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
    else:
        os.makedirs(save_path)

    image_list = self.image_list if self.image_list else [self.filename]
    save_images = save_images_checkbox.isChecked()
    split_enabled = split_section["enable"].isChecked()
    split_ratio = split_section["ratio"].value()
    split_seed = split_section["seed"].value()
    _save_split_prefs(self, split_enabled, split_ratio, split_seed)

    split_lists = None
    split_info = None
    if split_enabled:
        try:
            split_result, _, split_info = _plan_export_split(
                image_list, label_dir_path, split_ratio, split_seed
            )
        except ValueError as error:
            popup = Popup(
                str(error),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")
            return
        split_lists = (split_result.train, split_result.val)

    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    self.export_thread = ExportThread(
        converter,
        image_list,
        label_dir_path,
        save_path,
        mode,
        split_lists=split_lists,
        split_info=split_info,
        save_images=save_images,
    )

    def on_export_finished(success, error_msg):
        progress_dialog.close()
        if success:
            template = self.tr(
                "Exporting annotations successfully!\n"
                "Results have been saved to:\n"
                "%s"
            )
            message_text = template % save_path
            popup = Popup(
                message_text,
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = (
                f"Error occurred while exporting annotations: {str(error_msg)}"
            )
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.export_thread.finished.connect(on_export_finished)

    progress_dialog.show()
    self.export_thread.start()

    progress_dialog.canceled.connect(self.export_thread.terminate)


def export_dota_annotation(self):
    if not _check_filename_exist(self):
        return

    filter = "Classes Files (*.txt);;All Files (*)"
    self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific classes file"),
        "",
        filter,
    )
    if not self.classes_file:
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(osp.dirname(self.filename), "..", "labelTxt"))
    )
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()

    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(self.tr("Directory already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Yes    - Merge with existing files\n"
                "• No     - Delete existing directory\n"
                "• Cancel - Abort export"
            )
        )

        msg_box.addButton(
            self.tr("Yes"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        no_button = msg_box.addButton(
            self.tr("No"), QtWidgets.QMessageBox.ButtonRole.NoRole
        )
        cancel_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == no_button:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        elif clicked_button == cancel_button:
            return
    else:
        os.makedirs(save_path)

    converter = LabelConverter(classes_file=self.classes_file)

    image_list = self.image_list if self.image_list else [self.filename]

    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        for i, image_file in enumerate(image_list):
            image_file_name = osp.basename(image_file)
            label_file_name = osp.splitext(image_file_name)[0] + ".json"
            dst_file_name = osp.splitext(image_file_name)[0] + ".txt"

            if self.output_dir:
                src_file = osp.join(self.output_dir, label_file_name)
            else:
                src_file = osp.join(osp.dirname(image_file), label_file_name)
            dst_file = osp.join(save_path, dst_file_name)

            if not osp.exists(src_file):
                pathlib.Path(dst_file).touch()
            else:
                converter.custom_to_dota(src_file, dst_file)

            progress_dialog.setValue(i)
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = self.tr(
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % save_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = f"Error occurred while exporting annotations: {str(e)}"
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def export_wpod_annotation(self):
    """Export ``quadrilateral`` shapes to WPOD/IWPOD blocked quad format.

    Same options as the YOLO export: export labels alongside images
    (in-place), copy images, skip empty labels, and train/val split.
    Camera/scene subdirectories are preserved as relative paths under the
    export root. Writes one ``4,x1,x2,x3,x4,y1,y2,y3,y4,,`` line per
    quadrilateral; no classes file is needed since the format carries no
    label column.
    """
    if not _check_filename_exist(self):
        return

    converter = LabelConverter()
    source_root = _get_yolo_source_root(
        self.filename, getattr(self, "last_open_dir", None)
    )

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(
        QCoreApplication.translate("LabelingWidget", "Export options")
    )
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(
        QCoreApplication.translate("LabelingWidget", "Export path")
    )
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    default_labels_path = osp.realpath(
        osp.join(source_root, "..", "wpod_labels")
    )
    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(default_labels_path)
    path_edit.setPlaceholderText(
        QCoreApplication.translate("LabelingWidget", "Select Export Directory")
    )

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            QCoreApplication.translate(
                "LabelingWidget", "Select Export Directory"
            ),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(
        QCoreApplication.translate("LabelingWidget", "Browse")
    )
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    options_label = QtWidgets.QLabel(
        QCoreApplication.translate("LabelingWidget", "Export Options")
    )
    layout.addWidget(options_label)

    in_place_checkbox = QtWidgets.QCheckBox(
        QCoreApplication.translate(
            "LabelingWidget", "Export labels alongside images (in-place)?"
        )
    )
    in_place_checkbox.setChecked(False)
    layout.addWidget(in_place_checkbox)

    save_images_checkbox = QtWidgets.QCheckBox(
        QCoreApplication.translate("LabelingWidget", "Save with images?")
    )
    save_images_checkbox.setChecked(False)
    layout.addWidget(save_images_checkbox)

    skip_empty_files_checkbox = QtWidgets.QCheckBox(
        QCoreApplication.translate("LabelingWidget", "Skip empty labels?")
    )
    skip_empty_files_checkbox.setChecked(False)
    layout.addWidget(skip_empty_files_checkbox)

    split_section = _create_split_section(
        dialog,
        self,
        image_provider=lambda: (
            list(self.image_list) if self.image_list else [self.filename]
        ),
        output_dir_provider=lambda: getattr(self, "output_dir", None),
        skip_empty_provider=skip_empty_files_checkbox.isChecked,
    )
    layout.addWidget(split_section["enable"])
    layout.addLayout(split_section["row"])
    layout.addWidget(split_section["preview"])
    skip_empty_files_checkbox.toggled.connect(
        lambda _c: split_section["refresh"]()
    )

    in_place = False
    last_custom_path = default_labels_path
    split_path_toggler = _make_split_path_toggler(
        path_edit, default_labels_path
    )

    def on_in_place_toggled(checked):
        nonlocal in_place, last_custom_path
        in_place = checked
        if checked:
            last_custom_path = path_edit.text()
            path_edit.setText(source_root)
            path_edit.setEnabled(False)
            path_button.setEnabled(False)
            save_images_checkbox.setChecked(False)
            save_images_checkbox.setEnabled(False)
            split_section["enable"].setChecked(False)
            split_section["enable"].setEnabled(False)
        else:
            path_edit.setText(last_custom_path)
            path_edit.setEnabled(True)
            path_button.setEnabled(True)
            save_images_checkbox.setEnabled(True)
            split_section["enable"].setEnabled(True)

    in_place_checkbox.toggled.connect(on_in_place_toggled)

    def on_split_toggled(checked):
        if checked:
            in_place_checkbox.setChecked(False)
            in_place_checkbox.setEnabled(False)
        else:
            in_place_checkbox.setEnabled(True)
        split_path_toggler(checked)

    split_section["enable"].toggled.connect(on_split_toggled)
    if split_section["enable"].isChecked():
        # Persisted pref was on: apply split-mode UI state.
        on_split_toggled(True)
        split_section["refresh"]()

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(
        QCoreApplication.translate("LabelingWidget", "Cancel")
    )
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(
        QCoreApplication.translate("LabelingWidget", "OK")
    )
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    is_in_place = in_place or (
        osp.realpath(save_path) == osp.realpath(source_root)
    )
    save_images = save_images_checkbox.isChecked() and not is_in_place
    skip_empty_files = skip_empty_files_checkbox.isChecked()
    split_enabled = split_section["enable"].isChecked() and not is_in_place
    split_ratio = split_section["ratio"].value()
    split_seed = split_section["seed"].value()
    _save_split_prefs(self, split_enabled, split_ratio, split_seed)
    image_list = self.image_list if self.image_list else [self.filename]

    split_result = None
    if split_enabled:
        try:
            split_result, _, _ = _plan_export_split(
                image_list,
                getattr(self, "output_dir", None),
                split_ratio,
                split_seed,
                skip_empty_files,
            )
        except ValueError as error:
            _show_yolo_export_error(self, None, error)
            return

    try:
        _validate_yolo_export_path(
            source_root, save_path, allow_same_dir=is_in_place
        )
        if split_result is not None:
            export_files = _get_wpod_export_files(
                split_result.train, source_root, save_path, layout="train"
            ) + _get_wpod_export_files(
                split_result.val, source_root, save_path, layout="val"
            )
        else:
            export_files = _get_wpod_export_files(
                image_list, source_root, save_path
            )
    except ValueError as error:
        _show_yolo_export_error(self, None, error)
        return

    def get_label_file(image_file):
        label_file_name = osp.splitext(osp.basename(image_file))[0] + ".json"
        label_dir = self.output_dir or osp.dirname(image_file)
        return osp.join(label_dir, label_file_name)

    if osp.exists(save_path) and not is_in_place:
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(
            QCoreApplication.translate(
                "LabelingWidget", "Output Directory Exists!"
            )
        )
        msg_box.setText(
            QCoreApplication.translate(
                "LabelingWidget",
                "Directory already exists. Choose an action:",
            )
        )
        msg_box.setInformativeText(
            QCoreApplication.translate(
                "LabelingWidget",
                "• Yes    - Merge with existing files\n"
                "• No     - Delete existing directory\n"
                "• Cancel - Abort export",
            )
        )

        msg_box.addButton(
            QCoreApplication.translate("LabelingWidget", "Yes"),
            QtWidgets.QMessageBox.ButtonRole.YesRole,
        )
        no_button = msg_box.addButton(
            QCoreApplication.translate("LabelingWidget", "No"),
            QtWidgets.QMessageBox.ButtonRole.NoRole,
        )
        cancel_button = msg_box.addButton(
            QCoreApplication.translate("LabelingWidget", "Cancel"),
            QtWidgets.QMessageBox.ButtonRole.RejectRole,
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == no_button:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        elif clicked_button == cancel_button:
            return
    elif not osp.exists(save_path):
        os.makedirs(save_path)

    progress_dialog = QProgressDialog(
        QCoreApplication.translate("LabelingWidget", "Exporting..."),
        QCoreApplication.translate("LabelingWidget", "Cancel"),
        0,
        len(export_files),
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(
        QCoreApplication.translate("LabelingWidget", "Progress")
    )
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    current_image_file = None
    skipped_empty: set = set()
    try:
        for i, (image_file, dst_file, image_dst) in enumerate(export_files):
            current_image_file = image_file
            src_file = get_label_file(image_file)
            os.makedirs(osp.dirname(dst_file), exist_ok=True)

            is_empty_file = converter.custom_to_wpod(
                src_file,
                dst_file,
                skip_empty_files=skip_empty_files,
            )

            if save_images and not (skip_empty_files and is_empty_file):
                if osp.realpath(image_file) != osp.realpath(image_dst):
                    os.makedirs(osp.dirname(image_dst), exist_ok=True)
                    shutil.copy(image_file, image_dst)

            if skip_empty_files and is_empty_file and osp.exists(dst_file):
                os.remove(dst_file)
            if skip_empty_files and is_empty_file:
                skipped_empty.add(image_file)

            progress_dialog.setValue(i)
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

        if split_result is not None:
            # WPOD split layout: train/<scene>/ + val/<scene>/ with *.jpg
            # and *.txt side by side, plus train.txt/val.txt. No
            # dataset.yaml or classes.txt: the quad format carries no
            # label column.
            image_dests = {
                src: image_dst for src, _label, image_dst in export_files
            }
            # Labels filtered out as empty mid-export must not linger in
            # the split lists.
            final_train = [
                p for p in split_result.train if p not in skipped_empty
            ]
            final_val = [
                p for p in split_result.val if p not in skipped_empty
            ]
            if save_images:
                train_images = [image_dests[p] for p in final_train]
                val_images = [image_dests[p] for p in final_val]
            else:
                train_images = list(final_train)
                val_images = list(final_val)
            write_list_file(
                osp.join(save_path, "train.txt"),
                train_images,
                relto=save_path if save_images else None,
            )
            write_list_file(
                osp.join(save_path, "val.txt"),
                val_images,
                relto=save_path if save_images else None,
            )

        current_image_file = None
        progress_dialog.close()
        template = QCoreApplication.translate(
            "LabelingWidget",
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s",
        )
        message_text = template % save_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        progress_dialog.close()
        failed_image_path = (
            osp.abspath(current_image_file) if current_image_file else None
        )
        if failed_image_path:
            logger.error(
                "Error occurred while exporting annotations for image:\n"
                f"{failed_image_path}\n{e}"
            )
        else:
            logger.error(f"Error occurred while exporting annotations: {e}")

        _show_yolo_export_error(self, current_image_file, e)

def _export_mask_files(
    converter,
    image_list,
    output_dir,
    save_path,
    mapping_table,
    include_null_images,
    only_checked_images,
    progress_dialog,
):
    for i, image_file in enumerate(image_list):
        image_file_name = osp.basename(image_file)
        label_file_name = osp.splitext(image_file_name)[0] + ".json"
        dst_file_name = osp.splitext(image_file_name)[0] + ".png"

        if output_dir:
            src_file = osp.join(output_dir, label_file_name)
        else:
            src_file = osp.join(osp.dirname(image_file), label_file_name)
        dst_file = osp.join(save_path, dst_file_name)

        if osp.exists(src_file):
            if (
                not only_checked_images
                or converter.read_json(src_file).get("checked", False) is True
            ):
                converter.custom_to_mask(src_file, dst_file, mapping_table)
        elif include_null_images and not only_checked_images:
            converter.custom_image_to_empty_mask(
                image_file, dst_file, mapping_table
            )

        progress_dialog.setValue(i + 1)
        QtWidgets.QApplication.processEvents()
        if progress_dialog.wasCanceled():
            break


def export_mask_annotation(self):
    if not _check_filename_exist(self):
        return

    filter = "JSON Files (*.json);;All Files (*)"
    color_map_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific color_map file"),
        "",
        filter,
    )
    if not color_map_file:
        return

    with open(color_map_file, "r", encoding="utf-8") as f:
        mapping_table = json.load(f)

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.realpath(osp.join(label_dir_path, "..", "masks")))
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    options_label = QtWidgets.QLabel(self.tr("Export Options"))
    layout.addWidget(options_label)

    include_null_images_checkbox = QtWidgets.QCheckBox(
        self.tr("Include images without labels?")
    )
    include_null_images_checkbox.setChecked(False)
    layout.addWidget(include_null_images_checkbox)

    only_checked_images_checkbox = QtWidgets.QCheckBox(
        self.tr("Only export checked images?")
    )
    only_checked_images_checkbox.setChecked(False)
    layout.addWidget(only_checked_images_checkbox)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    include_null_images = include_null_images_checkbox.isChecked()
    only_checked_images = only_checked_images_checkbox.isChecked()
    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(self.tr("Directory already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Overwrite - Overwrite existing directory\n"
                "• Cancel - Abort export"
            )
        )

        msg_box.addButton(
            self.tr("Overwrite"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        cancel_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == cancel_button:
            return
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
    else:
        os.makedirs(save_path)

    converter = LabelConverter()
    image_list = self.image_list if self.image_list else [self.filename]

    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        _export_mask_files(
            converter,
            image_list,
            self.output_dir,
            save_path,
            mapping_table,
            include_null_images,
            only_checked_images,
            progress_dialog,
        )

        progress_dialog.close()
        template = self.tr(
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % save_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = f"Error occurred while exporting annotations: {str(e)}"
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def export_mot_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    converter = LabelConverter()
    if mode in ["mot", "mots", "odvg"]:
        filter = "Classes Files (*.txt);;All Files (*)"
        self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific classes file"),
            "",
            filter,
        )
        if not self.classes_file:
            return
        converter = LabelConverter(classes_file=self.classes_file)

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.realpath(osp.join(label_dir_path, "..", mode)))
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(self.tr("Directory already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Overwrite - Overwrite existing directory\n"
                "• Cancel - Abort export"
            )
        )

        msg_box.addButton(
            self.tr("Overwrite"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        cancel_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == cancel_button:
            return
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
    else:
        os.makedirs(save_path)

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    self.export_thread = ExportThread(
        converter, image_list, label_dir_path, save_path, mode
    )

    def on_export_finished(success, error_msg):
        progress_dialog.close()
        if success:
            template = self.tr(
                "Exporting annotations successfully!\n"
                "Results have been saved to:\n"
                "%s"
            )
            message_text = template % save_path
            popup = Popup(
                message_text,
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = (
                f"Error occurred while exporting annotations: {str(error_msg)}"
            )
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.export_thread.finished.connect(on_export_finished)

    progress_dialog.show()
    self.export_thread.start()

    progress_dialog.canceled.connect(self.export_thread.terminate)


def export_odvg_annotation(self):
    export_mot_annotation(self, "odvg")


def export_pporc_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(label_dir_path, "..", f"ppocr_{mode}"))
    )
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(self.tr("Directory already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Overwrite - Overwrite existing directory\n"
                "• Cancel - Abort export"
            )
        )

        msg_box.addButton(
            self.tr("Overwrite"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        cancel_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == cancel_button:
            return
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
    else:
        os.makedirs(save_path)

    if mode == "rec":
        save_crop_img_path = osp.join(save_path, "crop_img")
        if osp.exists(save_crop_img_path):
            shutil.rmtree(save_crop_img_path)
        os.makedirs(save_crop_img_path, exist_ok=True)
        for fname in ("Label.txt", "rec_gt.txt"):
            fpath = osp.join(save_path, fname)
            if osp.exists(fpath):
                os.remove(fpath)
    elif mode == "kie":
        total_class_set = set()
        class_list_file = osp.join(save_path, "class_list.txt")
        ppocr_kie_file = osp.join(save_path, "ppocr_kie.json")
        if osp.exists(ppocr_kie_file):
            os.remove(ppocr_kie_file)

    converter = LabelConverter()

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        for i, image_file in enumerate(image_list):
            image_file_name = osp.basename(image_file)
            label_file_name = osp.splitext(image_file_name)[0] + ".json"
            label_file = osp.join(osp.dirname(image_file), label_file_name)
            if mode == "rec":
                converter.custom_to_ppocr(
                    image_file, label_file, save_path, mode
                )
            elif mode == "kie":
                class_set = converter.custom_to_ppocr(
                    image_file, label_file, save_path, mode
                )
                total_class_set = total_class_set.union(class_set)

            progress_dialog.setValue(i)
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

        if mode == "kie":
            with open(class_list_file, "w") as f:
                for c in total_class_set:
                    f.writelines(f"{c.upper()}\n")

        progress_dialog.close()

        template = self.tr(
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % save_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = f"Error occurred while exporting annotations: {str(e)}"
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def export_vlm_r1_ovd_annotation(self):
    if not _check_filename_exist(self):
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export VLM-R1 OVD Annotation"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    main_layout = QVBoxLayout()
    main_layout.setContentsMargins(24, 24, 24, 24)
    main_layout.setSpacing(16)

    # --- File path selection ---
    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export to"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    # Default export path and filename
    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir
    default_export_path = osp.realpath(
        osp.join(label_dir_path, "..", "vlm_r1_ovd.jsonl")
    )

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(default_export_path)
    path_edit.setPlaceholderText(self.tr("Select Export File"))

    def browse_export_file():
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            dialog,
            self.tr("Select Export File"),
            path_edit.text(),
            "JSONL Files (*.jsonl)",
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            if not path.endswith(".jsonl"):
                path += ".jsonl"
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_file)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    main_layout.addLayout(path_layout)

    # --- Prefix input ---
    prefix_layout = QVBoxLayout()
    prefix_layout.setSpacing(8)

    prefix_label = QHBoxLayout()
    prefix_label.setSpacing(2)

    prefix_title_label = QtWidgets.QLabel(self.tr("Prefix:"))
    prefix_preview_label = QtWidgets.QLabel("")
    prefix_preview_label.setStyleSheet(
        "color: gray; font-style: italic; padding-left: 5px;"
    )

    prefix_label.addWidget(prefix_title_label)
    prefix_label.addWidget(prefix_preview_label)
    prefix_label.addStretch()

    prefix_edit = QtWidgets.QLineEdit()
    prefix_edit_placeholder = self.tr(
        "Optional prefix for image filenames (e.g., 'path/to/images/')"
    )
    prefix_edit.setPlaceholderText(prefix_edit_placeholder)

    prefix_layout.addLayout(prefix_label)
    prefix_layout.addWidget(prefix_edit)
    main_layout.addLayout(prefix_layout)

    def _update_preview():
        prefix = prefix_edit.text()
        if not prefix:
            prefix = "demo.jpg"
        else:
            prefix += "demo.jpg"
        preview_text = self.tr("{}").format(prefix)
        prefix_preview_label.setText(preview_text)

    prefix_edit.textChanged.connect(_update_preview)
    _update_preview()

    # --- Class Filtering ---
    self.classes_file = None

    # --- Class Label ---
    class_label = QtWidgets.QLabel(self.tr("Use specific classes? (Optional)"))
    main_layout.addWidget(class_label)

    # --- Class Path Layout ---
    class_path_layout = QHBoxLayout()
    class_path_layout.setSpacing(8)

    class_path_edit = QtWidgets.QLineEdit()
    class_path_edit.setPlaceholderText(
        self.tr("Select a specific classes file")
    )

    def _handle_class_file_upload():
        filter = "Classes Files (*.txt);;All Files (*)"
        classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific classes file"),
            "",
            filter,
        )
        class_path_edit.setText(classes_file)

    class_path_button = QtWidgets.QPushButton(self.tr("Upload"))
    class_path_edit.textChanged.connect(
        lambda text: setattr(self, "classes_file", text)
    )
    class_path_button.clicked.connect(_handle_class_file_upload)
    class_path_button.setStyleSheet(get_cancel_btn_style())

    class_path_layout.addWidget(class_path_edit)
    class_path_layout.addWidget(class_path_button)
    main_layout.addLayout(class_path_layout)

    # --- Hint Label ---
    class_hint_label = QtWidgets.QLabel(
        self.tr(
            "Hint: If you don't upload a specific classes file, all unique labels found in one of the annotations will be used for the export."
        )
    )
    class_hint_label.setStyleSheet(
        "color: gray; font-style: italic; padding-left: 5px;"
    )
    class_hint_label.setWordWrap(True)
    main_layout.addWidget(class_hint_label)

    # --- Buttons layout ---
    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    main_layout.addLayout(button_layout)

    dialog.setLayout(main_layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    prefix = prefix_edit.text().strip()

    # --- File Exists Check ---
    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("File Exists!"))
        msg_box.setText(self.tr("File already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Overwrite - Replace existing file\n"  # Escaped newline for informative text
                "• Cancel - Abort export"
            )
        )
        _ = msg_box.addButton(
            self.tr("Overwrite"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        cancel_msg_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        msg_box.setDefaultButton(cancel_msg_button)

        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == cancel_msg_button:
            return

    image_list = self.image_list if self.image_list else [self.filename]
    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    # --- Attempt to create LabelConverter first ---
    try:
        converter = LabelConverter(classes_file=self.classes_file)
    except Exception as e:
        logger.error(f"Failed to initialize LabelConverter: {e}")
        template = self.tr("Error initializing export: %s")
        popup = Popup(
            template % e,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    # --- Progress Dialog ---
    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    try:
        self.export_thread = ExportThread(
            converter,
            image_list,
            label_dir_path,
            save_path,
            "vlm_r1_ovd",
            prefix=prefix,
        )

        def on_export_finished(success, error_msg):
            progress_dialog.close()
            if success:
                template = self.tr(
                    "Exporting annotations successfully!\n"
                    "Results have been saved to:\n"
                    "%s"
                )
                message_text = template % save_path
                popup = Popup(
                    message_text,
                    self,
                    icon=new_icon_path("copy-green", "svg"),
                )
                popup.show_popup(self, popup_height=65, position="center")
            else:
                message = f"Error occurred while exporting annotations: {str(error_msg)}"
                logger.error(message)
                popup = Popup(
                    message,
                    self,
                    icon=new_icon_path("error", "svg"),
                )
                popup.show_popup(self, position="center")

        self.export_thread.finished.connect(on_export_finished)

        progress_dialog.show()
        self.export_thread.start()

        progress_dialog.canceled.connect(self.export_thread.terminate)

    except Exception as e:
        message = f"Error occurred while exporting annotations: {str(e)}"
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
