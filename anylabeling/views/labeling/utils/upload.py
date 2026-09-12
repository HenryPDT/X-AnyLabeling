import json
import jsonlines
import os
import os.path as osp
import time
import re
import yaml

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QProgressDialog,
)

from anylabeling.views.labeling.label_converter import LabelConverter
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.widgets import Popup
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import *
from anylabeling.views.labeling.utils.export import _check_filename_exist


def _refresh_uploaded_file_items(self):
    for index in range(self.file_list_widget.count()):
        item = self.file_list_widget.item(index)
        image_file = item.text()
        label_file = osp.splitext(image_file)[0] + ".json"
        if self.output_dir:
            label_file = osp.join(self.output_dir, osp.basename(label_file))
        check_state = (
            Qt.CheckState.Checked
            if osp.isfile(label_file)
            else Qt.CheckState.Unchecked
        )
        item.setCheckState(check_state)
        self._set_file_item_checked(item, self._label_file_checked(label_file))


def _refresh_after_annotation_upload(self):
    _refresh_uploaded_file_items(self)
    self.load_file(self.filename)


class UploadPPOCRThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, converter, input_file, output_path, image_path, mode):
        super().__init__()
        self.converter = converter
        self.input_file = input_file
        self.output_path = output_path
        self.image_path = image_path
        self.mode = mode

    def run(self):
        try:
            time.sleep(1)

            self.converter.ppocr_to_custom(
                input_file=self.input_file,
                output_path=self.output_path,
                image_path=self.image_path,
                mode=self.mode,
            )

            self.finished.emit(True, "")

        except Exception as e:
            self.finished.emit(False, str(e))


class UploadOdvgThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, converter, input_file, output_path):
        super().__init__()
        self.converter = converter
        self.input_file = input_file
        self.output_path = output_path

    def run(self):
        try:
            time.sleep(1)

            self.converter.odvg_to_custom(
                input_file=self.input_file,
                output_path=self.output_path,
            )

            self.finished.emit(True, "")

        except Exception as e:
            self.finished.emit(False, str(e))


class UploadMotThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, converter, gt_file, output_path, image_path):
        super().__init__()
        self.converter = converter
        self.gt_file = gt_file
        self.output_path = output_path
        self.image_path = image_path

    def run(self):
        try:
            time.sleep(1)

            self.converter.mot_to_custom(
                input_file=self.gt_file,
                output_path=self.output_path,
                image_path=self.image_path,
            )

            self.finished.emit(True, "")

        except Exception as e:
            self.finished.emit(False, str(e))


class UploadCocoThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, converter, input_file, output_dir_path, mode):
        super().__init__()
        self.converter = converter
        self.input_file = input_file
        self.output_dir_path = output_dir_path
        self.mode = mode

    def run(self):
        try:
            time.sleep(1)

            self.converter.coco_to_custom(
                input_file=self.input_file,
                output_dir_path=self.output_dir_path,
                mode=self.mode,
            )

            self.finished.emit(True, "")

        except Exception as e:
            self.finished.emit(False, str(e))


def upload_vlm_r1_ovd_annotation(self):
    if not _check_filename_exist(self):
        return

    filter = "Attribute Files (*.jsonl);;All Files (*)"
    input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a custom vlm_r1_ovd annotation file"),
        "",
        filter,
    )

    if not input_file:
        return

    output_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        output_dir_path = self.output_dir

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Cancel
        | QtWidgets.QMessageBox.StandardButton.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
        return

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    converter = LabelConverter()

    try:
        # parse input_data
        input_data = {}
        with jsonlines.open(input_file, "r") as reader:
            for data in list(reader):
                image_path = osp.basename(data["image"])
                input_data[image_path] = data["conversations"][1]["value"]

        for i, image_file in enumerate(image_list):
            image_filename = osp.basename(image_file)
            label_filename = osp.splitext(image_filename)[0] + ".json"
            output_file = osp.join(output_dir_path, label_filename)

            converter.vlm_r1_ovd_to_custom(
                input_data=input_data[image_filename],
                output_file=output_file,
                image_file=image_file,
            )

            progress_dialog.setValue(i)
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = self.tr(
            "Uploading annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % output_dir_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

        _refresh_after_annotation_upload(self)

    except Exception as e:
        progress_dialog.close()
        message = f"Error occurred while uploading annotations: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_ppocr_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    if mode == "rec":
        filter = "Attribute Files (*.txt);;All Files (*)"
        input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a custom annotation file (Label.txt)"),
            "",
            filter,
        )

        if not input_file:
            return

    elif mode == "kie":
        filter = "Attribute Files (*.json);;All Files (*)"
        input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a custom annotation file (ppocr_kie.json)"),
            "",
            filter,
        )

        if not input_file:
            return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Cancel
        | QtWidgets.QMessageBox.StandardButton.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    converter = LabelConverter()
    image_path = osp.dirname(self.filename)
    output_path = osp.dirname(self.filename)
    if self.output_dir:
        output_path = self.output_dir
    self.upload_thread = UploadPPOCRThread(
        converter, input_file, output_path, image_path, mode
    )

    def on_upload_finished(success, error_msg):
        progress_dialog.close()
        if success:
            _refresh_after_annotation_upload(self)

            popup = Popup(
                self.tr("Uploading annotations successfully!"),
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = (
                f"Error occurred while uploading annotations: {str(error_msg)}"
            )
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.upload_thread.finished.connect(on_upload_finished)

    progress_dialog.show()
    self.upload_thread.start()

    progress_dialog.canceled.connect(self.upload_thread.terminate)


def upload_odvg_annotation(self):
    if not _check_filename_exist(self):
        return

    filter = "OD Files (*.json *.jsonl);;All Files (*)"
    input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific OD file"),
        "",
        filter,
    )

    if not input_file:
        return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Cancel
        | QtWidgets.QMessageBox.StandardButton.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    converter = LabelConverter()
    output_path = osp.dirname(self.filename)
    if self.output_dir:
        output_path = self.output_dir
    self.upload_thread = UploadOdvgThread(converter, input_file, output_path)

    def on_upload_finished(success, error_msg):
        progress_dialog.close()
        if success:
            _refresh_after_annotation_upload(self)

            popup = Popup(
                self.tr("Uploading annotations successfully!"),
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = (
                f"Error occurred while uploading annotations: {str(error_msg)}"
            )
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.upload_thread.finished.connect(on_upload_finished)

    progress_dialog.show()
    self.upload_thread.start()

    progress_dialog.canceled.connect(self.upload_thread.terminate)


def upload_mmgd_annotation(self, LABEL_OPACITY):
    if not _check_filename_exist(self):
        return

    filter = "Classes Files (*.txt);;All Files (*)"
    classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific classes file"),
        "",
        filter,
    )
    if not classes_file:
        return

    try:
        with open(classes_file, "r", encoding="utf-8") as f:
            classes = f.read().splitlines()
    except Exception as e:
        popup = Popup(
            self.tr(f"Error reading classes file: {str(e)}"),
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    folder_layout = QVBoxLayout()
    folder_label = QtWidgets.QLabel(self.tr("Select Upload Folder"))
    folder_layout.addWidget(folder_label)

    folder_input_layout = QHBoxLayout()
    folder_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setPlaceholderText(
        self.tr("Please select a folder containing annotation files")
    )

    def browse_json_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            None,
            QtWidgets.QFileDialog.Option.ShowDirsOnly
            | QtWidgets.QFileDialog.Option.DontResolveSymlinks
            | QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    folder_button = QtWidgets.QPushButton(self.tr("Browse"))
    folder_button.clicked.connect(browse_json_folder)
    folder_button.setStyleSheet(get_cancel_btn_style())

    folder_input_layout.addWidget(path_edit)
    folder_input_layout.addWidget(folder_button)
    folder_layout.addLayout(folder_input_layout)
    layout.addLayout(folder_layout)

    category_layout = QVBoxLayout()
    category_label = QtWidgets.QLabel(self.tr("Category Settings"))
    category_layout.addWidget(category_label)

    scroll_area = QtWidgets.QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setMaximumHeight(200)
    scroll_widget = QtWidgets.QWidget()
    scroll_layout = QVBoxLayout(scroll_widget)

    category_widgets = {}
    for i, class_name in enumerate(classes):
        row_layout = QHBoxLayout()

        checkbox = QtWidgets.QCheckBox(class_name)
        checkbox.setChecked(True)

        threshold_label = QtWidgets.QLabel(self.tr("Threshold:"))
        threshold_input = QtWidgets.QDoubleSpinBox()
        threshold_input.setRange(0.0, 1.0)
        threshold_input.setSingleStep(0.01)
        threshold_input.setValue(0.20)
        threshold_input.setDecimals(2)

        row_layout.addWidget(checkbox)
        row_layout.addStretch()
        row_layout.addWidget(threshold_label)
        row_layout.addWidget(threshold_input)

        scroll_layout.addLayout(row_layout)
        category_widgets[class_name] = (checkbox, threshold_input)

    scroll_area.setWidget(scroll_widget)
    category_layout.addWidget(scroll_area)
    layout.addLayout(category_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))

    def on_ok_clicked():
        if not path_edit.text().strip():
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Upload folder path cannot be empty!"),
            )
            return

        dialog.accept()

    ok_button.clicked.connect(on_ok_clicked)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    labels = []
    thresholds = {}
    for class_name, (checkbox, threshold_input) in category_widgets.items():
        if checkbox.isChecked():
            labels.append(class_name)
            thresholds[class_name] = threshold_input.value()

    if not labels:
        popup = Popup(
            self.tr("Please select at least one category!"),
            self,
            icon=new_icon_path("warning", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    label_dir_path = path_edit.text()
    image_dir_path = osp.dirname(self.filename)
    image_file_list = os.listdir(image_dir_path)
    label_file_list = os.listdir(label_dir_path)
    output_dir_path = self.output_dir if self.output_dir else image_dir_path
    converter = LabelConverter(classes_file=classes_file)

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Cancel
        | QtWidgets.QMessageBox.StandardButton.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."),
        self.tr("Cancel"),
        0,
        len(image_file_list),
        self,
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
        for i, image_filename in enumerate(image_file_list):
            label_filename = osp.splitext(image_filename)[0] + ".json"
            if (
                image_filename.endswith(".json")
                or label_filename not in label_file_list
            ):
                continue

            converter.mmgd_to_custom(
                input_file=osp.join(label_dir_path, label_filename),
                output_file=osp.join(output_dir_path, label_filename),
                image_file=osp.join(image_dir_path, image_filename),
                labels=labels,
                thresholds=thresholds,
            )

            progress_dialog.setValue(i)
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = self.tr(
            "Uploading annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % output_dir_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

        for label in labels:
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, LABEL_OPACITY
                )

        _refresh_after_annotation_upload(self)

    except Exception as e:
        progress_dialog.close()
        message = f"Error occurred while uploading annotations: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_mot_annotation(self, LABEL_OPACITY):
    if not _check_filename_exist(self):
        return

    filter = "Classes Files (*.txt);;All Files (*)"
    classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific classes file"),
        "",
        filter,
    )

    if not classes_file:
        return

    filter = "Gt Files (*.txt);;All Files (*)"
    gt_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific gt file"),
        "",
        filter,
    )

    if not gt_file:
        popup = Popup(
            self.tr("Please select a specific gt file!"),
            self,
            icon=new_icon_path("warning", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Cancel
        | QtWidgets.QMessageBox.StandardButton.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
        return

    # Initialize unique labels
    with open(classes_file, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
        for label in labels:
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, LABEL_OPACITY
                )

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    converter = LabelConverter(classes_file=classes_file)
    image_path = osp.dirname(self.filename)
    output_path = image_path
    if self.output_dir:
        output_path = self.output_dir
    self.upload_thread = UploadMotThread(
        converter, gt_file, output_path, image_path
    )

    def on_upload_finished(success, error_msg):
        progress_dialog.close()
        if success:
            _refresh_after_annotation_upload(self)

            popup = Popup(
                self.tr("Uploading annotations successfully!"),
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = (
                f"Error occurred while uploading annotations: {str(error_msg)}"
            )
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.upload_thread.finished.connect(on_upload_finished)

    progress_dialog.show()
    self.upload_thread.start()

    progress_dialog.canceled.connect(self.upload_thread.terminate)


def upload_mask_annotation(self, LABEL_OPACITY):
    if not _check_filename_exist(self):
        return

    filter = "Attribute Files (*.json);;All Files (*)"
    color_map_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific color_map file"),
        "",
        filter,
    )

    if not color_map_file:
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Select Upload Folder"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.dirname(osp.dirname(self.filename)))

    def browse_upload_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.ShowDirsOnly
            | QtWidgets.QFileDialog.Option.DontResolveSymlinks
            | QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_upload_folder)
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

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Cancel
        | QtWidgets.QMessageBox.StandardButton.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
        return

    # Initialize unique labels
    with open(color_map_file, "r", encoding="utf-8") as f:
        mapping_table = json.load(f)
        classes = list(mapping_table["colors"].keys())
        for label in classes:
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, LABEL_OPACITY
                )

    label_dir_path = path_edit.text()
    image_dir_path = osp.dirname(self.filename)
    image_file_list = os.listdir(image_dir_path)
    label_file_list = os.listdir(label_dir_path)
    output_dir_path = self.output_dir if self.output_dir else image_dir_path
    converter = LabelConverter()

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, len(image_list), self
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
        for i, image_filename in enumerate(image_file_list):
            if image_filename.endswith(".json"):
                continue
            data_filename = osp.splitext(image_filename)[0] + ".json"
            if osp.splitext(image_filename)[0] + ".png" in label_file_list:
                label_filename = osp.splitext(image_filename)[0] + ".png"
            elif osp.splitext(image_filename)[0] + ".jpg" in label_file_list:
                label_filename = osp.splitext(image_filename)[0] + ".jpg"
            else:
                continue
            input_file = osp.join(label_dir_path, label_filename)
            output_file = osp.join(output_dir_path, data_filename)
            image_file = osp.join(image_dir_path, image_filename)
            converter.mask_to_custom(
                input_file=input_file,
                output_file=output_file,
                image_file=image_file,
                mapping_table=mapping_table,
            )

            progress_dialog.setValue(i)
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = self.tr(
            "Uploading annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % output_dir_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

        _refresh_after_annotation_upload(self)

    except Exception as e:
        progress_dialog.close()
        message = f"Error occurred while uploading annotations: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_dota_annotation(self):
    if not _check_filename_exist(self):
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Select Upload Folder"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.dirname(osp.dirname(self.filename)))

    def browse_upload_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.ShowDirsOnly
            | QtWidgets.QFileDialog.Option.DontResolveSymlinks
            | QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_upload_folder)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    # Button section
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

    label_dir_path = path_edit.text()
    image_dir_path = osp.dirname(self.filename)
    label_file_list = os.listdir(label_dir_path)
    output_dir_path = self.output_dir if self.output_dir else image_dir_path
    converter = LabelConverter()

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Cancel
        | QtWidgets.QMessageBox.StandardButton.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
        return

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, len(image_list), self
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
        for i, image_path in enumerate(image_list):
            image_filename = osp.basename(image_path)
            label_filename = osp.splitext(image_filename)[0] + ".txt"
            if label_filename not in label_file_list:
                continue

            input_file = osp.join(label_dir_path, label_filename)
            output_file = osp.join(
                output_dir_path, osp.splitext(image_filename)[0] + ".json"
            )
            image_file = osp.join(image_dir_path, image_filename)

            converter.dota_to_custom(
                input_file=input_file,
                output_file=output_file,
                image_file=image_file,
            )

            progress_dialog.setValue(i)
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = self.tr(
            "Uploading annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % output_dir_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

        _refresh_after_annotation_upload(self)

    except Exception as e:
        progress_dialog.close()
        message = f"Error occurred while uploading annotations: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def _upload_quad_txt_annotation(self, convert_fn):
    """Shared folder-picker + per-image loop for DOTA/WPOD txt uploads."""
    if not _check_filename_exist(self):
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Select Upload Folder"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.dirname(osp.dirname(self.filename)))

    def browse_upload_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.ShowDirsOnly
            | QtWidgets.QFileDialog.Option.DontResolveSymlinks
            | QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_upload_folder)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    # Button section
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

    label_dir_path = path_edit.text()
    label_index = {}
    for root, _, files in os.walk(label_dir_path):
        for name in files:
            if name.endswith(".txt") and name not in label_index:
                label_index[name] = osp.join(root, name)
    output_dir_path = self.output_dir if self.output_dir else None
    converter = LabelConverter()

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Cancel
        | QtWidgets.QMessageBox.StandardButton.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
        return

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, len(image_list), self
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
        imported, skipped = 0, 0
        for i, image_path in enumerate(image_list):
            image_filename = osp.basename(image_path)
            label_filename = osp.splitext(image_filename)[0] + ".txt"
            input_file = label_index.get(label_filename)
            if input_file is None:
                skipped += 1
                progress_dialog.setValue(i)
                QtWidgets.QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    break
                continue

            dest_dir = output_dir_path or osp.dirname(image_path)
            os.makedirs(dest_dir, exist_ok=True)
            output_file = osp.join(
                dest_dir, osp.splitext(image_filename)[0] + ".json"
            )

            convert_fn(
                converter,
                input_file=input_file,
                output_file=output_file,
                image_file=image_path,
            )
            imported += 1

            progress_dialog.setValue(i)
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = self.tr(
            "Uploading annotations successfully!\n"
            "Imported: %d, skipped (no label found): %d\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % (
            imported,
            skipped,
            output_dir_path or label_dir_path,
        )
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

        _refresh_after_annotation_upload(self)

    except Exception as e:
        progress_dialog.close()
        message = f"Error occurred while uploading annotations: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_wpod_annotation(self):
    """Upload WPOD/IWPOD quad annotations (``4,x1..x4,y1..y4,,``).

    Imports each sibling ``*.txt`` as ``quadrilateral`` shapes with the
    fixed ``plate`` label; out-of-range coords are clamped to [0,1].
    """
    _upload_quad_txt_annotation(
        self,
        lambda converter, input_file, output_file, image_file: converter.wpod_to_custom(
            input_file=input_file,
            output_file=output_file,
            image_file=image_file,
        ),
    )


def upload_coco_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    filter = "Attribute Files (*.json);;All Files (*)"
    input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a custom coco annotation file"),
        "",
        filter,
    )

    if not input_file:
        return

    output_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        output_dir_path = self.output_dir

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Cancel
        | QtWidgets.QMessageBox.StandardButton.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    self.upload_thread = UploadCocoThread(
        LabelConverter(), input_file, output_dir_path, mode
    )

    def on_upload_finished(success, error_msg):
        progress_dialog.close()
        if success:
            _refresh_after_annotation_upload(self)

            popup = Popup(
                self.tr("Uploading annotations successfully!"),
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = (
                f"Error occurred while uploading annotations: {str(error_msg)}"
            )
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.upload_thread.finished.connect(on_upload_finished)

    progress_dialog.show()
    self.upload_thread.start()

    progress_dialog.canceled.connect(self.upload_thread.terminate)


def upload_voc_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Select Upload Folder"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.dirname(osp.dirname(self.filename)))

    def browse_upload_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.ShowDirsOnly
            | QtWidgets.QFileDialog.Option.DontResolveSymlinks
            | QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_upload_folder)
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

    label_dir_path = path_edit.text()
    image_dir_path = osp.dirname(self.filename)
    label_file_list = os.listdir(label_dir_path)
    output_dir_path = self.output_dir if self.output_dir else image_dir_path
    converter = LabelConverter()

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Cancel
        | QtWidgets.QMessageBox.StandardButton.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
        return

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, len(image_list), self
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
        for i, image_path in enumerate(image_list):
            image_filename = osp.basename(image_path)
            label_filename = osp.splitext(image_filename)[0] + ".xml"
            if label_filename not in label_file_list:
                continue

            input_file = osp.join(label_dir_path, label_filename)
            output_file = osp.join(
                output_dir_path, osp.splitext(image_filename)[0] + ".json"
            )
            converter.voc_to_custom(
                input_file=input_file,
                output_file=output_file,
                image_filename=image_filename,
                mode=mode,
            )

            progress_dialog.setValue(i)
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = self.tr(
            "Uploading annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % output_dir_path
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

        _refresh_after_annotation_upload(self)

    except Exception as e:
        progress_dialog.close()
        message = f"Error occurred while uploading annotations: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_yolo_annotation(self, mode, LABEL_OPACITY):
    if not _check_filename_exist(self):
        return

    labels = []
    if mode == "pose":
        filter = "Classes Files (*.yaml);;All Files (*)"
        self.yaml_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific yolo-pose config file"),
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

        for class_name, keypoint_name in converter.pose_classes.items():
            labels.append(class_name)
            labels.extend(keypoint_name)

    elif mode in ["hbb", "obb", "seg"]:
        filter = "Classes Files (*.txt *.names);;All Files (*)"
        initial_file = ""
        if self.filename:
            img_dir = osp.dirname(self.filename)
            for cand in ["classes.txt", "obj.names", "coco.names"]:
                p = osp.join(img_dir, cand)
                if osp.exists(p):
                    initial_file = p
                    break
                p2 = osp.join(osp.dirname(img_dir), cand)
                if osp.exists(p2):
                    initial_file = p2
                    break

        self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a classes file (classes.txt / obj.names)"),
            initial_file,
            filter,
        )
        if not self.classes_file:
            return

        with open(self.classes_file, "r", encoding="utf-8") as f:
            labels = [
                line.strip() for line in f.read().splitlines() if line.strip()
            ]
        converter = LabelConverter(classes=labels)

    # Resolve dataset images and roots
    image_list = (
        list(self.image_list)
        if hasattr(self, "image_list") and self.image_list
        else [self.filename]
    )
    source_root = osp.dirname(osp.abspath(self.filename))
    if hasattr(self, "last_open_dir") and self.last_open_dir:
        try:
            if osp.commonpath(
                (self.last_open_dir, source_root)
            ) == osp.abspath(self.last_open_dir):
                source_root = osp.abspath(self.last_open_dir)
        except ValueError:
            pass

    # Auto-detect layout: inspect a sample of images
    sample_images = image_list[:25]
    darknet_hits = 0
    ultralytics_hits = 0
    detected_ultralytics_folder = ""

    cand_ultralytics_roots = []
    parent_source = osp.dirname(source_root)
    for cand in [
        osp.join(parent_source, "labels"),
        osp.join(source_root, "labels"),
    ]:
        if osp.exists(cand) and cand not in cand_ultralytics_roots:
            cand_ultralytics_roots.append(cand)

    for img in sample_images:
        txt_darknet = osp.splitext(img)[0] + ".txt"
        if osp.exists(txt_darknet):
            darknet_hits += 1

        txt_ultra1 = (
            re.sub(
                r"([/\\])images([/\\])", r"\1labels\2", osp.splitext(img)[0]
            )
            + ".txt"
        )
        if txt_ultra1 != txt_darknet and osp.exists(txt_ultra1):
            ultralytics_hits += 1
            m = re.search(r"(.*?)[/\\]labels[/\\]", txt_ultra1)
            if m:
                detected_ultralytics_folder = osp.join(m.group(1), "labels")

        try:
            rel = osp.relpath(img, source_root)
            rel_txt = osp.splitext(rel)[0] + ".txt"
            for u_root in cand_ultralytics_roots:
                if osp.exists(osp.join(u_root, rel_txt)):
                    ultralytics_hits += 1
                    if not detected_ultralytics_folder:
                        detected_ultralytics_folder = u_root
                    break
        except Exception:
            pass

    if ultralytics_hits > 0 and ultralytics_hits >= darknet_hits:
        detected_layout = "ultralytics"
        detected_path = detected_ultralytics_folder or (
            cand_ultralytics_roots[0]
            if cand_ultralytics_roots
            else osp.join(parent_source, "labels")
        )
    elif darknet_hits > 0:
        detected_layout = "darknet"
        detected_path = source_root
    elif cand_ultralytics_roots:
        detected_layout = "ultralytics"
        detected_path = cand_ultralytics_roots[0]
    else:
        detected_layout = "darknet"
        detected_path = source_root

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(540)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    # Info summary
    summary_text = (
        f"{len(labels)} classes loaded: "
        + ", ".join(labels[:5])
        + (f", ... (+{len(labels) - 5})" if len(labels) > 5 else "")
    )
    summary_label = QtWidgets.QLabel(summary_text)
    summary_label.setStyleSheet("font-size: 11px; opacity: 0.8;")
    layout.addWidget(summary_label)

    # Format layout selection
    format_label = QtWidgets.QLabel(
        self.tr("YOLO Layout (Auto-detected: %s):") % detected_layout.title()
    )
    format_label.setStyleSheet("font-weight: bold;")
    layout.addWidget(format_label)

    format_layout = QVBoxLayout()
    ultralytics_radio = QtWidgets.QRadioButton(
        self.tr("Ultralytics Format (Labels in separate labels/ folder)")
    )
    darknet_radio = QtWidgets.QRadioButton(
        self.tr("Darknet Format (Labels alongside images in same directory)")
    )
    custom_radio = QtWidgets.QRadioButton(self.tr("Custom Folder"))

    if detected_layout == "ultralytics":
        ultralytics_radio.setChecked(True)
    else:
        darknet_radio.setChecked(True)

    format_layout.addWidget(ultralytics_radio)
    format_layout.addWidget(darknet_radio)
    format_layout.addWidget(custom_radio)
    layout.addLayout(format_layout)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Labels Folder:"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(detected_path)

    def on_format_changed():
        if ultralytics_radio.isChecked():
            path_edit.setText(
                detected_ultralytics_folder
                or (
                    cand_ultralytics_roots[0]
                    if cand_ultralytics_roots
                    else osp.join(parent_source, "labels")
                )
            )
        elif darknet_radio.isChecked():
            path_edit.setText(source_root)

    ultralytics_radio.toggled.connect(on_format_changed)
    darknet_radio.toggled.connect(on_format_changed)

    def browse_upload_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.ShowDirsOnly
            | QtWidgets.QFileDialog.Option.DontResolveSymlinks
            | QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)
            custom_radio.setChecked(True)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_upload_folder)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    preserve_checkbox = QtWidgets.QCheckBox(
        self.tr("Preserve existing annotations")
    )
    preserve_checkbox.setChecked(False)
    layout.addWidget(preserve_checkbox)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("Upload"))
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

    selected_folder = path_edit.text()
    preserve_existing = preserve_checkbox.isChecked()
    layout_mode = (
        "ultralytics"
        if ultralytics_radio.isChecked()
        else ("darknet" if darknet_radio.isChecked() else "custom")
    )

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    response.setWindowTitle(self.tr("Warning"))
    if preserve_existing:
        response.setText(
            self.tr("New annotations will be merged with existing ones")
        )
        response.setInformativeText(
            self.tr(
                "You are going to add new annotations to this task. Existing annotations will be preserved. Continue?"
            )
        )
    else:
        response.setText(self.tr("Current annotations may be overwritten"))
        response.setInformativeText(
            self.tr(
                "You are going to upload new annotations to this task. Continue?"
            )
        )
    response.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Cancel
        | QtWidgets.QMessageBox.StandardButton.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Uploading annotations..."),
        self.tr("Cancel"),
        0,
        len(image_list),
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    uploaded_count = 0
    try:
        for i, image_file in enumerate(image_list):
            if image_file.endswith(".json"):
                continue

            try:
                rel_path = osp.relpath(image_file, source_root)
            except ValueError:
                rel_path = osp.basename(image_file)
            rel_base = osp.splitext(rel_path)[0]
            img_base = osp.splitext(osp.basename(image_file))[0]

            candidate_files = []
            if layout_mode == "darknet":
                candidate_files.append(osp.splitext(image_file)[0] + ".txt")
                candidate_files.append(
                    osp.join(selected_folder, rel_base + ".txt")
                )
                candidate_files.append(
                    osp.join(selected_folder, img_base + ".txt")
                )
            elif layout_mode == "ultralytics":
                candidate_files.append(
                    osp.join(selected_folder, rel_base + ".txt")
                )
                candidate_files.append(
                    re.sub(
                        r"([/\\])images([/\\])",
                        r"\1labels\2",
                        osp.splitext(image_file)[0],
                    )
                    + ".txt"
                )
                candidate_files.append(
                    osp.join(selected_folder, img_base + ".txt")
                )
            else:  # custom
                candidate_files.append(
                    osp.join(selected_folder, rel_base + ".txt")
                )
                candidate_files.append(
                    osp.join(selected_folder, img_base + ".txt")
                )

            input_file = None
            for cand in candidate_files:
                if osp.exists(cand):
                    input_file = cand
                    break

            if not input_file:
                progress_dialog.setValue(i)
                QtWidgets.QApplication.processEvents()
                continue

            if self.output_dir:
                output_file = osp.join(self.output_dir, rel_base + ".json")
            else:
                output_file = osp.splitext(image_file)[0] + ".json"

            os.makedirs(osp.dirname(output_file), exist_ok=True)

            existing_shapes = []
            if preserve_existing and osp.exists(output_file):
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    existing_shapes = existing_data.get("shapes", [])
                except Exception:
                    existing_shapes = []

            if mode in ["hbb", "seg"]:
                converter.yolo_to_custom(
                    input_file=input_file,
                    output_file=output_file,
                    image_file=image_file,
                    mode=mode,
                )
            elif mode == "obb":
                converter.yolo_obb_to_custom(
                    input_file=input_file,
                    output_file=output_file,
                    image_file=image_file,
                )
            elif mode == "pose":
                converter.yolo_pose_to_custom(
                    input_file=input_file,
                    output_file=output_file,
                    image_file=image_file,
                )

            if preserve_existing and existing_shapes:
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        new_data = json.load(f)
                    new_data["shapes"] = existing_shapes + new_data.get(
                        "shapes", []
                    )
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(new_data, f, indent=2, ensure_ascii=False)
                except Exception as err:
                    logger.warning(
                        f"Failed to merge existing shapes in {output_file}: {err}"
                    )

            uploaded_count += 1
            progress_dialog.setValue(i)
            QtWidgets.QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()

        if (
            labels
            and hasattr(self, "unique_label_list")
            and self.unique_label_list
        ):
            if hasattr(self.unique_label_list, "clear") and callable(
                self.unique_label_list.clear
            ):
                self.unique_label_list.clear()
            for idx, label in enumerate(labels):
                if hasattr(self.unique_label_list, "create_item_from_label"):
                    item = self.unique_label_list.create_item_from_label(label)
                    self.unique_label_list.addItem(item)
                    rgb = (
                        self._get_rgb_by_label(label)
                        if hasattr(self, "_get_rgb_by_label")
                        else (255, 0, 0)
                    )
                    self.unique_label_list.set_item_label(
                        item, label, rgb, LABEL_OPACITY, index=idx
                    )
            if hasattr(self.unique_label_list, "refresh_indices") and callable(
                self.unique_label_list.refresh_indices
            ):
                self.unique_label_list.refresh_indices()
            if hasattr(self, "_config") and isinstance(self._config, dict):
                self._config["labels"] = labels
                from anylabeling.config import save_config

                save_config(self._config)

        _refresh_after_annotation_upload(self)
        popup = Popup(
            self.tr(
                f"Upload completed successfully!\n"
                f"{uploaded_count} files uploaded with {len(labels)} classes."
            ),
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, position="center")

    except Exception as e:
        progress_dialog.close()
        message = f"Error occurred while uploading annotations: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_label_classes_file(self):
    filter = "Label Files (*.txt);;All Files (*)"
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific label classes file"),
        "",
        filter,
    )
    if not file_path:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]

        if not labels:
            popup = Popup(
                self.tr("No labels found in the file!"),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")
            return

        response = QtWidgets.QMessageBox()
        response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        response.setWindowTitle(self.tr("Warning"))
        response.setText(self.tr("Current labels will be lost"))
        response.setInformativeText(
            self.tr(
                "You are going to upload new labels to this task. Continue?"
            )
        )
        response.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Cancel
            | QtWidgets.QMessageBox.StandardButton.Ok
        )
        response.setStyleSheet(get_msg_box_style())

        if response.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
            return

        # Update unique_label_list
        self.unique_label_list.clear()
        self.load_labels(labels)

        # Update label_dialog.label_list
        self.label_dialog.label_list.clear()
        self.label_dialog.label_list.addItems(labels)
        if self.label_dialog._sort_labels:
            self.label_dialog.sort_labels()

        popup = Popup(
            self.tr(f"Successfully loaded {len(set(labels))} labels!"),
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, position="center")

    except Exception as e:
        message = (
            f"Error occurred while uploading label classes file: {str(e)}"
        )
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def validate_shape_attributes_config(attributes_data):
    """
    Validates and separates a shape attributes configuration.

    Args:
        attributes_data (dict): Parsed shape attributes configuration.

    Returns:
        tuple: Attribute definitions and widget type mappings.

    Raises:
        ValueError: If the configuration structure is invalid.
    """

    def invalid(label, property_name, widget_type, value, expected):
        raise ValueError(
            "Invalid shape attribute configuration: "
            f"label={label!r}, property={property_name!r}, "
            f"widget_type={widget_type!r}, "
            f"actual_type={type(value).__name__!r}, "
            f"expected={expected!r}"
        )

    if not isinstance(attributes_data, dict):
        invalid(
            "<root>",
            "<root>",
            "<none>",
            attributes_data,
            "a dictionary",
        )

    attributes = {
        label: properties
        for label, properties in attributes_data.items()
        if not label.startswith("__")
    }
    for label, properties in attributes.items():
        if not isinstance(properties, dict):
            invalid(
                label,
                "<all>",
                "<unknown>",
                properties,
                "a dictionary of attribute definitions",
            )

    widget_types = attributes_data.get("__widget_types__", {})
    if not isinstance(widget_types, dict):
        invalid(
            "__widget_types__",
            "<all>",
            "<mapping>",
            widget_types,
            "a dictionary of label widget mappings",
        )

    supported_widget_types = {
        "combobox",
        "radiobutton",
        "group_id",
        "lineedit",
    }
    for label, property_types in widget_types.items():
        if label not in attributes:
            invalid(
                label,
                "<all>",
                "<mapping>",
                None,
                "an existing attribute label",
            )
        if not isinstance(property_types, dict):
            invalid(
                label,
                "<all>",
                "<mapping>",
                property_types,
                "a dictionary of property widget types",
            )
        for property_name, widget_type in property_types.items():
            if property_name not in attributes[label]:
                invalid(
                    label,
                    property_name,
                    widget_type,
                    None,
                    "an existing attribute property",
                )
            if (
                not isinstance(widget_type, str)
                or widget_type not in supported_widget_types
            ):
                invalid(
                    label,
                    property_name,
                    widget_type,
                    widget_type,
                    "one of combobox, radiobutton, group_id, or lineedit",
                )

    for label, properties in attributes.items():
        property_types = widget_types.get(label, {})
        for property_name, value in properties.items():
            widget_type = property_types.get(property_name, "combobox")
            if widget_type in {"combobox", "radiobutton"}:
                if (
                    not isinstance(value, list)
                    or not value
                    or not all(isinstance(option, str) for option in value)
                ):
                    invalid(
                        label,
                        property_name,
                        widget_type,
                        value,
                        "a non-empty list of strings",
                    )
            elif widget_type == "lineedit" and not isinstance(value, str):
                invalid(
                    label,
                    property_name,
                    widget_type,
                    value,
                    "a string",
                )
            elif widget_type == "group_id" and value != []:
                invalid(
                    label,
                    property_name,
                    widget_type,
                    value,
                    "an empty list",
                )

    return attributes, widget_types


def upload_shape_attrs_file(self, LABEL_OPACITY):
    filter = "Shape Attributes Files (*.json);;All Files (*)"
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific shape attributes file"),
        "",
        filter,
    )
    if not file_path:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            attributes_data = json.load(f)

        attributes, widget_types = validate_shape_attributes_config(
            attributes_data
        )
        self.attributes = attributes
        self.attribute_widget_types = widget_types
        for label in self.attributes:
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, LABEL_OPACITY
                )

        # update the shape attributes dialog
        self.shape_attributes.show()
        selected_shapes = self.canvas.selected_shapes
        is_drawing_mode = (
            hasattr(self.canvas, "current") and self.canvas.current is not None
        )
        if (
            len(selected_shapes) == 1
            and selected_shapes[0] in self.canvas.shapes
            and not is_drawing_mode
        ):
            self.update_attributes(
                self.canvas.shapes.index(selected_shapes[0])
            )
        else:
            self.hide_attributes_panel()
        self.canvas.h_shape_is_hovered = False

        popup = Popup(
            self.tr("Uploading shape attributes file successfully!"),
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = (
            f"Error occurred while uploading shape attributes file: {str(e)}"
        )
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_label_flags_file(self, LABEL_OPACITY):
    filter = "Label Flags Files (*.yaml);;All Files (*)"
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific flags file"),
        "",
        filter,
    )
    if not file_path:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Each line in the file is an flag-level flag
            self.label_flags = yaml.safe_load(f)
            for label in list(self.label_flags.keys()):
                if not self.unique_label_list.find_items_by_label(label):
                    item = self.unique_label_list.create_item_from_label(label)
                    self.unique_label_list.addItem(item)
                    rgb = self._get_rgb_by_label(label)
                    self.unique_label_list.set_item_label(
                        item, label, rgb, LABEL_OPACITY
                    )

        # update the label dialog
        self.label_dialog.upload_flags(self.label_flags)

        popup = Popup(
            self.tr("Uploading flags file successfully!"),
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = f"Error occurred while uploading flags file: {str(e)}"
        logger.error(message)

        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def upload_image_flags_file(self):
    filter = "Image Flags Files (*.txt);;All Files (*)"
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific flags file"),
        "",
        filter,
    )
    if not file_path:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Each line in the file is an image-level flag
            self.image_flags = f.read().splitlines()
            self.load_flags({k: False for k in self.image_flags})
        self.flag_dock.show()

        self.load_file(self.filename)

        popup = Popup(
            self.tr("Uploading flags file successfully!"),
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = f"Error occurred while uploading flags file: {str(e)}"
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
