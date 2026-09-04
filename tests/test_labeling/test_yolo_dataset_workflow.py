import json
import os
import shutil
import tempfile
import unittest
from unittest import mock
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from anylabeling.views.labeling.label_converter import LabelConverter
from anylabeling.views.labeling.utils.export import (
    _get_yolo_export_files,
    _validate_yolo_export_path,
)
from anylabeling.views.labeling.widgets.label_dialog import LabelModifyDialog
from anylabeling.views.labeling.widgets.unique_label_qlist_widget import (
    UniqueLabelQListWidget,
)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class TestLabelConverterYolo(unittest.TestCase):
    def test_direct_classes_initialization(self):
        converter = LabelConverter(
            classes=["vehicle", "pedestrian", "cyclist"]
        )
        self.assertEqual(
            converter.classes, ["vehicle", "pedestrian", "cyclist"]
        )

    def test_custom_to_yolo_hbb_envelope(self):
        converter = LabelConverter(classes=["box", "poly"])
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "sample.json")
            txt_path = os.path.join(tmpdir, "sample.txt")

            data = {
                "imageWidth": 100,
                "imageHeight": 100,
                "shapes": [
                    {
                        "label": "box",
                        "shape_type": "rectangle",
                        "points": [[20, 20], [80, 20], [80, 60], [20, 60]],
                    },
                    {
                        "label": "poly",
                        "shape_type": "polygon",
                        "points": [[10, 10], [50, 90], [90, 50]],
                    },
                ],
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            converter.custom_to_yolo(json_path, txt_path, mode="hbb")

            self.assertTrue(os.path.exists(txt_path))
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [line.strip().split() for line in f.readlines()]

            self.assertEqual(len(lines), 2)
            # Check rectangle box (class 0, center=(0.5, 0.4), w=0.6, h=0.4)
            self.assertEqual(lines[0][0], "0")
            self.assertAlmostEqual(float(lines[0][1]), 0.5, places=3)
            self.assertAlmostEqual(float(lines[0][2]), 0.4, places=3)
            self.assertAlmostEqual(float(lines[0][3]), 0.6, places=3)
            self.assertAlmostEqual(float(lines[0][4]), 0.4, places=3)

            # Check polygon box (class 1, xs=[10, 50, 90] -> min=10, max=90 -> center=0.5, w=0.8;
            # ys=[10, 90, 50] -> min=10, max=90 -> center=0.5, h=0.8)
            self.assertEqual(lines[1][0], "1")
            self.assertAlmostEqual(float(lines[1][1]), 0.5, places=3)
            self.assertAlmostEqual(float(lines[1][2]), 0.5, places=3)
            self.assertAlmostEqual(float(lines[1][3]), 0.8, places=3)
            self.assertAlmostEqual(float(lines[1][4]), 0.8, places=3)


class TestYoloExportLayouts(unittest.TestCase):
    def test_get_yolo_export_files_in_place(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(tmpdir, "dataset")
            sub_dir = os.path.join(dataset_dir, "train")
            os.makedirs(sub_dir)

            img1 = os.path.join(sub_dir, "img1.jpg")
            open(img1, "w").close()

            # In-place export (save alongside images)
            export_files = _get_yolo_export_files(
                [img1],
                dataset_dir,
                dataset_dir,
            )
            self.assertEqual(len(export_files), 1)
            image_file, dst_file, image_dst = export_files[0]
            self.assertEqual(
                dst_file, os.path.join(dataset_dir, "train", "img1.txt")
            )
            self.assertEqual(image_dst, img1)

    def test_get_yolo_export_files_separate_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(tmpdir, "dataset")
            img_dir = os.path.join(dataset_dir, "images", "val")
            export_dir = os.path.join(tmpdir, "export")
            os.makedirs(img_dir)

            img1 = os.path.join(img_dir, "sample.png")
            open(img1, "w").close()

            export_files = _get_yolo_export_files(
                [img1],
                dataset_dir,
                export_dir,
            )
            self.assertEqual(len(export_files), 1)
            image_file, dst_file, image_dst = export_files[0]
            self.assertEqual(
                dst_file,
                os.path.join(export_dir, "images", "val", "sample.txt"),
            )
            self.assertEqual(
                image_dst,
                os.path.join(export_dir, "images", "val", "sample.png"),
            )

    def test_export_yolo_writes_classes_txt_only(self):
        from anylabeling.views.labeling.utils.export import (
            export_yolo_annotation,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = os.path.join(tmpdir, "images")
            os.makedirs(image_dir)
            img = os.path.join(image_dir, "img1.png")
            json_file = os.path.join(image_dir, "img1.json")
            open(img, "w").close()
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"imageWidth": 100, "imageHeight": 100, "shapes": []}, f
                )

            classes_file = os.path.join(tmpdir, "classes.txt")
            with open(classes_file, "w", encoding="utf-8") as f:
                f.write("car\nperson\n")

            widget = QtWidgets.QWidget()
            widget.filename = img
            widget.image_list = [img]
            widget.last_open_dir = image_dir
            widget.output_dir = None
            widget.may_continue = mock.Mock(return_value=True)

            with (
                mock.patch.object(
                    QtWidgets.QFileDialog,
                    "getOpenFileName",
                    return_value=(classes_file, ""),
                ),
                mock.patch.object(QtWidgets.QDialog, "exec", return_value=1),
                mock.patch("anylabeling.views.labeling.utils.export.Popup"),
            ):
                export_yolo_annotation(widget, "hbb")

            labels_dir = os.path.join(tmpdir, "labels")
            self.assertTrue(
                os.path.exists(os.path.join(labels_dir, "classes.txt"))
            )
            self.assertFalse(
                os.path.exists(os.path.join(labels_dir, "obj.names"))
            )
            self.assertFalse(
                os.path.exists(os.path.join(labels_dir, "data.yaml"))
            )
            widget.close()


class MockLabelingParent(QtWidgets.QWidget):
    def __init__(self, image_list, config_labels):
        super().__init__()
        self.output_dir = None
        self.image_list = image_list
        self.label_info = {}
        self._config = {"labels": config_labels}
        self.unique_label_list = UniqueLabelQListWidget()

    def _get_rgb_by_label(self, label):
        return (255, 0, 0)


class TestLabelManagerLogic(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.img1 = os.path.join(self.tmpdir, "img1.jpg")
        self.img2 = os.path.join(self.tmpdir, "img2.jpg")
        self.json1 = os.path.join(self.tmpdir, "img1.json")
        self.json2 = os.path.join(self.tmpdir, "img2.json")

        open(self.img1, "w").close()
        open(self.img2, "w").close()

        with open(self.json1, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "shapes": [
                        {
                            "label": "dog",
                            "shape_type": "rectangle",
                            "points": [[0, 0], [10, 10]],
                        },
                        {
                            "label": "cat",
                            "shape_type": "rectangle",
                            "points": [[10, 10], [20, 20]],
                        },
                    ]
                },
                f,
            )

        with open(self.json2, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "shapes": [
                        {
                            "label": "dog",
                            "shape_type": "rectangle",
                            "points": [[5, 5], [15, 15]],
                        },
                    ]
                },
                f,
            )

        self.mock_parent = MockLabelingParent(
            image_list=[self.img1, self.img2],
            config_labels=["dog", "cat"],
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_init_label_info_counts(self):
        dialog = LabelModifyDialog(parent=self.mock_parent)
        self.assertEqual(dialog.shape_counts["dog"], 2)
        self.assertEqual(dialog.image_counts["dog"], 2)
        self.assertEqual(dialog.shape_counts["cat"], 1)
        self.assertEqual(dialog.image_counts["cat"], 1)

    def test_unique_label_list_clean_display(self):
        widget = UniqueLabelQListWidget()
        for idx, lbl in enumerate(["first", "second", "third"]):
            item = widget.create_item_from_label(lbl)
            widget.addItem(item)
            widget.set_item_label(item, lbl, index=idx)

        widget.refresh_indices()
        label_widget = widget.itemWidget(widget.item(0))
        self.assertNotIn("[0]", label_widget.text())
        self.assertEqual(label_widget.text(), "first")
        self.assertEqual(
            widget.item(0).data(Qt.ItemDataRole.UserRole), "first"
        )

        label_widget1 = widget.itemWidget(widget.item(1))
        self.assertNotIn("[1]", label_widget1.text())
        self.assertEqual(label_widget1.text(), "second")
