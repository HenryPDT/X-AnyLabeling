import json
import os
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import yaml
from PyQt6 import QtWidgets

from anylabeling.views.labeling.utils.export import (
    ExportThread,
    _coco_split_filename,
    _export_mask_files,
    _get_yolo_export_files,
    _show_yolo_export_error,
    _validate_yolo_export_path,
    export_coco_annotation,
    export_yolo_annotation,
)
from anylabeling.views.labeling.label_converter import (
    LabelConverter,
    PoseClassError,
    PoseGroupError,
)
from anylabeling.views.labeling.utils.split import (
    collect_labels_and_names,
    stratified_split,
)


@pytest.mark.parametrize(
    ("checked", "expected"),
    [
        (True, True),
        (False, False),
        ("false", False),
        (1, False),
        (None, False),
    ],
)
def test_export_mask_files_only_accepts_checked_true(
    tmp_path, checked, expected
):
    image_file = tmp_path / "image.png"
    label_file = tmp_path / "image.json"
    image_file.touch()
    label_file.touch()
    converter = mock.Mock()
    converter.read_json.return_value = {"checked": checked}
    progress_dialog = mock.Mock()

    _export_mask_files(
        converter,
        [str(image_file)],
        None,
        str(tmp_path / "masks"),
        {"type": "grayscale", "colors": {}},
        include_null_images=False,
        only_checked_images=True,
        progress_dialog=progress_dialog,
    )

    assert converter.custom_to_mask.called is expected


@pytest.mark.parametrize(
    ("include_null_images", "only_checked_images", "expected"),
    [
        (False, False, False),
        (True, False, True),
        (True, True, False),
    ],
)
def test_export_mask_files_handles_images_without_labels(
    tmp_path, include_null_images, only_checked_images, expected
):
    image_file = tmp_path / "image.png"
    image_file.touch()
    converter = mock.Mock()
    progress_dialog = mock.Mock()

    _export_mask_files(
        converter,
        [str(image_file)],
        None,
        str(tmp_path / "masks"),
        {"type": "grayscale", "colors": {}},
        include_null_images=include_null_images,
        only_checked_images=only_checked_images,
        progress_dialog=progress_dialog,
    )

    assert converter.custom_image_to_empty_mask.called is expected


def test_export_mask_files_stops_after_cancellation(tmp_path):
    image_files = [tmp_path / "first.png", tmp_path / "second.png"]
    for image_file in image_files:
        image_file.touch()
    converter = mock.Mock()
    progress_dialog = mock.Mock()
    progress_dialog.wasCanceled.return_value = True

    _export_mask_files(
        converter,
        [str(image_file) for image_file in image_files],
        None,
        str(tmp_path / "masks"),
        {"type": "grayscale", "colors": {}},
        include_null_images=True,
        only_checked_images=False,
        progress_dialog=progress_dialog,
    )

    converter.custom_image_to_empty_mask.assert_called_once()
    progress_dialog.setValue.assert_called_once_with(1)


def test_yolo_export_does_not_blame_last_image_for_popup_error(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_file = image_dir / "image.png"
    image_file.touch()
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("person\n", encoding="utf-8")

    widget = QtWidgets.QWidget()
    widget.filename = str(image_file)
    widget.image_list = [str(image_file)]
    widget.output_dir = str(image_dir)
    widget.may_continue = mock.Mock(return_value=True)
    converter = mock.Mock()
    converter.custom_to_yolo.return_value = False
    popup = mock.Mock()
    popup.show_popup.side_effect = RuntimeError("Popup failed")

    with (
        mock.patch.object(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            return_value=(str(classes_file), ""),
        ),
        mock.patch.object(QtWidgets.QDialog, "exec", return_value=1),
        mock.patch(
            "anylabeling.views.labeling.utils.export.LabelConverter",
            return_value=converter,
        ),
        mock.patch(
            "anylabeling.views.labeling.utils.export.Popup",
            return_value=popup,
        ),
        mock.patch(
            "anylabeling.views.labeling.utils.export._show_yolo_export_error"
        ) as show_export_error,
    ):
        export_yolo_annotation(widget, "hbb")

    show_export_error.assert_called_once()
    parent, image_file, error = show_export_error.call_args.args
    assert parent is widget
    assert image_file is None
    assert isinstance(error, RuntimeError)
    widget.close()
    app.processEvents()


def test_yolo_export_reports_failed_image(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    current_image = image_dir / "current.png"
    failed_image = image_dir / "failed.png"
    current_image.touch()
    failed_image.touch()
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("person\n", encoding="utf-8")

    widget = QtWidgets.QWidget()
    widget.filename = str(current_image)
    widget.image_list = [str(failed_image)]
    widget.output_dir = str(image_dir)
    widget.may_continue = mock.Mock(return_value=True)
    converter = mock.Mock()
    converter.custom_to_yolo.side_effect = PoseGroupError(
        "group_id is None for pose annotation"
    )

    with (
        mock.patch.object(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            return_value=(str(classes_file), ""),
        ),
        mock.patch.object(QtWidgets.QDialog, "exec", return_value=1),
        mock.patch(
            "anylabeling.views.labeling.utils.export.LabelConverter",
            return_value=converter,
        ),
        mock.patch(
            "anylabeling.views.labeling.utils.export._show_yolo_export_error"
        ) as show_export_error,
    ):
        export_yolo_annotation(widget, "pose")

    show_export_error.assert_called_once_with(
        widget, str(failed_image), converter.custom_to_yolo.side_effect
    )
    widget.close()
    app.processEvents()


@pytest.mark.parametrize("mode", ["hbb", "obb", "seg", "pose"])
def test_yolo_export_preserves_nested_directories(tmp_path, mode):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image_dir = tmp_path / "images"
    first_image = image_dir / "first" / "image.png"
    second_image = image_dir / "second" / "image.png"
    for image_file in (first_image, second_image):
        image_file.parent.mkdir(parents=True, exist_ok=True)
        image_file.touch()
        image_file.with_suffix(".json").touch()
    config_file = tmp_path / ("pose.yaml" if mode == "pose" else "classes.txt")
    config_file.write_text(
        "classes:\n  person:\n    - nose\n" if mode == "pose" else "person\n",
        encoding="utf-8",
    )

    widget = QtWidgets.QWidget()
    widget.filename = str(first_image)
    widget.image_list = [str(first_image), str(second_image)]
    widget.last_open_dir = str(image_dir)
    widget.output_dir = None
    widget.may_continue = mock.Mock(return_value=True)
    converter = mock.Mock()
    converter.custom_to_yolo.return_value = False
    converter.read_json.return_value = {
        "imageWidth": 100,
        "imageHeight": 100,
        "shapes": [],
    }

    with (
        mock.patch.object(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            return_value=(str(config_file), ""),
        ),
        mock.patch.object(QtWidgets.QDialog, "exec", return_value=1),
        mock.patch.object(
            # Order: split preview refresh, persisted-pref check,
            # save-with-images, skip-empty, split enable.
            QtWidgets.QCheckBox,
            "isChecked",
            side_effect=(False, False, True, False, False),
        ),
        mock.patch(
            "anylabeling.views.labeling.utils.export.LabelConverter",
            return_value=converter,
        ),
        mock.patch("anylabeling.views.labeling.utils.export.Popup"),
    ):
        export_yolo_annotation(widget, mode)

    save_path = tmp_path / "labels"
    expected_calls = [
        mock.call(
            str(first_image.with_suffix(".json")),
            str(save_path / "first" / "image.txt"),
            mode,
            skip_empty_files=False,
            obb_boundary_policy="keep",
        ),
        mock.call(
            str(second_image.with_suffix(".json")),
            str(save_path / "second" / "image.txt"),
            mode,
            skip_empty_files=False,
            obb_boundary_policy="keep",
        ),
    ]
    assert converter.custom_to_yolo.call_args_list == expected_calls
    assert (save_path / "first").is_dir()
    assert (save_path / "second").is_dir()
    assert (save_path / "first" / "image.png").is_file()
    assert (save_path / "second" / "image.png").is_file()
    widget.close()
    app.processEvents()


def test_yolo_export_rejects_conflicting_label_paths(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    first_image = image_dir / "image.jpg"
    second_image = image_dir / "image.png"
    first_image.touch()
    second_image.touch()
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("person\n", encoding="utf-8")

    widget = QtWidgets.QWidget()
    widget.filename = str(first_image)
    widget.image_list = [str(first_image), str(second_image)]
    widget.last_open_dir = str(image_dir)
    widget.output_dir = None
    widget.may_continue = mock.Mock(return_value=True)
    converter = mock.Mock()

    with (
        mock.patch.object(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            return_value=(str(classes_file), ""),
        ),
        mock.patch.object(QtWidgets.QDialog, "exec", return_value=1),
        mock.patch(
            "anylabeling.views.labeling.utils.export.LabelConverter",
            return_value=converter,
        ),
        mock.patch(
            "anylabeling.views.labeling.utils.export._show_yolo_export_error"
        ) as show_export_error,
    ):
        export_yolo_annotation(widget, "hbb")

    show_export_error.assert_called_once()
    parent, image_file, error = show_export_error.call_args.args
    assert parent is widget
    assert image_file is None
    assert isinstance(error, ValueError)
    assert str(first_image) in str(error)
    assert str(second_image) in str(error)
    converter.custom_to_yolo.assert_not_called()
    assert not (tmp_path / "labels").exists()
    widget.close()
    app.processEvents()


@pytest.mark.parametrize("relative_path", [".", "nested", ".."])
def test_yolo_export_rejects_paths_overlapping_source(tmp_path, relative_path):
    source_root = tmp_path / "images"
    source_root.mkdir()
    save_path = source_root / relative_path

    with pytest.raises(ValueError):
        _validate_yolo_export_path(str(source_root), str(save_path))


def test_yolo_export_accepts_sibling_path(tmp_path):
    source_root = tmp_path / "images"
    source_root.mkdir()

    _validate_yolo_export_path(str(source_root), str(tmp_path / "labels"))


@pytest.mark.parametrize(
    ("error", "guidance"),
    [
        (
            PoseGroupError("Invalid pose group"),
            "Reason: Pose instance grouping is incomplete or mismatched.\n"
            "Please ensure that each instance has one bounding box and that "
            "its bounding box and keypoints use the same numeric group ID.",
        ),
        (
            PoseClassError("Invalid pose class"),
            "Reason: The bounding box label is not defined in the pose "
            "configuration.\nPlease ensure that the bounding box label is "
            "listed under classes in the pose YAML file.",
        ),
        (RuntimeError("Unexpected error"), "Reason: Unexpected error"),
    ],
)
def test_yolo_export_error_dialog_shows_actionable_guidance(
    tmp_path, error, guidance
):
    current_image = tmp_path / "current.png"
    failed_image = tmp_path / "failed.png"
    widget = mock.Mock()
    widget.filename = str(current_image)
    message_box = mock.Mock()

    with mock.patch(
        "anylabeling.views.labeling.utils.export.QtWidgets.QMessageBox",
        return_value=message_box,
    ) as message_box_class:
        _show_yolo_export_error(widget, str(failed_image), error)

    message_box_class.assert_called_once_with(widget)
    message_box.setWindowTitle.assert_called_once_with("Export Failed")
    expected_message = f"Failed on image: {failed_image}"
    if guidance:
        expected_message += f"\n\n{guidance}"
    message_box.setText.assert_called_once_with(expected_message)
    message_box.addButton.assert_called_once_with(
        message_box_class.StandardButton.Ok
    )
    widget.load_file.assert_called_once_with(str(failed_image))


def test_get_yolo_export_files_split_layout(tmp_path):
    src = tmp_path / "images"
    (src / "sub").mkdir(parents=True)
    first = src / "a.png"
    second = src / "sub" / "b.png"
    first.touch()
    second.touch()
    root = tmp_path / "dataset_split"

    train_files = _get_yolo_export_files(
        [str(first), str(second)], str(src), str(root), layout="train"
    )
    assert train_files[0][1] == str(root / "labels" / "train" / "a.txt")
    assert train_files[0][2] == str(root / "images" / "train" / "a.png")
    assert train_files[1][1] == str(
        root / "labels" / "train" / "sub" / "b.txt"
    )
    assert train_files[1][2] == str(
        root / "images" / "train" / "sub" / "b.png"
    )

    val_files = _get_yolo_export_files(
        [str(first)], str(src), str(root), layout="val"
    )
    assert val_files[0][1] == str(root / "labels" / "val" / "a.txt")

    flat_files = _get_yolo_export_files([str(first)], str(src), str(root))
    assert flat_files[0][1] == str(root / "a.txt")


def test_coco_split_filename_matches_single_export_stem():
    assert _coco_split_filename("rectangle", "train") == (
        "coco_detection_train.json"
    )
    assert _coco_split_filename("polygon", "val") == (
        "coco_instance_segmentation_val.json"
    )
    assert _coco_split_filename("pose", "train") == (
        "coco_keypoints_train.json"
    )


def _write_label_json(image_file, labels):
    label_file = image_file.with_suffix(".json")
    label_file.write_text(
        json.dumps(
            {
                "imageWidth": 100,
                "imageHeight": 100,
                "shapes": [
                    {
                        "label": label,
                        "shape_type": "rectangle",
                        "points": [[10, 10], [10, 60], [60, 60], [60, 10]],
                        "difficult": False,
                    }
                    for label in labels
                ],
            }
        ),
        encoding="utf-8",
    )


def test_yolo_export_split_builds_yolov5_layout(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    images = [image_dir / f"{name}.png" for name in ("a", "b", "c", "d")]
    for image_file, labels in zip(
        images, (["cat"], ["cat"], ["dog"], ["dog"])
    ):
        image_file.touch()
        _write_label_json(image_file, labels)
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("cat\ndog\n", encoding="utf-8")

    widget = QtWidgets.QWidget()
    widget.filename = str(images[0])
    widget.image_list = [str(image_file) for image_file in images]
    widget.last_open_dir = str(image_dir)
    widget.output_dir = str(image_dir)
    widget.may_continue = mock.Mock(return_value=True)
    converter = mock.Mock()
    converter.custom_to_yolo.return_value = False
    converter.classes = ["cat", "dog"]

    with (
        mock.patch.object(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            return_value=(str(classes_file), ""),
        ),
        mock.patch.object(QtWidgets.QDialog, "exec", return_value=1),
        mock.patch.object(
            # Order: preview enable/skip polls, persisted-pref check,
            # post-toggle refresh enable/skip polls, save-with-images,
            # skip-empty, split enable.
            QtWidgets.QCheckBox,
            "isChecked",
            side_effect=(True, False, True, True, False, True, False, True),
        ),
        mock.patch(
            "anylabeling.views.labeling.utils.export.LabelConverter",
            return_value=converter,
        ),
        mock.patch("anylabeling.views.labeling.utils.export.Popup"),
    ):
        export_yolo_annotation(widget, "hbb")

    root = tmp_path / "labels_split"
    assert (root / "labels" / "train").is_dir()
    assert (root / "labels" / "val").is_dir()
    assert (root / "images" / "train").is_dir()
    assert (root / "images" / "val").is_dir()

    label_calls = [
        call.args[1] for call in converter.custom_to_yolo.call_args_list
    ]
    assert len(label_calls) == 4
    assert sum("labels/train" in path for path in label_calls) == 2
    assert sum("labels/val" in path for path in label_calls) == 2

    train_lines = (root / "train.txt").read_text().splitlines()
    val_lines = (root / "val.txt").read_text().splitlines()
    assert len(train_lines) == 2
    assert len(val_lines) == 2
    assert set(train_lines).isdisjoint(val_lines)
    assert all(line.startswith("images/train/") for line in train_lines)
    assert all(line.startswith("images/val/") for line in val_lines)
    assert sorted(
        [line.split("/")[-1] for line in train_lines + val_lines]
    ) == ["a.png", "b.png", "c.png", "d.png"]

    dataset = yaml.safe_load((root / "dataset.yaml").read_text())
    assert dataset["train"] == "images/train"
    assert dataset["val"] == "images/val"
    assert dataset["nc"] == 2
    assert dataset["names"] == ["cat", "dog"]
    assert dataset["split"]["enabled"] is True
    assert dataset["split"]["seed"] == 42
    assert (root / "classes.txt").is_file()
    widget.close()
    app.processEvents()


def test_coco_export_split_thread_writes_two_jsons(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    images = [str(image_dir / f"{name}.png") for name in ("a", "b", "c", "d")]
    for image_file in images:
        Path(image_file).touch()
        _write_label_json(Path(image_file), ["cat"])
    save_path = tmp_path / "coco_split"
    save_path.mkdir()

    labels, _ = collect_labels_and_names(images, output_dir=str(image_dir))
    result = stratified_split(
        images, train_ratio=0.8, seed=42, labels_by_image=labels
    )
    assert len(result.train) == 3
    assert len(result.val) == 1

    converter = LabelConverter(classes=["cat"])
    thread = ExportThread(
        converter,
        images,
        str(image_dir),
        str(save_path),
        "rectangle",
        split_lists=(result.train, result.val),
        split_info={
            "enabled": True,
            "train_ratio": result.train_ratio,
            "seed": result.seed,
        },
        save_images=True,
    )
    thread.run()

    annotations_dir = save_path / "annotations"
    train_json = annotations_dir / "coco_detection_train.json"
    val_json = annotations_dir / "coco_detection_val.json"
    assert train_json.is_file()
    assert val_json.is_file()
    train_data = json.loads(train_json.read_text(encoding="utf-8"))
    val_data = json.loads(val_json.read_text(encoding="utf-8"))
    assert len(train_data["images"]) == 3
    assert len(val_data["images"]) == 1
    assert train_data["info"]["split"] == {
        "enabled": True,
        "train_ratio": 0.8,
        "seed": 42,
    }
    assert sorted((save_path / "train2017").iterdir()) != []
    assert len(list((save_path / "train2017").iterdir())) == 3
    assert len(list((save_path / "val2017").iterdir())) == 1
    app.processEvents()


def test_coco_export_dialog_passes_split_to_thread(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    images = [image_dir / f"{name}.png" for name in ("a", "b", "c", "d")]
    for image_file in images:
        image_file.touch()
        _write_label_json(image_file, ["cat"])
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("cat\n", encoding="utf-8")

    widget = QtWidgets.QWidget()
    widget.filename = str(images[0])
    widget.image_list = [str(image_file) for image_file in images]
    widget.output_dir = str(image_dir)
    widget.may_continue = mock.Mock(return_value=True)

    captured = {}

    class FakeThread:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            self.finished = mock.Mock()

        def start(self):
            pass

        def terminate(self):
            pass

    with (
        mock.patch.object(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            return_value=(str(classes_file), ""),
        ),
        mock.patch.object(QtWidgets.QDialog, "exec", return_value=1),
        mock.patch.object(QtWidgets.QMessageBox, "exec", return_value=0),
        mock.patch.object(
            # Order: preview enable poll, persisted-pref check,
            # post-toggle refresh poll, save-with-images, split.
            QtWidgets.QCheckBox,
            "isChecked",
            side_effect=(True, True, True, True, True),
        ),
        mock.patch(
            "anylabeling.views.labeling.utils.export.LabelConverter",
        ),
        mock.patch("anylabeling.views.labeling.utils.export.Popup"),
        mock.patch(
            "anylabeling.views.labeling.utils.export.ExportThread",
            FakeThread,
        ),
        mock.patch.object(
            QtWidgets.QProgressDialog, "show", return_value=None
        ),
    ):
        export_coco_annotation(widget, "rectangle")

    kwargs = captured["kwargs"]
    assert kwargs["save_images"] is True
    train, val = kwargs["split_lists"]
    assert len(train) == 3
    assert len(val) == 1
    assert set(train).isdisjoint(val)
    assert kwargs["split_info"]["enabled"] is True
    assert kwargs["split_info"]["seed"] == 42
    widget.close()
    app.processEvents()
