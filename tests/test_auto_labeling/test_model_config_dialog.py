import os
import pytest
import yaml
from PyQt6.QtWidgets import QApplication, QWidget

from anylabeling.views.labeling.widgets.model_config_dialog import (
    ModelConfigDialog,
    load_classes_from_file,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_load_classes_from_file(tmp_path):
    """Verify load_classes_from_file strips whitespace, empty lines, and comments."""
    assert load_classes_from_file(None) == []
    assert load_classes_from_file("/non/existent/path/classes.txt") == []

    class_file = tmp_path / "classes.txt"
    class_file.write_text(
        "# Comment header\n"
        "car\n"
        "\n"
        "motorcycle  \n"
        "# Another comment\n"
        "  truck  \n",
        encoding="utf-8",
    )
    classes = load_classes_from_file(str(class_file))
    assert classes == ["car", "motorcycle", "truck"]


def test_model_config_dialog_init_from_config(qapp, tmp_path):
    """Verify dialog populates fields from an existing config YAML."""
    model_file = tmp_path / "yolox.onnx"
    model_file.write_text("weights", encoding="utf-8")
    labels_file = tmp_path / "classes.txt"
    labels_file.write_text("person\nbicycle\n", encoding="utf-8")

    config_data = {
        "type": "deepstream_det",
        "name": "custom_yolox",
        "display_name": "Custom YOLOX",
        "model_path": str(model_file),
        "labelfile_path": str(labels_file),
        "conf_threshold": 0.3,
        "iou_threshold": 0.5,
        "containment_threshold": 0.0,
        "containment_keep": "area",
        "net_scale_factor": 0.00392156,
        "model_color_format": 1,
    }
    cfg_file = tmp_path / "model_config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    dialog = ModelConfigDialog(config_file=str(cfg_file))
    assert dialog.model_path_input.text() == str(model_file)
    assert dialog.class_path_input.text() == str(labels_file)
    assert dialog.display_name_input.text() == "Custom YOLOX"
    assert "✓ 2 classes" in dialog.preview_label.text()
    dialog.close()


def test_model_config_dialog_preserves_architecture_fields(qapp, tmp_path):
    """Verify applying changes updates model/class while keeping architecture fields intact."""
    old_model = tmp_path / "old.onnx"
    old_model.write_text("old weights", encoding="utf-8")
    new_model = tmp_path / "new.onnx"
    new_model.write_text("new weights", encoding="utf-8")

    old_labels = tmp_path / "old_labels.txt"
    old_labels.write_text("cat\ndog\n", encoding="utf-8")
    new_labels = tmp_path / "new_labels.txt"
    new_labels.write_text("car\nbus\ntruck\n", encoding="utf-8")

    cfg_file = tmp_path / "detector.yaml"
    initial_config = {
        "type": "deepstream_det",
        "name": "custom_detector",
        "display_name": "Old Detector",
        "model_path": str(old_model),
        "labelfile_path": str(old_labels),
        "conf_threshold": 0.35,
        "iou_threshold": 0.55,
        "containment_threshold": 0.0,
        "containment_keep": "area",
        "model_color_format": 0,
        "net_scale_factor": 1.0,
        "maintain_aspect_ratio": 1,
        "symmetric_padding": 1,
        "pad_value": 0,
        "cluster_mode": 2,
    }
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(initial_config, f)

    reloaded = []

    class MockParent(QWidget):
        def load_custom_model_config(self, path):
            reloaded.append(path)

    parent = MockParent()
    dialog = ModelConfigDialog(parent=parent, config_file=str(cfg_file))

    # User modifies weights, class file, and display name
    dialog.model_path_input.setText(str(new_model))
    dialog.class_path_input.setText(str(new_labels))
    dialog.display_name_input.setText("New Fleet Detector")

    # Apply changes
    dialog.on_apply()

    assert len(reloaded) == 1
    assert reloaded[0] == str(cfg_file)

    # Read back YAML and verify preservation
    with open(cfg_file, "r", encoding="utf-8") as f:
        saved = yaml.safe_load(f)

    # Updated fields
    assert saved["model_path"] == str(new_model)
    assert saved["labelfile_path"] == str(new_labels)
    assert saved["display_name"] == "New Fleet Detector"
    assert saved["classes"] == ["car", "bus", "truck"]

    # Preserved architecture dials
    assert saved["model_color_format"] == 0
    assert saved["net_scale_factor"] == 1.0
    assert saved["maintain_aspect_ratio"] == 1
    assert saved["symmetric_padding"] == 1
    assert saved["pad_value"] == 0
    assert saved["cluster_mode"] == 2
    assert saved["conf_threshold"] == 0.35
    assert saved["iou_threshold"] == 0.55
    assert saved["containment_threshold"] == 0.0

    dialog.close()
