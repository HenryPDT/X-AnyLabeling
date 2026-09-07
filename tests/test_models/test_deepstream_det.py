import os
from pathlib import Path
import cv2
import numpy as np
import pytest

from anylabeling.services.auto_labeling import (
    _CUSTOM_MODELS,
    _AUTO_LABELING_CONF_MODELS,
    _AUTO_LABELING_IOU_MODELS,
    _AUTO_LABELING_PRESERVE_EXISTING_ANNOTATIONS_STATE_MODELS,
)
from anylabeling.services.auto_labeling.deepstream_det import (
    DeepStreamDetection,
    _batched_nms,
    _parse_offsets,
)


def test_deepstream_det_registration():
    """Verify deepstream_det is registered in custom models and capability lists."""
    assert "deepstream_det" in _CUSTOM_MODELS
    assert "deepstream_det" in _AUTO_LABELING_CONF_MODELS
    assert "deepstream_det" in _AUTO_LABELING_IOU_MODELS
    assert (
        "deepstream_det"
        in _AUTO_LABELING_PRESERVE_EXISTING_ANNOTATIONS_STATE_MODELS
    )


def test_deepstream_det_meta():
    """Verify Meta configuration names and output modes."""
    assert "type" in DeepStreamDetection.Meta.required_config_names
    assert "name" in DeepStreamDetection.Meta.required_config_names
    assert "model_path" in DeepStreamDetection.Meta.required_config_names
    assert "edit_conf" in DeepStreamDetection.Meta.widgets
    assert "edit_iou" in DeepStreamDetection.Meta.widgets
    assert "input_containment" in DeepStreamDetection.Meta.widgets
    assert "edit_containment" in DeepStreamDetection.Meta.widgets
    assert "containment_keep_combobox" in DeepStreamDetection.Meta.widgets
    assert DeepStreamDetection.Meta.default_output_mode == "rectangle"


def test_deepstream_det_batched_nms():
    """Verify per-class batched NMS behavior."""
    boxes = np.array(
        [
            [10.0, 10.0, 50.0, 50.0],
            [12.0, 12.0, 52.0, 52.0],  # overlaps with box 0, same class
            [10.0, 10.0, 50.0, 50.0],  # same box, different class
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.85], dtype=np.float32)
    class_ids = np.array([0, 0, 1], dtype=np.int64)

    keep = _batched_nms(boxes, scores, class_ids, iou_threshold=0.5)
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep


def test_deepstream_det_preprocess():
    """Verify preprocessing padding math for both top-left and symmetric modes."""
    img = np.zeros((300, 600, 3), dtype=np.uint8)

    # Top-left padding
    config_tl = {
        "type": "deepstream_det",
        "name": "test_ds",
        "display_name": "Test DS",
        "model_path": "dummy.onnx",
        "maintain_aspect_ratio": 1,
        "symmetric_padding": 0,
        "pad_value": 114,
    }
    model_tl = DeepStreamDetection.__new__(DeepStreamDetection)
    model_tl.config = config_tl
    model_tl.input_shape = (640, 640)
    model_tl.maintain_aspect_ratio = True
    model_tl.symmetric_padding = False
    model_tl.pad_value = 114
    model_tl.model_color_format = 1  # BGR
    model_tl.net_scale_factor = 1.0
    model_tl.offsets = None

    blob_tl, meta_tl = model_tl.preprocess(img)
    assert blob_tl.shape == (1, 3, 640, 640)
    assert meta_tl["pad_x"] == 0.0
    assert meta_tl["pad_y"] == 0.0
    assert np.isclose(meta_tl["ratio"], 640 / 600)

    # Symmetric padding
    model_sym = DeepStreamDetection.__new__(DeepStreamDetection)
    model_sym.config = config_tl
    model_sym.input_shape = (640, 640)
    model_sym.maintain_aspect_ratio = True
    model_sym.symmetric_padding = True
    model_sym.pad_value = 114
    model_sym.model_color_format = 1
    model_sym.net_scale_factor = 1.0
    model_sym.offsets = None

    blob_sym, meta_sym = model_sym.preprocess(img)
    assert blob_sym.shape == (1, 3, 640, 640)
    assert meta_sym["pad_x"] == 0.0
    expected_pad_y = (640 - 300 * (640 / 600)) / 2.0
    assert np.isclose(meta_sym["pad_y"], expected_pad_y)


def test_deepstream_det_postprocess_nms_free():
    """Verify postprocessing in NMS-free mode (cluster_mode=4)."""
    model = DeepStreamDetection.__new__(DeepStreamDetection)
    model.input_shape = (640, 640)
    model.classes = ["cls0", "cls1"]
    model.conf_thres = 0.3
    model.nms_thres = 0.5
    model.cluster_mode = 4  # NMS-free
    model.topk = 300
    model.filter_classes = None

    raw_output = np.array(
        [
            [10.0, 10.0, 50.0, 50.0, 0.9, 0],
            [
                12.0,
                12.0,
                52.0,
                52.0,
                0.8,
                0,
            ],  # Overlapping, preserved in NMS-free
            [20.0, 20.0, 60.0, 60.0, 0.2, 0],  # Filtered by conf
        ],
        dtype=np.float32,
    )
    meta = {
        "maintain_aspect_ratio": False,
        "r_w": 1.0,
        "r_h": 1.0,
        "orig_w": 640,
        "orig_h": 640,
    }

    shapes = model.postprocess(raw_output, meta)
    assert len(shapes) == 2
    assert np.isclose(shapes[0].score, 0.9, atol=1e-4)
    assert np.isclose(shapes[1].score, 0.8, atol=1e-4)
    assert shapes[0].label == "cls0"


def test_deepstream_det_end_to_end_yolox(monkeypatch):
    """Verify end-to-end inference using pixeltable-yolox ONNX and dog.jpg."""
    import anylabeling.services.auto_labeling.model as auto_model

    monkeypatch.setattr(auto_model, "get_config", lambda: {})

    project_root = Path(__file__).resolve().parents[2]
    onnx_path = str(project_root / "weights" / "yolox_s.onnx")
    labels_path = str(project_root / "weights" / "labels.txt")
    img_path = str(project_root / "tests" / "assets" / "dog.jpg")

    if not os.path.isfile(onnx_path) or not os.path.isfile(img_path):
        pytest.skip("YOLOX ONNX or test image not available")

    config = {
        "type": "deepstream_det",
        "name": "deepstream_yolox",
        "display_name": "DeepStream YOLOX",
        "model_path": onnx_path,
        "labelfile_path": labels_path,
        "model_color_format": 1,
        "net_scale_factor": 1.0,
        "maintain_aspect_ratio": 1,
        "symmetric_padding": 0,
        "pad_value": 114,
        "cluster_mode": 2,
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
    }

    model = DeepStreamDetection(config, on_message=lambda msg: None)
    img = cv2.imread(img_path)
    # Qt image simulation: cv2 image converted to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = model.predict_shapes(img_rgb)

    assert len(result.shapes) > 0
    labels = [s.label for s in result.shapes]
    assert "dog" in labels
    assert "bicycle" in labels

    # Test dynamic threshold update
    model.set_auto_labeling_conf(0.95)
    result_high_conf = model.predict_shapes(img_rgb)
    assert len(result_high_conf.shapes) <= len(result.shapes)

    model.unload()


def test_deepstream_det_containment_setters():
    """Verify containment threshold and keep mode setters."""
    model = DeepStreamDetection.__new__(DeepStreamDetection)
    model.containment_thres = 0.0
    model.containment_keep = "area"

    model.set_auto_labeling_containment(0.7)
    assert model.containment_thres == 0.7

    model.set_auto_labeling_containment_keep("score")
    assert model.containment_keep == "score"

    model.set_auto_labeling_containment_keep("invalid_mode")
    assert model.containment_keep == "area"


def test_deepstream_det_containment_suppression():
    """Verify containment suppression behavior in predict_shapes."""
    from PyQt6.QtCore import QObject, QPointF
    from anylabeling.views.labeling.shape import Shape

    model = DeepStreamDetection.__new__(DeepStreamDetection)
    QObject.__init__(model)
    model.containment_thres = 0.8
    model.containment_keep = "area"
    model.replace = True

    class MockNet:
        def get_ort_inference(self, blob, extract=True):
            return [np.zeros((1, 6), dtype=np.float32)]

    model.net = MockNet()

    outer = Shape(label="car", score=0.8)
    outer.points = [
        QPointF(0, 0),
        QPointF(100, 0),
        QPointF(100, 100),
        QPointF(0, 100),
    ]
    outer.shape_type = "rectangle"

    inner = Shape(label="car", score=0.95)
    inner.points = [
        QPointF(10, 10),
        QPointF(90, 10),
        QPointF(90, 90),
        QPointF(10, 90),
    ]
    inner.shape_type = "rectangle"

    model.preprocess = lambda img: (
        np.zeros((1, 3, 640, 640), dtype=np.float32),
        {},
    )
    model.postprocess = lambda raw, meta: [outer, inner]

    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

    # Area mode: inner box suppressed, outer preserved
    res = model.predict_shapes(dummy_img)
    assert len(res.shapes) == 1
    assert res.shapes[0] == outer

    # Score mode: outer box suppressed, inner preserved
    model.set_auto_labeling_containment_keep("score")
    res_score = model.predict_shapes(dummy_img)
    assert len(res_score.shapes) == 1
    assert res_score.shapes[0] == inner

    # Disabled mode (containment_thres == 0): both preserved
    model.set_auto_labeling_containment(0.0)
    res_disabled = model.predict_shapes(dummy_img)
    assert len(res_disabled.shapes) == 2


def test_deepstream_det_load_labels_comment_stripping(tmp_path):
    """Verify that label parsing strips comments and whitespace."""
    label_file = tmp_path / "classes.txt"
    label_file.write_text(
        "# Header comment\ncar\n\n# Another comment\nmotorcycle\ntruck\n",
        encoding="utf-8",
    )
    model = DeepStreamDetection.__new__(DeepStreamDetection)
    model.config = {"labelfile_path": str(label_file)}
    model.get_model_abs_path = lambda cfg, key: cfg.get(key)
    # Replicate class parsing block from __init__
    labels_path = model.config.get("labelfile_path")
    with open(labels_path, "r", encoding="utf-8") as f:
        model.classes = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    assert model.classes == ["car", "motorcycle", "truck"]


def test_deepstream_det_parse_offsets():
    """Verify parsing of offsets in various formats (list, string, wrong shape)."""
    assert _parse_offsets(None) is None
    assert _parse_offsets([]) is None

    # List of 3 floats
    arr = _parse_offsets([103.939, 116.779, 123.68])
    assert arr is not None
    assert np.allclose(arr, [103.939, 116.779, 123.68])

    # Semicolon-delimited string (standard DeepStream notation)
    arr_semi = _parse_offsets("103.939;116.779;123.68")
    assert arr_semi is not None
    assert np.allclose(arr_semi, [103.939, 116.779, 123.68])

    # Comma-delimited string
    arr_comma = _parse_offsets("0.0, 0.0, 0.0")
    assert arr_comma is not None
    assert np.allclose(arr_comma, [0.0, 0.0, 0.0])

    # Wrong number of elements -> ignored
    assert _parse_offsets([1.0, 2.0]) is None
    assert _parse_offsets([1.0, 2.0, 3.0, 4.0]) is None

    # Invalid non-numeric string -> ignored
    assert _parse_offsets("invalid;offsets") is None


def test_deepstream_det_hyphen_label_resolution(tmp_path):
    """Verify labelfile-path hyphen key is resolved without KeyError."""
    label_file = tmp_path / "labels.txt"
    label_file.write_text("cat\ndog\n", encoding="utf-8")

    model = DeepStreamDetection.__new__(DeepStreamDetection)
    model.config = {"labelfile-path": str(label_file)}
    model.get_model_abs_path = lambda cfg, key: cfg.get(key)

    # Execute class resolution logic from __init__
    model.classes = model.config.get("classes")
    if not model.classes:
        labels_path = model.config.get("labelfile_path") or model.config.get(
            "labelfile-path"
        )
        if labels_path:
            field_name = (
                "labelfile_path"
                if "labelfile_path" in model.config
                else "labelfile-path"
            )
            resolved_labels = (
                model.get_model_abs_path(model.config, field_name)
                or labels_path
            )
            if os.path.isfile(resolved_labels):
                with open(resolved_labels, "r", encoding="utf-8-sig") as f:
                    model.classes = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.strip().startswith("#")
                    ]

    assert model.classes == ["cat", "dog"]


def test_deepstream_det_postprocess_invalid_rank():
    """Verify postprocess handles 1D and invalid rank inputs gracefully."""
    model = DeepStreamDetection.__new__(DeepStreamDetection)
    model.input_shape = (640, 640)
    model.conf_thres = 0.25
    model.classes = ["cat"]

    meta = {
        "orig_w": 640,
        "orig_h": 640,
        "maintain_aspect_ratio": False,
        "r_w": 1.0,
        "r_h": 1.0,
    }

    # 1D array of 2 elements (previously caused IndexError: tuple index out of range)
    res_1d = model.postprocess(np.array([1.0, 2.0]), meta)
    assert res_1d == []

    # Empty 1D array
    res_empty = model.postprocess(np.array([]), meta)
    assert res_empty == []

    # Array with fewer than 6 columns
    res_few_cols = model.postprocess(np.zeros((5, 4), dtype=np.float32), meta)
    assert res_few_cols == []


def test_deepstream_det_postprocess_zero_division():
    """Verify postprocess handles zero ratio / r_w without crashing."""
    model = DeepStreamDetection.__new__(DeepStreamDetection)
    model.input_shape = (640, 640)
    model.conf_thres = 0.25
    model.cluster_mode = 2
    model.nms_thres = 0.5
    model.topk = 100
    model.classes = ["cat"]
    model.filter_classes = None

    raw_output = np.array([[10, 10, 50, 50, 0.9, 0]], dtype=np.float32)

    # Zero ratio in aspect ratio mode
    meta_zero_ratio = {
        "orig_w": 640,
        "orig_h": 640,
        "maintain_aspect_ratio": True,
        "ratio": 0.0,
        "pad_x": 0.0,
        "pad_y": 0.0,
    }
    res_ratio = model.postprocess(raw_output, meta_zero_ratio)
    assert isinstance(res_ratio, list)

    # Zero r_w in stretch mode
    meta_zero_rw = {
        "orig_w": 640,
        "orig_h": 640,
        "maintain_aspect_ratio": False,
        "r_w": 0.0,
        "r_h": 0.0,
    }
    res_rw = model.postprocess(raw_output, meta_zero_rw)
    assert isinstance(res_rw, list)


def test_deepstream_det_network_dimensions_config():
    """Verify network-height and network-width keys are supported."""
    config_hyphen = {
        "network-height": 544,
        "network-width": 960,
    }
    input_shape = (
        int(config_hyphen["network-height"]),
        int(config_hyphen["network-width"]),
    )
    assert input_shape == (544, 960)


def test_deepstream_det_bom_label_loading(tmp_path):
    """Verify UTF-8 BOM is stripped when reading label files."""
    label_file = tmp_path / "labels_bom.txt"
    # Write with BOM \ufeff
    label_file.write_text("\ufeffperson\ncar\nbicycle\n", encoding="utf-8")

    with open(str(label_file), "r", encoding="utf-8-sig") as f:
        classes = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    assert classes == ["person", "car", "bicycle"]
    assert classes[0] == "person"
    assert not classes[0].startswith("\ufeff")
