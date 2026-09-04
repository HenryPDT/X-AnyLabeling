import numpy as np
import yaml

from anylabeling.services.auto_labeling.segment_anything_3 import (
    SegmentAnything3,
)


def test_sam3_config_is_registered():
    with open("anylabeling/configs/models.yaml", "r", encoding="utf-8") as f:
        models = yaml.safe_load(f)

    assert {
        "model_name": "sam3_vit_h-r20260426",
        "config_file": ":/sam3_vit_h.yaml",
    } in models

    with open(
        "anylabeling/configs/auto_labeling/sam3_vit_h.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    assert config["type"] == "segment_anything_3"
    assert config["encoder_model_path"].endswith("sam3_image_encoder.onnx")
    assert config["encoder_model_data_path"].endswith(
        "sam3_image_encoder.onnx.data"
    )
    assert config["decoder_model_path"].endswith("sam3_decoder.onnx")
    assert config["decoder_model_data_path"].endswith("sam3_decoder.onnx.data")
    assert config["language_encoder_path"].endswith(
        "sam3_language_encoder.onnx"
    )
    assert config["language_encoder_data_path"].endswith(
        "sam3_language_encoder.onnx.data"
    )


def test_sam3_post_process_polygon_from_multiple_masks():
    model = SegmentAnything3.__new__(SegmentAnything3)
    model.output_mode = "polygon"
    model.epsilon = 0.001
    masks = np.zeros((2, 1, 20, 20), dtype=np.bool_)
    masks[0, 0, 2:8, 3:10] = True
    masks[1, 0, 10:16, 11:18] = True

    shapes = model.post_process(masks, "truck")

    assert len(shapes) == 2
    assert all(shape.label == "truck" for shape in shapes)
    assert all(shape.shape_type == "polygon" for shape in shapes)
    assert all(shape.closed for shape in shapes)


def test_sam3_post_process_rectangle_output_mode():
    model = SegmentAnything3.__new__(SegmentAnything3)
    model.output_mode = "rectangle"
    model.epsilon = 0.001
    masks = np.zeros((1, 1, 20, 20), dtype=np.bool_)
    masks[0, 0, 4:9, 5:12] = True
    scores = np.array([0.88], dtype=np.float32)

    shapes = model.post_process(masks, "car", scores)

    assert len(shapes) == 1
    assert shapes[0].label == "car"
    assert shapes[0].score == float(scores[0])
    assert shapes[0].shape_type == "rectangle"


def test_sam3_splits_and_deduplicates_text_prompts():
    assert SegmentAnything3.split_text_prompts("person.car.person") == [
        "person",
        "car",
    ]
    assert SegmentAnything3.split_text_prompts(" person, car , ") == [
        "person",
        "car",
    ]


def test_sam3_class_agnostic_nms_keeps_higher_score_overlap():
    from PyQt6 import QtCore

    from anylabeling.services.auto_labeling.utils import (
        apply_class_agnostic_shape_nms,
    )
    from anylabeling.views.labeling.shape import Shape

    truck = Shape(flags={})
    truck.shape_type = "rectangle"
    truck.label = "truck"
    truck.score = 0.92
    for point in [(10, 10), (50, 10), (50, 40), (10, 40)]:
        truck.add_point(QtCore.QPointF(*point))

    bus = Shape(flags={})
    bus.shape_type = "rectangle"
    bus.label = "bus"
    bus.score = 0.71
    for point in [(12, 12), (48, 12), (48, 38), (12, 38)]:
        bus.add_point(QtCore.QPointF(*point))

    person = Shape(flags={})
    person.shape_type = "rectangle"
    person.label = "person"
    person.score = 0.80
    for point in [(80, 10), (100, 10), (100, 40), (80, 40)]:
        person.add_point(QtCore.QPointF(*point))

    kept = apply_class_agnostic_shape_nms([truck, bus, person], 0.5)

    assert len(kept) == 2
    assert {shape.label for shape in kept} == {"truck", "person"}


def test_sam3_class_agnostic_nms_disabled_when_iou_zero():
    from PyQt6 import QtCore

    from anylabeling.services.auto_labeling.utils import (
        apply_class_agnostic_shape_nms,
    )
    from anylabeling.views.labeling.shape import Shape

    truck = Shape(flags={})
    truck.shape_type = "rectangle"
    truck.label = "truck"
    truck.score = 0.92
    for point in [(10, 10), (50, 10), (50, 40), (10, 40)]:
        truck.add_point(QtCore.QPointF(*point))

    bus = Shape(flags={})
    bus.shape_type = "rectangle"
    bus.label = "bus"
    bus.score = 0.71
    for point in [(12, 12), (48, 12), (48, 38), (12, 38)]:
        bus.add_point(QtCore.QPointF(*point))

    kept = apply_class_agnostic_shape_nms([truck, bus], 0.0)

    assert len(kept) == 2
    assert {shape.label for shape in kept} == {"truck", "bus"}


def test_sam3_nms_preserves_box_and_mask_for_same_object():
    from PyQt6 import QtCore

    from anylabeling.services.auto_labeling.utils import (
        apply_class_agnostic_shape_nms,
    )
    from anylabeling.views.labeling.shape import Shape

    points = [(10, 10), (50, 10), (50, 40), (10, 40)]

    rect = Shape(flags={})
    rect.shape_type = "rectangle"
    rect.label = "truck"
    rect.score = 0.92
    for point in points:
        rect.add_point(QtCore.QPointF(*point))

    poly = Shape(flags={})
    poly.shape_type = "polygon"
    poly.label = "truck"
    poly.score = 0.92
    for point in points:
        poly.add_point(QtCore.QPointF(*point))

    bus = Shape(flags={})
    bus.shape_type = "rectangle"
    bus.label = "bus"
    bus.score = 0.71
    for point in [(12, 12), (48, 12), (48, 38), (12, 38)]:
        bus.add_point(QtCore.QPointF(*point))

    kept = apply_class_agnostic_shape_nms([rect, poly, bus], 0.5)

    assert len(kept) == 2
    assert {(shape.label, shape.shape_type) for shape in kept} == {
        ("truck", "rectangle"),
        ("truck", "polygon"),
    }


def test_sam3_same_label_containment_drops_nested_box():
    from PyQt6 import QtCore

    from anylabeling.services.auto_labeling.utils import (
        apply_class_agnostic_shape_nms,
    )
    from anylabeling.views.labeling.shape import Shape

    outer = Shape(flags={})
    outer.shape_type = "rectangle"
    outer.label = "motorbike and person"
    outer.score = 0.67
    for point in [(10, 10), (50, 10), (50, 80), (10, 80)]:
        outer.add_point(QtCore.QPointF(*point))

    nested = Shape(flags={})
    nested.shape_type = "rectangle"
    nested.label = "motorbike and person"
    nested.score = 0.34
    for point in [(15, 15), (40, 15), (40, 40), (15, 40)]:
        nested.add_point(QtCore.QPointF(*point))

    other = Shape(flags={})
    other.shape_type = "rectangle"
    other.label = "person"
    other.score = 0.80
    for point in [(15, 15), (40, 15), (40, 40), (15, 40)]:
        other.add_point(QtCore.QPointF(*point))

    kept = apply_class_agnostic_shape_nms(
        [outer, nested, other],
        iou_threshold=0.0,
        containment_threshold=0.5,
        containment_keep="score",
    )

    assert len(kept) == 2
    assert {shape.label for shape in kept} == {
        "motorbike and person",
        "person",
    }
    assert all(shape.score != 0.34 for shape in kept)


def test_sam3_containment_keep_area_prefers_larger_box():
    from PyQt6 import QtCore

    from anylabeling.services.auto_labeling.utils import (
        apply_class_agnostic_shape_nms,
    )
    from anylabeling.views.labeling.shape import Shape

    outer = Shape(flags={})
    outer.shape_type = "rectangle"
    outer.label = "motorbike and person"
    outer.score = 0.40
    for point in [(10, 10), (50, 10), (50, 80), (10, 80)]:
        outer.add_point(QtCore.QPointF(*point))

    nested = Shape(flags={})
    nested.shape_type = "rectangle"
    nested.label = "motorbike and person"
    nested.score = 0.95
    for point in [(15, 15), (40, 15), (40, 40), (15, 40)]:
        nested.add_point(QtCore.QPointF(*point))

    kept_score = apply_class_agnostic_shape_nms(
        [outer, nested],
        iou_threshold=0.0,
        containment_threshold=0.5,
        containment_keep="score",
    )
    assert len(kept_score) == 1
    assert kept_score[0].score == 0.95

    kept_area = apply_class_agnostic_shape_nms(
        [outer, nested],
        iou_threshold=0.0,
        containment_threshold=0.5,
        containment_keep="area",
    )
    assert len(kept_area) == 1
    assert kept_area[0].score == 0.40


def test_sam3_config_includes_iou_threshold():
    with open(
        "anylabeling/configs/auto_labeling/sam3_vit_h.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    assert config["iou_threshold"] == 0.5
    assert config["containment_threshold"] == 0.5
    assert config["containment_keep"] == "area"
    assert "input_conf" in SegmentAnything3.Meta.widgets
    assert "edit_conf" in SegmentAnything3.Meta.widgets
    assert "input_iou" in SegmentAnything3.Meta.widgets
    assert "edit_iou" in SegmentAnything3.Meta.widgets
    assert "input_containment" in SegmentAnything3.Meta.widgets
    assert "edit_containment" in SegmentAnything3.Meta.widgets
    assert "containment_keep_combobox" in SegmentAnything3.Meta.widgets


def test_sam3_nms_preserves_shapes_without_bounding_box():
    from PyQt6 import QtCore

    from anylabeling.services.auto_labeling.utils import (
        apply_class_agnostic_shape_nms,
    )
    from anylabeling.views.labeling.shape import Shape

    truck = Shape(flags={})
    truck.shape_type = "rectangle"
    truck.label = "truck"
    truck.score = 0.92
    for point in [(10, 10), (50, 10), (50, 40), (10, 40)]:
        truck.add_point(QtCore.QPointF(*point))

    point_shape = Shape(flags={})
    point_shape.shape_type = "point"
    point_shape.label = "metadata_marker"
    point_shape.points = []

    kept = apply_class_agnostic_shape_nms(
        [truck, point_shape],
        iou_threshold=0.5,
        containment_threshold=0.5,
    )

    assert len(kept) == 2
    assert point_shape in kept
    assert truck in kept
