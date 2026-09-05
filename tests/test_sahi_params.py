from anylabeling.services.sahi_params import (
    SAHI_DEFAULT_OVERLAP_RATIO,
    SAHI_DEFAULT_SLICE_HEIGHT,
    clamp_sahi_params,
    is_sahi_model_type,
    resolve_sahi_params,
)


def test_is_sahi_model_type():
    assert is_sahi_model_type("yolov8_sahi") is True
    assert is_sahi_model_type("yolov5_sahi") is True
    assert is_sahi_model_type("yolo11_sahi") is True
    assert is_sahi_model_type("yolo26_sahi") is True
    assert is_sahi_model_type("yolov8") is False
    assert is_sahi_model_type("grounding_dino") is False
    assert is_sahi_model_type("") is False
    assert is_sahi_model_type(None) is False


def test_clamp_sahi_params_defaults():
    assert clamp_sahi_params(512, 512, 0.2) == (512, 512, 0.2)


def test_clamp_sahi_params_bounds():
    slice_h, slice_w, overlap = clamp_sahi_params(10, 5000, 0.9)
    assert slice_h == 256
    assert slice_w == 2048
    assert overlap == 0.5


def test_clamp_sahi_params_invalid_fallback():
    slice_h, slice_w, overlap = clamp_sahi_params("x", None, "y")
    assert slice_h == SAHI_DEFAULT_SLICE_HEIGHT
    assert slice_w == 512
    assert overlap == SAHI_DEFAULT_OVERLAP_RATIO


def test_resolve_sahi_params_prefers_overrides():
    slice_h, slice_w, overlap = resolve_sahi_params(
        {"slice_height": 512, "slice_width": 512},
        {"slice_height": 640},
    )
    assert slice_h == 640
    assert slice_w == 512


def test_resolve_sahi_params_accepts_app_config_keys():
    # Regression: apply_sahi_config_to_model passes sahi_* keys.
    slice_h, slice_w, overlap = resolve_sahi_params(
        {
            "sahi_slice_height": 640,
            "sahi_slice_width": 768,
            "sahi_overlap_ratio": 0.3,
        }
    )
    assert (slice_h, slice_w, overlap) == (640, 768, 0.3)


def test_clamp_sahi_params_non_finite():
    inf = float("inf")
    nan = float("nan")
    assert clamp_sahi_params(inf, 512, 0.2)[0] == SAHI_DEFAULT_SLICE_HEIGHT
    assert clamp_sahi_params(512, inf, 0.2)[1] == 512
    assert clamp_sahi_params(512, 512, nan)[2] == SAHI_DEFAULT_OVERLAP_RATIO
    # Non-finite overlap falls back to default (not clamped to max).
    assert clamp_sahi_params(512, 512, inf)[2] == SAHI_DEFAULT_OVERLAP_RATIO


def test_clamp_sahi_params_string_float():
    assert clamp_sahi_params("640.7", "512", "0.3") == (640, 512, 0.3)


def test_sahi_model_setter_clamps():
    from anylabeling.services.auto_labeling.yolo11_sahi import YOLO11_SAHI
    from anylabeling.services.auto_labeling.yolo26_sahi import YOLO26_SAHI
    from anylabeling.services.auto_labeling.yolov5_sahi import YOLOv5_SAHI
    from anylabeling.services.auto_labeling.yolov8_sahi import YOLOv8_SAHI

    for cls in (YOLOv8_SAHI, YOLO11_SAHI, YOLO26_SAHI, YOLOv5_SAHI):
        obj = cls.__new__(cls)
        obj.slice_height = 512
        obj.slice_width = 512
        obj.overlap_height_ratio = 0.2
        obj.overlap_width_ratio = 0.2
        cls.set_auto_labeling_sahi_params(obj, 100, 3000, 0.9)
        assert obj.slice_height == 256
        assert obj.slice_width == 2048
        assert obj.overlap_height_ratio == 0.5
        assert obj.overlap_width_ratio == 0.5


def test_apply_sahi_params_helper():
    from anylabeling.services.sahi_params import apply_sahi_params_to

    class Fake:
        pass

    obj = Fake()
    applied = apply_sahi_params_to(obj, 640, 640, 0.3)
    assert applied == (640, 640, 0.3)
    assert obj.slice_height == 640
    assert obj.overlap_width_ratio == 0.3
