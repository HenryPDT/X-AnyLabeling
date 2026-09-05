from anylabeling.services.model_type import (
    normalize_model_type as shared_normalize,
)
from anylabeling.views.labeling.widgets.auto_labeling.auto_labeling import (
    AutoLabelingWidget,
    normalize_model_type,
)


def test_normalize_model_type_variants():
    assert normalize_model_type("grounding_dino") == "groundingdino"
    assert normalize_model_type("groundingdino") == "groundingdino"
    assert normalize_model_type("grounding-dino") == "groundingdino"
    assert normalize_model_type("grounding_dino_api") == "groundingdinoapi"
    assert normalize_model_type("remote_server") == "remoteserver"
    assert normalize_model_type("segment_anything_3") == "segmentanything3"
    assert normalize_model_type("") == ""
    assert normalize_model_type(None) == ""


def test_normalize_upn_florence2():
    assert normalize_model_type("upn") == "upn"
    assert normalize_model_type("florence2") == "florence2"


def test_normalize_shared_single_source():
    # auto_labeling re-exports the shared helper (no local duplicate).
    assert normalize_model_type is shared_normalize


class _FakeCombo:
    def __init__(self, items):
        self._items = list(items)
        self.current = 0
        self.blocked = False

    def findData(self, data):
        try:
            return self._items.index(data)
        except ValueError:
            return -1

    def blockSignals(self, blocked):
        self.blocked = bool(blocked)

    def setCurrentIndex(self, index):
        self.current = int(index)


class _FakeManager:
    def __init__(self, model):
        self.loaded_model_config = {
            "type": "grounding_dino_api",
            "model": model,
        }


def _widget_with_model(model, combo_items):
    widget = AutoLabelingWidget.__new__(AutoLabelingWidget)
    widget.model_manager = _FakeManager(model)
    widget.gd_select_combobox = _FakeCombo(combo_items)
    return widget


def test_update_groundingdino_mode_api_syncs_combobox():
    class ApiModel:
        model_name = "GroundingDino-1.6-Pro"

    widget = _widget_with_model(ApiModel(), ["GroundingDino_1_6_Pro", "x"])
    AutoLabelingWidget.update_groundingdino_mode_ui(widget)
    assert widget.gd_select_combobox.current == 0


def test_update_groundingdino_mode_local_is_noop():
    class LocalModel:
        pass

    widget = _widget_with_model(LocalModel(), ["GroundingDino_1_6_Pro"])
    AutoLabelingWidget.update_groundingdino_mode_ui(widget)
    # Local ONNX has no model_name: combobox untouched.
    assert widget.gd_select_combobox.current == 0


def test_update_groundingdino_mode_unknown_falls_back():
    class ApiModel:
        model_name = "Future-Model-X"

    widget = _widget_with_model(ApiModel(), ["Future-Model-X"])
    AutoLabelingWidget.update_groundingdino_mode_ui(widget)
    assert widget.gd_select_combobox.current == 0
