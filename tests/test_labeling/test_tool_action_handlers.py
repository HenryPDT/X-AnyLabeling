"""Regression: Tool menu actions must map to existing handlers.

Catches AttributeError at startup (e.g. missing dataset_split_builder).
Qt-free: checks class attributes without instantiating the widget.
"""

from anylabeling.views.labeling.label_widget import LabelingWidget


def test_tool_action_handlers_exist():
    for handler in [
        "overview",
        "dataset_statistics",
        "dataset_split_builder",
        "dataset_diagnostics",
        "annotation_review_gallery",
        "apply_file_sort_mode",
    ]:
        assert hasattr(LabelingWidget, handler), f"missing handler: {handler}"
        assert callable(getattr(LabelingWidget, handler))
