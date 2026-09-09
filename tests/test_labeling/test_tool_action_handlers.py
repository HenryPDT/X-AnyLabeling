"""Regression: Tool menu actions must map to existing handlers.

Catches AttributeError at startup (e.g. missing dataset_statistics).
Qt-free: checks class attributes without instantiating the widget.
"""

from anylabeling.views.labeling.label_widget import LabelingWidget


def test_tool_action_handlers_exist():
    for handler in [
        "overview",
        "dataset_statistics",
        "dataset_diagnostics",
        "annotation_review_gallery",
        "apply_file_sort_mode",
    ]:
        assert hasattr(LabelingWidget, handler), f"missing handler: {handler}"
        assert callable(getattr(LabelingWidget, handler))


def test_split_lives_in_export_dialogs_not_tool_menu():
    """Train/val split is an export-time option, not a Tool action."""
    assert not hasattr(LabelingWidget, "dataset_split_builder")
    from anylabeling.views.labeling.utils import export as export_module

    assert hasattr(export_module, "_create_split_section")
    assert hasattr(export_module, "_coco_split_filename")
