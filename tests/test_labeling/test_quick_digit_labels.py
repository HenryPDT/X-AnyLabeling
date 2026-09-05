import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    from anylabeling.views.labeling.label_widget import LabelingWidget
    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.widgets.canvas import Canvas
    from anylabeling.views.labeling.widgets.label_dialog import (
        DigitShortcutDialog,
    )
    from anylabeling.views.labeling.widgets.unique_label_qlist_widget import (
        UniqueLabelQListWidget,
    )

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for quick digit label tests"
)
class TestQuickDigitLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def _create_mock_widget(
        self, classes=None, digit_shortcuts=None, config=None
    ):
        widget = Mock(spec=LabelingWidget)
        widget.tr = lambda text: text
        widget.attributes = {}
        widget.reset_attribute = Mock(side_effect=lambda lbl, shape: lbl)
        widget.validate_label = Mock(return_value=True)
        widget.drawing_digit_shortcuts = digit_shortcuts or {}
        widget._config = {
            "quick_digit_labels": True,
            "canvas": {
                "crosshair": {
                    "show": True,
                    "width": 2.0,
                    "color": "#00FF00",
                    "opacity": 0.5,
                    "sync_label_color": True,
                }
            },
        }
        if config:
            widget._config.update(config)
        widget.digit_to_label = None

        # Real UniqueLabelQListWidget
        widget.unique_label_list = UniqueLabelQListWidget()
        if classes:
            for lbl in classes:
                item = widget.unique_label_list.create_item_from_label(lbl)
                widget.unique_label_list.addItem(item)
                widget.unique_label_list.set_item_label(item, lbl)

        # get_project_classes
        widget.get_project_classes = lambda: (
            LabelingWidget.get_project_classes(widget)
        )
        widget._get_target_label_for_digit = lambda digit: (
            LabelingWidget._get_target_label_for_digit(widget, digit)
        )
        widget._select_unique_label_item = lambda lbl: (
            LabelingWidget._select_unique_label_item(widget, lbl)
        )
        widget.assign_label_to_shapes = lambda shapes, lbl: (
            LabelingWidget.assign_label_to_shapes(widget, shapes, lbl)
        )
        widget.handle_digit_shortcut = lambda digit: (
            LabelingWidget.handle_digit_shortcut(widget, digit)
        )
        widget.create_digit_mode = lambda digit: (
            LabelingWidget.create_digit_mode(widget, digit)
        )
        widget.update_crosshair_color = lambda lbl=None: (
            LabelingWidget.update_crosshair_color(widget, lbl)
        )
        widget.on_unique_label_selection_changed = lambda: (
            LabelingWidget.on_unique_label_selection_changed(widget)
        )
        widget._update_shape_color = Mock()
        color_map = {
            "cat": (0, 0, 255),
            "dog": (255, 0, 0),
            "car": (255, 255, 0),
        }
        widget._get_rgb_by_label = Mock(
            side_effect=lambda lbl, **kw: color_map.get(lbl, (0, 255, 0))
        )
        widget.set_dirty = Mock()
        widget._refresh_shape_filters = Mock()
        widget.status = Mock()
        widget.toggle_draw_mode = Mock()

        # Canvas mock with real store_shapes
        widget.canvas = Canvas(parent=None)
        widget.canvas.selected_shapes = []

        # Label list mock
        widget.label_list = Mock()
        widget.label_list.find_item_by_shape = Mock(return_value=Mock())

        widget.label_dialog = Mock()
        widget.label_dialog.add_label_history = Mock()

        return widget

    def test_target_label_mapping_1_to_9_and_0(self):
        classes = [f"class_{i}" for i in range(1, 11)]  # 10 classes
        widget = self._create_mock_widget(classes=classes)

        # 1 -> 1st class (index 0)
        self.assertEqual(widget._get_target_label_for_digit(1), "class_1")
        # 2 -> 2nd class (index 1)
        self.assertEqual(widget._get_target_label_for_digit(2), "class_2")
        # 9 -> 9th class (index 8)
        self.assertEqual(widget._get_target_label_for_digit(9), "class_9")
        # 0 -> 10th class (index 9)
        self.assertEqual(widget._get_target_label_for_digit(0), "class_10")

    def test_target_label_out_of_range(self):
        classes = ["cat", "dog", "car"]
        widget = self._create_mock_widget(classes=classes)

        self.assertEqual(widget._get_target_label_for_digit(1), "cat")
        self.assertEqual(widget._get_target_label_for_digit(2), "dog")
        self.assertEqual(widget._get_target_label_for_digit(3), "car")
        self.assertIsNone(widget._get_target_label_for_digit(4))
        self.assertIsNone(widget._get_target_label_for_digit(0))

    def test_target_label_custom_override(self):
        classes = ["cat", "dog", "car"]
        # Custom override on digit 1 to be 'bicycle'
        digit_shortcuts = {1: {"label": "bicycle"}}
        widget = self._create_mock_widget(
            classes=classes, digit_shortcuts=digit_shortcuts
        )

        self.assertEqual(widget._get_target_label_for_digit(1), "bicycle")
        # Unoverridden digit 2 still maps to 'dog'
        self.assertEqual(widget._get_target_label_for_digit(2), "dog")

    def test_assign_label_to_selected_shape(self):
        classes = ["cat", "dog", "car"]
        widget = self._create_mock_widget(classes=classes)

        shape = Shape(label="cat", shape_type="rectangle")
        shape.points = [
            QtCore.QPointF(0, 0),
            QtCore.QPointF(10, 0),
            QtCore.QPointF(10, 10),
            QtCore.QPointF(0, 10),
        ]
        widget.canvas.shapes = [shape]
        widget.canvas.selected_shapes = [shape]

        # Press 2 -> dog
        widget.handle_digit_shortcut(2)

        self.assertEqual(shape.label, "dog")
        widget.set_dirty.assert_called_once()
        self.assertTrue(len(widget.canvas.shapes_backups) > 0)
        # Verify unique_label_list selection
        selected = widget.unique_label_list.selectedItems()
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected[0].data(QtCore.Qt.ItemDataRole.UserRole), "dog"
        )

    def test_assign_label_skips_locked_shape(self):
        classes = ["cat", "dog"]
        widget = self._create_mock_widget(classes=classes)

        shape = Shape(label="cat", shape_type="rectangle")
        shape.locked = True
        widget.canvas.shapes = [shape]
        widget.canvas.selected_shapes = [shape]

        widget.handle_digit_shortcut(2)

        # Label remains unchanged because shape is locked
        self.assertEqual(shape.label, "cat")
        widget.set_dirty.assert_not_called()

    def test_handle_digit_shortcut_no_selection_sets_active_label(self):
        classes = ["cat", "dog", "car"]
        widget = self._create_mock_widget(classes=classes)
        widget.canvas.selected_shapes = []

        # Press 3 -> selects 'car'
        widget.handle_digit_shortcut(3)

        self.assertEqual(widget.digit_to_label, "car")
        selected = widget.unique_label_list.selectedItems()
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected[0].data(QtCore.Qt.ItemDataRole.UserRole), "car"
        )

    def test_handle_digit_shortcut_explicit_mode_when_no_selection(self):
        classes = ["cat", "dog"]
        digit_shortcuts = {1: {"mode": "rectangle", "label": "custom_box"}}
        widget = self._create_mock_widget(
            classes=classes, digit_shortcuts=digit_shortcuts
        )
        widget.canvas.selected_shapes = []

        # Press 1 when no shape selected -> should switch to rectangle mode
        widget.handle_digit_shortcut(1)

        widget.toggle_draw_mode.assert_called_once_with(
            edit=False, create_mode="rectangle"
        )
        self.assertEqual(widget.digit_to_label, "custom_box")

    def test_handle_digit_shortcut_disabled_config(self):
        classes = ["cat", "dog"]
        config = {"quick_digit_labels": False}
        widget = self._create_mock_widget(classes=classes, config=config)

        shape = Shape(label="cat", shape_type="rectangle")
        widget.canvas.selected_shapes = [shape]

        # With quick_digit_labels=False, pressing 2 should NOT change shape label
        widget.handle_digit_shortcut(2)

        self.assertEqual(shape.label, "cat")

    def test_digit_shortcut_dialog_label_only(self):
        parent = QtWidgets.QWidget()
        parent.drawing_digit_shortcuts = {}
        dialog = DigitShortcutDialog(parent=parent)

        # Set digit 1: Mode = None (index 0), Label = "only_label"
        label_edit = dialog._get_label_edit(1)
        label_edit.setText("only_label")

        # Save settings
        with patch.object(dialog, "accept"):
            dialog.save_settings()

        self.assertIn(1, parent.drawing_digit_shortcuts)
        self.assertIsNone(parent.drawing_digit_shortcuts[1]["mode"])
        self.assertEqual(
            parent.drawing_digit_shortcuts[1]["label"], "only_label"
        )

    def test_assign_label_multiple_shapes(self):
        classes = ["cat", "dog", "car"]
        widget = self._create_mock_widget(classes=classes)

        shape1 = Shape(label="cat", shape_type="rectangle")
        shape2 = Shape(label="car", shape_type="rectangle")
        widget.canvas.shapes = [shape1, shape2]
        widget.canvas.selected_shapes = [shape1, shape2]

        # Press 2 -> dog
        widget.handle_digit_shortcut(2)

        self.assertEqual(shape1.label, "dog")
        self.assertEqual(shape2.label, "dog")
        widget.set_dirty.assert_called_once()

    def test_digit_label_canvas_undo(self):
        classes = ["cat", "dog"]
        widget = self._create_mock_widget(classes=classes)

        shape = Shape(label="cat", shape_type="rectangle")
        shape.points = [
            QtCore.QPointF(0, 0),
            QtCore.QPointF(10, 0),
            QtCore.QPointF(10, 10),
            QtCore.QPointF(0, 10),
        ]
        widget.canvas.shapes = [shape]
        widget.canvas.selected_shapes = [shape]

        # Initial backup
        widget.canvas.store_shapes()

        # Reclassify via shortcut
        widget.handle_digit_shortcut(2)
        self.assertEqual(widget.canvas.shapes[0].label, "dog")

        # Undo
        widget.canvas.restore_shape()
        self.assertEqual(widget.canvas.shapes[0].label, "cat")

    def test_out_of_bounds_status_message(self):
        classes = ["cat", "dog"]
        widget = self._create_mock_widget(classes=classes)
        shape = Shape(label="cat", shape_type="rectangle")
        widget.canvas.shapes = [shape]
        widget.canvas.selected_shapes = [shape]

        # Press 8 (out of range)
        widget.handle_digit_shortcut(8)

        # Label remains unchanged
        self.assertEqual(shape.label, "cat")
        widget.status.assert_called_with("No class defined for key '8'")

    def test_crosshair_color_matches_selected_shape(self):
        classes = ["cat", "dog"]  # cat -> blue #0000FF, dog -> red #FF0000
        widget = self._create_mock_widget(classes=classes)

        shape = Shape(label="dog", shape_type="rectangle")
        widget.canvas.shapes = [shape]
        widget.canvas.selected_shapes = [shape]

        widget.update_crosshair_color()
        self.assertEqual(widget.canvas.cross_line_color, "#FF0000")

    def test_crosshair_color_matches_digit_selection(self):
        classes = ["cat", "dog"]  # cat -> blue #0000FF
        widget = self._create_mock_widget(classes=classes)
        widget.canvas.selected_shapes = []

        # Press 1 -> selects 'cat'
        widget.handle_digit_shortcut(1)
        self.assertEqual(widget.canvas.cross_line_color, "#0000FF")

    def test_crosshair_sync_disabled_keeps_configured_color(self):
        classes = ["cat", "dog"]
        config = {
            "canvas": {
                "crosshair": {
                    "show": True,
                    "width": 2.0,
                    "color": "#00FF00",
                    "opacity": 0.5,
                    "sync_label_color": False,
                }
            }
        }
        widget = self._create_mock_widget(classes=classes, config=config)

        shape = Shape(label="dog", shape_type="rectangle")
        widget.canvas.selected_shapes = [shape]

        widget.update_crosshair_color()
        # Should stay green #00FF00 because sync_label_color is False
        self.assertEqual(widget.canvas.cross_line_color, "#00FF00")
