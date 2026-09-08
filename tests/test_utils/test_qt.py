import os
import tempfile
import unittest
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore

    from anylabeling.views.labeling import utils
    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.utils.qt import scan_all_images

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for Qt utility tests")
class TestQtUtils(unittest.TestCase):
    def test_scan_all_images_skips_delete_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            keep_path = os.path.join(tmp_dir, "keep.jpg")
            train_path = os.path.join(tmp_dir, "train", "active.jpg")
            trash_path = os.path.join(tmp_dir, "_delete_", "removed.jpg")
            os.makedirs(os.path.join(tmp_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(tmp_dir, "_delete_"), exist_ok=True)
            for path in (keep_path, train_path, trash_path):
                with open(path, "wb") as handle:
                    handle.write(b"img")

            images = scan_all_images(tmp_dir)

            self.assertIn(os.path.abspath(keep_path), images)
            self.assertIn(os.path.abspath(train_path), images)
            self.assertNotIn(os.path.abspath(trash_path), images)

    def test_distance_to_line_handles_2d_points_without_numpy_warning(self):
        point = QtCore.QPointF(5.0, 5.0)
        line = [QtCore.QPointF(0.0, 0.0), QtCore.QPointF(10.0, 0.0)]

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            distance = utils.distance_to_line(point, line)

        self.assertEqual(distance, 5.0)

    def test_nearest_edge_handles_line_shapes_without_numpy_warning(self):
        shape = Shape(label="line", shape_type="line")
        shape.points = [
            QtCore.QPointF(0.0, 0.0),
            QtCore.QPointF(10.0, 0.0),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            edge = shape.nearest_edge(QtCore.QPointF(5.0, 1.0), 2.0)

        self.assertIn(edge, (0, 1))
