import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image as PILImage

from anylabeling.services.auto_labeling.remote_server import RemoteServer


class TestRemoteServerClassFilter(unittest.TestCase):
    def setUp(self):
        with patch(
            "anylabeling.services.auto_labeling.model.get_config",
            return_value={"remote_server_settings": {}},
        ):
            self.model = RemoteServer(
                {
                    "type": "remote_server",
                    "display_name": "Remote Server",
                    "timeout": 30,
                },
                Mock(),
            )
        self.model.models_info = {
            "yolo": {
                "classes": ["person", "car", "dog"],
                "filter_classes": ["person", "dog"],
            }
        }

    def test_model_selection_loads_remote_class_metadata(self):
        self.model.set_model_id("yolo")

        self.assertEqual(self.model.classes, ["person", "car", "dog"])
        self.assertEqual(self.model.filter_classes, ["person", "dog"])

    def test_prediction_forwards_selected_classes(self):
        self.model.set_model_id("yolo")
        self.model.set_auto_labeling_filter_classes(["car"])
        response = Mock()
        response.json.return_value = {"data": {"shapes": []}}

        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            image_file.write(b"image")
            image_file.flush()
            with patch(
                "anylabeling.services.auto_labeling.remote_server.requests.post",
                return_value=response,
            ) as post:
                self.model.predict_shapes(object(), image_file.name)

        params = post.call_args.kwargs["json"]["params"]
        self.assertEqual(params["filter_classes"], ["car"])

    def test_selecting_all_classes_sends_empty_filter_override(self):
        self.model.set_model_id("yolo")
        self.model.set_auto_labeling_filter_classes(self.model.classes)
        response = Mock()
        response.json.return_value = {"data": {"shapes": []}}

        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            image_file.write(b"image")
            image_file.flush()
            with patch(
                "anylabeling.services.auto_labeling.remote_server.requests.post",
                return_value=response,
            ) as post:
                self.model.predict_shapes(object(), image_file.name)

        params = post.call_args.kwargs["json"]["params"]
        self.assertEqual(params["filter_classes"], [])

    def test_video_batch_range_limits_request_and_saved_frames(self):
        self.model.video_session_id = "session"
        self.model.video_prompt_frame = 0
        self.model._widget = SimpleNamespace(
            image_list=[f"/tmp/{index}.png" for index in range(5)],
            _batch_processing_active=True,
            _batch_start_index=1,
            _batch_end_index=3,
            cancel_processing=False,
        )
        mask = {"points": [[0, 0], [2, 2]], "label": "car"}
        event = {
            "type": "completed",
            "results": {str(i): {"masks": [mask]} for i in range(5)},
        }
        response = Mock()
        response.iter_lines.return_value = ["data: " + json.dumps(event)]

        with (
            patch(
                "anylabeling.services.auto_labeling.remote_server.requests.post",
                return_value=response,
            ) as post,
            patch(
                "anylabeling.views.labeling.utils.batch.save_auto_labeling_result"
            ) as save,
        ):
            self.model._handle_video_propagation()

        request = post.call_args.kwargs["json"]
        self.assertEqual(request["start_frame"], 1)
        self.assertEqual(request["end_frame"], 2)
        self.assertEqual(
            [call.args[1] for call in save.call_args_list],
            ["/tmp/1.png", "/tmp/2.png"],
        )


class TestRemoteServerImageDimensions(unittest.TestCase):
    def test_prefers_image_path_over_stale_qimage(self):
        """Batch Auto Run passes a stale UI image; path size must win."""
        stale_ui_image = SimpleNamespace(
            width=lambda: 1920, height=lambda: 1080
        )
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            PILImage.new("RGB", (2560, 1440), color=(0, 0, 0)).save(
                image_file.name
            )
            width, height = RemoteServer._get_image_dimensions(
                stale_ui_image, image_file.name
            )

        self.assertEqual((width, height), (2560.0, 1440.0))

    def test_falls_back_to_qimage_without_path(self):
        image = SimpleNamespace(width=lambda: 1280, height=lambda: 720)
        width, height = RemoteServer._get_image_dimensions(image, None)
        self.assertEqual((width, height), (1280.0, 720.0))

    def test_predict_shapes_does_not_clamp_to_stale_ui_height(self):
        with patch(
            "anylabeling.services.auto_labeling.model.get_config",
            return_value={"remote_server_settings": {}},
        ):
            model = RemoteServer(
                {
                    "type": "remote_server",
                    "display_name": "Remote Server",
                    "timeout": 30,
                },
                Mock(),
            )
        model.current_model_id = "sam3"
        model._apply_client_side_cleanup = lambda shapes: shapes

        response = Mock()
        response.json.return_value = {
            "data": {
                "shapes": [
                    {
                        "label": "object",
                        "shape_type": "rectangle",
                        "points": [
                            [100, 1200],
                            [200, 1200],
                            [200, 1350],
                            [100, 1350],
                        ],
                    }
                ]
            }
        }
        stale_ui_image = SimpleNamespace(
            width=lambda: 1920, height=lambda: 1080
        )

        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            PILImage.new("RGB", (2560, 1440), color=(0, 0, 0)).save(
                image_file.name
            )
            with patch(
                "anylabeling.services.auto_labeling.remote_server.requests.post",
                return_value=response,
            ):
                result = model.predict_shapes(stale_ui_image, image_file.name)

        self.assertEqual(len(result.shapes), 1)
        ys = [p.y() for p in result.shapes[0].points]
        self.assertGreater(max(ys), 1080.0)
        self.assertEqual(max(ys), 1350.0)


class TestRemoteServerClientSideContainment(unittest.TestCase):
    def test_is_client_side_containment_remote_model(self):
        from anylabeling.services.auto_labeling.remote_server import (
            is_client_side_containment_remote_model,
        )

        self.assertTrue(
            is_client_side_containment_remote_model("segment_anything_3")
        )
        self.assertTrue(
            is_client_side_containment_remote_model("deepstream_det_yolox")
        )
        self.assertTrue(
            is_client_side_containment_remote_model(
                "custom_det",
                {
                    "widgets": [
                        {"name": "edit_conf"},
                        {"name": "edit_iou"},
                    ]
                },
            )
        )
        self.assertFalse(
            is_client_side_containment_remote_model("classification_model", {})
        )

    def test_apply_client_side_containment_cleanup(self):
        from PyQt6.QtCore import QPointF
        from anylabeling.views.labeling.shape import Shape

        with patch(
            "anylabeling.services.auto_labeling.model.get_config",
            return_value={"remote_server_settings": {}},
        ):
            model = RemoteServer(
                {
                    "type": "remote_server",
                    "display_name": "Remote Server",
                    "timeout": 30,
                },
                Mock(),
            )
        model.current_model_id = "deepstream_det_test"
        model.containment_threshold = 0.8
        model.containment_keep = "area"

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

        # When containment_threshold > 0: inner is suppressed
        cleaned = model._apply_client_side_cleanup([outer, inner])
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0], outer)

        # When containment_threshold == 0: no suppression
        model.containment_threshold = 0.0
        not_cleaned = model._apply_client_side_cleanup([outer, inner])
        self.assertEqual(len(not_cleaned), 2)


if __name__ == "__main__":
    unittest.main()
