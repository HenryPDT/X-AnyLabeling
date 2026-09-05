import base64
import cv2
import json
import numpy as np
import os
import requests
from pathlib import Path

from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from anylabeling.views.labeling.utils.shape_geometry import (
    apply_class_agnostic_shape_nms,
    clamp_shapes_to_image_bounds,
)

# Client-side cleanup defaults for remote SAM3 image models.
_CLIENT_SIDE_IOU_DEFAULT = 0.5
_CLIENT_SIDE_CONTAINMENT_DEFAULT = 0.5
_CLIENT_SIDE_CONTAINMENT_KEEP_DEFAULT = "area"


def is_client_side_iou_remote_model(model_id, model_info=None):
    """Return True for remote SAM3 image models (not video)."""
    if not model_id:
        return False
    model_id_l = str(model_id).lower()
    if model_id_l == "segment_anything_3":
        return True
    if "segment_anything_3" in model_id_l and "video" not in model_id_l:
        return True

    info = model_info or {}
    display = str(info.get("display_name", "")).lower()
    if "segment anything 3" in display and "video" not in display:
        return True
    return False


class RemoteServer(Model):
    class Meta:
        required_config_names = [
            "type",
            "display_name",
        ]
        widgets = ["remote_server_select_combobox"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        remote_settings = self._config.get("remote_server_settings", {})
        self.server_url = remote_settings.get(
            "server_url",
            os.getenv("XANYLABELING_SERVER_URL", "http://localhost:8000"),
        )
        self.predict_url = f"{self.server_url}/v1/predict"

        api_key = remote_settings.get("api_key", "")
        self.headers = {
            "Content-Type": "application/json",
            "Token": api_key,
        }

        self.current_model_id = None
        self.timeout = self.config.get("timeout", 30)
        self.models_info = {}
        self.classes = []
        self.filter_classes = None

        self.marks = []
        self.conf_threshold = 0.0
        self.iou_threshold = 0.0
        self.containment_threshold = 0.0
        self.containment_keep = _CLIENT_SIDE_CONTAINMENT_KEEP_DEFAULT
        self.epsilon_factor = 0.001
        self.replace = True
        self.reset_tracker_flag = False

        self.current_task = None

        # Segment Anything 3
        self.label = None
        self.group_id = None
        self.video_session_id = None
        self.video_initialized = False
        self.video_prompt_frame = None
        self.video_session_image_list = None
        self.video_prompt_type = None

    def set_cache_auto_label(self, text, gid):
        """Set cache auto label"""
        self.label = text
        self.group_id = gid

    def set_model_id(self, model_id):
        self.current_model_id = model_id
        model_info = self.models_info.get(model_id, {})
        self.classes = model_info.get("classes", [])
        self.filter_classes = model_info.get("filter_classes")
        if (
            is_client_side_iou_remote_model(model_id, model_info)
            and self.iou_threshold <= 0
        ):
            self.iou_threshold = _CLIENT_SIDE_IOU_DEFAULT
        if (
            is_client_side_iou_remote_model(model_id, model_info)
            and self.containment_threshold <= 0
        ):
            self.containment_threshold = _CLIENT_SIDE_CONTAINMENT_DEFAULT
            self.containment_keep = _CLIENT_SIDE_CONTAINMENT_KEEP_DEFAULT

    def uses_client_side_iou_nms(self):
        """Whether current remote model applies client-side IoU NMS."""
        model_info = self.models_info.get(self.current_model_id, {})
        return is_client_side_iou_remote_model(
            self.current_model_id, model_info
        )

    def set_task(self, task_id):
        """Set task ID for the current model"""
        self.current_task = task_id

    def get_available_models(self):
        """Fetch available models from remote server"""
        try:
            models_url = f"{self.server_url}/v1/models"
            response = requests.get(
                url=models_url, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            self.models_info = result.get("data", {})
            return self.models_info
        except Exception as e:
            logger.error(f"Failed to fetch available models: {e}")
            return {}

    def get_batch_processing_mode(self):
        """Get batch processing mode for current model and task.

        Returns:
            str | None: Batch processing mode ("default", "text_prompt", "video", or None).
                 Returns None if mode is not supported (e.g., visual_prompting).
                 Falls back to global batch_processing_mode if task-specific mode
                 is not available.
        """
        if self.current_model_id is None:
            return "default"
        model_info = self.models_info.get(self.current_model_id, {})

        available_tasks = model_info.get("available_tasks", [])
        if available_tasks and self.current_task:
            for task in available_tasks:
                if task.get("id") == self.current_task:
                    task_mode = task.get("batch_processing_mode")
                    if "batch_processing_mode" in task:
                        return task_mode
                    break

        return model_info.get("batch_processing_mode", "default")

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def set_auto_labeling_conf(self, conf_thresh):
        self.conf_threshold = conf_thresh

    def set_auto_labeling_iou(self, iou_thresh):
        self.iou_threshold = iou_thresh

    def set_auto_labeling_containment(self, value):
        self.containment_threshold = value

    def set_auto_labeling_containment_keep(self, mode):
        mode = (mode or _CLIENT_SIDE_CONTAINMENT_KEEP_DEFAULT).lower()
        self.containment_keep = (
            mode
            if mode in {"score", "area"}
            else _CLIENT_SIDE_CONTAINMENT_KEEP_DEFAULT
        )

    def set_auto_labeling_filter_classes(self, class_names):
        """Set the active remote class filter by name."""
        if not class_names or len(class_names) == len(self.classes):
            self.filter_classes = []
        else:
            self.filter_classes = list(class_names)

    def set_mask_fineness(self, epsilon):
        self.epsilon_factor = epsilon

    def set_auto_labeling_reset_tracker(self):
        """Reset tracker state for tracking models."""
        self.reset_tracker_flag = True

    def _build_predict_params(self, text_prompt=None):
        params = {
            "conf_threshold": self.conf_threshold,
            "epsilon_factor": self.epsilon_factor,
        }
        if not self.uses_client_side_iou_nms():
            params["iou_threshold"] = self.iou_threshold
        if text_prompt:
            params["text_prompt"] = text_prompt.rstrip(".")
        if self.marks:
            params["marks"] = self.marks
        if self.reset_tracker_flag:
            params["reset_tracker"] = True
            self.reset_tracker_flag = False
        if self.current_task:
            params["current_task"] = self.current_task
        if self.filter_classes is not None:
            params["filter_classes"] = self.filter_classes
        return params

    def predict_shapes(
        self, image, image_path=None, text_prompt=None, run_tracker=False
    ):
        if image is None:
            return AutoLabelingResult([], replace=self.replace)

        if self.current_model_id is None:
            logger.warning("No model selected")
            return AutoLabelingResult([], replace=self.replace)

        batch_mode = self.get_batch_processing_mode()
        if batch_mode == "video" and text_prompt:
            return self._handle_video_prompt(image_path, text_prompt)
        elif batch_mode == "video" and self.marks:
            result = self._handle_video_point_prompt(image_path)
            if run_tracker:
                logger.info("Starting video propagation after point prompt...")
                return self._handle_video_propagation()
            return result
        elif batch_mode == "video" and run_tracker:
            logger.info("Starting video propagation...")
            return self._handle_video_propagation()

        try:
            img_data_uri = self._image_to_data_uri(image, image_path)
        except Exception as e:
            logger.warning(f"Could not process image: {e}")
            return AutoLabelingResult([], replace=self.replace)

        params = self._build_predict_params(text_prompt)

        payload = {
            "model": self.current_model_id,
            "image": img_data_uri,
            "params": params,
        }
        logger.debug(
            f"Sending request to {self.predict_url} with payload: "
            f"model: {self.current_model_id}, "
            f"parameters: {params}"
        )

        try:
            response = requests.post(
                url=self.predict_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Remote server prediction result: {result}")

            data = result.get("data", {})
            shapes = []
            for shape_data in data.get("shapes", []):
                shape = Shape(
                    label=shape_data["label"],
                    shape_type=shape_data["shape_type"],
                    attributes=shape_data.get("attributes", {}),
                    description=shape_data.get("description", None),
                    difficult=shape_data.get("difficult", False),
                    direction=shape_data.get("direction", 0),
                    flags=shape_data.get("flags", None),
                    group_id=shape_data.get("group_id", None),
                    kie_linking=shape_data.get("kie_linking", []),
                    score=shape_data.get("score", None),
                )

                for point in shape_data["points"]:
                    shape.add_point(QtCore.QPointF(point[0], point[1]))

                shapes.append(shape)

            img_w, img_h = self._get_image_dimensions(image, image_path)
            if img_w and img_h and img_w > 0 and img_h > 0:
                shapes = clamp_shapes_to_image_bounds(shapes, img_w, img_h)

            shapes = self._apply_client_side_cleanup(shapes)

            description = data.get("description", "")

            replace = data.get("replace")
            if replace is None:
                replace = self.replace

            return AutoLabelingResult(
                shapes, replace=replace, description=description
            )

        except Exception as e:
            logger.error(f"Remote server error: {e}")
            self.on_message(f"Remote server error: {e}")
            return AutoLabelingResult([], replace=self.replace)

    def _apply_client_side_cleanup(self, shapes):
        """Apply client-side NMS and containment cleanup for supported models."""
        if not self.uses_client_side_iou_nms():
            return shapes
        shapes_before_nms = len(shapes)
        shapes = apply_class_agnostic_shape_nms(
            shapes,
            self.iou_threshold,
            containment_threshold=self.containment_threshold,
            containment_keep=self.containment_keep,
        )
        logger.info(
            "Remote SAM3 client-side cleanup | "
            f"iou={self.iou_threshold:.4f} | "
            f"contain={self.containment_threshold:.4f}/"
            f"{self.containment_keep} | "
            f"shapes={shapes_before_nms}->{len(shapes)}"
        )
        return shapes

    @staticmethod
    def _get_image_dimensions(image, image_path):
        """Resolve image size for post-response clamping.

        Prefer ``image_path`` when available. Batch Auto Run reuses the UI
        ``QImage`` across files, so trusting that object can clamp boxes to a
        previous (often smaller) resolution and cut off detections.
        """
        if image_path and os.path.exists(image_path):
            try:
                from PIL import Image as PILImage

                with PILImage.open(image_path) as _im:
                    return float(_im.size[0]), float(_im.size[1])
            except Exception:
                pass
        if hasattr(image, "width") and hasattr(image, "height"):
            try:
                return float(image.width()), float(image.height())
            except Exception:
                pass
        return None, None

    def _image_to_data_uri(self, image, image_path):
        """Encode QImage or disk image file to base64 data URI."""
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".bmp": "image/bmp",
                ".webp": "image/webp",
            }.get(ext, "image/jpeg")
            return f"data:{mime_type};base64,{img_base64}"

        cv_image = qt_img_to_rgb_cv_img(image, image_path)
        cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        is_success, buffer = cv2.imencode(".png", cv_image_bgr)
        if not is_success:
            raise ValueError("Failed to encode image.")
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"

    def _handle_video_prompt(self, image_path, text_prompt):
        """Handle video prompt: initialize or reuse session and add prompt.

        Args:
            image_path: Path to current image file.
            text_prompt: Text prompt string.

        Returns:
            AutoLabelingResult with shapes from prompt frame.
        """
        if not image_path or not os.path.exists(image_path):
            return AutoLabelingResult([], replace=self.replace)

        try:
            image_list = []
            widget = getattr(self, "_widget", None)
            if widget:
                image_list = getattr(widget, "image_list", [])

            if not image_list:
                dir_path = Path(image_path).parent
                valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                all_images = [
                    str(file_path)
                    for file_path in dir_path.iterdir()
                    if file_path.is_file()
                    and file_path.suffix.lower() in valid_extensions
                ]

                try:
                    all_images.sort(
                        key=lambda p: int(
                            "".join(filter(str.isdigit, Path(p).name))
                        )
                    )
                except ValueError:
                    all_images.sort()

                image_list = all_images

            current_index = 0
            try:
                current_index = image_list.index(image_path)
            except ValueError:
                logger.warning(
                    f"Current image {image_path} not found in image list, using index 0"
                )

            logger.info(
                f"Video prompt: current_frame_index={current_index}, "
                f"image_path={image_path}, total_frames={len(image_list)}"
            )

            if not image_list:
                return AutoLabelingResult([], replace=self.replace)

            if (
                self.video_session_id
                and self.video_session_image_list != image_list
            ):
                logger.info("Image list changed, resetting video session")
                self._reset_video_session()

            if self.video_session_id and self.video_prompt_type != "text":
                logger.info(
                    "Switching from point prompt to text prompt, resetting video session"
                )
                self._reset_video_session()

            if not self.video_session_id:
                frames_data = []
                for frame_path in image_list:
                    if os.path.exists(frame_path):
                        with open(frame_path, "rb") as f:
                            frame_base64 = base64.b64encode(f.read()).decode(
                                "utf-8"
                            )
                        ext = os.path.splitext(frame_path)[1].lower()
                        mime_type = {
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".png": "image/png",
                            ".bmp": "image/bmp",
                            ".webp": "image/webp",
                        }.get(ext, "image/jpeg")
                        frames_data.append(
                            f"data:{mime_type};base64,{frame_base64}"
                        )

                message = self.tr(
                    "Packing completed, initializing video session... "
                    "(This may take some time, please wait patiently)"
                )
                logger.info(message)
                self.on_message(message)

                init_url = f"{self.server_url}/v1/video/init"
                init_response = requests.post(
                    url=init_url,
                    json={
                        "model": self.current_model_id,
                        "frames": frames_data,
                        "start_frame_index": 0,
                    },
                    headers=self.headers,
                    timeout=self.timeout * 2,
                )
                init_response.raise_for_status()
                init_result = init_response.json()
                self.video_session_id = init_result.get("data", {}).get(
                    "session_id"
                )
                self.video_initialized = True
                self.video_session_image_list = image_list.copy()
                logger.info(
                    f"Video session initialized: {self.video_session_id}"
                )

            prompt_url = f"{self.server_url}/v1/video/prompt"
            prompt_response = requests.post(
                url=prompt_url,
                json={
                    "session_id": self.video_session_id,
                    "model": self.current_model_id,
                    "text_prompt": text_prompt.rstrip("."),
                    "frame_index": current_index,
                    "params": {
                        "conf_threshold": self.conf_threshold,
                        "epsilon_factor": self.epsilon_factor,
                    },
                },
                headers=self.headers,
                timeout=self.timeout,
            )
            prompt_response.raise_for_status()
            prompt_result = prompt_response.json()
            data = prompt_result.get("data", {})

            self.video_prompt_frame = current_index
            self.video_prompt_type = "text"

            shapes = []
            for shape_data in data.get("masks", []):
                shape = Shape(
                    label=shape_data.get("label", "AUTOLABEL_OBJECT"),
                    shape_type=shape_data.get("shape_type", "rectangle"),
                    score=shape_data.get("score", None),
                    group_id=shape_data.get("group_id", None),
                )
                for point in shape_data.get("points", []):
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shapes.append(shape)

            img_w, img_h = self._get_image_dimensions(None, image_path)
            if img_w and img_h and img_w > 0 and img_h > 0:
                shapes = clamp_shapes_to_image_bounds(shapes, img_w, img_h)

            return AutoLabelingResult(
                shapes, replace=self.replace, description=""
            )

        except Exception as e:
            logger.error(f"Video prompt error: {e}")
            self.on_message(f"Video prompt error: {e}")
            self._reset_video_session()
            return AutoLabelingResult([], replace=self.replace)

    def _handle_video_point_prompt(self, image_path):  # noqa: C901
        """Handle video point prompt: initialize or reuse session and add point prompt.

        Args:
            image_path: Path to current image file.

        Returns:
            AutoLabelingResult with shapes from prompt frame.
        """
        if not image_path or not os.path.exists(image_path):
            return AutoLabelingResult([], replace=False)

        try:
            image_list = []
            widget = getattr(self, "_widget", None)
            if widget:
                image_list = getattr(widget, "image_list", [])

            if not image_list:
                dir_path = Path(image_path).parent
                valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                all_images = [
                    str(file_path)
                    for file_path in dir_path.iterdir()
                    if file_path.is_file()
                    and file_path.suffix.lower() in valid_extensions
                ]

                try:
                    all_images.sort(
                        key=lambda p: int(
                            "".join(filter(str.isdigit, Path(p).name))
                        )
                    )
                except ValueError:
                    all_images.sort()

                image_list = all_images

            current_index = 0
            try:
                current_index = image_list.index(image_path)
            except ValueError:
                logger.warning(
                    f"Current image {image_path} not found in image list, using index 0"
                )

            logger.info(
                f"Video point prompt: current_frame_index={current_index}, "
                f"image_path={image_path}, total_frames={len(image_list)}"
            )

            if not image_list:
                return AutoLabelingResult([], replace=False)

            if (
                self.video_session_id
                and self.video_session_image_list != image_list
            ):
                logger.info("Image list changed, resetting video session")
                self._reset_video_session()

            if self.video_session_id and self.video_prompt_type != "point":
                logger.info(
                    "Switching from text prompt to point prompt, resetting video session"
                )
                self._reset_video_session()

            if not self.video_session_id:
                frames_data = []
                for frame_path in image_list:
                    if os.path.exists(frame_path):
                        with open(frame_path, "rb") as f:
                            frame_base64 = base64.b64encode(f.read()).decode(
                                "utf-8"
                            )
                        ext = os.path.splitext(frame_path)[1].lower()
                        mime_type = {
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".png": "image/png",
                            ".bmp": "image/bmp",
                            ".webp": "image/webp",
                        }.get(ext, "image/jpeg")
                        frames_data.append(
                            f"data:{mime_type};base64,{frame_base64}"
                        )

                message = self.tr(
                    "Packing completed, initializing video session... "
                    "(This may take some time, please wait patiently)"
                )
                logger.info(message)
                self.on_message(message)

                init_url = f"{self.server_url}/v1/video/init"
                init_response = requests.post(
                    url=init_url,
                    json={
                        "model": self.current_model_id,
                        "frames": frames_data,
                        "start_frame_index": 0,
                    },
                    headers=self.headers,
                    timeout=self.timeout * 2,
                )
                init_response.raise_for_status()
                init_result = init_response.json()
                self.video_session_id = init_result.get("data", {}).get(
                    "session_id"
                )
                self.video_initialized = True
                self.video_session_image_list = image_list.copy()
                logger.info(
                    f"Video session initialized: {self.video_session_id}"
                )

            frame_img = cv2.imread(image_path)
            if frame_img is None:
                return AutoLabelingResult([], replace=False)
            img_height, img_width = frame_img.shape[:2]

            point_coords = []
            point_labels = []

            for mark in self.marks:
                mark_type = mark.get("type")
                if mark_type == "point":
                    data = mark.get("data", [])
                    if len(data) == 2:
                        point_coords.append(data)
                        label = mark.get("label", 1)
                        point_labels.append(label)

            if not point_coords:
                logger.warning("No valid point prompts provided")
                return AutoLabelingResult([], replace=False)

            points_abs = np.array(point_coords, dtype=np.float32)
            labels = np.array(point_labels, dtype=np.int32)

            points_rel = points_abs / np.array(
                [img_width, img_height], dtype=np.float32
            )
            points_tensor = points_rel.tolist()
            point_labels_tensor = labels.tolist()

            prompt_url = f"{self.server_url}/v1/video/prompt"
            prompt_response = requests.post(
                url=prompt_url,
                json={
                    "session_id": self.video_session_id,
                    "model": self.current_model_id,
                    "frame_index": current_index,
                    "points": points_tensor,
                    "point_labels": point_labels_tensor,
                    "obj_id": 10000,
                    "params": {
                        "conf_threshold": self.conf_threshold,
                        "epsilon_factor": self.epsilon_factor,
                    },
                },
                headers=self.headers,
                timeout=self.timeout,
            )
            prompt_response.raise_for_status()
            prompt_result = prompt_response.json()
            data = prompt_result.get("data", {})

            self.video_prompt_frame = current_index
            self.video_prompt_type = "point"

            shapes = []
            for shape_data in data.get("masks", []):
                shape = Shape(
                    label="AUTOLABEL_OBJECT",
                    shape_type=shape_data.get("shape_type", "rectangle"),
                    score=shape_data.get("score", None),
                    group_id=shape_data.get("group_id", None),
                )
                for point in shape_data.get("points", []):
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shapes.append(shape)

            img_w, img_h = self._get_image_dimensions(None, image_path)
            if img_w and img_h and img_w > 0 and img_h > 0:
                shapes = clamp_shapes_to_image_bounds(shapes, img_w, img_h)

            return AutoLabelingResult(shapes, replace=False, description="")

        except Exception as e:
            logger.error(f"Video point prompt error: {e}")
            self.on_message(f"Video point prompt error: {e}")
            self._reset_video_session()
            return AutoLabelingResult([], replace=False)

    def _handle_video_propagation(self):  # noqa: C901
        """Handle video propagation using SSE streaming."""
        if not self.video_session_id:
            logger.warning("No video session initialized")
            return AutoLabelingResult([], replace=False)

        widget = getattr(self, "_widget", None)
        image_list = getattr(widget, "image_list", []) if widget else []
        progress_dialog = (
            getattr(widget, "_progress_dialog", None) if widget else None
        )
        range_start = self.video_prompt_frame
        range_end = None
        if widget and getattr(widget, "_batch_processing_active", False):
            range_start = widget._batch_start_index
            range_end = widget._batch_end_index

        try:
            stream_url = f"{self.server_url}/v1/video/propagate/stream"
            logger.info(
                f"Starting streaming propagation: session_id={self.video_session_id}"
            )

            request_json = {
                "session_id": self.video_session_id,
                "model": self.current_model_id,
            }
            if range_start is not None:
                request_json["start_frame"] = range_start
            if range_end is not None:
                request_json["end_frame"] = range_end - 1

            response = requests.post(
                url=stream_url,
                json=request_json,
                headers=self.headers,
                stream=True,
                timeout=None,
            )
            response.raise_for_status()

            results = {}
            total_frames = 0
            is_first_progress = True

            cancelled = False
            for line in response.iter_lines(decode_unicode=True):
                if widget and getattr(widget, "cancel_processing", False):
                    logger.info("Propagation cancelled by user")
                    cancelled = True
                    response.close()
                    break

                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                if event_type == "started":
                    total_frames = event.get("total_frames", 0)
                    start_frame_index = event.get("start_frame_index", 0)
                    start_idx = range_start or start_frame_index
                    frames_to_process = (range_end or total_frames) - start_idx
                    if progress_dialog and frames_to_process > 0:
                        progress_dialog.setMaximum(frames_to_process)
                        progress_dialog.setValue(0)
                        progress_dialog.setLabelText(
                            self.tr("Model warming up, please wait...")
                        )
                        QApplication.processEvents()
                    logger.info(
                        f"Propagation started: total_frames={total_frames}, "
                        f"start_frame_index={start_frame_index}"
                    )

                elif event_type == "progress":
                    current_frame = event.get("current_frame", 0)
                    start_idx = range_start or 0
                    if range_end is not None and not (
                        start_idx <= current_frame < range_end
                    ):
                        continue
                    relative_frame = current_frame - start_idx
                    if relative_frame < 0:
                        relative_frame = 0
                    display_frame = relative_frame + 1
                    frames_to_process = (range_end or total_frames) - start_idx

                    if progress_dialog:
                        try:
                            if is_first_progress:
                                is_first_progress = False
                                QApplication.processEvents()

                            if display_frame <= frames_to_process:
                                progress_dialog.setValue(display_frame)
                            template = self.tr("Processing frame %s/%s")
                            message_text = template % (
                                display_frame,
                                frames_to_process,
                            )
                            progress_dialog.setLabelText(message_text)
                            QApplication.processEvents()
                        except Exception as e:
                            logger.warning(f"Error updating progress: {e}")

                elif event_type == "completed":
                    results = event.get("results", {})
                    logger.info(
                        f"Propagation completed: {len(results)} frame results"
                    )

                elif event_type == "error":
                    error_msg = event.get("message", "Unknown error")
                    logger.error(f"Propagation error: {error_msg}")
                    self.on_message(f"Propagation failed: {error_msg}")
                    self._reset_video_session()
                    return AutoLabelingResult([], replace=False)

            if cancelled:
                self._reset_video_session()
                return AutoLabelingResult([], replace=False)

            if not results:
                logger.warning("No results received from propagation")
                return AutoLabelingResult([], replace=False)

            if not widget or not image_list:
                logger.warning("Widget or image list not available")
                return AutoLabelingResult([], replace=False)

            saved_count = 0
            for frame_idx_str, frame_result in results.items():
                try:
                    frame_idx = int(frame_idx_str)
                    if range_end is not None and not (
                        range_start <= frame_idx < range_end
                    ):
                        continue
                    if 0 <= frame_idx < len(image_list):
                        frame_file = image_list[frame_idx]
                        masks = frame_result.get("masks", [])
                        if not masks:
                            continue

                        shapes = []
                        for shape_data in masks:
                            points = shape_data.get("points", [])
                            if not points:
                                continue
                            label = shape_data.get("label", "AUTOLABEL_OBJECT")
                            if self.label is not None:
                                label = self.label
                            group_id = shape_data.get("group_id", None)
                            if self.group_id is not None:
                                group_id = self.group_id
                            shape = Shape(
                                label=label,
                                shape_type=shape_data.get(
                                    "shape_type", "rectangle"
                                ),
                                score=shape_data.get("score", None),
                                group_id=group_id,
                            )
                            for point in points:
                                shape.add_point(
                                    QtCore.QPointF(point[0], point[1])
                                )
                            if shape.points:
                                shapes.append(shape)

                        if shapes:
                            from anylabeling.views.labeling.utils.batch import (
                                save_auto_labeling_result,
                            )

                            replace_value = (
                                False
                                if self.video_prompt_type == "point"
                                else self.replace
                            )
                            save_auto_labeling_result(
                                widget,
                                frame_file,
                                AutoLabelingResult(
                                    shapes, replace=replace_value
                                ),
                            )
                            saved_count += 1
                except (ValueError, KeyError) as e:
                    logger.warning(
                        f"Error processing frame {frame_idx_str}: {e}"
                    )

            self.label, self.group_id = None, None
            logger.info(f"Saved results for {saved_count} frames")
            return AutoLabelingResult([], replace=False)

        except requests.exceptions.RequestException as e:
            logger.error(f"Video propagation request error: {e}")
            self.on_message(f"Video propagation error: {e}")
            self._reset_video_session()
            self.label, self.group_id = None, None
            return AutoLabelingResult([], replace=False)
        except Exception as e:
            logger.error(f"Video propagation error: {e}", exc_info=True)
            self.on_message(f"Video propagation error: {e}")
            self._reset_video_session()
            self.label, self.group_id = None, None
            return AutoLabelingResult([], replace=False)

    def _reset_video_session(self):
        """Reset video session state and cleanup server session."""
        if self.video_session_id:
            try:
                cleanup_url = f"{self.server_url}/v1/video/cleanup/{self.video_session_id}"
                requests.post(
                    url=cleanup_url,
                    params={"model": self.current_model_id},
                    headers=self.headers,
                    timeout=self.timeout,
                )
                logger.debug(f"Cleaned up session {self.video_session_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup session: {e}")

        self.label = None
        self.group_id = None
        self.video_session_id = None
        self.video_initialized = False
        self.video_prompt_frame = None
        self.video_session_image_list = None
        self.video_prompt_type = None

    def unload(self):
        """Unload the model"""
        pass
