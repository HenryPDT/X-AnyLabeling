import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel


def _batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    topk: Optional[int] = None,
) -> np.ndarray:
    """Execute per-class batched NMS using OpenCV or pure numpy fallback."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    if hasattr(cv2.dnn, "NMSBoxesBatched"):
        boxes_xywh = boxes.copy()
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]
        indices = cv2.dnn.NMSBoxesBatched(
            boxes_xywh.tolist(),
            scores.tolist(),
            class_ids.astype(int).tolist(),
            score_threshold=0.0,
            nms_threshold=float(iou_threshold),
            top_k=int(topk) if topk is not None else 0,
        )
        if len(indices) > 0:
            return np.array(indices, dtype=np.int64).flatten()
        return np.array([], dtype=np.int64)

    max_coord = float(boxes.max()) if boxes.size > 0 else 0.0
    shifted_boxes = boxes + class_ids[:, None].astype(boxes.dtype) * (
        max_coord + 1.0
    )

    x1 = shifted_boxes[:, 0]
    y1 = shifted_boxes[:, 1]
    x2 = shifted_boxes[:, 2]
    y2 = shifted_boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if topk is not None and len(keep) >= topk:
            break
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.zeros_like(inter)
        np.divide(inter, union, out=iou, where=union > 0)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def _parse_offsets(raw_offsets: Any) -> Optional[np.ndarray]:
    """Parse offsets from list, tuple, or semicolon/comma-separated string."""
    if not raw_offsets:
        return None
    if isinstance(raw_offsets, str):
        delimiter = ";" if ";" in raw_offsets else ","
        try:
            parts = [
                float(p.strip())
                for p in raw_offsets.split(delimiter)
                if p.strip()
            ]
            arr = np.array(parts, dtype=np.float32)
        except Exception:
            logger.warning(f"Failed to parse offsets string: {raw_offsets}")
            return None
    else:
        try:
            arr = np.array(raw_offsets, dtype=np.float32)
        except Exception:
            logger.warning(
                f"Failed to convert offsets to float array: {raw_offsets}"
            )
            return None

    if arr.shape != (3,):
        logger.warning(
            f"DeepStream offsets must have 3 elements (got shape {arr.shape}). Ignoring offsets."
        )
        return None
    return arr


class DeepStreamDetection(Model):
    """Generic DeepStream-ONNX object detection model backend.

    Contract:
        ONNX output shape is [N, 6] or [1, N, 6] formatted as:
        [x1, y1, x2, y2, score, class_id]
        - Boxes are xyxy in network coordinate space
        - Scores are fused (objectness * class probability)
    """

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
        ]
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "input_iou",
            "edit_iou",
            "input_containment",
            "edit_containment",
            "containment_keep_combobox",
            "toggle_preserve_existing_annotations",
            "button_classes_filter",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config: Dict[str, Any], on_message) -> None:
        super().__init__(model_config, on_message)

        model_name = self.config.get("type", "deepstream_det")
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            cfg_file = self.config.get("config_file", "")
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Model weights file not found: '{model_abs_path or 'not configured'}'. "
                    f"Please verify model_path in {cfg_file or 'model config'}.",
                )
            )

        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)

        # Classes
        self.classes = self.config.get("classes")
        if not self.classes:
            labels_path = self.config.get("labelfile_path") or self.config.get(
                "labelfile-path"
            )
            if labels_path:
                field_name = (
                    "labelfile_path"
                    if "labelfile_path" in self.config
                    else "labelfile-path"
                )
                resolved_labels = (
                    self.get_model_abs_path(self.config, field_name)
                    or labels_path
                )
                if os.path.isfile(resolved_labels):
                    with open(resolved_labels, "r", encoding="utf-8-sig") as f:
                        self.classes = [
                            line.strip()
                            for line in f
                            if line.strip()
                            and not line.strip().startswith("#")
                        ]
        if not self.classes:
            self.classes = []

        # Determine input shape [height, width]
        raw_shape = self.net.get_input_shape()
        if (
            len(raw_shape) >= 4
            and isinstance(raw_shape[2], int)
            and isinstance(raw_shape[3], int)
        ):
            self.input_shape = (raw_shape[2], raw_shape[3])
        elif "input_shape" in self.config:
            self.input_shape = tuple(self.config["input_shape"])
        elif "input_size" in self.config:
            self.input_shape = tuple(self.config["input_size"])
        elif (
            "network_height" in self.config and "network_width" in self.config
        ):
            self.input_shape = (
                int(self.config["network_height"]),
                int(self.config["network_width"]),
            )
        elif (
            "network-height" in self.config and "network-width" in self.config
        ):
            self.input_shape = (
                int(self.config["network-height"]),
                int(self.config["network-width"]),
            )
        else:
            self.input_shape = (640, 640)
            logger.info(
                f"Input shape not explicitly specified for {model_name}; "
                f"defaulting to {self.input_shape}."
            )

        # Preprocessing settings
        color_format = self.config.get("model_color_format", 1)
        if isinstance(color_format, str):
            self.model_color_format = 0 if color_format.lower() == "rgb" else 1
        else:
            self.model_color_format = int(color_format)

        self.net_scale_factor = float(
            self.config.get(
                "net_scale_factor", self.config.get("net-scale-factor", 1.0)
            )
        )
        self.offsets = _parse_offsets(self.config.get("offsets"))

        mar = self.config.get(
            "maintain_aspect_ratio",
            self.config.get("maintain-aspect-ratio", 1),
        )
        self.maintain_aspect_ratio = bool(int(mar))

        sym_pad = self.config.get(
            "symmetric_padding", self.config.get("symmetric-padding", 0)
        )
        self.symmetric_padding = bool(int(sym_pad))

        self.pad_value = int(self.config.get("pad_value", 114))

        # Postprocessing settings
        if "cluster_mode" in self.config or "cluster-mode" in self.config:
            self.cluster_mode = int(
                self.config.get(
                    "cluster_mode", self.config.get("cluster-mode", 2)
                )
            )
        elif "nms" in self.config:
            self.cluster_mode = 2 if self.config.get("nms") else 4
        else:
            self.cluster_mode = 2

        self.conf_thres = float(
            self.config.get(
                "conf_threshold",
                self.config.get("pre_cluster_threshold", 0.25),
            )
        )
        self.nms_thres = float(
            self.config.get(
                "iou_threshold",
                self.config.get("nms_iou_threshold", 0.45),
            )
        )
        self.topk = (
            int(self.config["topk"])
            if self.config.get("topk") is not None
            else 300
        )

        self.containment_thres = float(
            self.config.get("containment_threshold", 0.0)
        )
        self.containment_keep = (
            str(self.config.get("containment_keep", "area")).lower()
            if str(self.config.get("containment_keep", "area")).lower()
            in {"score", "area"}
            else "area"
        )

        self.filter_classes = None
        self.replace = True

    def set_auto_labeling_conf(self, value):
        """Set auto labeling confidence threshold."""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_iou(self, value):
        """Set auto labeling IoU threshold."""
        if value > 0:
            self.nms_thres = value

    def set_auto_labeling_containment(self, value):
        """Set same-label nested-box containment threshold."""
        self.containment_thres = float(value)

    def set_auto_labeling_containment_keep(self, mode):
        """Set containment survivor mode: 'score' or 'area'."""
        mode = (mode or "area").lower()
        self.containment_keep = mode if mode in {"score", "area"} else "area"

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations."""
        self.replace = not state

    def set_auto_labeling_filter_classes(self, class_names):
        """Set filter classes by name."""
        if not class_names or len(class_names) == len(self.classes):
            self.filter_classes = None
        else:
            self.filter_classes = class_names

    def preprocess(
        self, input_image: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess RGB image from Qt into network input tensor."""
        orig_h, orig_w = input_image.shape[:2]
        net_h, net_w = self.input_shape

        if self.maintain_aspect_ratio:
            ratio = min(net_w / orig_w, net_h / orig_h)
            new_w = int(orig_w * ratio)
            new_h = int(orig_h * ratio)
            resized = cv2.resize(
                input_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

            if self.symmetric_padding:
                pad_x = (net_w - new_w) / 2.0
                pad_y = (net_h - new_h) / 2.0
            else:
                pad_x = 0.0
                pad_y = 0.0

            canvas = np.full((net_h, net_w, 3), self.pad_value, dtype=np.uint8)
            top = int(pad_y)
            left = int(pad_x)
            canvas[top : top + new_h, left : left + new_w] = resized

            meta = {
                "ratio": ratio,
                "pad_x": pad_x,
                "pad_y": pad_y,
                "maintain_aspect_ratio": True,
                "orig_w": orig_w,
                "orig_h": orig_h,
            }
        else:
            canvas = cv2.resize(
                input_image, (net_w, net_h), interpolation=cv2.INTER_LINEAR
            )
            meta = {
                "r_w": net_w / orig_w,
                "r_h": net_h / orig_h,
                "maintain_aspect_ratio": False,
                "orig_w": orig_w,
                "orig_h": orig_h,
            }

        # Input from Qt is RGB
        if self.model_color_format == 1:  # Target BGR
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        blob = canvas.astype(np.float32)
        if self.offsets is not None:
            blob -= self.offsets
        if self.net_scale_factor != 1.0:
            blob *= self.net_scale_factor

        blob = blob.transpose(2, 0, 1)[None, ...]
        blob = np.ascontiguousarray(blob, dtype=np.float32)
        return blob, meta

    def postprocess(
        self, raw_output: np.ndarray, meta: Dict[str, Any]
    ) -> List[Shape]:
        """Postprocess network output into Shapes."""
        preds = (
            raw_output[0]
            if raw_output.ndim == 3 and raw_output.shape[0] == 1
            else raw_output
        )
        if preds.ndim != 2 or preds.shape[0] == 0 or preds.shape[1] < 6:
            return []

        boxes = preds[:, :4]
        scores = preds[:, 4]
        class_ids = preds[:, 5]

        # 1. Filter by score
        mask = scores >= self.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        if len(boxes) == 0:
            return []

        # 2. Clamp to network size
        net_h, net_w = self.input_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, net_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, net_h)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        if len(boxes) == 0:
            return []

        # 3. NMS vs NMS-Free
        if self.cluster_mode == 2:
            keep = _batched_nms(
                boxes, scores, class_ids, self.nms_thres, self.topk
            )
            boxes = boxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]
        else:
            order = np.argsort(-scores)
            if self.topk is not None and len(order) > self.topk:
                order = order[: self.topk]
            boxes = boxes[order]
            scores = scores[order]
            class_ids = class_ids[order]

        if len(boxes) == 0:
            return []

        # 4. Rescale back to original image coordinates
        orig_w = max(1, int(meta.get("orig_w", 1)))
        orig_h = max(1, int(meta.get("orig_h", 1)))
        if meta.get("maintain_aspect_ratio"):
            ratio = float(meta.get("ratio", 1.0))
            if ratio > 0:
                pad_x = float(meta.get("pad_x", 0.0))
                pad_y = float(meta.get("pad_y", 0.0))
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        else:
            r_w = float(meta.get("r_w", 1.0))
            r_h = float(meta.get("r_h", 1.0))
            if r_w > 0 and r_h > 0:
                boxes[:, [0, 2]] /= r_w
                boxes[:, [1, 3]] /= r_h

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # 5. Construct Shape objects
        shapes = []
        for box, score, cid in zip(boxes, scores, class_ids):
            class_idx = int(cid)
            label = (
                str(self.classes[class_idx])
                if 0 <= class_idx < len(self.classes)
                else str(class_idx)
            )

            if self.filter_classes and label not in self.filter_classes:
                continue

            x1, y1, x2, y2 = box.tolist()
            rectangle_shape = Shape(
                label=label, score=float(score), shape_type="rectangle"
            )
            rectangle_shape.add_point(QtCore.QPointF(x1, y1))
            rectangle_shape.add_point(QtCore.QPointF(x2, y1))
            rectangle_shape.add_point(QtCore.QPointF(x2, y2))
            rectangle_shape.add_point(QtCore.QPointF(x1, y2))
            shapes.append(rectangle_shape)

        return shapes

    def predict_shapes(
        self, image, image_path: Optional[str] = None
    ) -> AutoLabelingResult:
        """Predict shapes from input image."""
        if image is None:
            return AutoLabelingResult([], replace=self.replace)

        if isinstance(image, np.ndarray):
            image_rgb = image
        else:
            try:
                image_rgb = qt_img_to_rgb_cv_img(image, image_path)
            except Exception as e:
                logger.warning(
                    "Could not convert image for DeepStream inference"
                )
                logger.warning(e)
                return AutoLabelingResult([], replace=self.replace)

        blob, meta = self.preprocess(image_rgb)
        raw_output = self.net.get_ort_inference(blob, extract=True)
        shapes = self.postprocess(raw_output, meta)
        if self.containment_thres > 0 and len(shapes) > 1:
            from anylabeling.views.labeling.utils.shape_geometry import (
                apply_same_label_containment_nms,
            )

            shapes = apply_same_label_containment_nms(
                shapes,
                self.containment_thres,
                keep_mode=self.containment_keep,
            )
        return AutoLabelingResult(shapes, replace=self.replace)

    def unload(self):
        """Unload ONNX model."""
        if hasattr(self, "net") and self.net is not None:
            del self.net
            self.net = None
