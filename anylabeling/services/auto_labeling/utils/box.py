import numpy as np

from .points_conversion import xywh2xyxy


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box1, box2):
    area1 = box_area(box1)  # N
    area2 = box_area(box2)  # M
    # broadcasting
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou  # NxM


def numpy_nms(boxes, scores, iou_threshold):
    idxs = scores.argsort()
    keep = []
    while idxs.size > 0:
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)
        if idxs.size == 1:
            break
        idxs = idxs[:-1]
        other_boxes = boxes[idxs]
        ious = box_iou(max_score_box, other_boxes)
        idxs = idxs[ious[0] <= iou_threshold]
    keep = np.array(keep)
    return keep


def shape_to_xyxy(shape):
    """Return axis-aligned [x1, y1, x2, y2] from a Shape's points."""
    if not getattr(shape, "points", None):
        return None
    xs = [point.x() for point in shape.points]
    ys = [point.y() for point in shape.points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _box_intersection_area(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _containment_ratio(inner_box, outer_box):
    """Fraction of ``inner_box`` area that lies inside ``outer_box``."""
    area = _box_area(inner_box)
    if area <= 0:
        return 0.0
    return _box_intersection_area(inner_box, outer_box) / area


def apply_class_agnostic_shape_nms(
    shapes,
    iou_threshold,
    group_by_shape_type=True,
    containment_threshold=None,
    containment_keep="score",
):
    """Client-side box cleanup for multi-prompt SAM3 outputs.

    1. Class-agnostic IoU NMS when ``iou_threshold`` > 0
       (e.g. truck vs bus on the same region).
    2. Same-label containment suppression when ``containment_threshold`` > 0
       (e.g. a nested ``motorbike and person`` fragment).

    ``containment_keep``:
      - ``"score"``: keep the higher-confidence box
      - ``"area"``: keep the larger box

    When ``group_by_shape_type`` is True, each ``shape_type`` is processed
    separately so remote SAM3 box+mask pairs are not collapsed.
    """
    if not shapes:
        return shapes

    kept = list(shapes)
    if iou_threshold is not None and iou_threshold > 0:
        if group_by_shape_type:
            grouped = {}
            for shape in kept:
                grouped.setdefault(
                    getattr(shape, "shape_type", None), []
                ).append(shape)
            next_kept = []
            for group_shapes in grouped.values():
                next_kept.extend(
                    _apply_class_agnostic_shape_nms_group(
                        group_shapes, iou_threshold
                    )
                )
            kept = next_kept
        else:
            kept = _apply_class_agnostic_shape_nms_group(kept, iou_threshold)

    if containment_threshold is not None and containment_threshold > 0:
        if group_by_shape_type:
            grouped = {}
            for shape in kept:
                grouped.setdefault(
                    getattr(shape, "shape_type", None), []
                ).append(shape)
            next_kept = []
            for group_shapes in grouped.values():
                next_kept.extend(
                    apply_same_label_containment_nms(
                        group_shapes,
                        containment_threshold,
                        keep_mode=containment_keep,
                    )
                )
            kept = next_kept
        else:
            kept = apply_same_label_containment_nms(
                kept,
                containment_threshold,
                keep_mode=containment_keep,
            )
    return kept


def apply_same_label_containment_nms(
    shapes, containment_threshold, keep_mode="score"
):
    """Drop same-label nested shapes using score or area priority.

    A shape is suppressed when most of its box lies inside (or contains)
    another same-label box. ``keep_mode`` selects which one survives:
    ``"score"`` (higher confidence) or ``"area"`` (larger box).
    """
    if (
        not shapes
        or containment_threshold is None
        or containment_threshold <= 0
    ):
        return shapes

    keep_mode = (keep_mode or "score").lower()
    if keep_mode not in {"score", "area"}:
        keep_mode = "score"

    boxes = []
    scores = []
    areas = []
    labels = []
    valid_shapes = []
    passthrough_shapes = []
    for shape in shapes:
        box = shape_to_xyxy(shape)
        if box is None:
            passthrough_shapes.append(shape)
            continue
        boxes.append(box)
        scores.append(float(shape.score) if shape.score is not None else 0.0)
        areas.append(_box_area(box))
        labels.append(getattr(shape, "label", None))
        valid_shapes.append(shape)

    if not valid_shapes:
        return list(shapes)

    priority = areas if keep_mode == "area" else scores
    order = np.argsort(-np.asarray(priority, dtype=np.float32), kind="stable")
    keep = []
    for index in order:
        candidate_box = boxes[int(index)]
        candidate_label = labels[int(index)]
        suppressed = False
        for kept_index in keep:
            if labels[kept_index] != candidate_label:
                continue
            kept_box = boxes[kept_index]
            candidate_in_kept = _containment_ratio(candidate_box, kept_box)
            kept_in_candidate = _containment_ratio(kept_box, candidate_box)
            if (
                max(candidate_in_kept, kept_in_candidate)
                > containment_threshold
            ):
                suppressed = True
                break
        if not suppressed:
            keep.append(int(index))

    keep_set = set(keep)
    result = [
        shape for index, shape in enumerate(valid_shapes) if index in keep_set
    ]
    result.extend(passthrough_shapes)
    return result


def _apply_class_agnostic_shape_nms_group(shapes, iou_threshold):
    boxes = []
    scores = []
    valid_shapes = []
    passthrough_shapes = []
    for shape in shapes:
        box = shape_to_xyxy(shape)
        if box is None:
            passthrough_shapes.append(shape)
            continue
        boxes.append(box)
        scores.append(float(shape.score) if shape.score is not None else 0.0)
        valid_shapes.append(shape)

    if not valid_shapes:
        return list(shapes)

    keep = numpy_nms(
        np.asarray(boxes, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        iou_threshold,
    )
    result = [valid_shapes[int(index)] for index in keep]
    result.extend(passthrough_shapes)
    return result


def numpy_nms_rotated(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)

    sorted_idx = np.argsort(scores)[::-1]
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes)
    ious = np.triu(ious, k=1)
    pick = np.nonzero(np.max(ious, axis=0) < iou_threshold)[0]
    return sorted_idx[pick]


def batch_probiou(obb1, obb2, eps=1e-7):
    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in np.split(obb2[..., :2], 2, axis=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))
    t1 = (
        (
            (a1 + a2) * (np.power(y1 - y2, 2))
            + (b1 + b2) * (np.power(x1 - x2, 2))
        )
        / ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)) + eps)
    ) * 0.25
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2))
        / ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)) + eps)
    ) * 0.5

    t3 = (
        np.log(
            ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)))
            / (
                4
                * np.sqrt(
                    (a1 * b1 - np.power(c1, 2)).clip(0)
                    * (a2 * b2 - np.power(c2, 2)).clip(0)
                )
                + eps
            )
            + eps
        )
        * 0.5
    )
    bd = t1 + t2 + t3
    bd = np.clip(bd, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def _get_covariance_matrix(boxes):
    gbbs = np.concatenate(
        (np.power(boxes[:, 2:4], 2) / 12, boxes[:, 4:]), axis=-1
    )
    a, b, c = np.split(gbbs, [1, 2], axis=-1)
    return (
        a * np.cos(c) ** 2 + b * np.sin(c) ** 2,
        a * np.sin(c) ** 2 + b * np.cos(c) ** 2,
        a * np.cos(c) * np.sin(c) - b * np.sin(c) * np.cos(c),
    )


def non_max_suppression_v5(
    prediction,
    task="det",
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_nms=30000,
    max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, \
        with support for masks and multiple labels per box.

    Arguments:
        prediction (np.array):
            A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks.
            The tensor should be in the format output by a model, such as YOLO.
        task: `det` | `seg` | `track`
        conf_thres (float):
            The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float):
            The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider.
            If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes,
            and all classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, np.array]]]):
            A list of lists, where each inner list contains the apriori labels \
            for a given image. The list should be in the format output by a dataloader, \
            with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. \
            Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into numpy_nms.
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[np.array]):
            A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes,
            with columns (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, \
        valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, \
        valid values are between 0.0 and 1.0"
    if task == "seg" and nc == 0:
        raise ValueError("The value of nc must be set when the mode is 'seg'.")
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # select only inference output
    bs = prediction.shape[0]  # batch size
    if task in ["det", "track"]:
        nc = prediction.shape[2] - 5  # number of classes

    nm = prediction.shape[2] - nc - 5
    mi = 5 + nc  # mask start index
    xc = prediction[..., 4] > conf_thres  # candidates

    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
    output = [np.zeros((0, 6 + nm))] * bs

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) |
        # (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[np.arange(len(lb)), lb[:, 0].astype(int) + 4] = 1.0  # cls
            x = np.concatenate((x[xc], v), axis=0)

        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        box = x[:, :4]
        mask = x[:, mi:]
        cls = x[:, 5:mi]

        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate(
                (box[i], x[i, 5 + j, None], j[:, None].astype(float), mask[i]),
                axis=1,
            )
        else:  # best class only
            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True)
            x = np.concatenate((box, conf, j.astype(float), mask), axis=1)[
                conf.flatten() > conf_thres
            ]
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[np.argsort(x[:, 4])[::-1][:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = numpy_nms(boxes, scores, iou_thres)
        i = i[:max_det]
        if merge and (1 < n < 3e3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = np.dot(weights, x[:, :4]) / weights.sum(
                1, keepdims=True
            )
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]

    return output


def non_max_suppression_v8(
    prediction,
    task="det",
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_nms=30000,
    max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, \
        with support for masks and multiple labels per box.

    Arguments:
        prediction (np.array):
            A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks.
            The tensor should be in the format output by a model, such as YOLO.
        task: `det` | `seg` | `track` | `obb`
        conf_thres (float):
            The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float):
            The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider.
            If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes,
            and all classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, np.array]]]):
            A list of lists, where each inner list contains the apriori labels \
            for a given image. The list should be in the format output by a dataloader, \
            with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. \
            Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into numpy_nms.
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[np.array]):
            A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes,
            with columns (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, \
        valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, \
        valid values are between 0.0 and 1.0"
    if task == "seg" and nc == 0:
        raise ValueError("The value of nc must be set when the mode is 'seg'.")
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # select only inference output
    bs = prediction.shape[0]  # batch size
    if task in ["det", "track"]:
        nc = prediction.shape[1] - 4  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = np.amax(prediction[:, 4:mi], axis=1) > conf_thres  # candidates

    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    # shape(1,84,6300) to shape(1,6300,84)
    prediction = np.transpose(prediction, (0, 2, 1))
    if task != "obb":
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
    output = [np.zeros((0, 6 + nm))] * bs

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) |
        # (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        if labels and len(labels[xi]) and task != "obb":
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[np.arange(len(lb)), lb[:, 0].astype(int) + 4] = 1.0  # cls
            x = np.concatenate((x[xc], v), axis=0)

        if not x.shape[0]:
            continue

        box = x[:, :4]
        cls = x[:, 4 : 4 + nc]
        mask = x[:, 4 + nc : 4 + nc + nm]

        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate(
                (box[i], x[i, 4 + j, None], j[:, None].astype(float), mask[i]),
                axis=1,
            )
        else:  # best class only
            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True)
            x = np.concatenate((box, conf, j.astype(float), mask), axis=1)[
                conf.flatten() > conf_thres
            ]
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[np.argsort(x[:, 4])[::-1][:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        if task == "obb":
            boxes = np.concatenate(
                (x[:, :2] + c, x[:, 2:4], x[:, -1:]), axis=-1
            )  # xywhr
            i = numpy_nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c
            i = numpy_nms(boxes, scores, iou_thres)
        i = i[:max_det]
        # if merge and (1 < n < 3e3):
        #     iou = box_iou(boxes[i], boxes) > iou_thres
        #     weights = iou * scores[None]
        #     x[i, :4] = np.dot(weights, x[:, :4]) / weights.sum(
        #         1, keepdims=True
        #     )
        #     if redundant:
        #         i = i[iou.sum(1) > 1]

        output[xi] = x[i]

    return output


def non_max_suppression_end2end(
    prediction,
    task="det",
    conf_thres=0.25,
    classes=None,
    max_det=300,
    nm=0,  # number of masks (for seg task)
    nkpt=0,  # number of keypoints (for pose task)
    ndim=3,  # keypoint dimensions (for pose task)
):
    """
    Process end-to-end model output (no NMS needed).

    End-to-end models like YOLO26 output predictions that are already
    post-processed, so this function only performs confidence filtering,
    class filtering, and max_det limiting.

    Arguments:
        prediction (np.array):
            End-to-end model output with shape (batch_size, num_boxes, num_features)
            - det: (batch, num_boxes, 6) where 6 = [x1, y1, x2, y2, score, class_id]
            - obb: (batch, num_boxes, 7) where 7 = [x, y, w, h, score, class_id, angle]
            - seg: (batch, num_boxes, 6+nm) where 6+nm = [x1, y1, x2, y2, score, class_id, mask_coeffs...]
            - pose: (batch, num_boxes, 6+nkpt*ndim) where 6+nkpt*ndim = [x1, y1, x2, y2, score, class_id, keypoints...]
        task: `det` | `seg` | `obb` | `pose`
        conf_thres (float):
            The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider.
            If None, all classes will be considered.
        max_det (int): The maximum number of boxes to keep.
        nm (int): Number of mask coefficients (for seg task).
        nkpt (int): Number of keypoints (for pose task).
        ndim (int): Keypoint dimensions, typically 2 (x,y) or 3 (x,y,visibility) (for pose task).

    Returns:
        (List[np.array]):
            A list of length batch_size, where each element is a tensor of
            shape (num_boxes, num_features) containing the kept boxes.
            - det: (num_boxes, 6) with columns [x1, y1, x2, y2, confidence, class]
            - obb: (num_boxes, 7) with columns [x, y, w, h, confidence, class, angle]
            - seg: (num_boxes, 6+nm) with columns [x1, y1, x2, y2, confidence, class, mask_coeffs...]
            - pose: (num_boxes, 6+nkpt*ndim) with columns [x1, y1, x2, y2, confidence, class, keypoints...]
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, \
        valid values are between 0.0 and 1.0"

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # select only inference output

    # Handle shape: ensure (batch, num_boxes, features)
    if len(prediction.shape) == 2:
        prediction = prediction[np.newaxis, ...]  # add batch dimension

    bs = prediction.shape[0]  # batch size

    # Determine output feature size based on task
    if task == "obb":
        out_features = 7  # x, y, w, h, conf, class, angle
    elif task == "seg":
        out_features = 6 + nm  # x1, y1, x2, y2, conf, class, mask_coeffs...
    elif task == "pose":
        out_features = (
            6 + nkpt * ndim
        )  # x1, y1, x2, y2, conf, class, keypoints...
    else:  # det
        out_features = 6  # x1, y1, x2, y2, conf, class

    output = [np.zeros((0, out_features))] * bs

    for xi, x in enumerate(prediction):  # image index, image inference
        # Filter by confidence threshold (score is at index 4)
        mask = x[:, 4] >= conf_thres
        x = x[mask]

        if not x.shape[0]:
            continue

        # Filter by classes (class_id is at index 5)
        if classes is not None:
            class_mask = np.isin(x[:, 5].astype(int), classes)
            x = x[class_mask]

        if not x.shape[0]:
            continue

        # Sort by confidence and limit to max_det
        if x.shape[0] > max_det:
            indices = np.argsort(x[:, 4])[::-1][:max_det]
            x = x[indices]

        output[xi] = x

    return output
