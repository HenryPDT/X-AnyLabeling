from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from PIL import Image

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.shape_geometry import (
    clamp_shape_to_image_bounds,
    detect_duplicate_shapes,
    shape_to_xyxy,
)
from anylabeling.views.labeling.utils.qt import get_image_delete_trash_dir


@dataclass
class ScanIssue:
    """Represents an issue found during annotation diagnostics."""

    image_path: str
    label_file: Optional[str]
    issue_type: str  # "duplicate_shape", "degenerate_geometry", "micro_noise", "missing_label", "out_of_bounds", "duplicate_image", "missing_image"
    severity: str  # "error", "warning", "info"
    details: str
    shape_index: Optional[int] = None
    shape_data: Optional[Dict[str, Any]] = None
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScanReport:
    """Summary of a dataset diagnostics scan."""

    total_images: int = 0
    total_annotations: int = 0
    empty_images: int = 0
    verified_empty_images: int = 0
    missing_labels: int = 0
    duplicate_shapes_count: int = 0
    corrupt_shapes_count: int = 0
    micro_shapes_count: int = 0
    out_of_bounds_count: int = 0
    duplicate_images_count: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    issues: List[ScanIssue] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        return len(self.issues)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["total_issues"] = self.total_issues
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def get_label_file_path(
    image_path: str, output_dir: Optional[str] = None
) -> str:
    """Resolve the expected annotation JSON file path for a given image path."""
    json_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    if output_dir:
        return os.path.join(output_dir, json_filename)
    return os.path.splitext(image_path)[0] + ".json"


def compute_file_md5(file_path: str, chunk_size: int = 65536) -> Optional[str]:
    """Compute MD5 hex digest of a file in chunks."""
    if not os.path.isfile(file_path):
        return None
    try:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as exc:
        logger.warning(f"Failed to compute MD5 for {file_path}: {exc}")
        return None


def _check_point_coordinates(
    points: Any, shape_index: int
) -> Optional[Tuple[str, str, str, str]]:
    """Validate point list integrity and numeric finiteness."""
    if not points or not isinstance(points, list):
        return (
            "degenerate_geometry",
            "error",
            f"Shape #{shape_index + 1} has no coordinates or empty points list.",
            "Delete or re-draw the shape.",
        )

    for pt_idx, pt in enumerate(points):
        try:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                raise TypeError("invalid vertex container")
            x_val = float(pt[0])
            y_val = float(pt[1])
            if not math.isfinite(x_val) or not math.isfinite(y_val):
                raise ValueError("non-finite coordinates")
        except (TypeError, ValueError, IndexError):
            return (
                "degenerate_geometry",
                "error",
                f"Shape #{shape_index + 1} vertex #{pt_idx + 1} contains non-finite (NaN/Inf) coordinates.",
                "Remove corrupt coordinates.",
            )
    return None


def _check_shape_type_geometry(
    shape: Dict[str, Any],
    shape_type: str,
    points: List[Any],
    shape_index: int,
) -> List[Tuple[str, str, str, str]]:
    """Validate shape vertices according to shape type."""
    issues: List[Tuple[str, str, str, str]] = []

    if shape_type == "rectangle":
        if len(points) not in (2, 4):
            issues.append(
                (
                    "degenerate_geometry",
                    "error",
                    f"Rectangle shape #{shape_index + 1} has {len(points)} points (expected 2 or 4).",
                    "Re-draw rectangle with standard vertices.",
                )
            )
        else:
            xyxy = shape_to_xyxy(shape)
            if xyxy:
                w = max(0.0, xyxy[2] - xyxy[0])
                h = max(0.0, xyxy[3] - xyxy[1])
                if w <= 0.0 or h <= 0.0:
                    issues.append(
                        (
                            "degenerate_geometry",
                            "error",
                            f"Rectangle shape #{shape_index + 1} has zero width or height.",
                            "Delete zero-dimension rectangle.",
                        )
                    )
    elif shape_type == "polygon":
        if len(points) < 3:
            issues.append(
                (
                    "degenerate_geometry",
                    "error",
                    f"Polygon shape #{shape_index + 1} has only {len(points)} vertices (minimum 3 required).",
                    "Add vertices or remove degenerate polygon.",
                )
            )
        else:
            has_dup_consecutive = any(
                points[i][0] == points[i + 1][0]
                and points[i][1] == points[i + 1][1]
                for i in range(len(points) - 1)
            )
            if has_dup_consecutive:
                issues.append(
                    (
                        "degenerate_geometry",
                        "warning",
                        f"Polygon shape #{shape_index + 1} has duplicate consecutive vertices.",
                        "Simplify polygon or remove duplicate vertices.",
                    )
                )
    elif shape_type == "rotation" and len(points) != 4:
        issues.append(
            (
                "degenerate_geometry",
                "error",
                f"Rotated box shape #{shape_index + 1} has {len(points)} points (expected 4).",
                "Re-draw rotated bounding box.",
            )
        )
    elif shape_type == "point" and len(points) != 1:
        issues.append(
            (
                "degenerate_geometry",
                "error",
                f"Point shape #{shape_index + 1} has {len(points)} points (expected 1).",
                "Re-draw point with a single vertex.",
            )
        )
    elif shape_type in ("line", "linestrip") and len(points) < 2:
        issues.append(
            (
                "degenerate_geometry",
                "error",
                f"Line shape #{shape_index + 1} has {len(points)} points (minimum 2 required).",
                "Add vertices or remove degenerate line.",
            )
        )
    elif shape_type == "circle" and len(points) != 2:
        issues.append(
            (
                "degenerate_geometry",
                "error",
                f"Circle shape #{shape_index + 1} has {len(points)} points (expected 2: center + rim).",
                "Re-draw circle with standard vertices.",
            )
        )
    elif shape_type == "mask" and len(points) < 3:
        issues.append(
            (
                "degenerate_geometry",
                "error",
                f"Mask shape #{shape_index + 1} has only {len(points)} vertices (minimum 3 required).",
                "Add vertices or remove degenerate mask.",
            )
        )

    return issues


def validate_shape(
    shape: Dict[str, Any],
    shape_index: int,
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
    min_size_px: float = 5.0,
    min_area_px: float = 16.0,
) -> List[Tuple[str, str, str, str]]:
    """Validate a single shape dictionary.

    Returns list of tuples: (issue_type, severity, details, suggestion)
    """
    issues: List[Tuple[str, str, str, str]] = []

    if not isinstance(shape, dict):
        issues.append(
            (
                "degenerate_geometry",
                "error",
                f"Shape #{shape_index + 1} is corrupt (not a dict).",
                "Delete or re-draw the corrupt shape.",
            )
        )
        return issues

    # Coerce image dims (JSON may store them as str)
    def _coerce_dim(v: Any) -> Optional[float]:
        try:
            f = float(v)
            return f if math.isfinite(f) and f > 0 else None
        except (TypeError, ValueError):
            return None

    img_w = _coerce_dim(image_width) if image_width is not None else None
    img_h = _coerce_dim(image_height) if image_height is not None else None

    # 1. Label check
    label = shape.get("label")
    if label is None or not str(label).strip():
        issues.append(
            (
                "missing_label",
                "error",
                f"Shape #{shape_index + 1} has an empty or missing class label.",
                "Assign a valid class label to the shape.",
            )
        )

    # 2. Points coordinate checks
    points = shape.get("points")
    pt_issue = _check_point_coordinates(points, shape_index)
    if pt_issue:
        issues.append(pt_issue)
        return issues

    shape_type = str(shape.get("shape_type", "polygon")).lower()
    issues.extend(
        _check_shape_type_geometry(shape, shape_type, points, shape_index)
    )

    # 3. Micro-noise check (exempt point/line/linestrip from height gate)
    xyxy = shape_to_xyxy(shape)
    if xyxy and shape_type not in ("point", "line", "linestrip"):
        w = max(0.0, xyxy[2] - xyxy[0])
        h = max(0.0, xyxy[3] - xyxy[1])
        area = w * h
        if w < min_size_px or h < min_size_px or area < min_area_px:
            issues.append(
                (
                    "micro_noise",
                    "warning",
                    f"Shape #{shape_index + 1} is micro-noise ({w:.1f}x{h:.1f}px, area={area:.1f}px² < {min_area_px}px²).",
                    "Purge or resize micro-noise shape.",
                )
            )
    elif xyxy and shape_type in ("line", "linestrip"):
        # Lines: only flag if length is tiny
        w = max(0.0, xyxy[2] - xyxy[0])
        h = max(0.0, xyxy[3] - xyxy[1])
        length = math.hypot(w, h)
        if length < min_size_px:
            issues.append(
                (
                    "micro_noise",
                    "warning",
                    f"Shape #{shape_index + 1} is micro-noise (length={length:.1f}px < {min_size_px}px).",
                    "Purge or resize micro-noise shape.",
                )
            )

    # 4. Out of bounds check
    if xyxy and img_w is not None and img_h is not None:
        x1, y1, x2, y2 = xyxy
        margin = 0.5
        if (
            x1 < -margin
            or y1 < -margin
            or x2 > (img_w + margin)
            or y2 > (img_h + margin)
        ):
            issues.append(
                (
                    "out_of_bounds",
                    "warning",
                    f"Shape #{shape_index + 1} extends outside image bounds ({int(img_w)}x{int(img_h)}): [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}].",
                    "Clamp coordinates inside image boundaries.",
                )
            )

    return issues


class AnnotationDiagnosticsScanner:
    """Diagnostic scanner for datasets in X-AnyLabeling format."""

    def __init__(
        self,
        iou_duplicate_threshold: float = 0.85,
        containment_duplicate_threshold: float = 0.90,
        min_size_px: float = 5.0,
        min_area_px: float = 16.0,
        check_image_hashes: bool = True,
        same_label_only: bool = False,
    ):
        self.iou_duplicate_threshold = iou_duplicate_threshold
        self.containment_duplicate_threshold = containment_duplicate_threshold
        self.min_size_px = min_size_px
        self.min_area_px = min_area_px
        self.check_image_hashes = check_image_hashes
        self.same_label_only = same_label_only

    def _check_image_duplicate(
        self,
        image_path: str,
        image_hashes: Dict[str, str],
        report: ScanReport,
    ) -> None:
        """Check if image is an exact binary duplicate of an earlier image."""
        if not self.check_image_hashes:
            return
        file_hash = compute_file_md5(image_path)
        if not file_hash:
            return
        if file_hash in image_hashes:
            orig_file = image_hashes[file_hash]
            report.duplicate_images_count += 1
            report.issues.append(
                ScanIssue(
                    image_path=image_path,
                    label_file=None,
                    issue_type="duplicate_image",
                    severity="warning",
                    details=f"Exact binary duplicate of '{os.path.basename(orig_file)}'.",
                    suggestion="Remove redundant duplicate image file.",
                )
            )
        else:
            image_hashes[file_hash] = image_path

    def _check_image_shapes(
        self,
        image_path: str,
        label_file: str,
        shapes: List[Dict[str, Any]],
        img_width: Optional[float],
        img_height: Optional[float],
        report: ScanReport,
    ) -> None:
        """Inspect and validate individual shapes on an image."""
        for shape_idx, shape in enumerate(shapes):
            try:
                if not isinstance(shape, dict):
                    raise TypeError("shape entry is not a dict")
                lbl = shape.get("label")
                if lbl and str(lbl).strip():
                    lbl_str = str(lbl).strip()
                    report.class_counts[lbl_str] = (
                        report.class_counts.get(lbl_str, 0) + 1
                    )

                findings = validate_shape(
                    shape=shape,
                    shape_index=shape_idx,
                    image_width=img_width,
                    image_height=img_height,
                    min_size_px=self.min_size_px,
                    min_area_px=self.min_area_px,
                )
            except Exception as exc:
                report.corrupt_shapes_count += 1
                report.issues.append(
                    ScanIssue(
                        image_path=image_path,
                        label_file=label_file,
                        issue_type="degenerate_geometry",
                        severity="error",
                        details=(
                            f"Shape #{shape_idx + 1} is corrupt/unparseable: {exc}"
                        ),
                        shape_index=shape_idx,
                        shape_data=shape if isinstance(shape, dict) else None,
                        suggestion="Delete or re-draw the corrupt shape.",
                    )
                )
                continue
            for issue_type, severity, details, suggestion in findings:
                if issue_type == "micro_noise":
                    report.micro_shapes_count += 1
                elif issue_type == "out_of_bounds":
                    report.out_of_bounds_count += 1
                elif issue_type in ("degenerate_geometry", "missing_label"):
                    report.corrupt_shapes_count += 1

                report.issues.append(
                    ScanIssue(
                        image_path=image_path,
                        label_file=label_file,
                        issue_type=issue_type,
                        severity=severity,
                        details=details,
                        shape_index=shape_idx,
                        shape_data=shape,
                        suggestion=suggestion,
                    )
                )

        if len(shapes) >= 2:
            self._check_duplicate_shapes(
                image_path, label_file, shapes, report
            )

    def _check_duplicate_shapes(
        self,
        image_path: str,
        label_file: str,
        shapes: List[Dict[str, Any]],
        report: ScanReport,
    ) -> None:
        """Detect duplicate and high-containment shapes on the same image."""
        try:
            duplicates = detect_duplicate_shapes(
                shapes,
                iou_threshold=self.iou_duplicate_threshold,
                containment_threshold=self.containment_duplicate_threshold,
                same_label_only=self.same_label_only,
            )
        except Exception as exc:
            logger.warning(
                f"Duplicate detection failed for {image_path}: {exc}"
            )
            return
        for dup in duplicates:
            orig_idx = dup["index_a"]
            dup_idx = dup["index_b"]
            iou_val = dup["metrics"]["iou"]
            cont_val = dup["metrics"]["containment"]
            lbl1 = dup.get("label_a") or "unknown"
            lbl2 = dup.get("label_b") or "unknown"
            # Cross-class overlap is ambiguity to review, not auto-delete
            cross_class = str(lbl1) != str(lbl2)
            report.duplicate_shapes_count += 1
            report.issues.append(
                ScanIssue(
                    image_path=image_path,
                    label_file=label_file,
                    issue_type="duplicate_shape",
                    severity="info" if cross_class else "warning",
                    details=(
                        f"Shape #{dup_idx + 1} ('{lbl2}') overlaps with "
                        f"shape #{orig_idx + 1} ('{lbl1}') (IoU={iou_val:.2f}, Containment={cont_val:.2f})."
                        + (
                            " [cross-class — review, not auto-delete]"
                            if cross_class
                            else ""
                        )
                    ),
                    shape_index=dup_idx,
                    shape_data=shapes[dup_idx]
                    if 0 <= dup_idx < len(shapes)
                    else None,
                    suggestion="Remove redundant duplicate shape."
                    if not cross_class
                    else "Review ambiguous overlapping labels.",
                )
            )

    def scan(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> ScanReport:
        """Scan a list of image paths and their corresponding label files."""
        report = ScanReport(total_images=len(image_paths))
        image_hashes: Dict[str, str] = {}
        total = len(image_paths)

        for i, image_path in enumerate(image_paths):
            if cancel_callback and cancel_callback():
                break

            if progress_callback:
                progress_callback(
                    i + 1, total, f"Scanning: {os.path.basename(image_path)}"
                )

            if not os.path.isfile(image_path):
                report.issues.append(
                    ScanIssue(
                        image_path=image_path,
                        label_file=None,
                        issue_type="missing_image",
                        severity="error",
                        details=f"Image file does not exist on disk: {image_path}",
                        suggestion="Verify dataset paths or restore missing image.",
                    )
                )
                continue

            self._check_image_duplicate(image_path, image_hashes, report)

            label_file = get_label_file_path(image_path, output_dir=output_dir)
            if not os.path.isfile(label_file):
                report.missing_labels += 1
                report.empty_images += 1
                report.issues.append(
                    ScanIssue(
                        image_path=image_path,
                        label_file=None,
                        issue_type="missing_label",
                        severity="info",
                        details="No annotation JSON file found for image.",
                        suggestion="Create annotations or mark as verified negative background.",
                    )
                )
                continue

            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                report.corrupt_shapes_count += 1
                report.issues.append(
                    ScanIssue(
                        image_path=image_path,
                        label_file=label_file,
                        issue_type="degenerate_geometry",
                        severity="error",
                        details=f"Failed to parse annotation JSON: {exc}",
                        suggestion="Fix or regenerate corrupted JSON file.",
                    )
                )
                continue

            if not isinstance(data, dict):
                report.corrupt_shapes_count += 1
                report.issues.append(
                    ScanIssue(
                        image_path=image_path,
                        label_file=label_file,
                        issue_type="degenerate_geometry",
                        severity="error",
                        details="Annotation JSON root is not an object.",
                        suggestion="Fix or regenerate corrupted JSON file.",
                    )
                )
                continue

            shapes = data.get("shapes", [])
            if not isinstance(shapes, list):
                report.corrupt_shapes_count += 1
                report.issues.append(
                    ScanIssue(
                        image_path=image_path,
                        label_file=label_file,
                        issue_type="degenerate_geometry",
                        severity="error",
                        details="Annotation 'shapes' field is not a list.",
                        suggestion="Fix or regenerate corrupted JSON file.",
                    )
                )
                continue
            img_width = data.get("imageWidth")
            img_height = data.get("imageHeight")

            if not shapes:
                if data.get("verified_empty") is True:
                    report.verified_empty_images += 1
                else:
                    report.empty_images += 1
                continue

            report.total_annotations += len(shapes)
            self._check_image_shapes(
                image_path, label_file, shapes, img_width, img_height, report
            )

        return report


def deduplicate_dataset_shapes(
    image_paths: List[str],
    output_dir: Optional[str] = None,
    iou_threshold: float = 0.85,
    containment_threshold: float = 0.90,
    same_label_only: bool = True,
) -> int:
    """Remove duplicate shapes across all dataset label files.

    Only same-label duplicates are removed by default to avoid
    cross-class data loss. Survivor is picked by score then area.
    Returns total count of duplicate shapes removed.
    """
    from anylabeling.views.labeling.utils.shape_geometry import (
        box_area,
        get_shape_score,
    )

    total_removed = 0

    for image_path in image_paths:
        label_file = get_label_file_path(image_path, output_dir=output_dir)
        if not os.path.isfile(label_file):
            continue

        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(
                f"Failed to read {label_file} during deduplication: {exc}"
            )
            continue

        if not isinstance(data, dict):
            continue
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list) or len(shapes) < 2:
            continue

        try:
            duplicates = detect_duplicate_shapes(
                shapes,
                iou_threshold=iou_threshold,
                containment_threshold=containment_threshold,
                same_label_only=same_label_only,
            )
        except Exception as exc:
            logger.warning(
                f"Duplicate detection failed for {label_file}: {exc}"
            )
            continue
        if not duplicates:
            continue

        # Pick survivor deterministically: higher score wins, then larger area.
        # Build removal set so we never delete both sides of a pair.
        def _priority(idx: int) -> Tuple[float, float]:
            s = shapes[idx] if 0 <= idx < len(shapes) else {}
            score = get_shape_score(s)
            try:
                b = shape_to_xyxy(s)
                area = box_area(b) if b else 0.0
            except Exception:
                area = 0.0
            return (
                float(score) if score is not None else 0.0,
                float(area),
            )

        indices_to_remove: set[int] = set()
        for d in duplicates:
            a_idx = d["index_a"]
            b_idx = d["index_b"]
            if a_idx in indices_to_remove or b_idx in indices_to_remove:
                continue
            # Keep higher priority, remove lower
            if _priority(a_idx) >= _priority(b_idx):
                indices_to_remove.add(b_idx)
            else:
                indices_to_remove.add(a_idx)
        if not indices_to_remove:
            continue

        try:
            new_shapes = [
                s
                for idx, s in enumerate(shapes)
                if idx not in indices_to_remove
            ]
            data["shapes"] = new_shapes
            _atomic_write_json(label_file, data)
        except Exception as exc:
            logger.warning(f"Failed to deduplicate {label_file}: {exc}")
            continue

        total_removed += len(indices_to_remove)

    return total_removed


def _resolve_label_file_path(
    image_path: str, output_dir: Optional[str] = None
) -> str:
    """Resolve label JSON path, matching LabelWidget.delete_image_file fallback."""
    label_file = get_label_file_path(image_path, output_dir=output_dir)
    if os.path.isfile(label_file):
        return label_file
    sibling = os.path.splitext(image_path)[0] + ".json"
    if sibling != label_file and os.path.isfile(sibling):
        return sibling
    return label_file


def delete_image_files(
    image_paths: List[str],
    output_dir: Optional[str] = None,
    dataset_root: Optional[str] = None,
) -> Tuple[int, List[str]]:
    """Remove image files from the dataset and delete paired JSON labels.

    Images are moved to ``{dataset_root}/_delete_/`` when dataset_root is
    known; otherwise ``../_delete_/`` relative to each image directory.
    Label JSON files are permanently removed.
    Returns (removed_count, failed_image_paths).
    """
    removed_count = 0
    failed: List[str] = []
    seen: Set[str] = set()
    unique_paths: List[str] = []
    for image_path in image_paths:
        if image_path and image_path not in seen:
            seen.add(image_path)
            unique_paths.append(image_path)

    for image_path in unique_paths:
        if not image_path or not os.path.isfile(image_path):
            continue
        try:
            trash_dir = get_image_delete_trash_dir(
                image_path, dataset_root=dataset_root
            )
            os.makedirs(trash_dir, exist_ok=True)
            dest_image = _unique_dest_path(
                trash_dir, os.path.basename(image_path)
            )
            shutil.move(image_path, dest_image)
            logger.info(f"Image file is moved to: {dest_image}")
        except Exception as exc:
            logger.warning(f"Failed to delete image {image_path}: {exc}")
            failed.append(image_path)
            continue

        label_file = _resolve_label_file_path(
            image_path, output_dir=output_dir
        )
        if os.path.isfile(label_file):
            try:
                os.remove(label_file)
                logger.info(f"Label file is removed: {label_file}")
            except Exception as exc:
                logger.warning(f"Failed to delete label {label_file}: {exc}")

        removed_count += 1

    return removed_count, failed


def delete_duplicate_images(
    image_paths: List[str],
    output_dir: Optional[str] = None,
    dataset_root: Optional[str] = None,
) -> int:
    """Remove MD5-identical duplicate images, keeping the first occurrence.

    Duplicate images are soft-deleted under the dataset _delete_ folder;
    paired JSON labels are permanently removed.
    Returns count of duplicate images removed.
    """
    image_hashes: Dict[str, str] = {}
    to_delete: List[str] = []

    for image_path in image_paths:
        if not os.path.isfile(image_path):
            continue
        file_hash = compute_file_md5(image_path)
        if not file_hash:
            continue
        if file_hash in image_hashes:
            to_delete.append(image_path)
        else:
            image_hashes[file_hash] = image_path

    deleted, _ = delete_image_files(
        to_delete, output_dir=output_dir, dataset_root=dataset_root
    )
    return deleted


def purge_micro_shapes(
    image_paths: List[str],
    output_dir: Optional[str] = None,
    min_size_px: float = 5.0,
    min_area_px: float = 16.0,
) -> int:
    """Purge micro-noise shapes whose width, height, or area falls below threshold.

    Returns total count of shapes purged.
    """
    total_purged = 0

    for image_path in image_paths:
        label_file = get_label_file_path(image_path, output_dir=output_dir)
        if not os.path.isfile(label_file):
            continue

        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to read {label_file} during purge: {exc}")
            continue

        shapes = data.get("shapes", [])
        if not isinstance(shapes, list) or not shapes:
            continue

        surviving_shapes = []
        purged_in_file = 0

        for shape in shapes:
            if not isinstance(shape, dict):
                # Corrupt entry: count as purged only if explicitly degenerate;
                # otherwise keep for manual review via scanner.
                surviving_shapes.append(shape)
                continue
            shape_type = str(shape.get("shape_type", "polygon")).lower()
            if shape_type == "point":
                surviving_shapes.append(shape)
                continue

            xyxy = shape_to_xyxy(shape)
            if xyxy is None:
                # Unparseable geometry — leave for manual review, do not
                # silently keep-or-delete as micro-noise.
                surviving_shapes.append(shape)
                continue

            w = max(0.0, xyxy[2] - xyxy[0])
            h = max(0.0, xyxy[3] - xyxy[1])
            area = w * h

            is_micro = False
            if shape_type in ("line", "linestrip"):
                is_micro = math.hypot(w, h) < min_size_px
            else:
                is_micro = (
                    w < min_size_px or h < min_size_px or area < min_area_px
                )

            if is_micro:
                purged_in_file += 1
            else:
                surviving_shapes.append(shape)

        if purged_in_file > 0:
            try:
                data["shapes"] = surviving_shapes
                _atomic_write_json(label_file, data)
            except Exception as exc:
                logger.warning(f"Failed to purge {label_file}: {exc}")
                continue

            total_purged += purged_in_file

    return total_purged


def _identify_out_of_bounds_indices(
    shapes: List[Dict[str, Any]], img_w: float, img_h: float
) -> set[int]:
    """Identify indices of shapes whose bounds exceed image dimensions."""
    out_of_bounds_indices = set()
    margin = 0.5
    for idx, shape in enumerate(shapes):
        xyxy = shape_to_xyxy(shape)
        if not xyxy:
            continue
        x1, y1, x2, y2 = xyxy
        if (
            x1 < -margin
            or y1 < -margin
            or x2 > (img_w + margin)
            or y2 > (img_h + margin)
        ):
            out_of_bounds_indices.add(idx)
    return out_of_bounds_indices


def clamp_out_of_bounds_shapes(
    image_paths: List[str],
    output_dir: Optional[str] = None,
) -> Tuple[int, int]:
    """Clamp out-of-bounds shape coordinates to image boundaries across dataset label files.

    Performs in-place coordinate clamping to [0, W] and [0, H], discarding degenerates.
    Returns (clamped_count, deleted_count) separately so callers do not
    misreport deletes as clamps.
    """
    total_clamped = 0
    total_deleted = 0

    for image_path in image_paths:
        label_file = get_label_file_path(image_path, output_dir=output_dir)
        if not os.path.isfile(label_file):
            continue

        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(
                f"Failed to read {label_file} during out-of-bounds clamping: {exc}"
            )
            continue

        if not isinstance(data, dict):
            continue
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list) or not shapes:
            continue

        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        try:
            img_w = float(img_w) if img_w is not None else None
            img_h = float(img_h) if img_h is not None else None
        except (TypeError, ValueError):
            img_w, img_h = None, None
        if (
            img_w is None or img_h is None or img_w <= 0 or img_h <= 0
        ) and os.path.isfile(image_path):
            try:
                with Image.open(image_path) as im:
                    img_w, img_h = im.size
                    data["imageWidth"] = img_w
                    data["imageHeight"] = img_h
            except Exception as exc:
                logger.warning(
                    f"Failed to read image dimensions for {image_path}: {exc}"
                )
                continue

        if not img_w or not img_h or img_w <= 0 or img_h <= 0:
            continue

        out_of_bounds_indices = _identify_out_of_bounds_indices(
            shapes, img_w, img_h
        )
        if not out_of_bounds_indices:
            continue

        new_shapes = []
        file_clamped = 0
        file_deleted = 0
        for idx, shape in enumerate(shapes):
            if idx in out_of_bounds_indices:
                try:
                    clamped_shape = clamp_shape_to_image_bounds(
                        shape, img_w, img_h
                    )
                except Exception as exc:
                    logger.warning(
                        f"Clamp failed for {label_file}#{idx}: {exc}"
                    )
                    new_shapes.append(shape)
                    continue
                if clamped_shape is None:
                    file_deleted += 1
                else:
                    file_clamped += 1
                    new_shapes.append(clamped_shape)
            else:
                new_shapes.append(shape)

        # No effective change (all clamps failed) — skip write to avoid
        # mtime/reformat churn.
        if file_clamped == 0 and file_deleted == 0:
            continue

        try:
            data["shapes"] = new_shapes
            _atomic_write_json(label_file, data)
        except Exception as exc:
            logger.warning(f"Failed to write clamped {label_file}: {exc}")
            continue

        total_clamped += file_clamped
        total_deleted += file_deleted

    return total_clamped, total_deleted


def _unique_dest_path(directory: str, basename: str) -> str:
    """Collision-safe destination path with numeric suffix."""
    candidate = os.path.join(directory, basename)
    if not os.path.exists(candidate):
        return candidate
    stem, ext = os.path.splitext(basename)
    counter = 1
    while True:
        suffixed = f"{stem}_{counter:03d}{ext}"
        candidate = os.path.join(directory, suffixed)
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _is_empty_label_for_move(
    label_file: str, include_verified_negatives: bool
) -> Optional[bool]:
    """Return True if empty, False if non-empty, None if skip/corrupt."""
    try:
        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(f"Skipping corrupt {label_file} in move: {exc}")
        return None
    if not isinstance(data, dict):
        return None
    if not include_verified_negatives and data.get("verified_empty") is True:
        return None
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        return None
    return len(shapes) == 0


def move_empty_images(
    image_paths: List[str],
    destination_dir: str,
    output_dir: Optional[str] = None,
    move_label_files: bool = True,
    include_verified_negatives: bool = False,
) -> int:
    """Move unannotated/empty images to a designated destination directory.

    Guard-only YOLO safety: sibling .txt means non-native dataset — skip.
    Returns count of moved images.
    """
    os.makedirs(destination_dir, exist_ok=True)
    dest_abs = os.path.abspath(destination_dir)
    moved_count = 0

    for image_path in image_paths:
        if not os.path.isfile(image_path):
            continue
        try:
            abs_img = os.path.abspath(image_path)
            if abs_img.startswith(dest_abs + os.sep):
                continue
            if os.path.dirname(abs_img) == dest_abs:
                continue
        except Exception:
            pass

        label_file = get_label_file_path(image_path, output_dir=output_dir)

        if not os.path.isfile(label_file):
            # Guard: YOLO-only folder (has .txt but no .json) is not empty
            yolo_txt = os.path.splitext(image_path)[0] + ".txt"
            if os.path.isfile(yolo_txt):
                continue
            is_empty = True
        else:
            empty_flag = _is_empty_label_for_move(
                label_file, include_verified_negatives
            )
            if empty_flag is None:
                continue
            is_empty = empty_flag

        if is_empty:
            try:
                dest_img = _unique_dest_path(
                    destination_dir, os.path.basename(image_path)
                )
                shutil.move(image_path, dest_img)

                if move_label_files and os.path.isfile(label_file):
                    dest_lbl = _unique_dest_path(
                        destination_dir, os.path.basename(label_file)
                    )
                    shutil.move(label_file, dest_lbl)

                moved_count += 1
            except Exception as exc:
                logger.warning(f"Failed to move empty {image_path}: {exc}")
                continue

    return moved_count


def _atomic_write_json(label_file: str, data: Dict[str, Any]) -> None:
    """Write a label JSON atomically via tmp-file replacement.

    Uses a pid-suffixed tmp name so concurrent writers do not collide.
    fsyncs the file before replace. Cleans up the tmp file on failure.
    Raises on error so callers decide whether to skip the file or abort.
    """
    tmp_path = f"{label_file}.{os.getpid()}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, label_file)
    finally:
        try:
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def apply_shape_modifications(
    label_file: str,
    deletions: Set[int] | List[int],
    reclasses: Dict[int, str],
    expected_labels: Optional[Dict[int, str]] = None,
) -> bool:
    """Apply a batch of deletions and reclassifications to a label file safely.

    Deletions are applied in descending index order to prevent index-drift.
    Writes are atomic using a temporary file replacement.
    Out-of-range indices in ``deletions``/``reclasses`` are ignored.
    If ``expected_labels`` maps an index to its expected ``label``, a
    mismatch aborts with ``False`` instead of deleting the wrong shape
    (stale gallery / concurrent edit guard).
    """
    if not os.path.isfile(label_file):
        return False
    try:
        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return False
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            return False

        # Stale-index guard: verify labels before touching anything.
        if expected_labels:
            for idx, exp_lbl in expected_labels.items():
                if not (0 <= idx < len(shapes)):
                    logger.error(
                        f"Stale shape index {idx} for {label_file} "
                        "(out of range, gallery is stale)"
                    )
                    return False
                existing = shapes[idx]
                if isinstance(existing, dict) and isinstance(exp_lbl, str):
                    if str(existing.get("label", "")) != exp_lbl:
                        logger.error(
                            f"Label mismatch at index {idx} for {label_file} "
                            f"(expected '{exp_lbl}', found "
                            f"'{existing.get('label')}') — gallery is stale"
                        )
                        return False

        # Apply reclassifications first (indices refer to original
        # positions, so reclass must precede deletion shifts).
        changed = False
        for idx, new_lbl in reclasses.items():
            if 0 <= idx < len(shapes) and isinstance(shapes[idx], dict):
                if str(shapes[idx].get("label", "")) != str(new_lbl):
                    shapes[idx]["label"] = new_lbl
                    changed = True

        # Apply deletions in descending index order to avoid shifting earlier indices
        valid_deletions = sorted(
            {d for d in set(deletions) if 0 <= d < len(shapes)},
            reverse=True,
        )
        for idx in valid_deletions:
            shapes.pop(idx)
        if valid_deletions:
            changed = True

        if not changed:
            return True

        data["shapes"] = shapes
        _atomic_write_json(label_file, data)
        return True
    except Exception as exc:
        logger.error(f"Failed applying modifications to {label_file}: {exc}")
        return False


def export_report_to_file(
    report: ScanReport, file_path: str, export_format: str = "json"
) -> None:
    """Export diagnostics scan report to JSON or CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    if export_format.lower() == "csv":
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Severity",
                    "Issue Type",
                    "Image Path",
                    "Label File",
                    "Shape Index",
                    "Class Label",
                    "Details",
                    "Suggestion",
                ]
            )
            for issue in report.issues:
                lbl = ""
                if isinstance(issue.shape_data, dict):
                    try:
                        lbl = issue.shape_data.get("label", "") or ""
                    except Exception:
                        lbl = ""
                shape_idx_str = (
                    str(issue.shape_index + 1)
                    if issue.shape_index is not None
                    else ""
                )
                writer.writerow(
                    [
                        issue.severity,
                        issue.issue_type,
                        issue.image_path,
                        issue.label_file or "",
                        shape_idx_str,
                        lbl,
                        issue.details,
                        issue.suggestion,
                    ]
                )
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report.to_json(indent=2))
