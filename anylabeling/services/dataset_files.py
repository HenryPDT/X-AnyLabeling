"""Qt-free helpers for dataset image/label file deletion and trash paths."""

from __future__ import annotations

import os
import shutil
from typing import List, Optional, Tuple

from anylabeling.services.dataset_meta import get_label_file_path
from anylabeling.views.labeling.logger import logger

IMAGE_DELETE_DIRNAME = "_delete_"


def get_image_delete_trash_dir(
    image_path: str, dataset_root: Optional[str] = None
) -> str:
    """Return the _delete_ folder for soft-deleted dataset images.

    When dataset_root is set and contains the image, trash is anchored to
    ``{dataset_root}/_delete_/``. Otherwise falls back to ``../_delete_/``
    relative to the image directory (legacy flat-folder behavior).
    """
    abs_image = os.path.abspath(image_path)
    if dataset_root:
        root = os.path.abspath(dataset_root)
        try:
            if os.path.commonpath((root, abs_image)) == root:
                return os.path.join(root, IMAGE_DELETE_DIRNAME)
        except ValueError:
            pass
    image_dir = os.path.dirname(abs_image)
    return os.path.abspath(os.path.join(image_dir, "..", IMAGE_DELETE_DIRNAME))


def resolve_label_file_path(
    image_path: str, output_dir: Optional[str] = None
) -> str:
    """Resolve label JSON path, including sibling fallback when output_dir differs."""
    label_file = get_label_file_path(image_path, output_dir=output_dir)
    if os.path.isfile(label_file):
        return label_file
    sibling = os.path.splitext(image_path)[0] + ".json"
    if sibling != label_file and os.path.isfile(sibling):
        return sibling
    return label_file


def unique_dest_path(directory: str, basename: str) -> str:
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


def delete_label_file(label_file: str) -> bool:
    """Permanently delete a label JSON file. Returns True on success."""
    if not label_file or not os.path.isfile(label_file):
        return False
    try:
        os.remove(label_file)
        logger.info(f"Label file is removed: {label_file}")
        return True
    except Exception as exc:
        logger.warning(f"Failed to delete label {label_file}: {exc}")
        return False


def soft_delete_dataset_images(
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
    seen: set[str] = set()
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
            dest_image = unique_dest_path(
                trash_dir, os.path.basename(image_path)
            )
            shutil.move(image_path, dest_image)
            logger.info(f"Image file is moved to: {dest_image}")
        except Exception as exc:
            logger.warning(f"Failed to delete image {image_path}: {exc}")
            failed.append(image_path)
            continue

        label_file = resolve_label_file_path(image_path, output_dir=output_dir)
        delete_label_file(label_file)

        removed_count += 1

    return removed_count, failed
