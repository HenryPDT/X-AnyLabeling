"""Train/val split builder (Qt-free, additive).

Deterministic seeded split with optional stratification by scene
(parent folder) and dominant class. Pure functions for testing;
dialog handles file I/O.
"""

from __future__ import annotations

import json
import operator
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from anylabeling.services.dataset_meta import get_label_file_path

#: Label used when an image has no usable annotation.
UNLABELED = "__unlabeled__"


def scene_key_for_image(image_path: str) -> str:
    """Return the normalized parent-folder key for scene-aware splitting."""
    parent = os.path.dirname(os.path.abspath(image_path))
    return os.path.normcase(os.path.normpath(parent))


@dataclass
class SplitResult:
    train: List[str]
    val: List[str]
    train_ratio: float
    seed: int


def _coerce_ratio(train_ratio: Any) -> float:
    """Validate ratio and clamp to [0.1, 0.9]."""
    try:
        ratio = float(train_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid train_ratio {train_ratio!r}: {exc}"
        ) from exc
    return max(0.1, min(0.9, ratio))


def _coerce_seed(seed: Any) -> int:
    """Validate seed as an integer."""
    try:
        return operator.index(seed)
    except TypeError as exc:
        raise ValueError(f"Invalid seed {seed!r}: must be integer") from exc


def stratified_split(
    image_paths: List[str],
    train_ratio: float = 0.8,
    seed: int = 42,
    labels_by_image: Optional[Dict[str, str]] = None,
    stratify_by_folder: bool = True,
) -> SplitResult:
    """Split images deterministically.

    Args:
        image_paths: Full list of image files.
        train_ratio: Fraction for train (0.1-0.9, else clamped).
        seed: Random seed for reproducibility.
        labels_by_image: Optional image -> dominant label for
            stratification. When None, plain shuffled split.
        stratify_by_folder: When True (default), stratify jointly by
            parent folder and label so each scene keeps the ratio
            instead of one scene collapsing into a single split.
            Flat datasets (one folder) are unaffected.

    A lone image (or lone group member) always goes to train: a
    single sample cannot validate.
    """
    ratio = _coerce_ratio(train_ratio)
    seed_int = _coerce_seed(seed)
    rng = random.Random(seed_int)
    images = list(image_paths)
    if not images:
        return SplitResult([], [], ratio, seed_int)
    if not labels_by_image:
        shuffled = list(images)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            return SplitResult(shuffled, [], ratio, seed_int)
        n_train = int(len(shuffled) * ratio)
        # Guarantee at least 1 val when possible.
        n_train = max(0, min(len(shuffled) - 1, n_train))
        return SplitResult(
            shuffled[:n_train], shuffled[n_train:], ratio, seed_int
        )
    # Stratify jointly by scene and dominant label.
    groups: Dict[Any, List[str]] = {}
    for img in images:
        label = str(labels_by_image.get(img, UNLABELED))
        key = (
            (scene_key_for_image(img), label) if stratify_by_folder else label
        )
        groups.setdefault(key, []).append(img)
    train: List[str] = []
    val: List[str] = []
    for _label in sorted(groups.keys()):
        group = list(groups[_label])
        rng.shuffle(group)
        if len(group) == 1:
            n_train = 1
        else:
            n_train = int(len(group) * ratio)
            n_train = max(1, min(len(group) - 1, n_train))
        train.extend(group[:n_train])
        val.extend(group[n_train:])
    rng.shuffle(train)
    rng.shuffle(val)
    return SplitResult(train, val, ratio, seed_int)


def build_dataset_yaml(
    train_list_path: str,
    val_list_path: str,
    names: List[str],
    project_root: Optional[str] = None,
) -> str:
    """Build minimal YOLO dataset.yaml content (safe-quoted)."""
    if project_root:
        try:
            train_ref = os.path.relpath(train_list_path, project_root)
            val_ref = os.path.relpath(val_list_path, project_root)
        except (TypeError, ValueError):
            train_ref = train_list_path
            val_ref = val_list_path
        payload = {
            "path": project_root,
            "train": train_ref,
            "val": val_ref,
            "nc": len(names),
            "names": list(names),
        }
    else:
        payload = {
            "train": train_list_path,
            "val": val_list_path,
            "nc": len(names),
            "names": list(names),
        }
    text = yaml.safe_dump(
        payload, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    if not names:
        text += "# no labeled classes found\n"
    return text


def write_list_file(file_path: str, image_paths: List[str]) -> None:
    """Write one absolute image path per line (creates dirs)."""
    directory = os.path.dirname(os.path.abspath(file_path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for img in image_paths:
            f.write(os.path.abspath(img) + "\n")


def dominant_label_for_image(
    shapes: List[Dict[str, Any]],
) -> str:
    """Return most frequent label in shape list or UNLABELED."""
    if not shapes:
        return UNLABELED
    counts: Dict[str, int] = {}
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "")).strip()
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return UNLABELED
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def collect_labels_and_names(
    image_paths: List[str], output_dir: Optional[str] = None
) -> Tuple[Dict[str, str], List[str]]:
    """Single-pass dominant-label mapping + sorted class names."""
    mapping: Dict[str, str] = {}
    names: set = set()
    for img in image_paths:
        label_file = get_label_file_path(img, output_dir=output_dir)
        if not os.path.isfile(label_file):
            mapping[img] = UNLABELED
            continue
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            mapping[img] = UNLABELED
            continue
        shapes = data.get("shapes", []) if isinstance(data, dict) else []
        if not isinstance(shapes, list):
            mapping[img] = UNLABELED
            continue
        mapping[img] = dominant_label_for_image(shapes)
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            label = str(shape.get("label", "")).strip()
            if label:
                names.add(label)
    return mapping, sorted(names)


def collect_names(
    image_paths: List[str], output_dir: Optional[str] = None
) -> List[str]:
    """Collect sorted unique class names from label JSONs."""
    _, names = collect_labels_and_names(image_paths, output_dir=output_dir)
    return names
