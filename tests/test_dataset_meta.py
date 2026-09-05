import json
import os

from anylabeling.services.dataset_meta import (
    get_dataset_meta,
    get_dataset_meta_list,
    get_label_file_path,
    is_label_file_checked_fast,
    is_label_file_verified_empty_fast,
)


def _write_label(path, checked=False, verified_empty=False, shapes=None):
    data = {
        "version": "1.0",
        "flags": {},
        "checked": checked,
        "shapes": shapes or [],
        "imagePath": os.path.basename(path).replace(".json", ".jpg"),
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 100,
    }
    if verified_empty:
        data["verified_empty"] = True
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_get_label_file_path(tmp_path):
    img = str(tmp_path / "a.jpg")
    assert get_label_file_path(img).endswith("a.json")
    out = get_label_file_path(img, output_dir=str(tmp_path / "out"))
    assert out == os.path.join(str(tmp_path / "out"), "a.json")


def test_checked_and_verified_fast(tmp_path):
    label = str(tmp_path / "img.json")
    _write_label(label, checked=True)
    assert is_label_file_checked_fast(label) is True
    assert is_label_file_verified_empty_fast(label) is False

    _write_label(label, checked=False, verified_empty=True)
    assert is_label_file_checked_fast(label) is False
    assert is_label_file_verified_empty_fast(label) is True


def test_missing_files_return_defaults(tmp_path):
    img = str(tmp_path / "missing.jpg")
    assert is_label_file_checked_fast(str(tmp_path / "x.json")) is False
    meta = get_dataset_meta(img)
    assert meta.exists is False
    assert meta.has_label_file is False
    assert meta.shape_count == 0


def test_get_dataset_meta_counts(tmp_path):
    img = str(tmp_path / "img.jpg")
    with open(img, "wb") as f:
        f.write(b"fake")
    label = str(tmp_path / "img.json")
    shapes = [
        {
            "label": "cat",
            "shape_type": "rectangle",
            "points": [[0, 0], [10, 10]],
        },
        {
            "label": "cat",
            "shape_type": "rectangle",
            "points": [[20, 20], [30, 30]],
        },
        {
            "label": "dog",
            "shape_type": "rectangle",
            "points": [[0, 0], [5, 5]],
        },
    ]
    _write_label(label, checked=True, shapes=shapes)
    meta = get_dataset_meta(img)
    assert meta.exists is True
    assert meta.has_label_file is True
    assert meta.checked is True
    assert meta.shape_count == 3
    assert meta.class_counts == {"cat": 2, "dog": 1}
    assert meta.corrupt is False


def test_get_dataset_meta_verified_empty(tmp_path):
    img = str(tmp_path / "bg.jpg")
    with open(img, "wb") as f:
        f.write(b"fake")
    label = str(tmp_path / "bg.json")
    _write_label(label, verified_empty=True, shapes=[])
    meta = get_dataset_meta(img)
    assert meta.shape_count == 0
    assert meta.verified_empty is True


def test_get_dataset_meta_corrupt(tmp_path):
    img = str(tmp_path / "c.jpg")
    with open(img, "wb") as f:
        f.write(b"fake")
    label = str(tmp_path / "c.json")
    with open(label, "w", encoding="utf-8") as f:
        f.write("{not-json")
    meta = get_dataset_meta(img)
    assert meta.has_label_file is True
    assert meta.corrupt is True


def test_get_dataset_meta_list_progress(tmp_path):
    imgs = []
    for name in ["a.jpg", "b.jpg"]:
        p = str(tmp_path / name)
        with open(p, "wb") as f:
            f.write(b"x")
        imgs.append(p)
    seen = []
    metas = get_dataset_meta_list(
        imgs, progress_callback=lambda a, b, c: seen.append((a, b))
    )
    assert len(metas) == 2
    assert len(seen) == 2

    cancelled = get_dataset_meta_list(imgs, cancel_callback=lambda: True)
    assert cancelled == []


def test_verified_empty_tail_scan_large_file(tmp_path):
    # Regression: text-mode seek by byte size could miss/split tokens.
    # verified_empty at tail with multi-hundred-KB imageData must be found.
    img = str(tmp_path / "big.jpg")
    with open(img, "wb") as f:
        f.write(b"x")
    label = str(tmp_path / "big.json")
    blob = "A" * (300 * 1024)
    data = {
        "version": "1.0",
        "flags": {},
        "checked": False,
        "shapes": [],
        "imagePath": "big.jpg",
        "imageData": blob,
        "imageHeight": 10,
        "imageWidth": 10,
        "verified_empty": True,
    }
    with open(label, "w", encoding="utf-8") as f:
        json.dump(data, f)
    assert is_label_file_verified_empty_fast(label) is True
    meta = get_dataset_meta(img)
    assert meta.verified_empty is True
    assert meta.shape_count == 0


def test_small_file_single_parse_checked_override(tmp_path):
    # Small files parse once; parsed JSON takes precedence over regex.
    img = str(tmp_path / "s.jpg")
    with open(img, "wb") as f:
        f.write(b"x")
    label = str(tmp_path / "s.json")
    _write_label(label, checked=True, shapes=[])
    meta = get_dataset_meta(img)
    assert meta.checked is True
    # Non-empty file can never be a verified background.
    _write_label(
        label,
        checked=True,
        shapes=[
            {
                "label": "cat",
                "shape_type": "rectangle",
                "points": [[0, 0], [5, 5]],
            }
        ],
    )
    with open(label, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["verified_empty"] = True
    with open(label, "w", encoding="utf-8") as f:
        json.dump(data, f)
    meta = get_dataset_meta(img)
    assert meta.verified_empty is False
    assert meta.shape_count == 1
