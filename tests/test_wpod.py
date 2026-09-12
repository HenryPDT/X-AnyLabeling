"""WPOD/IWPOD quad codec tests (blocked layout ``4,x1..x4,y1..y4,,``)."""

import json
import os

import pytest
from PIL import Image

from anylabeling.views.labeling.label_converter import LabelConverter


@pytest.fixture()
def tiny_image(tmp_path):
    img = tmp_path / "car.jpg"
    Image.new("RGB", (200, 100)).save(img)
    return str(img)


def _read_shapes(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)["shapes"]


def test_wpod_import_single_quad(tmp_path, tiny_image):
    txt = tmp_path / "car.txt"
    txt.write_text(
        "4,0.1,0.9,0.9,0.1,0.2,0.2,0.8,0.8,,\n", encoding="utf-8"
    )
    out = str(tmp_path / "car.json")
    LabelConverter().wpod_to_custom(str(txt), out, tiny_image)
    (shape,) = _read_shapes(out)
    assert shape["label"] == "plate"
    assert shape["shape_type"] == "quadrilateral"
    assert shape["points"] == [
        [20.0, 20.0],
        [180.0, 20.0],
        [180.0, 80.0],
        [20.0, 80.0],
    ]


def test_wpod_import_clamps_out_of_range(tmp_path, tiny_image):
    # Mirrors the 2 known-bad TK57DN files (y=1.01 / y=-0.003).
    txt = tmp_path / "bad.txt"
    txt.write_text(
        "4,0.4,0.6,0.62,0.41,0.88,0.87,1.01,0.99,,\n", encoding="utf-8"
    )
    out = str(tmp_path / "bad.json")
    LabelConverter().wpod_to_custom(str(txt), out, tiny_image)
    (shape,) = _read_shapes(out)
    ys = [p[1] for p in shape["points"]]
    assert max(ys) <= 100
    assert min(ys) >= 0


def test_wpod_import_rejects_malformed(tmp_path, tiny_image):
    txt = tmp_path / "bad.txt"
    txt.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        LabelConverter().wpod_to_custom(
            str(txt), str(tmp_path / "bad.json"), tiny_image
        )


def test_wpod_round_trip(tmp_path, tiny_image):
    txt = tmp_path / "car.txt"
    txt.write_text(
        "4,0.3081761006289308,0.6918238993710691,0.6918238993710691,"
        "0.3081761006289308,0.3836477987421384,0.3836477987421384,"
        "0.5283018867924528,0.5283018867924528,,\n",
        encoding="utf-8",
    )
    js = str(tmp_path / "car.json")
    LabelConverter().wpod_to_custom(str(txt), js, tiny_image)
    back = str(tmp_path / "car_out.txt")
    LabelConverter().custom_to_wpod(js, back)
    with open(back, encoding="utf-8") as f:
        (line,) = [ln for ln in f.read().splitlines() if ln.strip()]
    vals = [float(t) for t in line.split(",") if t.strip() != ""][1:]
    assert len(vals) == 8
    assert all(0.0 <= v <= 1.0 for v in vals)
    # Blocked order preserved: xs then ys.
    assert vals[0] == pytest.approx(0.3081761, abs=1e-4)
    assert vals[4] == pytest.approx(0.3836478, abs=1e-4)


def test_wpod_export_skips_non_quadrilaterals(tmp_path):
    data = {
        "version": "x",
        "flags": {},
        "shapes": [
            {
                "label": "plate",
                "points": [[1, 1], [9, 1], [9, 9], [1, 9]],
                "shape_type": "rectangle",
                "group_id": None,
                "difficult": False,
                "flags": {},
            }
        ],
        "imagePath": "a.jpg",
        "imageData": None,
        "imageHeight": 10,
        "imageWidth": 10,
    }
    js = tmp_path / "a.json"
    js.write_text(json.dumps(data), encoding="utf-8")
    out = str(tmp_path / "a.txt")
    is_empty = LabelConverter().custom_to_wpod(str(js), out)
    assert is_empty is True
    assert open(out, encoding="utf-8").read() == ""


def test_wpod_export_missing_input(tmp_path):
    out = str(tmp_path / "missing.txt")
    assert (
        LabelConverter().custom_to_wpod(
            str(tmp_path / "nope.json"), out
        )
        is True
    )
    assert os.path.exists(out)
    out2 = str(tmp_path / "missing2.txt")
    assert (
        LabelConverter().custom_to_wpod(
            str(tmp_path / "nope.json"), out2, skip_empty_files=True
        )
        is True
    )
    assert not os.path.exists(out2)


def test_wpod_export_preserves_scene_structure(tmp_path):
    """GUI export mapping keeps camera subdirs as relative paths."""
    from anylabeling.views.labeling.utils.export import (
        _get_wpod_export_files,
    )

    for cam in ("Access_Control", "Merthyl_9"):
        (tmp_path / "imgs" / cam).mkdir(parents=True)
        Image.new("RGB", (200, 100)).save(
            tmp_path / "imgs" / cam / f"{cam}_img0.jpg"
        )
    image_list = [
        str(tmp_path / "imgs" / "Access_Control" / "Access_Control_img0.jpg"),
        str(tmp_path / "imgs" / "Merthyl_9" / "Merthyl_9_img0.jpg"),
    ]
    # One json with a quad, one rect-only json (empty export).
    quad = {
        "version": "x",
        "flags": {},
        "shapes": [
            {
                "label": "plate",
                "points": [[20, 20], [180, 20], [180, 80], [20, 80]],
                "shape_type": "quadrilateral",
                "group_id": None,
                "difficult": False,
                "flags": {},
            }
        ],
        "imagePath": "Access_Control_img0.jpg",
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 200,
    }
    (tmp_path / "imgs" / "Access_Control" / "Access_Control_img0.json").write_text(
        json.dumps(quad), encoding="utf-8"
    )
    rect = dict(quad, shapes=[
        {
            "label": "plate",
            "points": [[1, 1], [9, 1], [9, 9], [1, 9]],
            "shape_type": "rectangle",
            "group_id": None,
            "difficult": False,
            "flags": {},
        }
    ])
    (tmp_path / "imgs" / "Merthyl_9" / "Merthyl_9_img0.json").write_text(
        json.dumps(rect), encoding="utf-8"
    )

    save_path = str(tmp_path / "wpod_labels")
    files = _get_wpod_export_files(
        image_list, str(tmp_path / "imgs"), save_path
    )
    assert files[0][1] == os.path.join(
        save_path, "Access_Control", "Access_Control_img0.txt"
    )
    assert files[1][1] == os.path.join(
        save_path, "Merthyl_9", "Merthyl_9_img0.txt"
    )

    converter = LabelConverter()
    for image_file, dst_file, _ in files:
        src = os.path.join(
            os.path.dirname(image_file),
            os.path.splitext(os.path.basename(image_file))[0] + ".json",
        )
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        empty = converter.custom_to_wpod(src, dst_file, skip_empty_files=True)
        if empty and os.path.exists(dst_file):
            os.remove(dst_file)
    assert open(
        os.path.join(save_path, "Access_Control", "Access_Control_img0.txt"),
        encoding="utf-8",
    ).read().startswith("4,")
    assert not os.path.exists(
        os.path.join(save_path, "Merthyl_9", "Merthyl_9_img0.txt")
    )


def test_wpod_split_layout_side_by_side(tmp_path):
    """Split export: train/<scene>/ + val/<scene>/ with jpg+txt siblings."""
    from anylabeling.views.labeling.utils.export import (
        _get_wpod_export_files,
    )

    source_root = str(tmp_path / "imgs")
    image_list = [
        os.path.join(source_root, "scene_001", "frame_0001.jpg"),
        os.path.join(source_root, "scene_003", "frame_0001.jpg"),
    ]
    save_path = str(tmp_path / "dataset")
    train_files = _get_wpod_export_files(
        [image_list[0]], source_root, save_path, layout="train"
    )
    val_files = _get_wpod_export_files(
        [image_list[1]], source_root, save_path, layout="val"
    )
    (train_img, train_txt, train_img_dst), = train_files
    (val_img, val_txt, val_img_dst), = val_files
    # Label and image destinations are side by side, no labels//images/ split.
    assert train_txt == os.path.join(
        save_path, "train", "scene_001", "frame_0001.txt"
    )
    assert train_img_dst == os.path.join(
        save_path, "train", "scene_001", "frame_0001.jpg"
    )
    assert val_txt == os.path.join(
        save_path, "val", "scene_003", "frame_0001.txt"
    )
    assert val_img_dst == os.path.join(
        save_path, "val", "scene_003", "frame_0001.jpg"
    )
    assert "labels" not in train_txt.split(os.sep)
    assert "images" not in train_img_dst.split(os.sep)


def test_wpod_tasks_registered():
    from anylabeling.views.common.converter import SUPPORTED_TASKS

    assert "wpod2xlabel" in SUPPORTED_TASKS
    assert "xlabel2wpod" in SUPPORTED_TASKS


def test_wpod_tasks_on_real_lp_sample():
    sample = (
        "/home/rackpc/Documents/LP_dataset/Access_Control/"
        "Access_Control_2023-03-25T15-15-00_000208_car_0.txt"
    )
    image = (
        "/home/rackpc/Documents/LP_dataset/Access_Control/"
        "Access_Control_2023-03-25T15-15-00_000208_car_0.jpg"
    )
    if not (os.path.exists(sample) and os.path.exists(image)):
        pytest.skip("LP_dataset sample not available")
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        js = os.path.join(d, "x.json")
        back = os.path.join(d, "x.txt")
        LabelConverter().wpod_to_custom(sample, js, image)
        shapes = _read_shapes(js)
        assert len(shapes) == 1
        assert shapes[0]["shape_type"] == "quadrilateral"
        LabelConverter().custom_to_wpod(js, back)
        assert open(back, encoding="utf-8").read().startswith("4,")
