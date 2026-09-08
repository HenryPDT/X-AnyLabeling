import json
import os
import tempfile

from anylabeling.services.dataset_files import (
    delete_label_file,
    get_image_delete_trash_dir,
    resolve_label_file_path,
    soft_delete_dataset_images,
    unique_dest_path,
)
from anylabeling.services.dataset_meta import get_label_file_path


def test_get_label_file_path():
    img_path = "/dataset/images/cat.png"
    assert get_label_file_path(img_path) == "/dataset/images/cat.json"
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = get_label_file_path(img_path, output_dir=tmp_dir)
        assert out == os.path.join(tmp_dir, "cat.json")


def test_get_image_delete_trash_dir_uses_dataset_root():
    with tempfile.TemporaryDirectory() as tmp_dir:
        nested = os.path.join(tmp_dir, "train", "scene_a")
        os.makedirs(nested)
        img_path = os.path.join(nested, "sample.jpg")
        trash_dir = get_image_delete_trash_dir(img_path, dataset_root=tmp_dir)
        assert trash_dir == os.path.join(tmp_dir, "_delete_")


def test_resolve_label_file_path_prefers_sibling():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "sample.jpg")
        sibling_lbl = os.path.join(tmp_dir, "sample.json")
        with open(sibling_lbl, "w", encoding="utf-8") as handle:
            json.dump({"shapes": []}, handle)
        output_lbl = os.path.join(tmp_dir, "labels", "sample.json")
        resolved = resolve_label_file_path(img_path, output_dir=output_lbl)
        assert resolved == sibling_lbl


def test_unique_dest_path_adds_suffix_on_collision():
    with tempfile.TemporaryDirectory() as tmp_dir:
        first = os.path.join(tmp_dir, "sample.jpg")
        with open(first, "wb") as handle:
            handle.write(b"a")
        second = unique_dest_path(tmp_dir, "sample.jpg")
        assert second == os.path.join(tmp_dir, "sample_001.jpg")


def test_delete_label_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        lbl_path = os.path.join(tmp_dir, "sample.json")
        with open(lbl_path, "w", encoding="utf-8") as handle:
            json.dump({"shapes": []}, handle)
        assert delete_label_file(lbl_path) is True
        assert not os.path.isfile(lbl_path)
        assert delete_label_file(lbl_path) is False


def test_soft_delete_dataset_images_moves_image_and_removes_label():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "sample.jpg")
        lbl_path = os.path.join(tmp_dir, "sample.json")
        with open(img_path, "wb") as handle:
            handle.write(b"img")
        with open(lbl_path, "w", encoding="utf-8") as handle:
            json.dump({"imagePath": "sample.jpg", "shapes": []}, handle)

        removed, failed = soft_delete_dataset_images(
            [img_path], dataset_root=tmp_dir
        )
        assert removed == 1
        assert failed == []
        assert not os.path.isfile(img_path)
        assert not os.path.isfile(lbl_path)
        assert os.path.isfile(os.path.join(tmp_dir, "_delete_", "sample.jpg"))


def test_soft_delete_dataset_images_uses_dataset_root_for_nested_images():
    with tempfile.TemporaryDirectory() as tmp_dir:
        nested_dir = os.path.join(tmp_dir, "train", "scene_a")
        os.makedirs(nested_dir)
        img_path = os.path.join(nested_dir, "sample.jpg")
        lbl_path = os.path.join(nested_dir, "sample.json")
        with open(img_path, "wb") as handle:
            handle.write(b"img")
        with open(lbl_path, "w", encoding="utf-8") as handle:
            json.dump({"imagePath": "sample.jpg", "shapes": []}, handle)

        removed, failed = soft_delete_dataset_images(
            [img_path], dataset_root=tmp_dir
        )
        assert removed == 1
        assert failed == []
        assert not os.path.isfile(img_path)
        assert not os.path.isfile(lbl_path)
        assert os.path.isfile(os.path.join(tmp_dir, "_delete_", "sample.jpg"))
        assert not os.path.isdir(os.path.join(tmp_dir, "train", "_delete_"))


def test_soft_delete_dataset_images_deduplicates_by_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "sample.jpg")
        with open(img_path, "wb") as handle:
            handle.write(b"img")

        removed, failed = soft_delete_dataset_images(
            [img_path, img_path], dataset_root=tmp_dir
        )
        assert removed == 1
        assert failed == []
        assert not os.path.isfile(img_path)
