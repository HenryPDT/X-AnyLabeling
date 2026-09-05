import pytest
import yaml

from anylabeling.views.labeling.utils.split import (
    UNLABELED,
    build_dataset_yaml,
    collect_labels_and_names,
    collect_names,
    dominant_label_for_image,
    scene_key_for_image,
    stratified_split,
)


def test_stratified_split_deterministic():
    imgs = [f"img{i}.jpg" for i in range(10)]
    first = stratified_split(imgs, train_ratio=0.8, seed=42)
    second = stratified_split(imgs, train_ratio=0.8, seed=42)
    assert first.train == second.train
    assert first.val == second.val
    assert len(first.train) + len(first.val) == 10
    assert len(first.train) == 8


def test_stratified_split_empty():
    result = stratified_split([], train_ratio=0.8, seed=1)
    assert result.train == [] and result.val == []


def test_stratified_split_single():
    result = stratified_split(["only.jpg"], train_ratio=0.8, seed=1)
    assert result.train == ["only.jpg"]
    assert result.val == []


def test_stratified_split_by_label():
    imgs = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    labels = {"a.jpg": "cat", "b.jpg": "cat", "c.jpg": "dog", "d.jpg": "dog"}
    result = stratified_split(
        imgs, train_ratio=0.5, seed=7, labels_by_image=labels
    )
    assert len(result.train) == 2
    assert len(result.val) == 2
    # Each split should contain both classes when stratified.
    train_labels = {labels[p] for p in result.train}
    val_labels = {labels[p] for p in result.val}
    assert train_labels == {"cat", "dog"}
    assert val_labels == {"cat", "dog"}
    assert result.train_ratio == 0.5
    # No overlap and full coverage.
    assert set(result.train).isdisjoint(result.val)
    assert sorted(result.train + result.val) == sorted(imgs)


def test_stratified_singleton_goes_to_train():
    result = stratified_split(
        ["only.jpg"],
        train_ratio=0.2,
        seed=1,
        labels_by_image={"only.jpg": "cat"},
    )
    assert result.train == ["only.jpg"]
    assert result.val == []


def test_stratified_split_invalid_inputs():
    with pytest.raises(ValueError):
        stratified_split(["a.jpg"], train_ratio="bad", seed=1)
    with pytest.raises(ValueError):
        stratified_split(["a.jpg"], train_ratio=0.8, seed="bad")


def test_build_dataset_yaml():
    text = build_dataset_yaml("train.txt", "val.txt", ["cat", "dog"])
    assert "train: train.txt" in text
    assert "nc: 2" in text
    assert "cat" in text


def test_build_dataset_yaml_quotes_special_chars(tmp_path):
    text = build_dataset_yaml(
        str(tmp_path / "train.txt"),
        str(tmp_path / "val.txt"),
        ["cat: tricky", "plain"],
        project_root=str(tmp_path),
    )
    parsed = yaml.safe_load(text)
    assert parsed["nc"] == 2
    assert parsed["names"] == ["cat: tricky", "plain"]
    assert parsed["path"] == str(tmp_path)


def test_build_dataset_yaml_empty_names():
    text = build_dataset_yaml("train.txt", "val.txt", [])
    assert yaml.safe_load(text)["nc"] == 0


def test_collect_labels_and_names_single_pass(tmp_path):
    import json

    img = str(tmp_path / "a.jpg")
    with open(img, "wb") as f:
        f.write(b"x")
    label = str(tmp_path / "a.json")
    with open(label, "w", encoding="utf-8") as f:
        json.dump(
            {
                "shapes": [
                    {"label": "cat"},
                    {"label": "cat"},
                    {"label": "dog"},
                ]
            },
            f,
        )
    mapping, names = collect_labels_and_names(
        [img, str(tmp_path / "missing.jpg")], output_dir=str(tmp_path)
    )
    assert mapping[img] == "cat"
    assert mapping[str(tmp_path / "missing.jpg")] == UNLABELED
    assert names == ["cat", "dog"]
    assert collect_names([img], output_dir=str(tmp_path)) == ["cat", "dog"]


def test_dominant_label():
    assert dominant_label_for_image([]) == "__unlabeled__"
    shapes = [
        {"label": "cat"},
        {"label": "dog"},
        {"label": "cat"},
    ]
    assert dominant_label_for_image(shapes) == "cat"


def test_scene_key_for_image(tmp_path):
    a = str(tmp_path / "scene1" / "a.jpg")
    b = str(tmp_path / "scene1" / "b.jpg")
    c = str(tmp_path / "scene2" / "a.jpg")
    assert scene_key_for_image(a) == scene_key_for_image(b)
    assert scene_key_for_image(a) != scene_key_for_image(c)


def test_stratified_split_per_scene_ratio(tmp_path):
    # 5 scenes x 10 images (5 cat + 5 dog each), 80/20 must hold per scene.
    imgs = []
    labels = {}
    for scene in range(5):
        for idx in range(10):
            img = str(tmp_path / f"scene{scene}" / f"img{idx}.jpg")
            imgs.append(img)
            labels[img] = "cat" if idx < 5 else "dog"
    result = stratified_split(
        imgs, train_ratio=0.8, seed=42, labels_by_image=labels
    )
    assert len(result.train) + len(result.val) == 50
    for scene in range(5):
        prefix = str(tmp_path / f"scene{scene}")
        scene_train = [p for p in result.train if p.startswith(prefix)]
        scene_val = [p for p in result.val if p.startswith(prefix)]
        assert len(scene_train) == 8
        assert len(scene_val) == 2
        assert {labels[p] for p in scene_train} == {"cat", "dog"}
        assert {labels[p] for p in scene_val} == {"cat", "dog"}


def test_stratified_split_flat_ignores_folder(tmp_path):
    # Single folder (or bare filenames) behaves like label-only split.
    imgs = [f"img{i}.jpg" for i in range(10)]
    labels = {img: ("cat" if i < 5 else "dog") for i, img in enumerate(imgs)}
    foldered = stratified_split(
        imgs, train_ratio=0.8, seed=42, labels_by_image=labels
    )
    ungrouped = stratified_split(
        imgs,
        train_ratio=0.8,
        seed=42,
        labels_by_image=labels,
        stratify_by_folder=False,
    )
    assert sorted(foldered.train) == sorted(ungrouped.train)
    assert sorted(foldered.val) == sorted(ungrouped.val)
