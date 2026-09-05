from anylabeling.services.dataset_meta import DatasetMeta
from anylabeling.services.dataset_stats_core import (
    collect_shape_sizes,
    compute_class_size_stats,
    compute_dataset_stats_from_metas,
)


def test_compute_class_size_stats_basic():
    widths = [10.0, 20.0, 30.0]
    heights = [10.0, 10.0, 10.0]
    areas = [100.0, 200.0, 300.0]
    stats = compute_class_size_stats("cat", widths, heights, areas, 2)
    assert stats.label == "cat"
    assert stats.count == 3
    assert stats.images == 2
    assert stats.min_width == 10.0
    assert stats.max_width == 30.0
    assert abs(stats.mean_width - 20.0) < 1e-6
    assert stats.stddev_width > 0
    assert stats.stddev_height == 0.0
    assert abs(stats.marks_per_image - 1.5) < 1e-6


def test_compute_class_size_stats_empty():
    stats = compute_class_size_stats("dog", [], [], [], 0)
    assert stats.count == 0
    assert stats.mean_width == 0.0
    assert stats.stddev_width == 0.0
    assert stats.marks_per_image == 0.0


def test_collect_shape_sizes_skips_invalid():
    shapes = [
        {"label": "cat", "points": [[0, 0], [10, 20]]},
        {"label": "cat", "points": []},
        {"label": "", "points": [[0, 0], [5, 5]]},
        "not-a-dict",
    ]
    grouped = collect_shape_sizes(shapes)
    assert "cat" in grouped
    assert len(grouped["cat"]["widths"]) == 1
    assert grouped["cat"]["widths"][0] == 10.0
    assert grouped["cat"]["heights"][0] == 20.0


def test_collect_shape_sizes_skips_degenerate():
    grouped = collect_shape_sizes(
        [{"label": "dot", "points": [[5, 5], [5, 5]]}]
    )
    assert grouped == {}


def test_box_size_true_geometry():
    from anylabeling.services.dataset_stats_core import box_size

    assert box_size([0, 0, 10, 20]) == (10.0, 20.0, 200.0)
    assert box_size([5, 5, 5, 5]) is None
    assert box_size(None) is None


def test_compute_dataset_stats_from_metas():
    metas = [
        DatasetMeta(
            image_path="a.jpg",
            has_label_file=True,
            shape_count=2,
            verified_empty=False,
        ),
        DatasetMeta(
            image_path="b.jpg",
            has_label_file=True,
            shape_count=0,
            verified_empty=True,
        ),
        DatasetMeta(
            image_path="c.jpg",
            has_label_file=False,
            shape_count=0,
            verified_empty=False,
        ),
    ]
    stats = compute_dataset_stats_from_metas(metas)
    assert stats.total_images == 3
    assert stats.total_annotations == 2
    assert stats.verified_empty_images == 1
    assert stats.empty_images == 1


def test_compute_dataset_stats_from_metas_verified_vs_missing():
    # Regression: `or True` previously made has_label branch dead.
    # Verified-empty and missing-label must be distinguished.
    metas = [
        DatasetMeta(
            image_path="v.jpg",
            has_label_file=True,
            shape_count=0,
            verified_empty=True,
        ),
        DatasetMeta(
            image_path="m.jpg",
            has_label_file=False,
            shape_count=0,
            verified_empty=False,
        ),
    ]
    stats = compute_dataset_stats_from_metas(metas, tiny_box_px=16.0)
    assert stats.verified_empty_images == 1
    assert stats.empty_images == 1
    assert stats.total_annotations == 0
