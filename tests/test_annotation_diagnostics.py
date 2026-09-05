import csv
import json
import os
import tempfile
import pytest

from anylabeling.services.annotation_diagnostics import (
    AnnotationDiagnosticsScanner,
    ScanIssue,
    ScanReport,
    clamp_out_of_bounds_shapes,
    compute_file_md5,
    deduplicate_dataset_shapes,
    export_report_to_file,
    get_label_file_path,
    move_empty_images,
    purge_micro_shapes,
    validate_shape,
)


def test_scan_issue_and_report_serialization():
    issue = ScanIssue(
        image_path="/path/img1.jpg",
        label_file="/path/img1.json",
        issue_type="duplicate_shape",
        severity="warning",
        details="Duplicate of shape #1",
        shape_index=1,
        suggestion="Remove duplicate",
    )
    issue_dict = issue.to_dict()
    assert issue_dict["issue_type"] == "duplicate_shape"
    assert issue_dict["severity"] == "warning"

    report = ScanReport(total_images=10, total_annotations=25)
    report.issues.append(issue)
    assert report.total_issues == 1
    rep_dict = report.to_dict()
    assert rep_dict["total_issues"] == 1
    assert "issues" in rep_dict
    json_str = report.to_json()
    assert '"total_images": 10' in json_str


def test_get_label_file_path():
    img_path = "/dataset/images/cat.png"
    # When output_dir is None
    assert get_label_file_path(img_path) == "/dataset/images/cat.json"

    with tempfile.TemporaryDirectory() as tmp_dir:
        out = get_label_file_path(img_path, output_dir=tmp_dir)
        assert out == os.path.join(tmp_dir, "cat.json")


def test_validate_shape_missing_label():
    shape = {
        "shape_type": "rectangle",
        "points": [[0, 0], [10, 10]],
        "label": "",
    }
    issues = validate_shape(shape, 0)
    assert any(i[0] == "missing_label" and i[1] == "error" for i in issues)


def test_validate_shape_degenerate_geometry():
    # Empty points
    issues = validate_shape(
        {"shape_type": "rectangle", "points": [], "label": "dog"}, 0
    )
    assert any(i[0] == "degenerate_geometry" for i in issues)

    # Non-finite coordinates
    issues = validate_shape(
        {
            "shape_type": "rectangle",
            "points": [[float("nan"), 0], [10, 10]],
            "label": "dog",
        },
        0,
    )
    assert any(i[0] == "degenerate_geometry" for i in issues)

    # Polygon with < 3 points
    issues = validate_shape(
        {
            "shape_type": "polygon",
            "points": [[0, 0], [10, 10]],
            "label": "dog",
        },
        0,
    )
    assert any(i[0] == "degenerate_geometry" for i in issues)

    # Polygon with duplicate consecutive vertices
    issues = validate_shape(
        {
            "shape_type": "polygon",
            "points": [[0, 0], [0, 0], [10, 0], [5, 5]],
            "label": "dog",
        },
        0,
    )
    assert any(
        i[0] == "degenerate_geometry" and i[1] == "warning" for i in issues
    )

    # Zero size rectangle
    issues = validate_shape(
        {
            "shape_type": "rectangle",
            "points": [[10, 10], [10, 10]],
            "label": "dog",
        },
        0,
    )
    assert any(i[0] == "degenerate_geometry" for i in issues)


def test_validate_shape_micro_noise_and_point_exception():
    # Micro-noise rectangle (2x2 px)
    micro_shape = {
        "shape_type": "rectangle",
        "points": [[10, 10], [12, 12]],
        "label": "car",
    }
    issues = validate_shape(micro_shape, 0, min_size_px=5.0, min_area_px=16.0)
    assert any(i[0] == "micro_noise" for i in issues)

    # Point shape (1x1 px) should NOT be flagged as micro-noise
    point_shape = {
        "shape_type": "point",
        "points": [[10, 10]],
        "label": "landmark",
    }
    issues = validate_shape(point_shape, 0, min_size_px=5.0, min_area_px=16.0)
    assert not any(i[0] == "micro_noise" for i in issues)


def test_validate_shape_out_of_bounds():
    shape = {
        "shape_type": "rectangle",
        "points": [[-10, 0], [50, 50]],
        "label": "car",
    }
    issues = validate_shape(shape, 0, image_width=100, image_height=100)
    assert any(i[0] == "out_of_bounds" for i in issues)

    shape_overflow = {
        "shape_type": "rectangle",
        "points": [[10, 10], [120, 50]],
        "label": "car",
    }
    issues = validate_shape(
        shape_overflow, 0, image_width=100, image_height=100
    )
    assert any(i[0] == "out_of_bounds" for i in issues)


def test_scanner_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create 3 dummy image files
        img1 = os.path.join(tmp_dir, "img1.jpg")
        img2 = os.path.join(tmp_dir, "img2.jpg")  # Binary duplicate of img1
        img3 = os.path.join(tmp_dir, "img3.jpg")  # Empty image
        missing_img = os.path.join(tmp_dir, "nonexistent.jpg")

        content_a = b"fake-image-content-identical"
        content_b = b"different-image-content"
        with open(img1, "wb") as f:
            f.write(content_a)
        with open(img2, "wb") as f:
            f.write(content_a)
        with open(img3, "wb") as f:
            f.write(content_b)

        # img1 label has duplicate shapes and micro noise
        lbl1 = os.path.join(tmp_dir, "img1.json")
        lbl1_data = {
            "imagePath": "img1.jpg",
            "imageWidth": 640,
            "imageHeight": 480,
            "shapes": [
                {
                    "label": "car",
                    "shape_type": "rectangle",
                    "points": [[10, 10], [100, 100]],
                    "score": 0.9,
                },
                {
                    "label": "car",
                    "shape_type": "rectangle",
                    "points": [[10, 10], [100, 100]],  # Exact duplicate
                    "score": 0.7,
                },
                {
                    "label": "noise",
                    "shape_type": "rectangle",
                    "points": [[200, 200], [202, 202]],  # Micro noise
                },
            ],
        }
        with open(lbl1, "w", encoding="utf-8") as f:
            json.dump(lbl1_data, f)

        # img2 has no label file
        # img3 has label file with 0 shapes
        lbl3 = os.path.join(tmp_dir, "img3.json")
        with open(lbl3, "w", encoding="utf-8") as f:
            json.dump({"imagePath": "img3.jpg", "shapes": []}, f)

        scanner = AnnotationDiagnosticsScanner(
            iou_duplicate_threshold=0.85,
            containment_duplicate_threshold=0.90,
            min_size_px=5.0,
            min_area_px=16.0,
            check_image_hashes=True,
        )

        progress_calls = []

        def on_progress(current, total, msg):
            progress_calls.append((current, total, msg))

        report = scanner.scan(
            [img1, img2, img3, missing_img],
            progress_callback=on_progress,
        )

        assert report.total_images == 4
        assert len(progress_calls) == 4
        assert report.duplicate_images_count == 1
        assert report.duplicate_shapes_count == 1
        assert report.micro_shapes_count == 1
        assert report.missing_labels >= 1
        assert report.empty_images >= 1
        assert report.class_counts.get("car") == 2

        # Check issues present
        issue_types = [issue.issue_type for issue in report.issues]
        assert "duplicate_image" in issue_types
        assert "duplicate_shape" in issue_types
        assert "micro_noise" in issue_types
        assert "missing_image" in issue_types
        assert "missing_label" in issue_types


def test_deduplicate_dataset_shapes_and_backup():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "sample.jpg")
        with open(img_path, "wb") as f:
            f.write(b"img")

        lbl_path = os.path.join(tmp_dir, "sample.json")
        lbl_data = {
            "imagePath": "sample.jpg",
            "shapes": [
                {
                    "label": "dog",
                    "shape_type": "rectangle",
                    "points": [[0, 0], [50, 50]],
                },
                {
                    "label": "dog",
                    "shape_type": "rectangle",
                    "points": [[1, 1], [50, 50]],
                },
            ],
        }
        with open(lbl_path, "w", encoding="utf-8") as f:
            json.dump(lbl_data, f)

        backup_folder = os.path.join(tmp_dir, "backups")
        removed = deduplicate_dataset_shapes(
            [img_path],
            iou_threshold=0.85,
            backup=True,
            backup_dir=backup_folder,
        )
        assert removed == 1

        # Check backup created
        assert os.path.exists(os.path.join(backup_folder, "sample.json"))

        # Check modified file now has 1 shape
        with open(lbl_path, "r", encoding="utf-8") as f:
            updated = json.load(f)
        assert len(updated["shapes"]) == 1


def test_purge_micro_shapes():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "sample.jpg")
        with open(img_path, "wb") as f:
            f.write(b"img")

        lbl_path = os.path.join(tmp_dir, "sample.json")
        lbl_data = {
            "imagePath": "sample.jpg",
            "shapes": [
                {
                    "label": "valid",
                    "shape_type": "rectangle",
                    "points": [[10, 10], [50, 50]],
                },
                {
                    "label": "tiny",
                    "shape_type": "rectangle",
                    "points": [[0, 0], [2, 2]],
                },
            ],
        }
        with open(lbl_path, "w", encoding="utf-8") as f:
            json.dump(lbl_data, f)

        purged = purge_micro_shapes(
            [img_path], min_size_px=5.0, min_area_px=16.0, backup=False
        )
        assert purged == 1

        with open(lbl_path, "r", encoding="utf-8") as f:
            updated = json.load(f)
        assert len(updated["shapes"]) == 1
        assert updated["shapes"][0]["label"] == "valid"


def test_move_empty_images():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_valid = os.path.join(tmp_dir, "valid.jpg")
        img_empty = os.path.join(tmp_dir, "empty.jpg")
        img_verified_bg = os.path.join(tmp_dir, "verified_bg.jpg")
        with open(img_valid, "wb") as f:
            f.write(b"valid")
        with open(img_empty, "wb") as f:
            f.write(b"empty")
        with open(img_verified_bg, "wb") as f:
            f.write(b"verified_bg")

        lbl_valid = os.path.join(tmp_dir, "valid.json")
        lbl_empty = os.path.join(tmp_dir, "empty.json")
        lbl_verified_bg = os.path.join(tmp_dir, "verified_bg.json")
        with open(lbl_valid, "w", encoding="utf-8") as f:
            json.dump(
                {"shapes": [{"label": "box", "points": [[0, 0], [10, 10]]}]}, f
            )
        with open(lbl_empty, "w", encoding="utf-8") as f:
            json.dump({"shapes": []}, f)
        with open(lbl_verified_bg, "w", encoding="utf-8") as f:
            json.dump({"shapes": [], "verified_empty": True}, f)

        dest_dir = os.path.join(tmp_dir, "empty_bucket")
        moved = move_empty_images(
            [img_valid, img_empty, img_verified_bg], dest_dir
        )
        assert moved == 1

        # Valid and verified background images stay
        assert os.path.exists(img_valid)
        assert os.path.exists(lbl_valid)
        assert os.path.exists(img_verified_bg)
        assert os.path.exists(lbl_verified_bg)

        # Empty unverified image moved
        assert not os.path.exists(img_empty)
        assert not os.path.exists(lbl_empty)
        assert os.path.exists(os.path.join(dest_dir, "empty.jpg"))
        assert os.path.exists(os.path.join(dest_dir, "empty.json"))


def test_export_report_to_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        report = ScanReport(total_images=2, total_annotations=5)
        report.issues.append(
            ScanIssue(
                image_path="/path/test.jpg",
                label_file="/path/test.json",
                issue_type="micro_noise",
                severity="warning",
                details="Noise box 2x2",
                shape_index=0,
                suggestion="Purge",
            )
        )

        # JSON Export
        json_path = os.path.join(tmp_dir, "report.json")
        export_report_to_file(report, json_path, export_format="json")
        assert os.path.exists(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_images"] == 2
        assert len(data["issues"]) == 1

        # CSV Export
        csv_path = os.path.join(tmp_dir, "report.csv")
        export_report_to_file(report, csv_path, export_format="csv")
        assert os.path.exists(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        assert len(reader) == 2  # Header + 1 issue row
        assert reader[0][0] == "Severity"
        assert reader[1][0] == "warning"


def test_clamp_out_of_bounds_shapes():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "test.jpg")
        with open(img_path, "wb") as f:
            f.write(b"dummy")

        lbl_path = os.path.join(tmp_dir, "test.json")
        initial_data = {
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [
                {
                    "label": "clamped_box",
                    "shape_type": "rectangle",
                    "points": [[-10.0, -5.0], [105.0, 95.0]],
                },
                {
                    "label": "valid_box",
                    "shape_type": "rectangle",
                    "points": [[10.0, 10.0], [50.0, 50.0]],
                },
                {
                    "label": "outside_box",
                    "shape_type": "rectangle",
                    "points": [[-50.0, -50.0], [-10.0, -10.0]],
                },
            ],
        }
        with open(lbl_path, "w", encoding="utf-8") as f:
            json.dump(initial_data, f)

        # Run scanner before clamping to verify out_of_bounds detected
        scanner = AnnotationDiagnosticsScanner()
        report_before = scanner.scan([img_path])
        assert report_before.out_of_bounds_count == 2

        # Perform in-place clamping (no backup for test isolation)
        result = clamp_out_of_bounds_shapes([img_path], backup=False)
        assert isinstance(result, tuple) and len(result) == 2
        clamped_count, deleted_count = result
        assert clamped_count == 1
        assert deleted_count == 1

        # Verify no backup directory is created when backup=False
        backup_dirs = [
            d for d in os.listdir(tmp_dir) if d.startswith(".backup_")
        ]
        assert len(backup_dirs) == 0

        # Verify updated label file (2-pt rects stay 2-pt)
        with open(lbl_path, "r", encoding="utf-8") as f:
            updated_data = json.load(f)
        assert len(updated_data["shapes"]) == 2
        assert updated_data["shapes"][0]["label"] == "clamped_box"
        assert updated_data["shapes"][0]["points"] == [
            [0.0, 0.0],
            [100.0, 95.0],
        ]
        assert updated_data["shapes"][1]["label"] == "valid_box"

        # Re-scan to verify 0 out-of-bounds remaining
        report_after = scanner.scan([img_path])
        assert report_after.out_of_bounds_count == 0
