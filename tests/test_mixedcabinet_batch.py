import json
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "manipulation"))

import mixedcabinet_batch as batch


def write_asset(root, asset_id, joints, annotations):
    asset_dir = root / asset_id
    asset_dir.mkdir()
    robot = ET.Element("robot", name=f"asset_{asset_id}")
    ET.SubElement(robot, "link", name="base")
    for joint in joints:
        ET.SubElement(robot, "link", name=joint["child"])
        joint_el = ET.SubElement(robot, "joint", name=joint["name"], type=joint["type"])
        ET.SubElement(joint_el, "parent", link=joint.get("parent", "base"))
        ET.SubElement(joint_el, "child", link=joint["child"])
        ET.SubElement(joint_el, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(joint_el, "axis", xyz="1 0 0")
        limit = joint.get("limit")
        if limit is not None:
            ET.SubElement(joint_el, "limit", lower=str(limit[0]), upper=str(limit[1]))
    for joint in joints:
        for fixed in joint.get("fixed_children", []):
            ET.SubElement(robot, "link", name=fixed)
            joint_el = ET.SubElement(robot, "joint", name=f"fixed_{fixed}", type="fixed")
            ET.SubElement(joint_el, "parent", link=joint["child"])
            ET.SubElement(joint_el, "child", link=fixed)
            ET.SubElement(joint_el, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.ElementTree(robot).write(asset_dir / "mobility_annotation_gapartnet.urdf")
    (asset_dir / "link_annotation_gapartnet.json").write_text(json.dumps(annotations))
    return asset_dir


def test_summarize_assets_maps_joint_types_and_part_ids(tmp_path):
    write_asset(
        tmp_path,
        "100",
        [
            {"name": "joint_0", "type": "continuous", "child": "link_2"},
            {"name": "joint_1", "type": "prismatic", "child": "link_4", "limit": [0.0, 0.2]},
        ],
        [
            {"is_gapart": True, "link_name": "link_2", "category": "hinge_door"},
            {"is_gapart": True, "link_name": "link_4", "category": "slider_drawer"},
        ],
    )

    summary = batch.summarize_assets(tmp_path)

    asset = summary["assets"]["100"]
    assert asset["revolute_count"] == 1
    assert asset["prismatic_count"] == 1
    assert asset["movable_part_count"] == 2
    assert asset["parts"][0]["part_id"] == 0
    assert asset["parts"][0]["joint_type"] == "revolute"
    assert asset["parts"][1]["part_id"] == 1
    assert asset["parts"][1]["joint_type"] == "prismatic"


def test_select_target_prefers_only_movable_or_highest_prismatic_handle_for_mixed():
    single_prismatic = {
        "asset_id": "1",
        "revolute_count": 0,
        "prismatic_count": 1,
        "parts": [{"part_id": 5, "joint_type": "prismatic"}],
    }
    single_revolute = {
        "asset_id": "2",
        "revolute_count": 1,
        "prismatic_count": 0,
        "parts": [{"part_id": 3, "joint_type": "revolute"}],
    }
    mixed = {
        "asset_id": "3",
        "revolute_count": 2,
        "prismatic_count": 2,
        "parts": [
            {"part_id": 4, "joint_type": "prismatic", "handle_center_z": 0.1},
            {"part_id": 1, "joint_type": "revolute"},
            {"part_id": 2, "joint_type": "prismatic", "handle_center_z": 0.8},
        ],
    }

    assert batch.select_test_target(single_prismatic)["part_id"] == 5
    assert batch.select_test_target(single_revolute)["part_id"] == 3
    target = batch.select_test_target(mixed)
    assert target["part_id"] == 2
    assert target["selection_reason"] == "mixed_asset_highest_prismatic_handle"


def test_summarize_asset_records_fixed_handle_height_for_prismatic_parts(tmp_path):
    asset_dir = write_asset(
        tmp_path,
        "45661_like",
        [
            {"name": "joint_0", "type": "revolute", "child": "link_0", "fixed_children": ["link_4"], "limit": [0.0, 1.0]},
            {"name": "joint_1", "type": "prismatic", "child": "link_1", "fixed_children": ["link_5"], "limit": [0.0, 0.66]},
            {"name": "joint_2", "type": "prismatic", "child": "link_2", "fixed_children": ["link_6"], "limit": [0.0, 0.66]},
        ],
        [
            {"is_gapart": True, "link_name": "link_0", "category": "hinge_door"},
            {"is_gapart": True, "link_name": "link_1", "category": "slider_drawer"},
            {"is_gapart": True, "link_name": "link_2", "category": "slider_drawer"},
            {"is_gapart": False, "link_name": "link_3", "category": ""},
            {"is_gapart": True, "link_name": "link_4", "category": "line_fixed_handle", "bbox": [[0, 0, 0.1]] * 8},
            {"is_gapart": True, "link_name": "link_5", "category": "line_fixed_handle", "bbox": [[0, 0, 0.3]] * 8},
            {"is_gapart": True, "link_name": "link_6", "category": "line_fixed_handle", "bbox": [[0, 0, 0.7]] * 8},
        ],
    )

    summary = batch.summarize_asset(asset_dir)
    target = batch.select_test_target(summary)

    assert target["part_id"] == 2
    assert target["joint_name"] == "joint_2"
    assert np.isclose(target["handle_center_z"], 0.7)


def test_evaluate_success_uses_requested_motion_thresholds_and_handle_check():
    revolute = {"joint_type": "revolute"}
    assert batch.evaluate_attempt(revolute, 0.0, np.deg2rad(31.0), True)["status"] == "success"
    fail_revolute = batch.evaluate_attempt(revolute, 0.0, np.deg2rad(29.0), True)
    assert fail_revolute["status"] == "failure"
    assert fail_revolute["failure_reason"] == "insufficient_motion"

    prismatic = {"joint_type": "prismatic", "lower": 0.0, "upper": 0.4}
    assert batch.evaluate_attempt(prismatic, 0.0, 0.21, True)["status"] == "success"
    fail_handle = batch.evaluate_attempt(prismatic, 0.0, 0.3, False)
    assert fail_handle["status"] == "failure"
    assert fail_handle["failure_reason"] == "gripper_not_on_handle"


def test_run_command_uses_manipulation_mp4_result_contract(tmp_path):
    target = {"asset_id": "100", "part_id": 2, "joint_type": "prismatic"}
    cmd = batch.build_run_command(
        target,
        dataset_root=tmp_path,
        output_root=pathlib.Path("output"),
    )

    assert "--save_video" in cmd
    assert "--part_id" in cmd
    assert str(tmp_path / "100") in cmd
    assert batch.VIDEO_NAME == "manipulation.mp4"
