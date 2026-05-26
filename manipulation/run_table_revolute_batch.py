import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np


DATASET = Path("/Titan/dataset/data_gapartnet/partnet_mobility_StorageFurniture_Table/revolute_joint")
REPO = Path("/Titan/code/robohike_ws/src/3D-Diffusion-Policy/third_party/GAPartNet")
MANIP = REPO / "manipulation"
TASK_ROOT = MANIP / "output" / "table_revolute"
AGGREGATE = MANIP / "output" / "table_revolute_results.json"
MARKDOWN = MANIP / "output" / "table_revolute_summary.md"
VERTICAL_THRESHOLD = 0.85


def parse_xyz(value, default=(0.0, 0.0, 0.0)):
    if not value:
        return np.array(default, dtype=np.float32)
    return np.array([float(x) for x in value.split()], dtype=np.float32)


def rotation_from_rpy(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def parse_urdf(asset_dir):
    root = ET.parse(asset_dir / "mobility_annotation_gapartnet.urdf").getroot()
    fixed_parent = {}
    movable = []
    for joint in root.findall("joint"):
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.attrib.get("link")
        child = child_el.attrib.get("link")
        joint_type = joint.attrib.get("type")
        if joint_type == "fixed":
            fixed_parent[child] = parent
            continue
        if joint_type not in {"revolute", "continuous", "prismatic"}:
            continue
        axis_el = joint.find("axis")
        origin_el = joint.find("origin")
        axis = parse_xyz(axis_el.attrib.get("xyz") if axis_el is not None else None, default=(1.0, 0.0, 0.0))
        rpy = parse_xyz(origin_el.attrib.get("rpy") if origin_el is not None else None)
        axis_world = rotation_from_rpy(rpy) @ axis
        norm = np.linalg.norm(axis_world)
        if norm > 1e-6:
            axis_world = axis_world / norm
        movable.append(
            {
                "name": joint.attrib.get("name"),
                "type": joint_type,
                "parent": parent,
                "child": child,
                "axis": axis_world.astype(float).tolist(),
                "vertical_score": float(abs(axis_world[2])),
            }
        )
    return fixed_parent, movable


def fixed_chain_root(link, fixed_parent):
    current = link
    visited = set()
    while current in fixed_parent and current not in visited:
        visited.add(current)
        current = fixed_parent[current]
    return current


def select_target(asset_dir):
    fixed_parent, movable = parse_urdf(asset_dir)
    revolute_joints = [
        joint
        for joint in movable
        if joint["type"] in {"revolute", "continuous"}
    ]
    if not revolute_joints:
        return None, "no_revolute_joint"
    joint = sorted(
        revolute_joints,
        key=lambda item: (-float(item.get("vertical_score", 0.0)), item.get("name") or "", item.get("child") or ""),
    )[0]
    anno_path = asset_dir / "link_annotation_gapartnet.json"
    if not anno_path.exists():
        return None, "missing_link_annotation"
    annos = json.loads(anno_path.read_text())
    valid_annos = [(idx, anno) for idx, anno in enumerate(annos) if anno.get("is_gapart")]
    handles = []
    movable_parts = []
    for raw_idx, anno in valid_annos:
        link_name = anno.get("link_name", "")
        if fixed_chain_root(link_name, fixed_parent) != joint["child"]:
            continue
        category = anno.get("category", "")
        if category.endswith("fixed_handle"):
            handles.append((raw_idx, anno))
        else:
            movable_parts.append((raw_idx, anno))
    candidates = sorted(handles, key=lambda item: item[0]) or sorted(movable_parts, key=lambda item: item[0])
    if not candidates:
        return None, "no_part_for_revolute_joint"
    part_id, anno = candidates[0]
    return {
        "asset_id": asset_dir.name,
        "asset_dir": str(asset_dir),
        "part_id": int(part_id),
        "selected_link": anno.get("link_name"),
        "selected_category": anno.get("category"),
        "joint_name": joint["name"],
        "joint_type": "revolute" if joint["type"] == "continuous" else joint["type"],
        "axis": joint["axis"],
        "vertical_score": joint["vertical_score"],
    }, None


def load_result(asset_id, part_id):
    result_dir = TASK_ROOT / f"revolute_joint_{asset_id}"
    result_path = result_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    return {
        "asset_id": asset_id,
        "tested_part_id": part_id,
        "status": "failure",
        "failure_reason": "missing_result_json",
        "save_root": str(result_dir),
        "video": str(result_dir / "manipulation.mp4"),
    }


def write_outputs(assets, targets, results, skipped):
    success_ids = sorted({str(result.get("asset_id")) for result in results if result.get("status") == "success"})
    failure_ids = sorted({str(result.get("asset_id")) for result in results if result.get("status") != "success"})
    skipped_ids = sorted({str(result.get("asset_id")) for result in skipped})
    payload = {
        "dataset_root": str(DATASET),
        "task_root": str(TASK_ROOT),
        "total_assets": len(assets),
        "target_count": len(targets),
        "tested": len(results),
        "success": len(success_ids),
        "failure": len(failure_ids),
        "skipped": len(skipped_ids),
        "targets": targets,
        "results": results,
        "skipped_results": skipped,
        "success_assets": success_ids,
        "failure_assets": failure_ids,
        "skipped_assets": skipped_ids,
    }
    MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    AGGREGATE.write_text(json.dumps(payload, indent=2, sort_keys=True))
    lines = [
        "# Table Revolute Smoke Test Summary",
        "",
        f"Dataset: `{DATASET}`",
        "",
        "Command: `python run.py --mode run_arti_open --headless --save_video --joint_type revolute --part_id <part_id> --object_path <asset_dir>`",
        "",
        f"Total assets: {len(assets)}",
        f"Tested: {len(results)}",
        f"Successful: {len(success_ids)}",
        f"Failed: {len(failure_ids)}",
        f"Skipped: {len(skipped_ids)}",
        "",
        "## Successful Assets",
        *[f"- {asset_id}" for asset_id in success_ids],
        "",
        "## Failed Assets",
        *[f"- {asset_id}" for asset_id in failure_ids],
        "",
        "## Skipped Assets",
        *[f"- {asset_id}" for asset_id in skipped_ids],
        "",
        "## Output Files",
        f"- Aggregate JSON: `{AGGREGATE}`",
        f"- Task root: `{TASK_ROOT}`",
        "",
    ]
    MARKDOWN.write_text("\n".join(lines))
    return payload


def main():
    assets = sorted([path for path in DATASET.iterdir() if path.is_dir() and path.name.isdigit()], key=lambda p: p.name)
    targets = []
    results = []
    skipped = []
    TASK_ROOT.mkdir(parents=True, exist_ok=True)
    for index, asset_dir in enumerate(assets, 1):
        target, skip_reason = select_target(asset_dir)
        if target is None:
            skipped.append({"asset_id": asset_dir.name, "status": "skipped", "failure_reason": skip_reason})
            write_outputs(assets, targets, results, skipped)
            print(f"[REV_BATCH] skip {asset_dir.name}: {skip_reason}", flush=True)
            continue
        targets.append(target)
        print(f"[REV_BATCH] {index}/{len(assets)} asset={asset_dir.name} part_id={target['part_id']} joint={target['joint_name']}", flush=True)
        cmd = [
            "conda",
            "run",
            "-n",
            "3d_dp",
            "python",
            "run.py",
            "--mode",
            "run_arti_open",
            "--headless",
            "--save_video",
            "--object_path",
            str(asset_dir),
            "--joint_type",
            "revolute",
            "--part_id",
            str(target["part_id"]),
            "--task_root",
            str(TASK_ROOT),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = "/root/isaacgym/python"
        start = time.time()
        proc = subprocess.run(cmd, cwd=str(MANIP), env=env)
        elapsed = time.time() - start
        result = load_result(asset_dir.name, target["part_id"])
        result["returncode"] = proc.returncode
        result["elapsed_sec"] = elapsed
        result["planned_target"] = target
        if proc.returncode != 0 and result.get("status") == "success":
            result["status"] = "failure"
            result["failure_reason"] = "run_command_failed"
        results.append(result)
        write_outputs(assets, targets, results, skipped)
        print(
            f"[REV_BATCH] done asset={asset_dir.name} status={result.get('status')} "
            f"reason={result.get('failure_reason')} elapsed={elapsed:.1f}s",
            flush=True,
        )
    payload = write_outputs(assets, targets, results, skipped)
    print(json.dumps({
        "aggregate": str(AGGREGATE),
        "markdown": str(MARKDOWN),
        "success_assets": payload["success_assets"],
        "failure_assets": payload["failure_assets"],
        "skipped_assets": payload["skipped_assets"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
