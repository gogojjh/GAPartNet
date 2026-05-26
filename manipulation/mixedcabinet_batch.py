import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path("/Titan/dataset/data_gapartnet/partnet_mobility_mixedcabinet")
OUTPUT_ROOT = SCRIPT_DIR / "output"
SUMMARY_NAME = "mixedcabinet_movable_parts_summary.json"
RESULTS_NAME = "mixedcabinet_open_test_results.json"
VIDEO_NAME = "manipulation.mp4"
REVOLUTE_SUCCESS_RAD = float(np.deg2rad(30.0))


def _canonical_joint_type(joint_type):
    if joint_type in {"revolute", "continuous"}:
        return "revolute"
    if joint_type == "prismatic":
        return "prismatic"
    return joint_type


def _parse_limit(joint_el):
    limit_el = joint_el.find("limit")
    if limit_el is None:
        return None, None
    lower = limit_el.attrib.get("lower")
    upper = limit_el.attrib.get("upper")
    return (
        float(lower) if lower is not None else None,
        float(upper) if upper is not None else None,
    )


def _load_link_part_ids(asset_dir):
    anno_path = asset_dir / "link_annotation_gapartnet.json"
    if not anno_path.exists():
        return {}
    annotations = json.loads(anno_path.read_text())
    return {
        anno.get("link_name"): idx
        for idx, anno in enumerate(annotations)
        if anno.get("is_gapart") and anno.get("link_name")
    }


def _load_fixed_handle_heights(asset_dir):
    anno_path = asset_dir / "link_annotation_gapartnet.json"
    if not anno_path.exists():
        return {}
    annotations = json.loads(anno_path.read_text())
    fixed_parent = {}
    urdf_path = asset_dir / "mobility_annotation_gapartnet.urdf"
    if urdf_path.exists():
        for joint_el in ET.parse(urdf_path).getroot().findall("joint"):
            if joint_el.attrib.get("type") != "fixed":
                continue
            parent_el = joint_el.find("parent")
            child_el = joint_el.find("child")
            if parent_el is None or child_el is None:
                continue
            fixed_parent[child_el.attrib.get("link")] = parent_el.attrib.get("link")

    heights = {}
    for anno in annotations:
        if not anno.get("is_gapart") or not anno.get("category", "").endswith("fixed_handle"):
            continue
        bbox = np.asarray(anno.get("bbox", []), dtype=np.float32)
        if bbox.size == 0:
            continue
        current = anno.get("link_name")
        visited = set()
        while current in fixed_parent and current not in visited:
            visited.add(current)
            current = fixed_parent[current]
        if current:
            heights[current] = max(heights.get(current, -float("inf")), float(np.mean(bbox[:, 2])))
    return heights


def summarize_asset(asset_dir):
    asset_dir = Path(asset_dir)
    urdf_path = asset_dir / "mobility_annotation_gapartnet.urdf"
    link_part_ids = _load_link_part_ids(asset_dir)
    handle_heights = _load_fixed_handle_heights(asset_dir)
    root = ET.parse(urdf_path).getroot()
    parts = []
    fallback_part_id = 0
    for joint_i, joint_el in enumerate(root.findall("joint")):
        joint_type = _canonical_joint_type(joint_el.attrib.get("type"))
        if joint_type not in {"revolute", "prismatic"}:
            continue
        child_el = joint_el.find("child")
        parent_el = joint_el.find("parent")
        child = child_el.attrib.get("link") if child_el is not None else None
        parent = parent_el.attrib.get("link") if parent_el is not None else None
        lower, upper = _parse_limit(joint_el)
        part_id = link_part_ids.get(child)
        if part_id is None:
            part_id = fallback_part_id
            fallback_part_id += 1
        parts.append({
            "part_id": int(part_id),
            "joint_order": int(joint_i),
            "joint_name": joint_el.attrib.get("name"),
            "joint_type": joint_type,
            "raw_joint_type": joint_el.attrib.get("type"),
            "parent": parent,
            "child_link": child,
            "handle_center_z": handle_heights.get(child),
            "lower": lower,
            "upper": upper,
        })
    parts.sort(key=lambda p: (p["part_id"], p["joint_order"]))
    revolute_count = sum(1 for p in parts if p["joint_type"] == "revolute")
    prismatic_count = sum(1 for p in parts if p["joint_type"] == "prismatic")
    return {
        "asset_id": asset_dir.name,
        "asset_dir": str(asset_dir),
        "movable_part_count": len(parts),
        "revolute_count": revolute_count,
        "prismatic_count": prismatic_count,
        "parts": parts,
    }


def summarize_assets(dataset_root):
    dataset_root = Path(dataset_root)
    assets = {}
    for urdf_path in sorted(dataset_root.glob("*/mobility_annotation_gapartnet.urdf")):
        asset = summarize_asset(urdf_path.parent)
        assets[asset["asset_id"]] = asset
    return {
        "dataset_root": str(dataset_root),
        "asset_count": len(assets),
        "assets": assets,
    }


def write_summary(dataset_root=DATASET_ROOT, output_path=None):
    dataset_root = Path(dataset_root)
    output_path = Path(output_path) if output_path is not None else dataset_root / SUMMARY_NAME
    summary = summarize_assets(dataset_root)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary, output_path


def select_test_target(asset_summary):
    parts = sorted(asset_summary.get("parts", []), key=lambda p: (p["part_id"], p.get("joint_order", 0)))
    revolute = [p for p in parts if p.get("joint_type") == "revolute"]
    prismatic = [p for p in parts if p.get("joint_type") == "prismatic"]
    target = None
    reason = None
    if len(revolute) == 0 and len(prismatic) == 1:
        target = prismatic[0]
        reason = "single_prismatic"
    elif len(revolute) == 1 and len(prismatic) == 0:
        target = revolute[0]
        reason = "single_revolute"
    elif len(revolute) >= 1 and len(prismatic) >= 1:
        target = max(
            prismatic,
            key=lambda p: (
                -float("inf") if p.get("handle_center_z") is None else float(p.get("handle_center_z")),
                -p["part_id"],
                -p.get("joint_order", 0),
            ),
        )
        reason = "mixed_asset_highest_prismatic_handle"
    if target is None:
        return None
    selected = dict(target)
    selected["asset_id"] = asset_summary["asset_id"]
    selected["asset_dir"] = asset_summary.get("asset_dir")
    selected["selection_reason"] = reason
    return selected


def build_targets(summary):
    targets = []
    skipped = []
    for asset_id in sorted(summary.get("assets", {})):
        target = select_test_target(summary["assets"][asset_id])
        if target is None:
            skipped.append({
                "asset_id": asset_id,
                "reason": "unsupported_movable_part_counts",
                "revolute_count": summary["assets"][asset_id].get("revolute_count"),
                "prismatic_count": summary["assets"][asset_id].get("prismatic_count"),
            })
        else:
            targets.append(target)
    return targets, skipped


def required_delta(part):
    joint_type = part.get("joint_type")
    if joint_type == "revolute":
        return REVOLUTE_SUCCESS_RAD
    if joint_type == "prismatic":
        lower = part.get("lower")
        upper = part.get("upper")
        if lower is None or upper is None:
            return None
        return 0.5 * abs(float(upper) - float(lower))
    return None


def evaluate_attempt(part, initial_dof, final_dof, gripper_on_handle):
    delta = abs(float(final_dof) - float(initial_dof))
    req = required_delta(part)
    if not gripper_on_handle:
        status = "failure"
        reason = "gripper_not_on_handle"
    elif req is None:
        status = "failure"
        reason = "missing_joint_limit"
    elif delta >= req:
        status = "success"
        reason = None
    else:
        status = "failure"
        reason = "insufficient_motion"
    return {
        "status": status,
        "failure_reason": reason,
        "delta": delta,
        "required_delta": req,
        "gripper_on_handle": bool(gripper_on_handle),
    }


def build_run_command(target, dataset_root=DATASET_ROOT, output_root=OUTPUT_ROOT, python_bin=None):
    dataset_root = Path(dataset_root)
    python_bin = python_bin or sys.executable
    return [
        str(python_bin),
        "run.py",
        "--mode",
        "run_arti_open",
        "--headless",
        "--save_video",
        "--object_path",
        str(dataset_root / str(target["asset_id"])),
        "--part_id",
        str(target["part_id"]),
        "--task_root",
        str(output_root),
    ]


def clean_old_outputs(output_root=OUTPUT_ROOT):
    output_root = Path(output_root)
    patterns = [
        "partnet_mobility_mixedcabinet_*",
        "revolute_success",
        "revolute_failure",
        "prismatic_success",
        "prismatic_failure",
        "classified_targets",
    ]
    removed = []
    for pattern in patterns:
        for path in output_root.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                removed.append(str(path))
            elif path.exists():
                path.unlink()
                removed.append(str(path))
    return removed


def load_run_result(target, output_root=OUTPUT_ROOT):
    result_path = Path(output_root) / f"partnet_mobility_mixedcabinet_{target['asset_id']}" / "result.json"
    if not result_path.exists():
        return {
            "asset_id": target["asset_id"],
            "tested_part_id": target["part_id"],
            "joint_type": target["joint_type"],
            "status": "failure",
            "failure_reason": "missing_result_json",
            "video": str(result_path.parent / VIDEO_NAME),
        }
    data = json.loads(result_path.read_text())
    data.setdefault("asset_id", target["asset_id"])
    data.setdefault("tested_part_id", target["part_id"])
    data.setdefault("joint_type", target["joint_type"])
    data.setdefault("video", str(result_path.parent / VIDEO_NAME))
    return data


def run_targets(targets, dataset_root=DATASET_ROOT, output_root=OUTPUT_ROOT, python_bin=None):
    results = []
    for target in targets:
        cmd = build_run_command(target, dataset_root=dataset_root, output_root=output_root, python_bin=python_bin)
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
        result = load_run_result(target, output_root=output_root)
        result["returncode"] = proc.returncode
        if proc.returncode != 0 and result.get("status") == "success":
            result["status"] = "failure"
            result["failure_reason"] = "run_command_failed"
        results.append(result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args(argv)

    if args.clean_output:
        removed = clean_old_outputs(args.output_root)
        print(json.dumps({"removed": removed}, indent=2))

    summary, summary_path = write_summary(args.dataset_root)
    targets, skipped = build_targets(summary)
    print(f"Wrote summary: {summary_path}")
    print(f"Selected {len(targets)} targets; skipped {len(skipped)} assets")
    if args.summary_only:
        return 0

    plan = {"targets": targets, "skipped": skipped}
    plan_path = args.dataset_root / "mixedcabinet_open_test_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True))
    print(f"Wrote test plan: {plan_path}")
    if args.dry_run:
        for target in targets:
            print(" ".join(build_run_command(target, args.dataset_root, args.output_root, python_bin=args.python_bin)))
        return 0

    results = run_targets(targets, args.dataset_root, args.output_root, python_bin=args.python_bin)
    results_path = args.dataset_root / RESULTS_NAME
    results_path.write_text(json.dumps({"results": results, "skipped": skipped}, indent=2, sort_keys=True))
    print(f"Wrote results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
