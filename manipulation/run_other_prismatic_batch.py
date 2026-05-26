import json
import pathlib
import subprocess
import sys
import time


DATASET = pathlib.Path("/Titan/dataset/data_gapartnet/partnet_mobility_mixedcabinet")
REPO = pathlib.Path("/Titan/code/robohike_ws/src/3D-Diffusion-Policy/third_party/GAPartNet")
MANIP = REPO / "manipulation"
OUTPUT = MANIP / "output"
SUMMARY = DATASET / "mixedcabinet_movable_parts_summary.json"
OUT = DATASET / "mixedcabinet_other_prismatic_open_test_results.json"
EXCLUDE = {"40147", "41510", "45146", "45168"}


def select_targets(summary):
    targets = []
    for asset_id, asset in sorted(summary["assets"].items()):
        if asset_id in EXCLUDE or asset.get("prismatic_count", 0) <= 0:
            continue
        parts = sorted(
            asset.get("parts", []),
            key=lambda part: (part.get("part_id", 10**9), part.get("joint_order", 10**9)),
        )
        prismatic_parts = [part for part in parts if part.get("joint_type") == "prismatic"]
        if not prismatic_parts:
            continue
        target = dict(prismatic_parts[0])
        target.update(
            {
                "asset_id": asset_id,
                "asset_dir": asset.get("asset_dir"),
                "revolute_count": asset.get("revolute_count"),
                "prismatic_count": asset.get("prismatic_count"),
                "movable_part_count": asset.get("movable_part_count"),
            }
        )
        if asset.get("revolute_count", 0) >= 1 and asset.get("prismatic_count", 0) >= 1:
            target["selection_reason"] = "mixed_sorted_first_prismatic"
        elif asset.get("prismatic_count", 0) == 1:
            target["selection_reason"] = "single_or_prismatic_only"
        else:
            target["selection_reason"] = "prismatic_sorted_first"
        targets.append(target)
    return targets


def load_result(target):
    result_path = OUTPUT / f"partnet_mobility_mixedcabinet_{target['asset_id']}" / "result.json"
    if not result_path.exists():
        return {
            "asset_id": target["asset_id"],
            "tested_part_id": target["part_id"],
            "status": "failure",
            "failure_reason": "missing_result_json",
            "video": str(result_path.parent / "manipulation.mp4"),
        }
    return json.loads(result_path.read_text())


def write_payload(targets, results):
    payload = {
        "source_summary": str(SUMMARY),
        "excluded_assets": sorted(EXCLUDE),
        "target_count": len(targets),
        "completed_count": len(results),
        "targets": targets,
        "results": results,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main():
    summary = json.loads(SUMMARY.read_text())
    targets = select_targets(summary)
    print(f"[BATCH] selected {len(targets)} remaining prismatic assets", flush=True)
    results = []
    for index, target in enumerate(targets, 1):
        asset_id = str(target["asset_id"])
        part_id = str(target["part_id"])
        print(
            f"[BATCH] {index}/{len(targets)} asset={asset_id} part_id={part_id} "
            f"joint={target.get('joint_name')} reason={target.get('selection_reason')}",
            flush=True,
        )
        cmd = [
            sys.executable,
            "run.py",
            "--mode",
            "run_arti_open",
            "--headless",
            "--save_video",
            "--object_path",
            str(DATASET / asset_id),
            "--part_id",
            part_id,
            "--task_root",
            "output",
        ]
        start = time.time()
        proc = subprocess.run(cmd, cwd=str(MANIP))
        elapsed = time.time() - start
        result = load_result(target)
        result["returncode"] = proc.returncode
        result["elapsed_sec"] = elapsed
        result["planned_target"] = target
        if proc.returncode != 0 and result.get("status") == "success":
            result["status"] = "failure"
            result["failure_reason"] = "run_command_failed"
        results.append(result)
        write_payload(targets, results)
        print(
            f"[BATCH] done asset={asset_id} status={result.get('status')} "
            f"delta={result.get('delta')} req={result.get('required_delta')} "
            f"grip={result.get('gripper_on_handle')} elapsed={elapsed:.1f}s",
            flush=True,
        )
    print(f"[BATCH] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
