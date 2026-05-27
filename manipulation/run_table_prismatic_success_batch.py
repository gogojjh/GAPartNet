#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MANIP_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path("/Titan/dataset/data_gapartnet/partnet_mobility_StorageFurniture_Table/prismatic_joint")
SUMMARY_MD = MANIP_DIR / "output" / "table_prismatic_summary.md"
TASK_ROOT = MANIP_DIR / "output" / "table_prismatic"
AGGREGATE_JSON = MANIP_DIR / "output" / "table_prismatic_success_retest_results.json"
SUMMARY_OUT = MANIP_DIR / "output" / "table_prismatic_success_retest_summary.md"


def parse_summary_markdown(path):
    text = path.read_text()
    lines = text.split("\n")

    in_success_section = False
    success_asset_ids = []
    for line in lines:
        if "## Failure Assets" in line or "## Enriched Assets" in line:
            in_success_section = False
        if in_success_section and line.strip().startswith("- "):
            asset_id = line.strip().lstrip("- ").strip()
            if asset_id.isdigit():
                success_asset_ids.append(asset_id)
        if "## Successful Assets" in line:
            in_success_section = True

    in_detail = False
    success_runs = []
    for line in lines:
        if "## Enriched Assets" in line:
            in_detail = False
        if in_detail and line.startswith("|") and "|---" not in line:
            cells = line.split("|")
            if len(cells) >= 5 and cells[4].strip() == "success":
                asset_id = cells[1].strip()
                part_id = cells[2].strip() if cells[2].strip() else None
                joint_name = cells[3].strip()
                success_runs.append({
                    "asset_id": asset_id,
                    "part_id": part_id,
                    "joint_name": joint_name,
                })
        if "## Detailed Results" in line:
            in_detail = True

    return success_asset_ids, success_runs


def load_existing_result(asset_id, part_id):
    candidates = [TASK_ROOT / f"prismatic_joint_{asset_id}"]
    if part_id:
        candidates.append(TASK_ROOT / f"prismatic_joint_{asset_id}_part{part_id}")
    for result_dir in candidates:
        result_path = result_dir / "result.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
    return {
        "asset_id": asset_id,
        "tested_part_id": part_id,
        "status": "failure",
        "failure_reason": "missing_result_json",
        "save_root": str(candidates[0]),
        "video": str(candidates[0] / "manipulation.mp4"),
    }


def build_command(asset_id, part_id):
    return [
        sys.executable,
        "run.py",
        "--mode",
        "run_arti_open",
        "--headless",
        "--save_video",
        "--object_path",
        str(DATASET_DIR / asset_id),
        "--task_root",
        str(TASK_ROOT),
    ] + (["--part_id", str(part_id)] if part_id is not None else [])


def write_outputs(success_asset_ids, success_runs, results, failures):
    success_result_ids = sorted({str(r.get("asset_id")) for r in results if r.get("status") == "success"})
    failure_result_ids = sorted({str(r.get("asset_id")) for r in results if r.get("status") != "success"})

    payload = {
        "source_summary": str(SUMMARY_MD),
        "task_root": str(TASK_ROOT),
        "successful_assets_from_summary": len(success_asset_ids),
        "successful_runs_from_summary": len(success_runs),
        "tested": len(results),
        "success_count": len(success_result_ids),
        "failure_count": len(failure_result_ids),
        "results": results,
        "failures": failures,
        "success_assets": success_result_ids,
        "failure_assets": failure_result_ids,
    }
    AGGREGATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    AGGREGATE_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))

    vid_table = ""
    for r in results:
        asset = r.get("asset_id", "")
        status = r.get("status", "")
        delta = r.get("delta", "")
        video = r.get("video", "")
        label = "[:movie_camera:]" if (video and Path(video).exists()) else "[:x:]"
        vid_table += f"|{asset}|{status}|{delta}|{label} {video}|\n"

    markdown_lines = [
        "# Table Prismatic Success Re-test Summary",
        "",
        f"Source: `{SUMMARY_MD}`",
        f"Dataset: `{DATASET_DIR}`",
        "",
        f"Successful assets in original summary: {len(success_asset_ids)}",
        f"Successful test runs in original summary: {len(success_runs)}",
        f"Tested in re-test: {len(results)}",
        f"Success in re-test: {len(success_result_ids)}",
        f"Failure in re-test: {len(failure_result_ids)}",
        "",
        "## Re-test Results",
        "",
        "|asset|status|delta|video|",
        "|---|---|---|---|",
    ]
    for line in vid_table.split("\n"):
        if line.strip():
            markdown_lines.append(line)
    markdown_lines.extend([
        "",
        "## Output Files",
        f"- Aggregate JSON: `{AGGREGATE_JSON}`",
        f"- Task root: `{TASK_ROOT}`",
        "",
    ])
    SUMMARY_OUT.write_text("\n".join(markdown_lines))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    success_asset_ids, success_runs = parse_summary_markdown(SUMMARY_MD)
    print(f"[BATCH] Parsed {len(success_asset_ids)} successful asset IDs from summary")
    print(f"[BATCH] Parsed {len(success_runs)} successful test runs from Detailed Results")
    for run in success_runs[:5]:
        print(f"  {run['asset_id']} part={run['part_id']} {run['joint_name']}")
    if len(success_runs) > 5:
        print(f"  ... and {len(success_runs) - 5} more")

    if args.dry_run:
        for i, t in enumerate(success_runs, 1):
            cmd = build_command(str(t["asset_id"]), str(t["part_id"]) if t["part_id"] else None)
            print(f"[DRY] {i}/{len(success_runs)} {' '.join(cmd)}")
        return

    TASK_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []

    for index, target in enumerate(success_runs, 1):
        asset_id = str(target["asset_id"])
        part_id = target["part_id"]
        if part_id is not None:
            part_id = str(part_id)

        print(
            f"[BATCH] {index}/{len(success_runs)} asset={asset_id} part={part_id or 'auto'} "
            f"joint={target['joint_name']}",
            flush=True,
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = "/root/isaacgym/python"

        cmd = build_command(asset_id, part_id)
        start = time.time()
        proc = subprocess.run(cmd, cwd=str(MANIP_DIR), env=env)
        elapsed = time.time() - start

        result = load_existing_result(asset_id, part_id)
        result["returncode"] = proc.returncode
        result["elapsed_sec"] = elapsed
        result["planned_part_id"] = part_id
        result["planned_joint_name"] = target["joint_name"]

        if result.get("status") is None and result.get("delta") is not None:
            result["status"] = "success"

        if proc.returncode != 0 and result.get("status") == "success":
            result["status"] = "failure"
            result["failure_reason"] = "run_command_failed"

        results.append(result)

        if result.get("status") != "success":
            failures.append({
                "asset_id": asset_id,
                "part_id": part_id,
                "joint_name": target["joint_name"],
                "status": result.get("status"),
                "failure_reason": result.get("failure_reason"),
                "returncode": proc.returncode,
                "elapsed_sec": elapsed,
            })

        write_outputs(success_asset_ids, success_runs, results, failures)

        print(
            f"[BATCH] done asset={asset_id} part={part_id or 'auto'} "
            f"status={result.get('status')} "
            f"delta={result.get('delta')} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    payload = write_outputs(success_asset_ids, success_runs, results, failures)
    success_count = len(payload["success_assets"])
    failure_count = len(payload["failure_assets"])
    print(f"[BATCH] Complete: {success_count} success, {failure_count} failure")
    print(f"[BATCH] Aggregate JSON: {AGGREGATE_JSON}")
    print(f"[BATCH] Summary: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
