import json
from collections import Counter
from pathlib import Path


DATASET_RESULT = Path(
    "/Titan/dataset/data_gapartnet/partnet_mobility_mixedcabinet/"
    "mixedcabinet_other_prismatic_open_test_results.json"
)
OUTPUT_ROOT = Path(
    "/Titan/code/robohike_ws/src/3D-Diffusion-Policy/third_party/GAPartNet/"
    "manipulation/output"
)
SUMMARY_COPY = OUTPUT_ROOT / "mixedcabinet_other_prismatic_open_test_summary.json"


def build_summary(data):
    results = data.get("results", [])
    failures = []
    successes = []
    missing_videos = []
    for result in results:
        asset_id = str(result.get("asset_id"))
        video_path = OUTPUT_ROOT / f"partnet_mobility_mixedcabinet_{asset_id}" / "manipulation.mp4"
        result["video_exists"] = video_path.exists()
        if not video_path.exists():
            missing_videos.append(asset_id)
        if result.get("status") == "success":
            successes.append(asset_id)
        else:
            failures.append(
                {
                    "asset_id": asset_id,
                    "part_id": result.get("tested_part_id"),
                    "joint": result.get("joint_name"),
                    "delta": result.get("delta"),
                    "required_delta": result.get("required_delta"),
                    "gripper_on_handle": result.get("gripper_on_handle"),
                    "failure_reason": result.get("failure_reason"),
                    "video": str(video_path),
                    "video_exists": video_path.exists(),
                }
            )
    return {
        "completed_count": len(results),
        "target_count": data.get("target_count"),
        "status_counts": dict(Counter(result.get("status") for result in results)),
        "success_count": len(successes),
        "failure_count": len(failures),
        "success_assets": successes,
        "failure_assets": [failure["asset_id"] for failure in failures],
        "missing_videos": missing_videos,
        "failures": failures,
    }


def main():
    data = json.loads(DATASET_RESULT.read_text())
    data["summary"] = build_summary(data)
    DATASET_RESULT.write_text(json.dumps(data, indent=2, sort_keys=True))
    SUMMARY_COPY.write_text(json.dumps(data["summary"], indent=2, sort_keys=True))
    print(json.dumps(data["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
