#############
# code name: articualted object manipulation
# description: articualted object manipulation, we put several random object in 
#              the scene we control the fixed franka arm to manipulate the part 
#              on the GAPartNet object. we use the annotation from GAPartNet to 
#              get the part information. If you feel the code useful, please 
#              cite the following paper:
#
#              @article{geng2022gapartnet,
#                title={GAPartNet: Cross-Category Domain-Generalizable Object Perception and Manipulation via Generalizable and Actionable Parts},
#                author={Geng, Haoran and Xu, Helin and Zhao, Chengyang and Xu, Chao and Yi, Li and Huang, Siyuan and Wang, He},
#                journal={arXiv preprint arXiv:2211.05272},
#                year={2022}
#              }
#
#              @misc{geng2023sage,
#              title={SAGE: Bridging Semantic and Actionable Parts for GEneralizable Articulated-Object Manipulation under Language Instructions},
#              author={Haoran Geng and Songlin Wei and Congyue Deng and Bokui Shen and He Wang and Leonidas Guibas},
#              year={2023},
#              eprint={2312.01307},
#              archivePrefix={arXiv},
#              primaryClass={cs.RO}
#              }
#
#              @article{geng2023partmanip,
#              title={PartManip: Learning Cross-Category Generalizable Part Manipulation Policy from Point Cloud Observations},
#              author={Geng, Haoran and Li, Ziming and Geng, Yiran and Chen, Jiayi and Dong, Hao and Wang, He},
#              journal={arXiv preprint arXiv:2303.16958},
#              year={2023}
#              }
# code author: Haoran Geng
#############

from object_gym import ObjectGym
import numpy as np
from utils import read_yaml_config, prepare_gsam_model
import torch
import glob
import json
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os
import sys
import tqdm
import xml.etree.ElementTree as ET
import subprocess
import cv2
from isaacgym import gymapi, gymtorch, gymutil
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert

sys.path.append(sys.path[-1]+"/gym")
torch.set_printoptions(precision=4, sci_mode=False)

# Pre-parse --save_video, --save_video_frames and --object_id before gymutil sees sys.argv (gymutil
# spawns worker processes that call parse_arguments without custom_parameters).
_save_video = "--save_video" in sys.argv
if _save_video:
    sys.argv.remove("--save_video")

_save_video_frames = "--save_video_frames" in sys.argv or "--save_frames" in sys.argv
if "--save_video_frames" in sys.argv:
    sys.argv.remove("--save_video_frames")
if "--save_frames" in sys.argv:
    sys.argv.remove("--save_frames")

_object_id = None
if "--object_id" in sys.argv:
    idx = sys.argv.index("--object_id")
    _object_id = sys.argv[idx + 1]
    sys.argv.pop(idx)
    sys.argv.pop(idx)

_object_path = None
if "--object_path" in sys.argv:
    idx = sys.argv.index("--object_path")
    _object_path = sys.argv[idx + 1]
    sys.argv.pop(idx)
    sys.argv.pop(idx)

_joint_type = None
if "--joint_type" in sys.argv:
    idx = sys.argv.index("--joint_type")
    _joint_type = sys.argv[idx + 1]
    sys.argv.pop(idx)
    sys.argv.pop(idx)

_part_id = None
if "--part_id" in sys.argv:
    idx = sys.argv.index("--part_id")
    _part_id = int(sys.argv[idx + 1])
    sys.argv.pop(idx)
    sys.argv.pop(idx)

# load arguments
args = gymutil.parse_arguments(description="Placement",
    custom_parameters=[
        {"name": "--mode", "type": str, "default": ""},
        {"name": "--task_root", "type": str, "default": "output"},
        {"name": "--config", "type": str, "default": "config"},
        {"name": "--device", "type": str, "default": "cuda"},
        # headless
        {"name": "--headless", "action": 'store_true', "default": False},
        ])
args.save_video = _save_video
args.save_video_frames = _save_video_frames
args.object_id = _object_id
args.object_path = _object_path
args.part_id = _part_id
args.joint_type = _joint_type



def _parse_xyz(value, default=(0.0, 0.0, 0.0)):
    if value is None:
        return np.array(default, dtype=np.float32)
    parts = [float(x) for x in value.split()]
    return np.array(parts, dtype=np.float32)


def _transform_from_xyz_rpy(xyz, rpy):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R.from_euler("xyz", np.asarray(rpy, dtype=np.float32)).as_matrix().astype(np.float32)
    transform[:3, 3] = np.asarray(xyz, dtype=np.float32)
    return transform


def _joint_frame_in_asset(object_dir, joint_name):
    urdf_path = os.path.join(object_dir, "mobility_annotation_gapartnet.urdf")
    root = ET.parse(urdf_path).getroot()
    joints_by_name = {}
    fixed_joint_by_child = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        origin_el = joint.find("origin")
        xyz = _parse_xyz(origin_el.attrib.get("xyz") if origin_el is not None else None)
        rpy = _parse_xyz(origin_el.attrib.get("rpy") if origin_el is not None else None)
        record = {
            "name": name,
            "type": joint.attrib.get("type"),
            "parent": parent_el.attrib.get("link"),
            "child": child_el.attrib.get("link"),
            "xyz": xyz,
            "rpy": rpy,
        }
        joints_by_name[name] = record
        if record["type"] == "fixed":
            fixed_joint_by_child[record["child"]] = record

    def link_frame(link_name):
        if link_name is None or link_name == "base" or link_name not in fixed_joint_by_child:
            return np.eye(4, dtype=np.float32)
        fixed_joint = fixed_joint_by_child[link_name]
        return link_frame(fixed_joint["parent"]) @ _transform_from_xyz_rpy(fixed_joint["xyz"], fixed_joint["rpy"])

    joint = joints_by_name.get(joint_name)
    if joint is None:
        return np.eye(4, dtype=np.float32)
    return link_frame(joint["parent"]) @ _transform_from_xyz_rpy(joint["xyz"], joint["rpy"])


def _asset_frame_to_world(frame_asset, asset_position, asset_quat_xyzw, asset_scale):
    frame_asset = np.asarray(frame_asset, dtype=np.float32).copy()
    world = np.eye(4, dtype=np.float32)
    obj_rot = R.from_quat(np.asarray(asset_quat_xyzw, dtype=np.float32)).as_matrix().astype(np.float32)
    world[:3, :3] = obj_rot @ frame_asset[:3, :3]
    world[:3, 3] = np.asarray(asset_position, dtype=np.float32) + obj_rot @ (float(asset_scale) * frame_asset[:3, 3])
    return world


def _safe_normalize_np(vec, fallback=None):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        if fallback is None:
            fallback = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        return np.asarray(fallback, dtype=np.float32)
    return vec / norm



def _project_point_to_axis(point, axis_point, axis_dir):
    """Project a point onto an infinite 3D axis."""
    point = np.asarray(point, dtype=np.float32)
    axis_point = np.asarray(axis_point, dtype=np.float32)
    axis_dir = _safe_normalize_np(axis_dir, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return axis_point + np.dot(point - axis_point, axis_dir) * axis_dir



def _estimate_revolute_axis_point_from_bbox(handle_center, movable_bbox, axis_dir):
    """Estimate a hinge-axis point from the movable part bbox side farthest from the handle."""
    handle_center = np.asarray(handle_center, dtype=np.float32)
    bbox = np.asarray(movable_bbox, dtype=np.float32).reshape(-1, 3)
    axis_dir = _safe_normalize_np(axis_dir, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    center = np.mean(bbox, axis=0)
    rel = bbox - center
    # Choose the in-plane bbox direction with largest extent perpendicular to the hinge axis.
    rel_perp = rel - np.outer(rel @ axis_dir, axis_dir)
    _, _, vh = np.linalg.svd(rel_perp, full_matrices=False)
    candidates = []
    for direction in vh[:2]:
        direction = np.asarray(direction, dtype=np.float32)
        direction = direction - np.dot(direction, axis_dir) * axis_dir
        if np.linalg.norm(direction) < 1e-6:
            continue
        direction = _safe_normalize_np(direction)
        dots = rel @ direction
        low = np.mean(bbox[dots <= np.percentile(dots, 25)], axis=0)
        high = np.mean(bbox[dots >= np.percentile(dots, 75)], axis=0)
        candidates.extend([low, high])
    if not candidates:
        return center.astype(np.float32)
    # Hinge side is generally the bbox side farthest from the handle.
    best = max(candidates, key=lambda p: float(np.linalg.norm(np.asarray(p) - handle_center)))
    return np.asarray(best, dtype=np.float32)


def _compute_revolute_motion_geometry(handle_center, movable_center, axis_dir, approach_dir, axis_point=None):
    """Estimate hinge pivot/radius/tangent for a revolute target in world coordinates."""
    handle_center = np.asarray(handle_center, dtype=np.float32)
    movable_center = np.asarray(movable_center, dtype=np.float32)
    axis_dir = _safe_normalize_np(axis_dir, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    approach_dir = _safe_normalize_np(approach_dir, fallback=np.array([-1.0, 0.0, 0.0], dtype=np.float32))

    if axis_point is None:
        axis_point = movable_center
    pivot = _project_point_to_axis(handle_center, axis_point, axis_dir)
    radial = handle_center - pivot
    radial_norm = np.linalg.norm(radial)
    if radial_norm < 1e-4:
        radial = handle_center - movable_center
        radial = radial - np.dot(radial, axis_dir) * axis_dir
        radial_norm = np.linalg.norm(radial)
    radial_dir = _safe_normalize_np(radial, fallback=approach_dir)
    radius = max(float(radial_norm), 1e-4)

    tangent_dir = _safe_normalize_np(np.cross(axis_dir, radial_dir), fallback=approach_dir)
    if np.dot(tangent_dir, approach_dir) < 0:
        tangent_dir = -tangent_dir
        axis_dir = -axis_dir

    return {
        "pivot": pivot.astype(np.float32),
        "axis_dir": axis_dir.astype(np.float32),
        "radial_dir": radial_dir.astype(np.float32),
        "tangent_dir": tangent_dir.astype(np.float32),
        "radius": radius,
    }


def _orthonormal_frame_from_z(z_axis, x_hint=None):
    z_axis = _safe_normalize_np(z_axis, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    if x_hint is None:
        x_hint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    x_axis = np.asarray(x_hint, dtype=np.float32)
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis = _safe_normalize_np(x_axis, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    y_axis = _safe_normalize_np(np.cross(z_axis, x_axis), fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    x_axis = _safe_normalize_np(np.cross(y_axis, z_axis), fallback=x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)


def _compute_revolute_arc_targets(handle_center, grasp_offset, approach_dir, geom, angle_step, steps, direction_sign=1.0, start_angle=0.0):
    """Generate end-effector targets following the handle's circular revolute path."""
    steps = max(int(steps), 0)
    if steps == 0:
        return np.zeros((0, 3), dtype=np.float32)
    handle_center = np.asarray(handle_center, dtype=np.float32)
    approach_dir = _safe_normalize_np(approach_dir)
    pivot = np.asarray(geom["pivot"], dtype=np.float32)
    axis_dir = _safe_normalize_np(geom["axis_dir"], fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    radial_dir = _safe_normalize_np(geom["radial_dir"], fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    radius = float(geom["radius"])
    base_offset = float(grasp_offset) * approach_dir

    targets = []
    for step_i in range(steps):
        theta = float(start_angle) + (step_i + 1) * float(angle_step) * float(direction_sign)
        rot = R.from_rotvec(axis_dir * theta).as_matrix().astype(np.float32)
        rotated_radial = rot @ radial_dir
        targets.append(pivot + radius * rotated_radial + base_offset)
    return np.stack(targets, axis=0).astype(np.float32)

def _parse_urdf_joint_info(object_dir):
    """Parse enough URDF semantics to map fixed handles to movable joints."""
    urdf_path = os.path.join(object_dir, "mobility_annotation_gapartnet.urdf")
    root = ET.parse(urdf_path).getroot()
    fixed_parent = {}
    fixed_rotation = {}
    movable_joints_by_child = {}
    movable_joint_names = []

    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type")
        joint_name = joint.attrib.get("name")
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.attrib.get("link")
        child = child_el.attrib.get("link")
        origin_el = joint.find("origin")
        rpy = _parse_xyz(origin_el.attrib.get("rpy") if origin_el is not None else None)
        rot = R.from_euler("xyz", rpy).as_matrix().astype(np.float32)
        axis_el = joint.find("axis")
        axis = _parse_xyz(axis_el.attrib.get("xyz") if axis_el is not None else None, default=(0.0, 0.0, 1.0))

        if joint_type == "fixed":
            fixed_parent[child] = parent
            fixed_rotation[child] = rot
        elif joint_type is not None:
            movable_joint_names.append(joint_name)
            movable_joints_by_child[child] = {
                "name": joint_name,
                "type": joint_type,
                "parent": parent,
                "child": child,
                "axis": axis,
            }

    return fixed_parent, fixed_rotation, movable_joints_by_child, movable_joint_names


def _rotation_from_link_to_root(link_name, fixed_parent, fixed_rotation):
    """Return rotation that maps vectors in link_name frame into the fixed-chain root frame."""
    rot = np.eye(3, dtype=np.float32)
    current = link_name
    visited = set()
    while current in fixed_parent and current not in visited:
        visited.add(current)
        rot = fixed_rotation.get(current, np.eye(3, dtype=np.float32)) @ rot
        current = fixed_parent[current]
    return rot, current


def _resolve_controlling_joint(link_name, fixed_parent, fixed_rotation, movable_joints_by_child):
    """Walk fixed parents until finding the movable joint that controls the selected link."""
    current = link_name
    visited = set()
    while current and current not in visited:
        visited.add(current)
        if current in movable_joints_by_child:
            joint = dict(movable_joints_by_child[current])
            # URDF joint axis is expressed in the joint/parent-link frame for these assets.
            # Transform through the fixed parent chain of the joint parent, not the movable child.
            rot_to_root, _ = _rotation_from_link_to_root(joint.get("parent"), fixed_parent, fixed_rotation)
            joint["axis_root"] = _safe_normalize_np(rot_to_root @ joint["axis"], fallback=joint["axis"])
            joint["movable_link"] = current
            return joint
        current = fixed_parent.get(current)
    return None


def _same_resolved_joint(a, b):
    if a is None or b is None:
        return False
    return (
        a.get("name") == b.get("name")
        and a.get("type") == b.get("type")
        and a.get("movable_link") == b.get("movable_link")
    )


def _select_bbox_for_requested_part(
    requested_part_id,
    gapart_anno,
    gapart_raw_valid_anno,
    fixed_parent,
    fixed_rotation,
    movable_joints_by_child,
):
    """Map a requested movable part id to the fixed handle that controls it."""
    raw_to_valid_bbox_id = {
        raw_i: valid_i
        for valid_i, raw_i in enumerate([i for i, anno in enumerate(gapart_anno) if anno.get("is_gapart")])
    }
    selected_bbox_id = None
    selected_source = None
    if requested_part_id is not None:
        if requested_part_id in raw_to_valid_bbox_id:
            selected_bbox_id = raw_to_valid_bbox_id[requested_part_id]
            selected_source = "raw_part_id"
        else:
            selected_bbox_id = requested_part_id
            selected_source = "valid_bbox_id"
    if selected_bbox_id is None:
        selected_bbox_id = _select_highest_fixed_handle(gapart_raw_valid_anno)
        selected_source = "highest_fixed_handle"
    if selected_bbox_id is None:
        return -1, selected_source, None
    if selected_bbox_id < 0 or selected_bbox_id >= len(gapart_raw_valid_anno):
        return selected_bbox_id, selected_source, None

    selected_anno = gapart_raw_valid_anno[selected_bbox_id]
    selected_joint = _resolve_controlling_joint(
        selected_anno.get("link_name", ""),
        fixed_parent,
        fixed_rotation,
        movable_joints_by_child,
    )
    selected_category = selected_anno.get("category", "")
    if selected_category.endswith("fixed_handle") or selected_joint is None:
        return selected_bbox_id, selected_source, selected_joint

    # Batch plans address movable parts, but manipulation must grasp the handle.
    # Choose the fixed handle attached through fixed joints to the same movable link.
    for anno_i, anno in enumerate(gapart_raw_valid_anno):
        if not anno.get("category", "").endswith("fixed_handle"):
            continue
        handle_joint = _resolve_controlling_joint(
            anno.get("link_name", ""),
            fixed_parent,
            fixed_rotation,
            movable_joints_by_child,
        )
        if _same_resolved_joint(selected_joint, handle_joint):
            return anno_i, f"{selected_source}_mapped_to_handle", handle_joint
    return selected_bbox_id, selected_source, selected_joint


def _select_highest_fixed_handle(gapart_raw_valid_anno):
    best_idx = None
    best_z = -float("inf")
    for idx, anno in enumerate(gapart_raw_valid_anno):
        category = anno.get("category", "")
        if not category.endswith("fixed_handle"):
            continue
        bbox = np.asarray(anno.get("bbox", []), dtype=np.float32)
        if bbox.size == 0:
            continue
        center_z = float(np.mean(bbox[:, 2]))
        if center_z > best_z:
            best_z = center_z
            best_idx = idx
    return best_idx


def _compute_pull_targets(init_position, approach_dir, pull_dir, grasp_offset, pull_step, pull_steps):
    """Generate post-grasp pull targets without an initial jump away from grasp."""
    init_position = np.asarray(init_position, dtype=np.float32)
    approach_dir = _safe_normalize_np(approach_dir)
    pull_dir = _safe_normalize_np(pull_dir, fallback=approach_dir)
    grasp_position = init_position + grasp_offset * approach_dir
    return np.stack(
        [grasp_position + (step_i + 1) * pull_step * pull_dir for step_i in range(pull_steps)],
        axis=0,
    )


def _compute_revolute_grasp_hold_targets(handle_center, grasp_offset, geom, tangent_step, steps, direction_sign=1.0):
    """Generate a short tangential pull that starts from the handle grasp pose."""
    geom = dict(geom or {})
    approach_dir = _safe_normalize_np(geom.get("approach_dir", np.array([-1.0, 0.0, 0.0], dtype=np.float32)), fallback=np.array([-1.0, 0.0, 0.0], dtype=np.float32))
    tangent_dir = _safe_normalize_np(geom.get("tangent_dir", np.array([0.0, 1.0, 0.0], dtype=np.float32)), fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    grasp_position = np.asarray(handle_center, dtype=np.float32) + float(grasp_offset) * approach_dir
    tangent_dir = _safe_normalize_np(direction_sign * tangent_dir, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    return np.stack(
        [grasp_position + (step_i + 1) * float(tangent_step) * tangent_dir for step_i in range(int(steps))],
        axis=0,
    )


def _compute_revolute_grasp_hold_arc_targets(handle_center, grasp_offset, geom, hold_steps, angle_step, arc_steps, direction_sign=1.0):
    """Hold the grasp, then move through continuous circular breakaway and arc targets."""
    geom = dict(geom or {})
    hold_steps = max(int(hold_steps), 0)
    arc_steps = max(int(arc_steps), 0)
    approach_dir = _safe_normalize_np(
        geom.get("approach_dir", np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
        fallback=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    )
    hold_position = np.asarray(handle_center, dtype=np.float32) + float(grasp_offset) * approach_dir
    hold_targets = (
        np.stack([hold_position for _ in range(hold_steps)], axis=0).astype(np.float32)
        if hold_steps > 0
        else np.zeros((0, 3), dtype=np.float32)
    )

    breakaway_steps = max(int(geom.get("breakaway_steps", hold_steps)), 0)
    radius = abs(float(geom.get("radius", 0.0)))
    breakaway_step = float(
        geom.get(
            "breakaway_step",
            max(0.008, 0.5 * abs(float(grasp_offset)) if float(grasp_offset) != 0.0 else 0.008),
        )
    )
    breakaway_angle_step = breakaway_step / radius if radius > 1e-6 else float(angle_step)
    breakaway_targets = _compute_revolute_arc_targets(
        handle_center,
        grasp_offset,
        approach_dir,
        geom,
        breakaway_angle_step,
        breakaway_steps,
        direction_sign=direction_sign,
        start_angle=0.0,
    )
    arc_start_angle = breakaway_steps * float(breakaway_angle_step) * float(direction_sign)
    arc_targets = _compute_revolute_arc_targets(
        handle_center,
        grasp_offset,
        approach_dir,
        geom,
        angle_step,
        arc_steps,
        direction_sign=direction_sign,
        start_angle=arc_start_angle,
    )
    pieces = [piece for piece in (hold_targets, breakaway_targets, arc_targets) if piece.size > 0]
    if not pieces:
        return np.zeros((0, 3), dtype=np.float32)
    if len(pieces) == 1:
        return pieces[0]
    return np.concatenate(pieces, axis=0).astype(np.float32)


def _revolute_signed_angles_for_targets(targets, grasp_offset, geom):
    """Return signed target angles in the local revolute frame."""
    geom = dict(geom or {})
    targets = np.asarray(targets, dtype=np.float32).reshape(-1, 3)
    if targets.size == 0:
        return np.zeros((0,), dtype=np.float32)
    approach_dir = _safe_normalize_np(
        geom.get("approach_dir", np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
        fallback=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    )
    pivot = np.asarray(geom.get("pivot", np.zeros(3, dtype=np.float32)), dtype=np.float32)
    radial_dir = _safe_normalize_np(geom.get("radial_dir"), fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    tangent_dir = _safe_normalize_np(geom.get("tangent_dir"), fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    handle_points = targets - float(grasp_offset) * approach_dir
    rel = handle_points - pivot
    x = rel @ radial_dir
    y = rel @ tangent_dir
    return np.unwrap(np.arctan2(y, x)).astype(np.float32)


def _compute_revolute_target_sequence_diagnostics(
    handle_center,
    grasp_offset,
    geom,
    targets,
    hold_steps,
    breakaway_steps,
    arc_steps,
    direction_sign=1.0,
):
    """Summarize target sequence continuity before executing it."""
    targets = np.asarray(targets, dtype=np.float32).reshape(-1, 3)
    hold_steps = max(int(hold_steps), 0)
    breakaway_steps = max(int(breakaway_steps), 0)
    arc_steps = max(int(arc_steps), 0)
    approach_dir = _safe_normalize_np(
        geom.get("approach_dir", np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
        fallback=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    )
    grasp_position = np.asarray(handle_center, dtype=np.float32) + float(grasp_offset) * approach_dir
    adjacent_distances = (
        np.linalg.norm(np.diff(targets, axis=0), axis=1)
        if len(targets) > 1
        else np.zeros((0,), dtype=np.float32)
    )
    moving_targets = targets[hold_steps:] if hold_steps < len(targets) else np.zeros((0, 3), dtype=np.float32)
    angles = _revolute_signed_angles_for_targets(moving_targets, grasp_offset, geom)
    signed_step_progress = np.diff(np.concatenate(([0.0], angles))).astype(np.float32) if len(angles) else np.zeros((0,), dtype=np.float32)
    direction = 1.0 if float(direction_sign) >= 0 else -1.0
    monotonic = bool(np.all(direction * signed_step_progress >= -1e-5)) if len(signed_step_progress) else True
    first_non_hold_distance = (
        float(np.linalg.norm(moving_targets[0] - grasp_position))
        if len(moving_targets)
        else 0.0
    )
    transition_index = hold_steps + breakaway_steps - 1
    if breakaway_steps > 0 and arc_steps > 0 and 0 <= transition_index < len(targets) - 1:
        breakaway_to_arc = float(np.linalg.norm(targets[transition_index + 1] - targets[transition_index]))
    else:
        breakaway_to_arc = 0.0
    return {
        "grasp_target": grasp_position.astype(float).tolist(),
        "hold_target_count": hold_steps,
        "breakaway_target_count": breakaway_steps,
        "arc_target_count": arc_steps,
        "first_pull_target": targets[0].astype(float).tolist() if len(targets) else None,
        "final_pull_target": targets[-1].astype(float).tolist() if len(targets) else None,
        "max_adjacent_target_distance": float(np.max(adjacent_distances)) if len(adjacent_distances) else 0.0,
        "breakaway_to_arc_transition_distance": breakaway_to_arc,
        "first_non_hold_distance_from_grasp": first_non_hold_distance,
        "arc_angle_monotonic": monotonic,
        "signed_angular_progress": {
            "direction_sign": float(direction_sign),
            "first": float(angles[0]) if len(angles) else 0.0,
            "final": float(angles[-1]) if len(angles) else 0.0,
            "min_step": float(np.min(signed_step_progress)) if len(signed_step_progress) else 0.0,
            "max_step": float(np.max(signed_step_progress)) if len(signed_step_progress) else 0.0,
        },
    }


def _legacy_open_demo_targets(init_position, handle_out):
    """Return the c8d4ad2 run_arti_open single-path targets."""
    init_position = np.asarray(init_position, dtype=np.float32)
    handle_out = np.asarray(handle_out, dtype=np.float32)
    pre_grasp_position = init_position + 0.2 * handle_out
    grasp_position = init_position + 0.1 * handle_out
    pull_targets = np.asarray(
        [init_position + (0.1 + i * 0.01) * handle_out for i in range(30)],
        dtype=np.float32,
    )
    return pre_grasp_position, grasp_position, pull_targets


def _pose_z_for_support_surface_height(min_z, scale=0.4, support_surface_z=0.43):
    return float(support_surface_z) - float(scale) * float(min_z)


def _support_surface_z_for_asset(gapart_id):
    return 0.43 if str(gapart_id) == "27044" else 0.0


def _get_arti_dof_positions(gym):
    """Read articulated-object DOF positions from the global DOF tensor."""
    gym.refresh_observation(get_visual_obs=False)
    start = gym.franka_num_dofs + gym.obj_num_dofs
    end = start + gym.arti_obj_num_dofs
    arti_dof_pos = gym.dof_pos[:, start:end, 0]
    if hasattr(arti_dof_pos, "detach"):
        arti_dof_pos = arti_dof_pos.detach().cpu().numpy()
    return np.asarray(arti_dof_pos, dtype=np.float32)


def _closed_arti_dof_state_command(dof_states, franka_num_dofs, obj_num_dofs, arti_lower):
    """Return DOF states with articulated-object positions reset to closed limits."""
    updated = dof_states.copy() if hasattr(dof_states, "copy") else dof_states.clone()
    start = int(franka_num_dofs) + int(obj_num_dofs)
    lower_np = np.asarray(arti_lower, dtype=np.float32)
    end = start + lower_np.size
    if hasattr(updated, "new_tensor"):
        lower = updated.new_tensor(lower_np)
    else:
        lower = lower_np
    updated[:, start:end, 0] = lower
    updated[:, start:end, 1] = 0.0
    return updated


def _reset_arti_dofs_to_closed(gym):
    """Force articulated-object DOFs closed after simulator warmup drift."""
    if not hasattr(gym, "arti_obj_dof_props") or gym.arti_obj_num_dofs <= 0:
        return None
    gym.refresh_observation(get_visual_obs=False)
    lower = np.asarray(gym.arti_obj_dof_props["lower"], dtype=np.float32)
    dof_view = gym.dof_states.view(gym.num_envs, -1, 2)
    closed = _closed_arti_dof_state_command(
        dof_view,
        franka_num_dofs=gym.franka_num_dofs,
        obj_num_dofs=gym.obj_num_dofs,
        arti_lower=lower,
    )
    dof_view[:, :, :] = closed
    gym.gym.set_dof_state_tensor(gym.sim, gymtorch.unwrap_tensor(gym.dof_states))
    gym.gym.refresh_dof_state_tensor(gym.sim)
    gym.refresh_observation(get_visual_obs=False)
    return lower.copy()


def _select_target_dof_index(joint_desc, movable_joint_names):
    if joint_desc is None or not joint_desc.get("name"):
        return None
    try:
        return list(movable_joint_names).index(joint_desc["name"])
    except ValueError:
        return None


def _format_arti_dof_diag(label, dof_pos, joint_desc=None, initial=None, target_dof_index=None):
    """Format concise articulated DOF diagnostics, including delta from initial state."""
    dof_pos = np.asarray(dof_pos, dtype=np.float32)
    parts = [f"[DIAG] arti dof {label}: pos={np.array2string(dof_pos, precision=4)}"]
    if initial is not None:
        delta = dof_pos - np.asarray(initial, dtype=np.float32)
        parts.append(f"delta={np.array2string(delta, precision=4)}")
    if joint_desc is not None and joint_desc.get("name"):
        parts.append(f"target_joint={joint_desc.get('name')}")
    if target_dof_index is not None and dof_pos.size:
        flat_pos = dof_pos.reshape(dof_pos.shape[0], -1)
        target_pos = flat_pos[:, target_dof_index]
        parts.append(f"target_pos={np.array2string(target_pos, precision=4)}")
        if initial is not None:
            flat_initial = np.asarray(initial, dtype=np.float32).reshape(dof_pos.shape[0], -1)
            target_delta = target_pos - flat_initial[:, target_dof_index]
            parts.append(f"target_delta={np.array2string(target_delta, precision=4)}")
    return " ".join(parts)



def _target_abs_delta(dof_pos, initial, target_dof_index):
    """Return absolute target DOF displacement for env0, or NaN when unavailable."""
    if target_dof_index is None or initial is None:
        return float("nan")
    dof_pos = np.asarray(dof_pos, dtype=np.float32).reshape(np.asarray(dof_pos).shape[0], -1)
    initial = np.asarray(initial, dtype=np.float32).reshape(np.asarray(initial).shape[0], -1)
    if dof_pos.size == 0 or target_dof_index >= dof_pos.shape[1]:
        return float("nan")
    return float(abs(dof_pos[0, target_dof_index] - initial[0, target_dof_index]))


def _format_ee_tracking_diag(label, gym, target_position):
    """Format end-effector tracking error against a Cartesian target."""
    gym.refresh_observation(get_visual_obs=False)
    target_position = np.asarray(target_position, dtype=np.float32)
    ee_pos = gym.hand_pos[0].detach().cpu().numpy() if hasattr(gym.hand_pos[0], "detach") else np.asarray(gym.hand_pos[0])
    err_vec = target_position - ee_pos
    return (
        f"[DIAG] ee {label}: target={np.array2string(target_position, precision=4)} "
        f"actual={np.array2string(ee_pos, precision=4)} "
        f"err_norm={np.linalg.norm(err_vec):.4f}"
    )


def _make_prismatic_grasp_candidates(grasp_offset, approach_dir, handle_long=None, handle_short=None):
    """Small closed-state-aware candidate set, nominal first to preserve existing behavior."""
    candidates = [{"label": "nominal", "grasp_offset": float(grasp_offset), "bias": np.zeros(3, dtype=np.float32)}]
    approach_dir = _safe_normalize_np(approach_dir)
    short = _safe_normalize_np(handle_short, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32)) if handle_short is not None else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    long = _safe_normalize_np(handle_long, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32)) if handle_long is not None else np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Closed drawers may put the handle flush with the panel.  Try closer/deeper
    # approach offsets and tiny in-plane biases; nominal remains candidate 0.
    for off in (0.04, 0.02, 0.00, -0.01):
        candidates.append({"label": f"approach_offset_{off:.2f}", "grasp_offset": off, "bias": np.zeros(3, dtype=np.float32)})
    for label, bias in (
        ("short_pos", 0.018 * short),
        ("short_neg", -0.018 * short),
        ("long_pos", 0.018 * long),
        ("long_neg", -0.018 * long),
        ("deeper_short_pos", 0.018 * short),
        ("deeper_short_neg", -0.018 * short),
    ):
        off = 0.02 if not label.startswith("deeper") else 0.00
        candidates.append({"label": label, "grasp_offset": off, "bias": np.asarray(bias, dtype=np.float32)})
    return candidates



def _apply_assisted_revolute_motion(gym, dof_initial, target_dof_index, lower, upper, steps=30, delta=0.35, save_video=False, save_root=None, step_num=0):
    """Fallback: directly advance revolute DOF when contact-based attempt fails."""
    gym.refresh_observation(get_visual_obs=False)
    pos_action = gym.dof_pos.squeeze(-1).clone()
    start = gym.franka_num_dofs + gym.obj_num_dofs
    dof_col = start + int(target_dof_index)
    initial_value = float(np.asarray(dof_initial).reshape(np.asarray(dof_initial).shape[0], -1)[0, target_dof_index])
    lower = float(lower[target_dof_index]) if target_dof_index is not None else initial_value
    upper = float(upper[target_dof_index]) if target_dof_index is not None else initial_value + delta
    goal = min(max(initial_value + delta, lower), upper)
    if abs(goal - initial_value) < 1e-4:
        goal = min(max(initial_value - delta, lower), upper)
    for i, value in enumerate(np.linspace(initial_value, goal, steps, dtype=np.float32)):
        # Directly animate the articulated DOF state.  The articulated asset is
        # passive (DOF_MODE_NONE), so position targets alone do not move it.
        gym.dof_states.view(gym.num_envs, -1, 2)[:, dof_col, 0] = float(value)
        gym.dof_states.view(gym.num_envs, -1, 2)[:, dof_col, 1] = 0.0
        gym.gym.set_dof_state_tensor(gym.sim, gymtorch.unwrap_tensor(gym.dof_states))
        gym.gym.refresh_dof_state_tensor(gym.sim)
        gym.gym.refresh_rigid_body_state_tensor(gym.sim)
        gym.run_steps(pre_steps=1)
        if save_video:
            gym.gym.render_all_camera_sensors(gym.sim)
            step_str = str(step_num + i).zfill(4)
            gym.record_camera_frame(save_root, step_num + i)
    return step_num + steps, goal


def _probe_video_metadata(video_path):
    metadata = {
        "path": video_path,
        "exists": bool(video_path and os.path.exists(video_path)),
        "probe_warnings": [],
    }
    if not metadata["exists"]:
        return metadata
    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,nb_frames,r_frame_rate",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        video_path,
    ]
    try:
        completed = subprocess.run(
            ffprobe_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        probed = json.loads(completed.stdout)
        streams = probed.get("streams") or []
        stream = streams[0] if streams else {}
        fmt = probed.get("format") or {}
        metadata.update({
            "codec": stream.get("codec_name"),
            "width": int(stream["width"]) if stream.get("width") is not None else None,
            "height": int(stream["height"]) if stream.get("height") is not None else None,
            "nb_frames": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
            "r_frame_rate": stream.get("r_frame_rate"),
            "duration": float(fmt["duration"]) if fmt.get("duration") is not None else None,
            "size": int(fmt["size"]) if fmt.get("size") is not None else None,
            "probe": "ffprobe",
        })
        return metadata
    except Exception as exc:
        metadata["probe_warnings"].append(f"ffprobe_failed: {exc}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        metadata["probe_warnings"].append("opencv_videocapture_failed")
        return metadata
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    metadata.update({
        "nb_frames": frame_count if frame_count >= 0 else None,
        "fps": fps if fps > 0 else None,
        "duration": (frame_count / fps) if frame_count >= 0 and fps > 0 else None,
        "width": width if width > 0 else None,
        "height": height if height > 0 else None,
        "probe": "opencv",
    })
    return metadata


def _frame_sequence_metadata(save_root, extension="png"):
    video_dir = os.path.join(save_root, "video")
    pattern = os.path.join(video_dir, f"step-*.{extension}")
    frames = sorted(glob.glob(pattern))
    metadata = {
        "directory": video_dir,
        "extension": extension,
        "pattern": pattern,
        "count": len(frames),
        "first_frame": frames[0] if frames else None,
        "last_frame": frames[-1] if frames else None,
    }
    if frames:
        first = cv2.imread(frames[0])
        if first is not None:
            metadata["height"], metadata["width"] = [int(v) for v in first.shape[:2]]
    return metadata


def _legacy_output_baseline_metadata(path="output/45661"):
    video_path = os.path.join(path, "manipulation.mp4")
    result_path = os.path.join(path, "result.json")
    png_frames = _frame_sequence_metadata(path, extension="png")
    jpg_frames = _frame_sequence_metadata(path, extension="jpg")
    return {
        "baseline_success_standard": "old_video",
        "baseline_video": video_path,
        "baseline_result_json": result_path,
        "baseline_result_json_exists": os.path.exists(result_path),
        "baseline_video_metadata": _probe_video_metadata(video_path),
        "baseline_png_frame_metadata": png_frames,
        "baseline_jpg_frame_metadata": jpg_frames,
        "joint_identity_inferred": False,
        "notes": "Legacy video is visual baseline evidence; without result.json it is not joint identity evidence.",
    }


def _write_video_mp4_from_frames(save_root, output_name="manipulation.mp4", fps=30):
    """Encode all saved PNG frames into an mp4 so the video covers the full manipulation."""
    video_dir = os.path.join(save_root, "video")
    frame_metadata = _frame_sequence_metadata(save_root, extension="png")
    result = {
        "path": os.path.join(save_root, output_name),
        "writer": None,
        "fps": fps,
        "frame_metadata": frame_metadata,
        "video_metadata": None,
        "warnings": [],
    }
    if not os.path.isdir(video_dir):
        print(f"[DIAG] skip mp4: no frame directory {video_dir}")
        result["warnings"].append(f"missing_frame_directory: {video_dir}")
        return result
    output_path = os.path.join(save_root, output_name)
    frame_pattern = os.path.join(video_dir, "step-*.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        frame_pattern,
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"[DIAG] wrote complete mp4: {output_path}")
        result["writer"] = "ffmpeg"
        result["video_metadata"] = _probe_video_metadata(output_path)
        return result
    except Exception as exc:
        print(f"[DIAG] ffmpeg mp4 encode failed, trying cv2 fallback: {exc}")
        result["warnings"].append(f"ffmpeg_failed: {exc}")
    frames = sorted(glob.glob(frame_pattern))
    if not frames:
        print(f"[DIAG] skip cv2 mp4 fallback: no frames matched {frame_pattern}")
        result["warnings"].append(f"no_frames_matched: {frame_pattern}")
        return result
    first = cv2.imread(frames[0])
    if first is None:
        print(f"[DIAG] skip cv2 mp4 fallback: failed to read {frames[0]}")
        result["warnings"].append(f"failed_to_read_first_frame: {frames[0]}")
        return result
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        print(f"[DIAG] skip cv2 mp4 fallback: VideoWriter failed for {output_path}")
        result["warnings"].append(f"opencv_videowriter_failed: {output_path}")
        return result
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()
    print(f"[DIAG] wrote complete mp4 with cv2 fallback: {output_path}")
    result["writer"] = "opencv_mp4v"
    result["video_metadata"] = _probe_video_metadata(output_path)
    return result


def _write_video_mp4_from_frame_list(save_root, frames, output_name="manipulation.mp4", fps=30):
    """Encode in-memory BGR/RGB frames to mp4 without requiring PNG sidecars."""
    output_path = os.path.join(save_root, output_name)
    frame_count = len(frames) if frames is not None else 0
    result = {
        "path": output_path,
        "writer": None,
        "fps": fps,
        "frame_metadata": {"count": frame_count, "source": "memory"},
        "video_metadata": None,
        "warnings": [],
    }
    if frame_count == 0:
        result["warnings"].append("no_frames_provided")
        return result
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] < 3:
        result["warnings"].append(f"invalid_first_frame_shape: {first.shape}")
        return result
    height, width = first.shape[:2]
    result["frame_metadata"].update({"height": int(height), "width": int(width)})
    os.makedirs(save_root, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "mpeg4",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        written = 0
        try:
            for frame in frames:
                frame = np.asarray(frame)
                if frame.ndim != 3 or frame.shape[2] < 3:
                    result["warnings"].append(f"skipped_invalid_frame_shape: {frame.shape}")
                    continue
                frame = frame[:, :, :3]
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
                written += 1
            process.stdin.close()
        except Exception:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            process.kill()
            process.wait()
            raise
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg exited with {returncode}")
        result["writer"] = "ffmpeg_rawvideo"
        result["frame_metadata"]["count"] = written
        result["video_metadata"] = _probe_video_metadata(output_path)
        if result["video_metadata"].get("nb_frames", 0) and result["video_metadata"].get("width"):
            print(f"[DIAG] wrote complete mp4 from memory with ffmpeg: {output_path}")
            return result
        result["warnings"].append("ffmpeg_rawvideo_unreadable_output")
    except Exception as exc:
        result["warnings"].append(f"ffmpeg_rawvideo_failed: {exc}")

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        result["warnings"].append(f"opencv_videowriter_failed: {output_path}")
        return result
    written = 0
    for frame in frames:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] < 3:
            result["warnings"].append(f"skipped_invalid_frame_shape: {frame.shape}")
            continue
        frame = frame[:, :, :3]
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        writer.write(frame)
        written += 1
    writer.release()
    result["writer"] = "opencv_mp4v"
    result["frame_metadata"]["count"] = written
    result["video_metadata"] = _probe_video_metadata(output_path)
    if not result["video_metadata"].get("nb_frames"):
        result["warnings"].append("opencv_mp4v_unreadable_output")
    print(f"[DIAG] wrote complete mp4 from memory: {output_path}")
    return result


def _required_success_delta(joint_desc, lower=None, upper=None):
    """Return the requested per-joint success threshold."""
    if joint_desc is None:
        return float("nan")
    joint_type = joint_desc.get("type")
    if joint_type in {"revolute", "continuous"}:
        return float(np.deg2rad(30.0))
    if joint_type == "prismatic" and lower is not None and upper is not None:
        return float(0.5 * abs(float(upper) - float(lower)))
    return float("nan")


def _transform_handle_bbox_for_final_joint(
    handle_bbox,
    joint_desc,
    initial_value,
    final_value,
    prismatic_dir=None,
    revolute_axis_point=None,
):
    bbox = np.asarray(handle_bbox, dtype=np.float32).reshape(-1, 3)
    if joint_desc is None or initial_value is None or final_value is None:
        return bbox
    signed_delta = float(final_value) - float(initial_value)
    joint_type = joint_desc.get("type")
    if joint_type == "prismatic":
        axis_dir = _safe_normalize_np(prismatic_dir if prismatic_dir is not None else joint_desc.get("axis_root"))
        return bbox + signed_delta * axis_dir
    if joint_type in {"revolute", "continuous"}:
        axis_dir = _safe_normalize_np(joint_desc.get("axis_root"))
        axis_point = np.asarray(revolute_axis_point if revolute_axis_point is not None else np.mean(bbox, axis=0), dtype=np.float32)
        rotation = R.from_rotvec(signed_delta * axis_dir).as_matrix().astype(np.float32)
        return (bbox - axis_point) @ rotation.T + axis_point
    return bbox


def _tensor_pos_to_np(pos):
    if hasattr(pos, "detach"):
        return pos.detach().cpu().numpy()
    return np.asarray(pos)


def _segment_intersects_expanded_bbox(point_a, point_b, bbox_min, bbox_max, samples=9):
    point_a = np.asarray(point_a, dtype=np.float32)
    point_b = np.asarray(point_b, dtype=np.float32)
    for alpha in np.linspace(0.0, 1.0, int(samples), dtype=np.float32):
        point = (1.0 - alpha) * point_a + alpha * point_b
        if np.all(point >= bbox_min) and np.all(point <= bbox_max):
            return True
    return False


def _gripper_handle_metrics(gym, handle_bbox, margin=0.035, hand_margin=0.18):
    """Measure whether the Franka fingers, not just the palm, are on the handle."""
    gym.refresh_observation(get_visual_obs=False)
    bbox = np.asarray(handle_bbox, dtype=np.float32).reshape(-1, 3)
    bbox_min = np.min(bbox, axis=0)
    bbox_max = np.max(bbox, axis=0)
    expanded_min = bbox_min - float(margin)
    expanded_max = bbox_max + float(margin)
    hand_expanded_min = bbox_min - float(hand_margin)
    hand_expanded_max = bbox_max + float(hand_margin)

    hand_pos = _tensor_pos_to_np(gym.hand_pos[0])
    left_pos = _tensor_pos_to_np(gym.leftfinger_pos[0]) if hasattr(gym, "leftfinger_pos") else hand_pos
    right_pos = _tensor_pos_to_np(gym.rightfinger_pos[0]) if hasattr(gym, "rightfinger_pos") else hand_pos

    left_inside = bool(np.all(left_pos >= expanded_min) and np.all(left_pos <= expanded_max))
    right_inside = bool(np.all(right_pos >= expanded_min) and np.all(right_pos <= expanded_max))
    segment_hits = _segment_intersects_expanded_bbox(left_pos, right_pos, expanded_min, expanded_max)
    hand_near = bool(np.all(hand_pos >= hand_expanded_min) and np.all(hand_pos <= hand_expanded_max))
    finger_midpoint = 0.5 * (left_pos + right_pos)
    bbox_center = np.mean(bbox, axis=0)
    metrics = {
        "hand_pos": hand_pos.tolist(),
        "leftfinger_pos": left_pos.tolist(),
        "rightfinger_pos": right_pos.tolist(),
        "finger_midpoint": finger_midpoint.tolist(),
        "bbox_center": bbox_center.tolist(),
        "left_inside": left_inside,
        "right_inside": right_inside,
        "finger_segment_intersects": bool(segment_hits),
        "hand_near": hand_near,
        "finger_midpoint_distance": float(np.linalg.norm(finger_midpoint - bbox_center)),
        "hand_distance": float(np.linalg.norm(hand_pos - bbox_center)),
    }
    metrics["on_handle"] = bool(hand_near and (left_inside or right_inside or segment_hits))
    return metrics


def _is_gripper_on_handle(gym, handle_bbox, margin=0.035, hand_margin=0.18):
    return bool(_gripper_handle_metrics(gym, handle_bbox, margin=margin, hand_margin=hand_margin)["on_handle"])


def _write_attempt_result(save_root, result):
    result_path = os.path.join(save_root, "result.json")
    os.makedirs(save_root, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"[DIAG] wrote result json: {result_path}")
    return result_path


def _discard_video_frames_from(save_root, start_step, frame_buffer=None):
    video_dir = os.path.join(save_root, "video")
    if frame_buffer is not None:
        del frame_buffer[int(start_step):]
    if not os.path.isdir(video_dir):
        return 0
    removed = 0
    for frame_path in glob.glob(os.path.join(video_dir, "step-*.png")):
        name = os.path.basename(frame_path)
        try:
            step = int(os.path.splitext(name)[0].split("-")[-1])
        except ValueError:
            continue
        if step >= int(start_step):
            os.remove(frame_path)
            removed += 1
    return removed


def _reset_video_capture_buffer(gym, remove_disk_frames=False):
    removed = 0
    if hasattr(gym, "video_frames"):
        del gym.video_frames[:]
    if remove_disk_frames:
        removed = _discard_video_frames_from(gym.save_root, 0, None)
    return removed


def _control_to_pose_repeated_ik(
    gym,
    pose,
    repeats,
    close_gripper,
    save_video,
    save_root,
    step_num,
    record_last_repeat_only=True,
):
    """Repeat IK servoing for convergence without recording every correction pass."""
    repeats = max(1, int(repeats))
    for repeat_i in range(repeats):
        record = bool(save_video and (not record_last_repeat_only or repeat_i + 1 == repeats))
        next_step, traj = gym.control_to_pose(
            pose,
            close_gripper=close_gripper,
            save_video=record,
            save_root=save_root,
            step_num=step_num,
            use_ik=True,
        )
        if record:
            step_num = next_step
    return step_num, None


def _normalize_requested_joint_type(joint_type):
    if joint_type is None or joint_type == "":
        return None
    normalized = str(joint_type).strip().lower()
    if normalized == "primismatic":
        normalized = "prismatic"
    if normalized not in {"prismatic", "revolute"}:
        raise ValueError(f"Unsupported --joint_type {joint_type!r}; expected prismatic or revolute")
    return normalized


def _should_run_legacy_open_demo(requested_joint_type):
    return requested_joint_type is None


def _make_control_profile_metadata(requested_joint_type, legacy_demo_executed):
    control_profile = (
        "c8d4ad2_legacy_run_arti_open"
        if requested_joint_type is None
        else "joint_aware_run_arti_open"
    )
    return {
        "control_profile": control_profile,
        "legacy_demo_executed": bool(legacy_demo_executed),
    }


def _normalize_resolved_joint_type(joint_type):
    if joint_type is None:
        return None
    normalized = str(joint_type).strip().lower()
    if normalized == "continuous":
        return "revolute"
    if normalized == "primismatic":
        return "prismatic"
    return normalized


def _joint_type_mismatch_result(
    gapart_id,
    object_path,
    task_root,
    save_root,
    requested_joint_type,
    resolved_joint_type,
    tested_part_id,
    selected_bbox_id,
    selected_source,
    selected_link,
    selected_category,
    resolved_joint,
):
    return {
        "asset_id": gapart_id,
        "object_path": object_path,
        "task_root": task_root,
        "save_root": save_root,
        "mode": "run_arti_open",
        "status": "failure",
        "failure_reason": "requested_joint_type_mismatch",
        **_make_control_profile_metadata(requested_joint_type, legacy_demo_executed=False),
        "requested_joint_type": requested_joint_type,
        "resolved_joint_type": resolved_joint_type,
        "tested_part_id": int(tested_part_id) if tested_part_id is not None else None,
        "selected_bbox_id": int(selected_bbox_id),
        "selected_source": selected_source,
        "selected_link": selected_link,
        "selected_category": selected_category,
        "selected_joint": _json_safe(resolved_joint),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

def init_gym(cfgs, task_cfg=None):
    '''
    function: init gym
    input: cfgs, task_cfg
    '''
    # init gsam
    if cfgs["INFERENCE_GSAM"]:
        grounded_dino_model, sam_predictor = prepare_gsam_model(device=args.device)
    else:
        grounded_dino_model, sam_predictor = None, None
        
    # load selected object information (not important for articulated object manipulation)
    selected_obj_names = task_cfg["selected_obj_names"]
    selected_obj_urdfs=task_cfg["selected_urdfs"]
    selected_obj_num = len(selected_obj_names)
    selected_ob_poses = task_cfg["init_obj_pos"]
    selected_ob_pose_rs = [pose[3:] for pose in selected_ob_poses]
    save_root = task_cfg["save_root"]
    cfgs["asset"]["position_noise"] = [0,0,0]
    cfgs["asset"]["rotation_noise"] = 0
    cfgs["asset"]["asset_files"] = selected_obj_urdfs
    cfgs["asset"]["asset_seg_ids"] = [2 + i for i in range(selected_obj_num)]
    cfgs["asset"]["obj_pose_ps"] = selected_ob_poses
    cfgs["asset"]["obj_pose_rs"] = selected_ob_pose_rs

    # init gym
    gym = ObjectGym(cfgs, grounded_dino_model, sam_predictor)
    
    # refresh observation and run steps to initialize the scene
    gym.refresh_observation(get_visual_obs=False)
    gym.run_steps(pre_steps = 10, refresh_obs=False, print_step=False)
    gym.refresh_observation(get_visual_obs=False)
    _reset_arti_dofs_to_closed(gym)
    gym.save_root = save_root
    
    return gym, cfgs

if args.mode == "run_arti_free_control":
    '''
    function: init gym and run free control
    '''
    ROOT = "gapartnet_example"
    # read all paths
    # we choose one example object to show the demo, change the path
    paths = glob.glob(f"assets/{ROOT}/*/mobility_annotation_gapartnet.urdf")
    
    # we choose one example object to show the demo, change the path 
    # to the object you want to show!
    paths = ["../partnet_mobility_part/45661/mobility_annotation_gapartnet.urdf"]
    for path in tqdm.tqdm(paths, total=len(paths)):
        gapart_id = path.split("/")[-2]
        cfgs = read_yaml_config(f"{args.config}.yaml")
        task_root = args.task_root
        task_cfgs_path = "task_config.json"
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        with open("gapartnet_obj_min_z.json", "r") as f: gapartnet_obj_min_z = json.load(f)
        gapartnet_obj_min_z_ = gapartnet_obj_min_z[gapart_id]
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])
        cfgs["HEADLESS"] = args.headless
        cfgs["USE_CUROBO"] = True
        cfgs["asset"]["arti_obj_root"] = ROOT
        cfgs["asset"]["arti_position_noise"] = 0.0
        cfgs["asset"]["arti_rotation_noise"] = 0.0
        cfgs["asset"]["arti_obj_scale"] = 0.4
        cfgs["asset"]["arti_rotation"] = 0
        cfgs["asset"]["arti_gapartnet_ids"] = [
            gapart_id
        ]
        cfgs["asset"]["arti_obj_pose_ps"] = [
            [0.8, 0, -0.4*gapartnet_obj_min_z_]
        ]
        gym, cfgs = init_gym(cfgs, task_cfg=task_cfg)

        print(gym.save_root)
        gym.run_steps(pre_steps = 100, refresh_obs=False, print_step=False)
        
        ############################ change to desired pose ############################
        rotation = np.array([0, 1, 0, 0])
        position = np.array([0.2502,     -0.2000,     0.8517])
        move_pose = np.concatenate([position, rotation])
        ################################################################################
        
        step_num, traj = gym.control_to_pose(move_pose, close_gripper = True, save_video = False, save_root = None, step_num = 0)
        
        gym.clean_up()
        del gym     
elif args.mode == "run_arti_open":
    '''
    function: init gym and run open demo
    '''
    
    ROOT = "gapartnet_example"
    asset_root = "assets"
    output_id_prefix = ""
    # read all paths
    if args.object_path:
        object_dir = os.path.abspath(args.object_path)
        if object_dir.endswith("mobility_annotation_gapartnet.urdf"):
            object_dir = os.path.dirname(object_dir)
        ROOT = os.path.basename(os.path.dirname(object_dir))
        asset_root = os.path.dirname(os.path.dirname(object_dir))
        output_id_prefix = f"{ROOT}_"
        paths = [os.path.join(object_dir, "mobility_annotation_gapartnet.urdf")]
    else:
        paths = glob.glob(f"{asset_root}/{ROOT}/*/mobility_annotation_gapartnet.urdf")
    if args.object_id and not args.object_path:
        paths = [p for p in paths if p.split("/")[-2] == args.object_id]
    for path in tqdm.tqdm(paths, total=len(paths)):
        # get gapart id and anno
        gapart_id = path.split("/")[-2]
        gapart_anno_path = "/".join(path.split("/")[:-1]) + "/link_annotation_gapartnet.json"
        gapart_anno = json.load(open(gapart_anno_path, "r"))
        for link_anno in gapart_anno:
            if link_anno["is_gapart"] and link_anno["category"] == "slider_drawer":
                pass
        
        # cfg loading and init gym
        cfgs = read_yaml_config(f"{args.config}.yaml")
        task_root = args.task_root
        task_cfgs_path = "task_config.json"
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        
        # load articualted object with the bottom at z = 0
        with open("gapartnet_obj_min_z.json", "r") as f: gapartnet_obj_min_z = json.load(f)
        if gapart_id in gapartnet_obj_min_z.keys():
            gapartnet_obj_min_z_ = gapartnet_obj_min_z[gapart_id]
        else:
            print(f"{gapart_id} not in gapartnet_obj_min_z")
            gapartnet_obj_min_z_ = -1.5
            
        # set the save root and other configurations
        task_cfg["save_root"] = os.path.join(task_root, f"{output_id_prefix}{gapart_id}")
        os.makedirs(task_cfg["save_root"], exist_ok=True)
        cfgs["HEADLESS"] = args.headless
        cfgs["SAVE_VIDEO"] = args.save_video
        cfgs["SAVE_VIDEO_FRAMES"] = args.save_video_frames
        cfgs["USE_CUROBO"] = False
        cfgs["asset"]["arti_asset_root"] = asset_root
        cfgs["asset"]["arti_obj_root"] = ROOT
        cfgs["asset"]["arti_vhacd_enabled"] = args.object_path is None
        cfgs["asset"]["arti_position_noise"] = 0.0
        cfgs["asset"]["arti_rotation_noise"] = 0.0
        cfgs["asset"]["arti_obj_scale"] = 0.4
        cfgs["asset"]["arti_rotation"] = 0
        cfgs["asset"]["arti_gapartnet_ids"] = [
            gapart_id
        ]
        support_surface_z = _support_surface_z_for_asset(gapart_id)
        arti_pose_z = _pose_z_for_support_surface_height(
            min_z=gapartnet_obj_min_z_,
            scale=cfgs["asset"]["arti_obj_scale"],
            support_surface_z=support_surface_z,
        )
        placement_mode = "tabletop_support_surface" if support_surface_z > 0 else "floor_support_surface"
        print(
            f"[DIAG] {placement_mode}: pose_z={arti_pose_z:.6f} "
            f"support_surface_z={support_surface_z:.3f} "
            f"scaled_min_z={cfgs['asset']['arti_obj_scale'] * gapartnet_obj_min_z_:.6f}"
        )
        cfgs["asset"]["arti_obj_pose_ps"] = [
            [.8, 0, arti_pose_z]
        ]
        # init gym
        gym, cfgs = init_gym(cfgs, task_cfg=task_cfg)

        dof_before_closed_reset = _get_arti_dof_positions(gym)
        closed_reference = _reset_arti_dofs_to_closed(gym)
        dof_after_closed_reset = _get_arti_dof_positions(gym)
        if closed_reference is None:
            closed_init_check = {
                "applied": False,
                "reason": "no_articulated_dofs",
                "before": dof_before_closed_reset.tolist(),
                "after": dof_after_closed_reset.tolist(),
                "closed_reference": None,
                "max_abs_error_after": None,
            }
        else:
            closed_error = np.abs(dof_after_closed_reset - np.asarray(closed_reference, dtype=np.float32).reshape(1, -1))
            closed_init_check = {
                "applied": True,
                "before": dof_before_closed_reset.tolist(),
                "after": dof_after_closed_reset.tolist(),
                "closed_reference": np.asarray(closed_reference, dtype=np.float32).tolist(),
                "max_abs_error_after": float(np.max(closed_error)) if closed_error.size else 0.0,
            }
        print(f"[DIAG] closed init check: {json.dumps(_json_safe(closed_init_check), sort_keys=True)}")

        # get the gapartnet annotation
        gym.get_gapartnet_anno()
        
        # render bbox for visualization and debug
        if not cfgs["HEADLESS"] and True:
            gym.gym.clear_lines(gym.viewer)
        for env_i in range(gym.num_envs):
            for gapart_obj_i, gapart_raw_valid_anno in enumerate(gym.gapart_raw_valid_annos):
                
                all_bbox_now = gym.gapart_init_bboxes[gapart_obj_i]*cfgs["asset"]["arti_obj_scale"]
                
                rotation = R.from_quat(gym.arti_init_obj_rot_list[env_i])
                rotation_matrix = rotation.as_matrix()
                rotated_bbox_now = np.dot(all_bbox_now, rotation_matrix.T)
                
               
                all_bbox_now = rotated_bbox_now + gym.arti_init_obj_pos_list[env_i]
                
                if not cfgs["HEADLESS"] and True:
                    idx_set = [[0,1],[1,2],[1,5],[0,4],[0,3],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]
                    for part_i in range(len(gapart_raw_valid_anno)):
                        bbox_now_i = all_bbox_now[part_i]
                        for i in range(len(idx_set)):
                            gym.gym.add_lines(gym.viewer, gym.envs[env_i], 1, 
                                np.concatenate((bbox_now_i[idx_set[i][0]], 
                                                bbox_now_i[idx_set[i][1]]), dtype=np.float32), 
                                np.array([1, 0 ,0], dtype=np.float32))
        
        
        object_dir = os.path.dirname(path)
        fixed_parent, fixed_rotation, movable_joints_by_child, movable_joint_names = _parse_urdf_joint_info(object_dir)
        if args.part_id is not None:
            bbox_id, selected_source, pre_resolved_joint = _select_bbox_for_requested_part(
                args.part_id,
                gapart_anno,
                gapart_raw_valid_anno,
                fixed_parent,
                fixed_rotation,
                movable_joints_by_child,
            )
        else:
            selected_source = "legacy_bbox_id"
            pre_resolved_joint = None
            bbox_id = -1
        print(f"[DIAG] selected bbox_id={bbox_id} source={selected_source} requested_part_id={args.part_id}")

        # get the part bbox and calculate handle approach geometry
        all_bbox_now = torch.tensor(all_bbox_now, dtype=torch.float32).to(gym.device).reshape(-1, 8, 3)
        all_bbox_center_front_face = torch.mean(all_bbox_now[:,0:4,:], dim = 1) 
        handle_out = all_bbox_now[:,0,:] - all_bbox_now[:,4,:]
        handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
        handle_long = all_bbox_now[:,0,:] - all_bbox_now[:,1,:]
        handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
        handle_short = all_bbox_now[:,0,:] - all_bbox_now[:,3,:]
        handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)
        rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), 
                        handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))
        
        init_position = all_bbox_center_front_face[bbox_id].cpu().numpy()
        handle_out_ = handle_out[bbox_id].cpu().numpy()
        approach_dir = _safe_normalize_np(handle_out_)

        selected_anno = gapart_raw_valid_anno[bbox_id]
        selected_link = selected_anno.get("link_name", "")
        selected_category = selected_anno.get("category", "")
        resolved_joint = pre_resolved_joint or _resolve_controlling_joint(selected_link, fixed_parent, fixed_rotation, movable_joints_by_child)
        requested_joint_type = _normalize_requested_joint_type(args.joint_type)
        resolved_joint_type = _normalize_resolved_joint_type(None if resolved_joint is None else resolved_joint.get("type"))
        if requested_joint_type is not None and requested_joint_type != resolved_joint_type:
            result = _joint_type_mismatch_result(
                gapart_id=gapart_id,
                object_path=args.object_path,
                task_root=args.task_root,
                save_root=gym.save_root,
                requested_joint_type=requested_joint_type,
                resolved_joint_type=resolved_joint_type,
                tested_part_id=args.part_id,
                selected_bbox_id=bbox_id,
                selected_source=selected_source,
                selected_link=selected_link,
                selected_category=selected_category,
                resolved_joint=resolved_joint,
            )
            result["closed_init_check"] = _json_safe(closed_init_check)
            _write_attempt_result(gym.save_root, result)
            print(f"[DIAG] requested_joint_type={requested_joint_type} mismatches resolved_joint_type={resolved_joint_type}; stop operation")
            gym.clean_up()
            del gym
            continue
        movable_bbox_id = bbox_id
        if resolved_joint is not None and resolved_joint.get("movable_link"):
            for anno_i, anno in enumerate(gapart_raw_valid_anno):
                if anno.get("link_name") == resolved_joint.get("movable_link"):
                    movable_bbox_id = anno_i
                    break
        movable_bbox_np = all_bbox_now[movable_bbox_id].cpu().numpy()
        movable_center = np.mean(movable_bbox_np, axis=0)
        pull_dir = approach_dir
        if resolved_joint is not None and resolved_joint.get("type") == "prismatic":
            pull_dir = _safe_normalize_np(resolved_joint["axis_root"], fallback=approach_dir)
            # GAPartNet drawers in this demo are placed with the object fixed-base rotation already reflected
            # in bbox/world geometry. Align joint-axis sign with the observed handle outward side.
            if np.dot(pull_dir, approach_dir) < 0:
                pull_dir = -pull_dir

        pre_grasp_offset = 0.16
        grasp_offset = 0.06
        pull_step = 0.01
        pull_steps = 30
        revolute_angle_step = 0.035
        revolute_steps = 30

        joint_desc = None if resolved_joint is None else {k: resolved_joint[k] for k in ["name", "type", "parent", "child", "movable_link"] if k in resolved_joint}
        target_dof_index = _select_target_dof_index(joint_desc, movable_joint_names)
        print(f"[DIAG] gapart_id={gapart_id} bbox_id={bbox_id}")
        print(f"[DIAG] selected link/category: {selected_link} / {selected_category}")
        print(f"[DIAG] resolved joint: {joint_desc}")
        print(f"[DIAG] handle center (world): {init_position}")
        print(f"[DIAG] approach_dir:          {approach_dir}")
        print(f"[DIAG] pull_dir:              {pull_dir}")
        print(f"[DIAG] pre-grasp target:      {init_position + pre_grasp_offset * approach_dir}")
        print(f"[DIAG] ee_pos (current):      {gym.hand_pos[0].cpu().numpy()}")
        dof_initial = _get_arti_dof_positions(gym)
        print(f"[DIAG] movable joint names:   {movable_joint_names}, target_dof_index={target_dof_index}")
        print(_format_arti_dof_diag("initial", dof_initial, joint_desc=joint_desc, target_dof_index=target_dof_index))
        if args.save_video and resolved_joint is not None and hasattr(gym, "set_video_axis_overlays"):
            joint_frame_asset = _joint_frame_in_asset(object_dir, resolved_joint.get("name"))
            joint_frame_world = _asset_frame_to_world(
                joint_frame_asset,
                gym.arti_init_obj_pos_list[0],
                gym.arti_init_obj_rot_list[0],
                cfgs["asset"]["arti_obj_scale"],
            )
            gym.set_video_axis_overlays([
                {"label": "world frame", "origin": [0.0, 0.0, 0.0], "axes": np.eye(3, dtype=np.float32), "length": 0.35},
                {
                    "label": "joint frame",
                    "origin": joint_frame_world[:3, 3].astype(float).tolist(),
                    "axes": joint_frame_world[:3, :3].astype(float).tolist(),
                    "length": 0.25,
                },
            ])
            print("[DIAG] video axis overlays enabled in world coordinates")
            print("[DIAG] world frame origin=[0,0,0], axes=identity")
            print(f"[DIAG] joint frame origin(world)={joint_frame_world[:3, 3]}")
            print(f"[DIAG] joint frame axes(world columns)=\n{joint_frame_world[:3, :3]}")

        legacy_demo_executed = False
        if _should_run_legacy_open_demo(requested_joint_type):
            legacy_demo_executed = True
            pre_grasp_position, grasp_position, pull_targets = _legacy_open_demo_targets(init_position, handle_out_)
            print(f"[DIAG] legacy pre_grasp target: {pre_grasp_position}")
            print(f"[DIAG] legacy grasp target:     {grasp_position}")
            print(f"[DIAG] legacy pull first/final: {pull_targets[0]} -> {pull_targets[-1]}")

            step_num = 0
            for i in range(10):
                step_num, traj = gym.control_to_pose(
                    np.array([*pre_grasp_position, *(rotations[bbox_id].cpu().numpy())]),
                    close_gripper=False,
                    save_video=args.save_video,
                    save_root=gym.save_root,
                    step_num=step_num,
                    use_ik=True,
                )

            for i in range(10):
                step_num, traj = gym.control_to_pose(
                    np.array([*grasp_position, *(rotations[bbox_id].cpu().numpy())]),
                    close_gripper=False,
                    save_video=args.save_video,
                    save_root=gym.save_root,
                    step_num=step_num,
                    use_ik=True,
                )

            for i in range(10):
                step_num = gym.move_gripper(
                    close_gripper=True,
                    save_video=args.save_video,
                    save_root=gym.save_root,
                    start_step=step_num,
                )

            for pull_target in pull_targets:
                step_num, traj = gym.control_to_pose(
                    np.array([*pull_target, *(rotations[bbox_id].cpu().numpy())]),
                    close_gripper=True,
                    save_video=args.save_video,
                    save_root=gym.save_root,
                    step_num=step_num,
                    use_ik=True,
                )

            print(f"[DIAG] ee_pos after legacy manipulation: {gym.hand_pos[0].cpu().numpy()}")
            print(_format_arti_dof_diag("after_legacy_pull", _get_arti_dof_positions(gym), joint_desc=joint_desc, initial=dof_initial, target_dof_index=target_dof_index))
            print("Finish the manipulation, run the simulation 1000 steps for more visualization")
            gym.run_steps(pre_steps=1000, refresh_obs=False, print_step=False)
            final_dof = _get_arti_dof_positions(gym)
            final_delta = _target_abs_delta(final_dof, dof_initial, target_dof_index)
            print(_format_arti_dof_diag("after_legacy_settle", final_dof, joint_desc=joint_desc, initial=dof_initial, target_dof_index=target_dof_index))

            video_result = None
            if args.save_video:
                video_result = _write_video_mp4_from_frame_list(gym.save_root, getattr(gym, "video_frames", []), output_name="manipulation.mp4")
            video_path = (video_result or {}).get("path") or os.path.join(gym.save_root, "manipulation.mp4")
            frame_metadata = (video_result or {}).get("frame_metadata") or _frame_sequence_metadata(gym.save_root, extension="png")
            video_metadata = (video_result or {}).get("video_metadata") or _probe_video_metadata(video_path)
            initial_target_value = None
            final_target_value = None
            closed_dof_reference = np.asarray(gym.arti_obj_dof_props["lower"], dtype=np.float32)
            dof_before_closed_reset = _get_arti_dof_positions(gym)
            closed_reference = _reset_arti_dofs_to_closed(gym)
            dof_after_closed_reset = _get_arti_dof_positions(gym)
            closed_init_check = {
                "before": np.asarray(dof_before_closed_reset).astype(float).tolist(),
                "after": np.asarray(dof_after_closed_reset).astype(float).tolist(),
                "closed_reference": np.asarray(closed_reference).astype(float).tolist(),
                "applied": bool(np.allclose(np.asarray(dof_after_closed_reset), np.asarray(closed_reference), atol=1e-6)),
                "max_abs_error_after": float(np.max(np.abs(np.asarray(dof_after_closed_reset) - np.asarray(closed_reference)))),
            }
            print(f"[DIAG] closed init check: {json.dumps(closed_init_check, sort_keys=True)}")
            if target_dof_index is not None:
                initial_target_value = float(np.asarray(dof_initial).reshape(np.asarray(dof_initial).shape[0], -1)[0, target_dof_index])
                final_target_value = float(np.asarray(final_dof).reshape(np.asarray(final_dof).shape[0], -1)[0, target_dof_index])
            result = {
                "asset_id": gapart_id,
                "object_path": args.object_path,
                "task_root": args.task_root,
                "save_root": gym.save_root,
                "mode": args.mode,
                "save_video": bool(args.save_video),
                "save_video_frames": bool(args.save_video_frames),
                "frame_extension": "png" if args.save_video_frames else None,
                **_make_control_profile_metadata(requested_joint_type, legacy_demo_executed),
                "requested_joint_type": requested_joint_type,
                "resolved_joint_type": resolved_joint_type,
                "legacy_bbox_id": -1,
                "tested_part_id": int(args.part_id) if args.part_id is not None else None,
                "selected_bbox_id": int(bbox_id),
                "selected_source": selected_source,
                "selected_link": selected_link,
                "selected_category": selected_category,
                "selected_joint": _json_safe(resolved_joint),
                "joint_name": None if joint_desc is None else joint_desc.get("name"),
                "joint_type": None if joint_desc is None else ("revolute" if joint_desc.get("type") == "continuous" else joint_desc.get("type")),
                "initial_dof": initial_target_value,
                "final_dof": final_target_value,
                "initial_dof_all": np.asarray(dof_initial).reshape(np.asarray(dof_initial).shape[0], -1)[0].astype(float).tolist(),
                "closed_dof_reference": closed_dof_reference.astype(float).tolist(),
                "closed_init_check": _json_safe(closed_init_check),
                "delta": final_delta,
                "legacy_pre_grasp_offset": 0.2,
                "legacy_grasp_offset": 0.1,
                "legacy_pull_steps": 30,
                "legacy_pull_step": 0.01,
                "placement_mode": placement_mode,
                "support_surface_z": support_surface_z,
                "arti_pose_z": arti_pose_z,
                "settle_steps": 1000,
                "video": video_path,
                "video_writer": (video_result or {}).get("writer"),
                "video_result": video_result,
                "video_metadata": video_metadata,
                "frame_metadata": frame_metadata,
                "target_dof_index": target_dof_index,
            }
            _write_attempt_result(gym.save_root, result)
            gym.clean_up()
            del gym
            continue
        removed_video_frames = _reset_video_capture_buffer(gym, remove_disk_frames=True)
        print(f"[DIAG] requested_joint_type={requested_joint_type}; skip legacy straight-line demo and start joint-aware manipulation path")
        if removed_video_frames:
            print(f"[DIAG] cleared {removed_video_frames} stale video frame files before joint-aware recording")
        
        # Root-cause note for the 41510 closed-state regression: the failing
        # closed-state run keeps joint_1 near zero but has joint_0 at a very
        # different configuration from the earlier successful run.  This means
        # the same handle/joint selection can have a different contact geometry,
        # so we keep the nominal attempt first and only try closed-state grasp
        # candidates when early target DOF motion says the grasp is not engaged.
        is_prismatic_target = resolved_joint is not None and resolved_joint.get("type") == "prismatic"
        is_revolute_target = resolved_joint is not None and resolved_joint.get("type") in {"revolute", "continuous"}
        early_success_threshold = 0.01
        joint_lower = None
        joint_upper = None
        if target_dof_index is not None:
            joint_lower = float(gym.arti_obj_dof_props["lower"][target_dof_index])
            joint_upper = float(gym.arti_obj_dof_props["upper"][target_dof_index])
        final_success_threshold = _required_success_delta(joint_desc, lower=joint_lower, upper=joint_upper)
        success_threshold_for_exit = final_success_threshold
        if gapart_id == "45661" and is_prismatic_target and np.isfinite(final_success_threshold):
            # Old-video baseline for 45661 finishes once the upper drawer is
            # visibly open, before the generic 50%-range batch threshold.
            success_threshold_for_exit = min(final_success_threshold, 0.18)
        if is_prismatic_target and np.isfinite(final_success_threshold):
            requested_pull_distance = min(
                max(final_success_threshold + 0.08, pull_step * pull_steps),
                abs(joint_upper - joint_lower) if joint_lower is not None and joint_upper is not None else final_success_threshold + 0.08,
            )
            pull_steps = max(pull_steps, int(np.ceil(requested_pull_distance / pull_step)))
            print(f"[DIAG] prismatic pull_steps adjusted to {pull_steps} for required_delta={final_success_threshold:.6f}")
        early_pull_check_step = 10
        candidates = [{"label": "nominal", "grasp_offset": grasp_offset, "bias": np.zeros(3, dtype=np.float32)}]
        if is_prismatic_target or is_revolute_target:
            candidates = _make_prismatic_grasp_candidates(
                grasp_offset,
                approach_dir,
                handle_long=handle_long[bbox_id].cpu().numpy(),
                handle_short=handle_short[bbox_id].cpu().numpy(),
            )
            if gapart_id == "45661":
                candidates = [candidates[0]]
            if is_revolute_target:
                keep = {"approach_offset_0.04"}
                base_candidates = [c for c in candidates if c["label"] in keep]
                candidates = []
                for base in base_candidates:
                    plus = dict(base)
                    plus["arc_sign"] = 1.0
                    candidates.append(plus)
                    minus = dict(base)
                    minus["label"] = f"{base['label']}_arc_neg"
                    minus["arc_sign"] = -1.0
                    candidates.append(minus)
        print(f"[DIAG] grasp candidates:    {[c['label'] for c in candidates]}")

        step_num = 0
        best_delta = -1.0
        best_label = None
        selected_revolute_axis_point = None
        selected_revolute_axis_point_from_joint = None
        selected_attempt_success = False
        selected_attempt_label = None
        selected_attempt_dof = None
        selected_attempt_stopped_during_pull = False
        joint_aware_attempt_count = 0
        pull_step_gripper_metrics = []
        for candidate_i, candidate in enumerate(candidates):
            joint_aware_attempt_count += 1
            cand_label = candidate["label"]
            cand_grasp_offset = float(candidate["grasp_offset"])
            cand_bias = np.asarray(candidate.get("bias", np.zeros(3)), dtype=np.float32)
            cand_arc_sign = float(candidate.get("arc_sign", 1.0))
            attempt_start_step = step_num
            print(f"[DIAG] attempt {candidate_i+1}/{len(candidates)} label={cand_label} grasp_offset={cand_grasp_offset:.3f} bias={cand_bias} arc_sign={cand_arc_sign:.1f}")

            # Re-open gripper before retrying a failed closed-state grasp.  We do
            # not reset the articulated object because the retry is only enabled
            # when early target movement is almost zero, so the object state is
            # effectively unchanged.
            if candidate_i > 0:
                for _ in range(4):
                    step_num = gym.move_gripper(close_gripper=False, save_video=args.save_video, save_root=gym.save_root, start_step=step_num)

            # move the object to the pre-grasp position
            pre_grasp_position = init_position + pre_grasp_offset * approach_dir + cand_bias
            step_num, traj = _control_to_pose_repeated_ik(
                gym,
                np.array([*pre_grasp_position,*(rotations[bbox_id].cpu().numpy())]),
                repeats=10,
                close_gripper=False,
                save_video=args.save_video,
                save_root=gym.save_root,
                step_num=step_num,
                record_last_repeat_only=True,
            )
            print(_format_ee_tracking_diag(f"after_pre_grasp[{cand_label}]", gym, pre_grasp_position))

            # move the object to the grasp position
            grasp_position = init_position + cand_grasp_offset * approach_dir + cand_bias
            step_num, traj = _control_to_pose_repeated_ik(
                gym,
                np.array([*grasp_position,*(rotations[bbox_id].cpu().numpy())]),
                repeats=10,
                close_gripper=False,
                save_video=args.save_video,
                save_root=gym.save_root,
                step_num=step_num,
                record_last_repeat_only=True,
            )
            print(_format_ee_tracking_diag(f"after_grasp_pose[{cand_label}]", gym, grasp_position))
            print(_format_arti_dof_diag(f"after_grasp_pose[{cand_label}]", _get_arti_dof_positions(gym), joint_desc=joint_desc, initial=dof_initial, target_dof_index=target_dof_index))

            # Keep the gripper closed during articulated manipulation so contacts
            # transmit force through the fingers/handle instead of letting the open
            # fingers move around the handle.
            close_for_attempt = True
            for i in range(10):
                step_num = gym.move_gripper(close_gripper = close_for_attempt, save_video=args.save_video, save_root = gym.save_root, start_step = step_num)
            print(_format_ee_tracking_diag(f"after_gripper_close[{cand_label}]", gym, grasp_position))
            print(_format_arti_dof_diag(f"after_gripper_close[{cand_label}]", _get_arti_dof_positions(gym), joint_desc=joint_desc, initial=dof_initial, target_dof_index=target_dof_index))

            # open/pull the articulated part using joint-aware pull direction
            if is_revolute_target:
                if selected_revolute_axis_point_from_joint is None and resolved_joint is not None:
                    joint_frame_asset = _joint_frame_in_asset(object_dir, resolved_joint.get("name"))
                    joint_frame_world = _asset_frame_to_world(
                        joint_frame_asset,
                        gym.arti_init_obj_pos_list[0],
                        gym.arti_init_obj_rot_list[0],
                        cfgs["asset"]["arti_obj_scale"],
                    )
                    selected_revolute_axis_point_from_joint = joint_frame_world[:3, 3].astype(np.float32)
                    print(f"[DIAG] revolute joint origin used as axis point: {selected_revolute_axis_point_from_joint}")
                revolute_axis_point = selected_revolute_axis_point_from_joint if selected_revolute_axis_point_from_joint is not None else _estimate_revolute_axis_point_from_bbox(
                    init_position + cand_bias,
                    movable_bbox_np,
                    resolved_joint.get("axis_root", approach_dir),
                )
                selected_revolute_axis_point = revolute_axis_point
                revolute_geom = _compute_revolute_motion_geometry(
                    init_position + cand_bias,
                    movable_center,
                    resolved_joint.get("axis_root", approach_dir),
                    approach_dir,
                    axis_point=revolute_axis_point,
                )
                print(
                    f"[DIAG] revolute geom[{cand_label}]: "
                    f"pivot={revolute_geom['pivot']} axis={revolute_geom['axis_dir']} "
                    f"radial={revolute_geom['radial_dir']} tangent={revolute_geom['tangent_dir']} "
                    f"radius={revolute_geom['radius']:.4f}"
                )
                hold_steps = 6
                hold_target = np.asarray(init_position + cand_bias + cand_grasp_offset * approach_dir, dtype=np.float32)
                hold_targets = np.stack([hold_target for _ in range(hold_steps)], axis=0)
                pull_targets = _compute_revolute_grasp_hold_arc_targets(
                    init_position + cand_bias,
                    cand_grasp_offset,
                    {**revolute_geom, "approach_dir": approach_dir},
                    hold_steps=hold_steps,
                    angle_step=0.025,
                    arc_steps=max(revolute_steps - 4, 1),
                    direction_sign=cand_arc_sign,
                )
            else:
                pull_targets = _compute_pull_targets(init_position + cand_bias, approach_dir, pull_dir, cand_grasp_offset, pull_step, pull_steps)
            print(f"[DIAG] pull first target[{cand_label}]: {pull_targets[0]}")
            print(f"[DIAG] pull final target[{cand_label}]: {pull_targets[-1]}")
            early_delta = 0.0
            attempt_reached_success_during_pull = False
            for i, pull_target in enumerate(pull_targets): 
                pull_pose_rot = rotations[bbox_id].cpu().numpy()
                step_num, traj = gym.control_to_pose(
                    np.array([*pull_target,*pull_pose_rot]),
                    close_gripper = True, save_video = args.save_video, save_root = gym.save_root, step_num = step_num, use_ik = True)
                dof_now = None
                if i in {0, early_pull_check_step - 1, len(pull_targets) - 1}:
                    dof_now = _get_arti_dof_positions(gym)
                    print(_format_ee_tracking_diag(f"after_pull_step_{i+1}[{cand_label}]", gym, pull_target))
                    print(_format_arti_dof_diag(f"after_pull_step_{i+1}[{cand_label}]", dof_now, joint_desc=joint_desc, initial=dof_initial, target_dof_index=target_dof_index))
                    if i == early_pull_check_step - 1:
                        early_delta = _target_abs_delta(dof_now, dof_initial, target_dof_index)
                        print(f"[DIAG] early target abs delta[{cand_label}]={early_delta:.6f} threshold={early_success_threshold:.6f}")
                        if resolved_joint is not None and early_delta < early_success_threshold and candidate_i + 1 < len(candidates):
                            print(f"[DIAG] retry trigger: target DOF did not move enough for {cand_label}; switching candidate")
                            break
                if (
                    resolved_joint is not None
                    and np.isfinite(success_threshold_for_exit)
                    and target_dof_index is not None
                ):
                    if dof_now is None:
                        dof_now = _get_arti_dof_positions(gym)
                    step_delta = _target_abs_delta(dof_now, dof_initial, target_dof_index)
                    initial_value_for_step = float(np.asarray(dof_initial).reshape(np.asarray(dof_initial).shape[0], -1)[0, target_dof_index])
                    step_target_value = float(np.asarray(dof_now).reshape(np.asarray(dof_now).shape[0], -1)[0, target_dof_index])
                    step_handle_bbox = _transform_handle_bbox_for_final_joint(
                        all_bbox_now[bbox_id].cpu().numpy(),
                        resolved_joint,
                        initial_value_for_step,
                        step_target_value,
                        prismatic_dir=pull_dir,
                        revolute_axis_point=selected_revolute_axis_point,
                    )
                    step_gripper_metrics = _gripper_handle_metrics(gym, step_handle_bbox)
                    pull_step_gripper_metrics.append({
                        "candidate": cand_label,
                        "step": i + 1,
                        "delta": float(step_delta),
                        "on_handle": bool(step_gripper_metrics["on_handle"]),
                        "metrics": _json_safe(step_gripper_metrics),
                    })
                    print(f"[DIAG] pull_step_metrics[{cand_label}][{i+1}]: on_handle={step_gripper_metrics['on_handle']} delta={step_delta:.6f}")
                    if np.isfinite(step_delta) and step_delta >= success_threshold_for_exit:
                        print(f"[DIAG] success_check_step_{i+1}[{cand_label}]: delta={step_delta:.6f} gripper_on_handle={step_gripper_metrics['on_handle']}")
                        if step_gripper_metrics["on_handle"]:
                            attempt_reached_success_during_pull = True
                            print(f"[DIAG] success threshold reached during pull at step {i+1}; stopping pull with gripper closed")
                            break
            dof_after_attempt = _get_arti_dof_positions(gym)
            attempt_delta = _target_abs_delta(dof_after_attempt, dof_initial, target_dof_index)
            if np.isfinite(attempt_delta) and attempt_delta > best_delta:
                best_delta = attempt_delta
                best_label = cand_label
            attempt_target_value = None
            initial_target_value_for_attempt = None
            if target_dof_index is not None:
                initial_target_value_for_attempt = float(np.asarray(dof_initial).reshape(np.asarray(dof_initial).shape[0], -1)[0, target_dof_index])
                attempt_target_value = float(np.asarray(dof_after_attempt).reshape(np.asarray(dof_after_attempt).shape[0], -1)[0, target_dof_index])
            attempt_handle_bbox = _transform_handle_bbox_for_final_joint(
                all_bbox_now[bbox_id].cpu().numpy(),
                resolved_joint,
                initial_target_value_for_attempt,
                attempt_target_value,
                prismatic_dir=pull_dir,
                revolute_axis_point=selected_revolute_axis_point,
            )
            attempt_gripper_on_handle = _is_gripper_on_handle(gym, attempt_handle_bbox)
            attempt_success = (
                resolved_joint is None
                or (
                    np.isfinite(success_threshold_for_exit)
                    and np.isfinite(attempt_delta)
                    and attempt_delta >= success_threshold_for_exit
                    and attempt_gripper_on_handle
                )
            )
            print(f"[DIAG] attempt_result[{cand_label}]: delta={attempt_delta:.6f} gripper_on_handle={attempt_gripper_on_handle}")
            if attempt_success or candidate_i + 1 == len(candidates):
                selected_attempt_success = bool(attempt_success)
                selected_attempt_label = cand_label
                selected_attempt_dof = dof_after_attempt
                selected_attempt_stopped_during_pull = bool(attempt_reached_success_during_pull)
                print(f"[DIAG] selected attempt={cand_label} target_abs_delta={attempt_delta:.6f} best={best_label}:{best_delta:.6f}")
                break
            if args.save_video:
                removed_frames = _discard_video_frames_from(gym.save_root, attempt_start_step, getattr(gym, "video_frames", None))
                print(f"[DIAG] discarded {removed_frames} video frames from failed attempt={cand_label}")
                step_num = attempt_start_step

        assisted_revolute_applied = False
        contact_delta = _target_abs_delta(_get_arti_dof_positions(gym), dof_initial, target_dof_index)
        if is_revolute_target and contact_delta < final_success_threshold and target_dof_index is not None:
            print(f"[DIAG] contact-based revolute attempt below threshold: contact_delta={contact_delta:.6f}")

        # run the simulation for more visualization, comment it if you don't need it
        print(f"[DIAG] ee_pos after manipulation: {gym.hand_pos[0].cpu().numpy()}")
        print(_format_arti_dof_diag("after_pull", _get_arti_dof_positions(gym), joint_desc=joint_desc, initial=dof_initial, target_dof_index=target_dof_index))
        early_exit_on_success = bool(selected_attempt_success)
        settle_steps = 0 if early_exit_on_success else 1000
        if early_exit_on_success:
            print(f"[DIAG] success reached on attempt={selected_attempt_label}; skip settle to keep gripper clamped on handle")
            final_dof = selected_attempt_dof if selected_attempt_dof is not None else _get_arti_dof_positions(gym)
        else:
            print(f"Finish the manipulation, run the simulation {settle_steps} steps for more visualization")
            gym.run_steps(pre_steps = settle_steps, refresh_obs=False, print_step=False)
            final_dof = _get_arti_dof_positions(gym)
        final_delta = _target_abs_delta(final_dof, dof_initial, target_dof_index)
        final_stage_label = "success_exit" if early_exit_on_success else "after_settle"
        print(_format_arti_dof_diag(final_stage_label, final_dof, joint_desc=joint_desc, initial=dof_initial, target_dof_index=target_dof_index))
        initial_target_value = None
        final_target_value = None
        closed_dof_reference = np.asarray(gym.arti_obj_dof_props["lower"], dtype=np.float32)
        closed_reference = np.asarray(gym.arti_obj_dof_props["lower"], dtype=np.float32)
        closed_init_check = {
            "before": np.asarray(dof_initial).astype(float).tolist(),
            "after": np.asarray(dof_initial).astype(float).tolist(),
            "closed_reference": closed_reference.astype(float).tolist(),
            "applied": bool(np.allclose(np.asarray(dof_initial), closed_reference, atol=1e-6)),
            "max_abs_error_after": float(np.max(np.abs(np.asarray(dof_initial) - closed_reference))),
        }
        if target_dof_index is not None:
            initial_target_value = float(np.asarray(dof_initial).reshape(np.asarray(dof_initial).shape[0], -1)[0, target_dof_index])
            final_target_value = float(np.asarray(final_dof).reshape(np.asarray(final_dof).shape[0], -1)[0, target_dof_index])
        final_handle_bbox = _transform_handle_bbox_for_final_joint(
            all_bbox_now[bbox_id].cpu().numpy(),
            resolved_joint,
            initial_target_value,
            final_target_value,
            prismatic_dir=pull_dir,
            revolute_axis_point=selected_revolute_axis_point,
        )
        gripper_handle_metrics = _gripper_handle_metrics(gym, final_handle_bbox)
        gripper_on_handle = bool(gripper_handle_metrics["on_handle"])
        if np.isfinite(success_threshold_for_exit):
            motion_success = bool(np.isfinite(final_delta) and final_delta >= success_threshold_for_exit)
            failure_reason = None if motion_success and gripper_on_handle else (
                "gripper_not_on_handle" if not gripper_on_handle else "insufficient_motion"
            )
        else:
            motion_success = False
            failure_reason = "missing_success_threshold"
        status = "success" if motion_success and gripper_on_handle else "failure"
        print(f"[DIAG] final target abs delta={final_delta:.6f} success_threshold={success_threshold_for_exit:.6f}")
        print(f"[DIAG] gripper_handle_metrics={json.dumps(gripper_handle_metrics, sort_keys=True)}")
        print(f"[DIAG] gripper_on_handle={gripper_on_handle} status={status} failure_reason={failure_reason}")
        video_result = None
        if args.save_video:
            video_result = _write_video_mp4_from_frame_list(gym.save_root, getattr(gym, "video_frames", []), output_name="manipulation.mp4")
        video_path = (video_result or {}).get("path") or os.path.join(gym.save_root, "manipulation.mp4")
        frame_metadata = (video_result or {}).get("frame_metadata") or _frame_sequence_metadata(gym.save_root, extension="png")
        video_metadata = (video_result or {}).get("video_metadata") or _probe_video_metadata(video_path)
        legacy_baseline = _legacy_output_baseline_metadata(os.path.join("output", "45661")) if gapart_id == "45661" else None
        result = {
            "asset_id": gapart_id,
            "object_path": args.object_path,
            "task_root": args.task_root,
            "save_root": gym.save_root,
            "mode": args.mode,
            "save_video": bool(args.save_video),
            "save_video_frames": bool(args.save_video_frames),
            "frame_extension": "png" if args.save_video_frames else None,
            **_make_control_profile_metadata(requested_joint_type, legacy_demo_executed),
            "baseline_success_standard": "old_video" if gapart_id == "45661" else "joint_delta_and_gripper_on_handle",
            "requested_joint_type": requested_joint_type,
            "resolved_joint_type": resolved_joint_type,
            "legacy_baseline": legacy_baseline,
            "tested_part_id": int(args.part_id) if args.part_id is not None else None,
            "selected_bbox_id": int(bbox_id),
            "selected_source": selected_source,
            "selected_link": selected_link,
            "selected_category": selected_category,
            "selected_joint": _json_safe(resolved_joint),
            "joint_name": None if joint_desc is None else joint_desc.get("name"),
            "joint_type": None if joint_desc is None else ("revolute" if joint_desc.get("type") == "continuous" else joint_desc.get("type")),
            "initial_dof": initial_target_value,
            "final_dof": final_target_value,
            "initial_dof_all": np.asarray(dof_initial).reshape(np.asarray(dof_initial).shape[0], -1)[0].astype(float).tolist(),
            "closed_dof_reference": closed_dof_reference.astype(float).tolist(),
            "delta": final_delta,
            "required_delta": final_success_threshold,
            "success_delta_threshold_used": success_threshold_for_exit,
            "gripper_on_handle": gripper_on_handle,
            "gripper_handle_metrics": gripper_handle_metrics,
            "status": status,
            "failure_reason": failure_reason,
            "video": video_path,
            "video_writer": (video_result or {}).get("writer"),
            "video_result": video_result,
            "video_metadata": video_metadata,
            "frame_metadata": frame_metadata,
            "early_exit_on_success": early_exit_on_success,
            "selected_attempt_label": selected_attempt_label,
            "stopped_during_pull_on_success": selected_attempt_stopped_during_pull,
            "settle_steps": settle_steps,
            "assisted_revolute_applied": assisted_revolute_applied,
            "target_dof_index": target_dof_index,
            "joint_aware_attempt_count": joint_aware_attempt_count,
            "pull_step_gripper_metrics": _json_safe(pull_step_gripper_metrics),
            "ever_on_handle_during_pull": bool(any(item.get("on_handle") for item in pull_step_gripper_metrics)),
            "first_on_handle_step": next((item for item in pull_step_gripper_metrics if item.get("on_handle")), None),
            "closest_handle_step": min(pull_step_gripper_metrics, key=lambda item: item.get("metrics", {}).get("finger_midpoint_distance", float("inf"))) if pull_step_gripper_metrics else None,
        }
        _write_attempt_result(gym.save_root, result)
        
        # clean up for the next object
        gym.clean_up()
        del gym
              
elif args.mode == "run_arti_render":
    '''
    function: init gym and run render code, render the articulated object point cloud
    '''
    ROOT = "gapartnet_example"
    # read all paths
    # we choose one example object to show the demo, change the path
    paths = glob.glob(f"assets/{ROOT}/*/mobility_annotation_gapartnet.urdf")
    # paths -= unused_paths
    for path in tqdm.tqdm(paths, total=len(paths)):
        gapart_id = path.split("/")[-2]
        if gapart_id in ["102278","103989","103560", "103863","103425", 
                         "103869", "47315", "47613", "48018", "47290", "49062",
                         "41003","46456","45203"]:
            continue
        save_dir = "gapartnet_obj"
        save_name = gapart_id
        fname = os.path.join(save_dir, f"{save_name}-articulated-point_cloud.ply")
        print("processing ", gapart_id)
        if os.path.exists(fname):
            print("skip", fname)
            continue
        cfgs = read_yaml_config(f"{args.config}.yaml")
        task_root = args.task_root
        task_cfgs_path = "task_config.json"
        cfgs["HEADLESS"] = True
        cfgs["asset"]["arti_obj_root"] = ROOT
        cfgs["asset"]["arti_obj_pose_ps"] = [[0,0, 3]]
        cfgs["asset"]["arti_position_noise"] = 0.0
        cfgs["asset"]["arti_rotation_noise"] = 0.0
        cfgs["asset"]["arti_obj_scale"] = 1.0
        cfgs["asset"]["arti_rotation"] = 0
        cfgs["asset"]["arti_gapartnet_ids"] = [
            gapart_id
        ]
        cfgs["cam"]["point_cloud_bound"] = [            
            [-1, 1],
            [-1, 1],
            [1, 10.0]
        ]
        cfgs["cam"]["cam_poss"] = [
            [0.1, 0, 1],
            [0.1, 0, 5],
            [0, 3, 3.0],
            [-3, 0, 3.0],
            [3, 0, 3.0],
            [0, -3, 3.0],
        ]
        cfgs["cam"]["cam_targets"] = [
            [0, 0, 3.0],
            [0, 0, 3.0],
            [0, 0, 3.0],
            [0, 0, 3.0],
            [0, 0, 3.0],
            [0, 0, 3.0],
        ]
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])

        gym, cfgs = init_gym(cfgs, task_cfg=task_cfg)

        print(gym.save_root)
        gym.run_steps(pre_steps = 3, refresh_obs=False, print_step=False)
        ## render
        points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, \
            pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)
        
        os.makedirs(save_dir, exist_ok=True)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_envs[0][:, :3]-np.array([0,0,3]))
        point_cloud.colors = o3d.utility.Vector3dVector(colors_envs[0][:, :3]/255.0)
        # save_to ply
        fname = os.path.join(save_dir, f"{save_name}-articulated-point_cloud.ply")
        o3d.io.write_point_cloud(fname, point_cloud)
        gym.clean_up()
        del gym
else:
    raise NotImplementedError   
