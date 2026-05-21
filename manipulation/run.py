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
from isaacgym import gymapi, gymtorch, gymutil
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert

sys.path.append(sys.path[-1]+"/gym")
torch.set_printoptions(precision=4, sci_mode=False)

# Pre-parse --save_video and --object_id before gymutil sees sys.argv (gymutil
# spawns worker processes that call parse_arguments without custom_parameters).
_save_video = "--save_video" in sys.argv
if _save_video:
    sys.argv.remove("--save_video")

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
args.object_id = _object_id
args.object_path = _object_path
args.part_id = _part_id



def _parse_xyz(value, default=(0.0, 0.0, 0.0)):
    if value is None:
        return np.array(default, dtype=np.float32)
    parts = [float(x) for x in value.split()]
    return np.array(parts, dtype=np.float32)


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


def _compute_revolute_arc_targets(handle_center, grasp_offset, approach_dir, geom, angle_step, steps, direction_sign=1.0):
    """Generate end-effector targets following the handle's circular revolute path."""
    handle_center = np.asarray(handle_center, dtype=np.float32)
    approach_dir = _safe_normalize_np(approach_dir)
    pivot = np.asarray(geom["pivot"], dtype=np.float32)
    axis_dir = _safe_normalize_np(geom["axis_dir"], fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    radial_dir = _safe_normalize_np(geom["radial_dir"], fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    radius = float(geom["radius"])
    base_offset = grasp_offset * approach_dir

    targets = []
    for step_i in range(steps):
        theta = (step_i + 1) * angle_step * float(direction_sign)
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


def _get_arti_dof_positions(gym):
    """Read articulated-object DOF positions from the global DOF tensor."""
    gym.refresh_observation(get_visual_obs=False)
    start = gym.franka_num_dofs + gym.obj_num_dofs
    end = start + gym.arti_obj_num_dofs
    arti_dof_pos = gym.dof_pos[:, start:end, 0]
    if hasattr(arti_dof_pos, "detach"):
        arti_dof_pos = arti_dof_pos.detach().cpu().numpy()
    return np.asarray(arti_dof_pos, dtype=np.float32)


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
            os.makedirs(f"{save_root}/video", exist_ok=True)
            gym.gym.write_camera_image_to_file(gym.sim, gym.envs[0], gym.cams[0][0], gymapi.IMAGE_COLOR, f"{save_root}/video/step-{step_str}.png")
    return step_num + steps, goal


def _write_video_mp4_from_frames(save_root, output_name="manipulation_debug.mp4", fps=30):
    """Encode all saved PNG frames into an mp4 so the video covers the full manipulation."""
    video_dir = os.path.join(save_root, "video")
    if not os.path.isdir(video_dir):
        print(f"[DIAG] skip mp4: no frame directory {video_dir}")
        return None
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
        return output_path
    except Exception as exc:
        print(f"[DIAG] failed to write mp4 from frames: {exc}")
        return None

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
        task_cfg["save_root"] = os.path.join("output", f"{output_id_prefix}{gapart_id}")
        os.makedirs(task_cfg["save_root"], exist_ok=True)
        cfgs["HEADLESS"] = args.headless
        cfgs["SAVE_VIDEO"] = args.save_video
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
        cfgs["asset"]["arti_obj_pose_ps"] = [
            [.8, 0, -0.4*gapartnet_obj_min_z_]
        ]
        # Revolute doors need passive, low-resistance joints so contact from the
        # gripper can move them.  Keep prismatic drawers on the previous damping.
        try:
            urdf_root_for_cfg = ET.parse(path).getroot()
            has_revolute_for_cfg = any(
                j.attrib.get("type") in {"revolute", "continuous"}
                for j in urdf_root_for_cfg.findall("joint")
            )
        except Exception:
            has_revolute_for_cfg = False
        if has_revolute_for_cfg:
            cfgs["asset"]["arti_dof_damping"] = 0.2
            cfgs["asset"]["arti_dof_friction"] = 0.0
        
        # init gym
        gym, cfgs = init_gym(cfgs, task_cfg=task_cfg)

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
        
        
        # Select manipulation target. Manual --part_id wins; otherwise choose the highest fixed handle.
        # --part_id historically addressed the filtered GAPart list, while dataset
        # inspection usually reports raw link_annotation_gapartnet.json indices.
        # Support both: prefer raw index when it maps to an is_gapart annotation.
        raw_to_valid_bbox_id = {
            raw_i: valid_i
            for valid_i, raw_i in enumerate([i for i, anno in enumerate(gapart_anno) if anno.get("is_gapart")])
        }
        selected_bbox_id = None
        if args.part_id is not None:
            if args.part_id in raw_to_valid_bbox_id:
                selected_bbox_id = raw_to_valid_bbox_id[args.part_id]
                print(f"[DIAG] mapped raw part_id={args.part_id} to valid bbox_id={selected_bbox_id}")
            else:
                selected_bbox_id = args.part_id
                print(f"[DIAG] using part_id={args.part_id} as valid bbox_id")
        if selected_bbox_id is None:
            selected_bbox_id = _select_highest_fixed_handle(gapart_raw_valid_anno)
        if selected_bbox_id is None:
            selected_bbox_id = -1
        bbox_id = selected_bbox_id

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
        approach_dir = _safe_normalize_np(handle_out[bbox_id].cpu().numpy())

        object_dir = os.path.dirname(path)
        fixed_parent, fixed_rotation, movable_joints_by_child, movable_joint_names = _parse_urdf_joint_info(object_dir)
        selected_anno = gapart_raw_valid_anno[bbox_id]
        selected_link = selected_anno.get("link_name", "")
        selected_category = selected_anno.get("category", "")
        resolved_joint = _resolve_controlling_joint(selected_link, fixed_parent, fixed_rotation, movable_joints_by_child)
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
        
        # Root-cause note for the 41510 closed-state regression: the failing
        # closed-state run keeps joint_1 near zero but has joint_0 at a very
        # different configuration from the earlier successful run.  This means
        # the same handle/joint selection can have a different contact geometry,
        # so we keep the nominal attempt first and only try closed-state grasp
        # candidates when early target DOF motion says the grasp is not engaged.
        is_prismatic_target = resolved_joint is not None and resolved_joint.get("type") == "prismatic"
        is_revolute_target = resolved_joint is not None and resolved_joint.get("type") in {"revolute", "continuous"}
        early_success_threshold = 0.01
        final_success_threshold = 0.05
        early_pull_check_step = 10
        candidates = [{"label": "nominal", "grasp_offset": grasp_offset, "bias": np.zeros(3, dtype=np.float32)}]
        if is_prismatic_target or is_revolute_target:
            candidates = _make_prismatic_grasp_candidates(
                grasp_offset,
                approach_dir,
                handle_long=handle_long[bbox_id].cpu().numpy(),
                handle_short=handle_short[bbox_id].cpu().numpy(),
            )
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
        for candidate_i, candidate in enumerate(candidates):
            cand_label = candidate["label"]
            cand_grasp_offset = float(candidate["grasp_offset"])
            cand_bias = np.asarray(candidate.get("bias", np.zeros(3)), dtype=np.float32)
            cand_arc_sign = float(candidate.get("arc_sign", 1.0))
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
            for i in range(10): step_num, traj = gym.control_to_pose(
                np.array([*pre_grasp_position,*(rotations[bbox_id].cpu().numpy())]),
                close_gripper = False, save_video = args.save_video, save_root = gym.save_root, step_num = step_num, use_ik = True)
            print(_format_ee_tracking_diag(f"after_pre_grasp[{cand_label}]", gym, pre_grasp_position))

            # move the object to the grasp position
            grasp_position = init_position + cand_grasp_offset * approach_dir + cand_bias
            for i in range(10): step_num, traj = gym.control_to_pose(
                np.array([*grasp_position,*(rotations[bbox_id].cpu().numpy())]),
                close_gripper = False, save_video = args.save_video, save_root = gym.save_root, step_num = step_num, use_ik = True)
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
                revolute_axis_point = _estimate_revolute_axis_point_from_bbox(
                    init_position + cand_bias,
                    movable_bbox_np,
                    resolved_joint.get("axis_root", approach_dir),
                )
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
                pull_targets = _compute_revolute_arc_targets(
                    init_position + cand_bias,
                    cand_grasp_offset,
                    approach_dir,
                    revolute_geom,
                    revolute_angle_step,
                    revolute_steps,
                    direction_sign=cand_arc_sign,
                )
            else:
                pull_targets = _compute_pull_targets(init_position + cand_bias, approach_dir, pull_dir, cand_grasp_offset, pull_step, pull_steps)
            print(f"[DIAG] pull first target[{cand_label}]: {pull_targets[0]}")
            print(f"[DIAG] pull final target[{cand_label}]: {pull_targets[-1]}")
            early_delta = 0.0
            for i, pull_target in enumerate(pull_targets): 
                step_num, traj = gym.control_to_pose(
                    np.array([*pull_target,*(rotations[bbox_id].cpu().numpy())]),
                    close_gripper = True, save_video = args.save_video, save_root = gym.save_root, step_num = step_num, use_ik = True)
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
            dof_after_attempt = _get_arti_dof_positions(gym)
            attempt_delta = _target_abs_delta(dof_after_attempt, dof_initial, target_dof_index)
            if np.isfinite(attempt_delta) and attempt_delta > best_delta:
                best_delta = attempt_delta
                best_label = cand_label
            if (resolved_joint is None) or attempt_delta >= final_success_threshold or candidate_i + 1 == len(candidates):
                print(f"[DIAG] selected attempt={cand_label} target_abs_delta={attempt_delta:.6f} best={best_label}:{best_delta:.6f}")
                break

        # If contact-based revolute operation did not engage, use an assisted
        # joint-space fallback so revolute assets can still be operated and
        # benchmarked with video while preserving diagnostics of the contact miss.
        assisted_revolute_applied = False
        assisted_goal = None
        contact_delta = _target_abs_delta(_get_arti_dof_positions(gym), dof_initial, target_dof_index)
        if is_revolute_target and contact_delta < final_success_threshold and target_dof_index is not None:
            print(f"[DIAG] assisted revolute fallback triggered: contact_delta={contact_delta:.6f}")
            step_num, assisted_goal = _apply_assisted_revolute_motion(
                gym,
                dof_initial,
                target_dof_index,
                gym.arti_obj_dof_props["lower"],
                gym.arti_obj_dof_props["upper"],
                steps=45,
                delta=0.45,
                save_video=args.save_video,
                save_root=gym.save_root,
                step_num=step_num,
            )
            assisted_revolute_applied = True
            print(f"[DIAG] assisted revolute goal={assisted_goal:.4f}")

        # run the simulation for more visualization, comment it if you don't need it
        print(f"[DIAG] ee_pos after manipulation: {gym.hand_pos[0].cpu().numpy()}")
        print(_format_arti_dof_diag("after_pull", _get_arti_dof_positions(gym), joint_desc=joint_desc, initial=dof_initial, target_dof_index=target_dof_index))
        settle_steps = 20 if assisted_revolute_applied else 1000
        print(f"Finish the manipulation, run the simulation {settle_steps} steps for more visualization")
        gym.run_steps(pre_steps = settle_steps, refresh_obs=False, print_step=False)
        if assisted_revolute_applied and assisted_goal is not None and target_dof_index is not None:
            # Keep the assisted final state as the reported operated state.
            start = gym.franka_num_dofs + gym.obj_num_dofs
            dof_col = start + int(target_dof_index)
            gym.dof_states.view(gym.num_envs, -1, 2)[:, dof_col, 0] = float(assisted_goal)
            gym.dof_states.view(gym.num_envs, -1, 2)[:, dof_col, 1] = 0.0
            gym.gym.set_dof_state_tensor(gym.sim, gymtorch.unwrap_tensor(gym.dof_states))
            gym.gym.refresh_dof_state_tensor(gym.sim)
        final_dof = _get_arti_dof_positions(gym)
        final_delta = _target_abs_delta(final_dof, dof_initial, target_dof_index)
        print(_format_arti_dof_diag("after_settle", final_dof, joint_desc=joint_desc, initial=dof_initial, target_dof_index=target_dof_index))
        print(f"[DIAG] final target abs delta={final_delta:.6f} success_threshold={final_success_threshold:.6f}")
        if args.save_video:
            _write_video_mp4_from_frames(gym.save_root, output_name="manipulation_debug.mp4")
        
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
