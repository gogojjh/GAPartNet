import ast
import pathlib
import types
import numpy as np

RUN_PATH = pathlib.Path(__file__).resolve().parents[1] / "manipulation" / "run.py"
HELPERS = {
    "_safe_normalize_np",
    "_compute_pull_targets",
    "_get_arti_dof_positions",
    "_format_arti_dof_diag",
    "_select_target_dof_index",
    "_target_abs_delta",
    "_make_prismatic_grasp_candidates",
    "_project_point_to_axis",
    "_compute_revolute_motion_geometry",
    "_compute_revolute_arc_targets",
    "_compute_revolute_target_sequence_diagnostics",
    "_revolute_signed_angles_for_targets",
    "_compute_revolute_grasp_hold_targets",
    "_compute_revolute_grasp_hold_arc_targets",
    "_estimate_revolute_axis_point_from_bbox",
    "_orthonormal_frame_from_z",
    "_rotation_from_link_to_root",
    "_resolve_controlling_joint",
    "_required_success_delta",
    "_is_gripper_on_handle",
    "_tensor_pos_to_np",
    "_segment_intersects_expanded_bbox",
    "_gripper_handle_metrics",
    "_transform_handle_bbox_for_final_joint",
    "_same_resolved_joint",
    "_select_bbox_for_requested_part",
    "_select_highest_fixed_handle",
    "_legacy_open_demo_targets",
    "_pose_z_for_support_surface_height",
    "_support_surface_z_for_asset",
    "_closed_arti_dof_state_command",
    "_probe_video_metadata",
    "_frame_sequence_metadata",
    "_write_video_mp4_from_frame_list",
    "_legacy_output_baseline_metadata",
    "_json_safe",
    "_discard_video_frames_from",
    "_normalize_requested_joint_type",
    "_normalize_resolved_joint_type",
    "_joint_type_mismatch_result",
    "_should_run_legacy_open_demo",
    "_reset_video_capture_buffer",
    "_make_control_profile_metadata",
}


def load_helpers():
    tree = ast.parse(RUN_PATH.read_text())
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            any(alias.name in {"numpy", "torch", "scipy.spatial.transform", "os", "glob", "json", "subprocess", "cv2"} for alias in node.names)
            or (isinstance(node, ast.ImportFrom) and node.module == "scipy.spatial.transform")
        )
    ]
    selected += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
    module = types.ModuleType("run_helpers_for_test")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(RUN_PATH), "exec"), module.__dict__)
    return module


def test_pull_targets_start_from_grasp_without_extra_jump():
    helpers = load_helpers()
    init = np.array([0.57185966, 0.0017532, 0.46060535], dtype=np.float32)
    approach = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    pull = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    grasp_offset = 0.10
    pull_step = 0.005

    targets = helpers._compute_pull_targets(init, approach, pull, grasp_offset, pull_step, 3)
    grasp = init + grasp_offset * approach

    assert np.allclose(targets[0], grasp + pull_step * pull)
    assert np.allclose(targets[1], grasp + 2 * pull_step * pull)
    # Regression guard: old formula jumped by pull_start_offset=0.10 on first pull step.
    assert np.linalg.norm(targets[0] - grasp) < 0.02


def test_articulated_dof_positions_slice_after_franka_and_rigid_objects():
    helpers = load_helpers()

    class DummyGym:
        franka_num_dofs = 9
        obj_num_dofs = 2
        arti_obj_num_dofs = 3
        def __init__(self):
            self.dof_pos = np.arange(14, dtype=np.float32).reshape(1, 14, 1)
            self.refreshed = False
        def refresh_observation(self, get_visual_obs=False):
            self.refreshed = True

    dummy = DummyGym()
    pos = helpers._get_arti_dof_positions(dummy)

    assert dummy.refreshed
    assert np.allclose(pos, [[11.0, 12.0, 13.0]])


def test_closed_arti_dof_state_command_sets_all_articulated_dofs_to_lower_and_zero_velocity():
    helpers = load_helpers()
    dof_states = np.zeros((1, 14, 2), dtype=np.float32)
    dof_states[0, :, 0] = np.arange(14, dtype=np.float32)
    lower = np.array([0.0, -0.2, 0.1], dtype=np.float32)

    updated = helpers._closed_arti_dof_state_command(
        dof_states,
        franka_num_dofs=9,
        obj_num_dofs=2,
        arti_lower=lower,
    )

    assert np.allclose(updated[0, 11:14, 0], lower)
    assert np.allclose(updated[0, 11:14, 1], 0.0)
    assert np.allclose(updated[0, :11, 0], np.arange(11, dtype=np.float32))


def test_format_articulated_dof_diag_includes_named_joint_delta():
    helpers = load_helpers()
    text = helpers._format_arti_dof_diag(
        "after_pull",
        np.array([[0.0, 0.25]], dtype=np.float32),
        joint_desc={"name": "joint_1"},
        initial=np.array([[0.0, 0.10]], dtype=np.float32),
    )
    assert "after_pull" in text
    assert "joint_1" in text
    assert "delta" in text
    assert "0.15" in text


def test_select_target_dof_index_uses_resolved_joint_order():
    helpers = load_helpers()
    assert helpers._select_target_dof_index({"name": "joint_1"}, ["joint_0", "joint_1"]) == 1
    assert helpers._select_target_dof_index({"name": "joint_9"}, ["joint_0", "joint_1"]) is None
    assert helpers._select_target_dof_index(None, ["joint_0", "joint_1"]) is None



def test_target_abs_delta_uses_absolute_env0_displacement():
    helpers = load_helpers()
    delta = helpers._target_abs_delta(
        np.array([[0.0, -0.05]], dtype=np.float32),
        np.array([[0.0, 0.10]], dtype=np.float32),
        1,
    )
    assert np.isclose(delta, 0.15)


def test_prismatic_grasp_candidates_keep_nominal_first():
    helpers = load_helpers()
    candidates = helpers._make_prismatic_grasp_candidates(
        0.06,
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        handle_long=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        handle_short=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert candidates[0]["label"] == "nominal"
    assert np.isclose(candidates[0]["grasp_offset"], 0.06)
    assert any(c["grasp_offset"] < 0.06 for c in candidates[1:])
    assert any(np.linalg.norm(c["bias"]) > 0 for c in candidates[1:])


def test_prismatic_pull_step_count_covers_success_threshold_formula():
    pull_step = 0.01
    pull_steps = 30
    final_success_threshold = 0.33
    joint_lower = 0.0
    joint_upper = 0.66

    requested_pull_distance = min(
        max(final_success_threshold + 0.08, pull_step * pull_steps),
        abs(joint_upper - joint_lower),
    )
    adjusted_pull_steps = max(pull_steps, int(np.ceil(requested_pull_distance / pull_step)))

    assert adjusted_pull_steps == 41
    assert adjusted_pull_steps * pull_step > final_success_threshold



def test_project_point_to_axis_projects_along_normalized_axis():
    helpers = load_helpers()
    point = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis_point = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    axis_dir = np.array([0.0, 0.0, 10.0], dtype=np.float32)

    projected = helpers._project_point_to_axis(point, axis_point, axis_dir)

    assert np.allclose(projected, [0.0, 2.0, 3.0])


def test_compute_revolute_motion_geometry_returns_tangent_aligned_with_approach():
    helpers = load_helpers()
    handle_center = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    movable_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    axis_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    approach_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    geom = helpers._compute_revolute_motion_geometry(
        handle_center,
        movable_center,
        axis_dir,
        approach_dir,
    )

    assert np.allclose(geom["pivot"], [0.0, 0.0, 0.0])
    assert np.allclose(geom["radial_dir"], [1.0, 0.0, 0.0])
    assert np.allclose(geom["tangent_dir"], [0.0, 1.0, 0.0])
    assert np.isclose(geom["radius"], 1.0)


def test_compute_revolute_arc_targets_follow_circular_arc():
    helpers = load_helpers()
    geom = {
        "pivot": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "axis_dir": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "radial_dir": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "radius": 1.0,
    }
    targets = helpers._compute_revolute_arc_targets(
        handle_center=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        grasp_offset=0.0,
        approach_dir=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        geom=geom,
        angle_step=0.1,
        steps=2,
    )

    assert targets.shape == (2, 3)
    assert np.allclose(targets[0], [np.cos(0.1), np.sin(0.1), 0.0], atol=1e-5)
    assert np.allclose(targets[1], [np.cos(0.2), np.sin(0.2), 0.0], atol=1e-5)


def test_revolute_arc_targets_zero_steps_do_not_require_geometry():
    helpers = load_helpers()

    targets = helpers._compute_revolute_arc_targets(
        handle_center=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        grasp_offset=0.0,
        approach_dir=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        geom={},
        angle_step=0.1,
        steps=0,
    )

    assert targets.shape == (0, 3)
    assert targets.dtype == np.float32


def test_revolute_hold_only_targets_do_not_require_arc_geometry():
    helpers = load_helpers()

    targets = helpers._compute_revolute_grasp_hold_arc_targets(
        handle_center=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        grasp_offset=0.05,
        geom={
            "approach_dir": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "breakaway_steps": 0,
        },
        hold_steps=2,
        angle_step=0.025,
        arc_steps=0,
    )

    assert targets.shape == (2, 3)
    assert np.allclose(targets, [[1.0, 0.0, 0.05], [1.0, 0.0, 0.05]])


def test_revolute_breakaway_step_sets_circular_breakaway_angle():
    helpers = load_helpers()
    geom = {
        "pivot": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "axis_dir": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "radial_dir": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "tangent_dir": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "radius": 2.0,
        "approach_dir": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "breakaway_steps": 2,
        "breakaway_step": 0.1,
    }

    targets = helpers._compute_revolute_grasp_hold_arc_targets(
        handle_center=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        grasp_offset=0.05,
        geom=geom,
        hold_steps=0,
        angle_step=0.2,
        arc_steps=1,
        direction_sign=1.0,
    )
    angles = helpers._revolute_signed_angles_for_targets(targets, 0.05, geom)

    assert np.allclose(angles, [0.05, 0.10, 0.30], atol=1e-5)


def test_revolute_hold_breakaway_arc_targets_are_continuous():
    helpers = load_helpers()
    geom = {
        "pivot": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "axis_dir": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "radial_dir": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "tangent_dir": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "radius": 1.0,
        "approach_dir": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "breakaway_steps": 2,
    }

    targets = helpers._compute_revolute_grasp_hold_arc_targets(
        handle_center=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        grasp_offset=0.05,
        geom=geom,
        hold_steps=2,
        angle_step=0.025,
        arc_steps=4,
        direction_sign=1.0,
    )
    grasp = np.array([1.0, 0.0, 0.05], dtype=np.float32)
    diagnostics = helpers._compute_revolute_target_sequence_diagnostics(
        handle_center=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        grasp_offset=0.05,
        geom=geom,
        targets=targets,
        hold_steps=2,
        breakaway_steps=2,
        arc_steps=4,
        direction_sign=1.0,
    )

    assert targets.shape == (8, 3)
    assert np.allclose(targets[0], grasp)
    assert np.allclose(targets[1], grasp)
    assert diagnostics["first_non_hold_distance_from_grasp"] < 0.03
    assert diagnostics["breakaway_to_arc_transition_distance"] < 0.03
    assert diagnostics["max_adjacent_target_distance"] < 0.03
    assert diagnostics["arc_angle_monotonic"] is True
    assert diagnostics["hold_target_count"] == 2
    assert diagnostics["breakaway_target_count"] == 2
    assert diagnostics["arc_target_count"] == 4


def test_revolute_target_sequence_diagnostics_support_negative_direction():
    helpers = load_helpers()
    geom = {
        "pivot": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "axis_dir": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "radial_dir": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "tangent_dir": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "radius": 1.0,
        "approach_dir": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "breakaway_steps": 2,
    }

    targets = helpers._compute_revolute_grasp_hold_arc_targets(
        handle_center=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        grasp_offset=0.05,
        geom=geom,
        hold_steps=1,
        angle_step=0.025,
        arc_steps=4,
        direction_sign=-1.0,
    )
    diagnostics = helpers._compute_revolute_target_sequence_diagnostics(
        handle_center=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        grasp_offset=0.05,
        geom=geom,
        targets=targets,
        hold_steps=1,
        breakaway_steps=2,
        arc_steps=4,
        direction_sign=-1.0,
    )

    assert diagnostics["arc_angle_monotonic"] is True
    assert diagnostics["signed_angular_progress"]["final"] < 0.0
    assert diagnostics["signed_angular_progress"]["direction_sign"] == -1.0



def test_estimate_revolute_axis_point_from_bbox_uses_side_far_from_handle():
    helpers = load_helpers()
    # Door panel spans x=[0, 1], y=[-0.1, 0.1], z=[0, 2]. Handle is near x=1,
    # so the hinge-side axis point should be centered near x=0.
    bbox = np.array([
        [0.0, -0.1, 0.0], [0.0, 0.1, 0.0], [1.0, 0.1, 0.0], [1.0, -0.1, 0.0],
        [0.0, -0.1, 2.0], [0.0, 0.1, 2.0], [1.0, 0.1, 2.0], [1.0, -0.1, 2.0],
    ], dtype=np.float32)
    axis_point = helpers._estimate_revolute_axis_point_from_bbox(
        handle_center=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        movable_bbox=bbox,
        axis_dir=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert np.allclose(axis_point[:2], [0.0, 0.0], atol=1e-5)
    assert np.isclose(axis_point[2], 1.0)


def test_required_success_delta_matches_requested_thresholds():
    helpers = load_helpers()
    assert np.isclose(helpers._required_success_delta({"type": "revolute"}), np.deg2rad(30.0))
    assert np.isclose(helpers._required_success_delta({"type": "continuous"}), np.deg2rad(30.0))
    assert np.isclose(helpers._required_success_delta({"type": "prismatic"}, lower=-0.1, upper=0.3), 0.2)


def test_is_gripper_on_handle_uses_finger_geometry_not_palm_center():
    helpers = load_helpers()

    class DummyTensor:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float32)
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self.value

    class DummyGym:
        def __init__(self, hand_pos, leftfinger_pos, rightfinger_pos):
            self.hand_pos = [DummyTensor(hand_pos)]
            self.leftfinger_pos = [DummyTensor(leftfinger_pos)]
            self.rightfinger_pos = [DummyTensor(rightfinger_pos)]
            self.refreshed = False
        def refresh_observation(self, get_visual_obs=False):
            self.refreshed = True

    bbox = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.1, 0.1, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
        [0.1, 0.0, 0.1],
        [0.1, 0.1, 0.1],
        [0.0, 0.1, 0.1],
    ], dtype=np.float32)

    # The palm can be outside the handle bbox while the fingers straddle it.
    on_handle = DummyGym(
        hand_pos=[0.17, 0.05, 0.05],
        leftfinger_pos=[0.05, 0.05, 0.05],
        rightfinger_pos=[0.15, 0.05, 0.05],
    )
    far_from_handle = DummyGym(
        hand_pos=[0.3, 0.05, 0.05],
        leftfinger_pos=[0.24, 0.05, 0.05],
        rightfinger_pos=[0.26, 0.05, 0.05],
    )

    assert helpers._is_gripper_on_handle(on_handle, bbox, margin=0.03)
    assert on_handle.refreshed
    metrics = helpers._gripper_handle_metrics(on_handle, bbox, margin=0.03)
    assert metrics["left_inside"]
    assert metrics["finger_segment_intersects"]
    assert not helpers._is_gripper_on_handle(far_from_handle, bbox, margin=0.03)


def test_transform_handle_bbox_for_final_prismatic_joint_tracks_moved_handle():
    helpers = load_helpers()
    bbox = np.zeros((8, 3), dtype=np.float32)
    bbox[:, 0] = np.linspace(0.0, 0.1, 8)

    moved = helpers._transform_handle_bbox_for_final_joint(
        bbox,
        {"type": "prismatic", "axis_root": np.array([1.0, 0.0, 0.0], dtype=np.float32)},
        initial_value=0.0,
        final_value=0.25,
    )

    assert np.allclose(moved, bbox + np.array([0.25, 0.0, 0.0], dtype=np.float32))


def test_transform_handle_bbox_for_final_revolute_joint_rotates_around_axis():
    helpers = load_helpers()
    bbox = np.array([[1.0, 0.0, 0.0]] * 8, dtype=np.float32)

    moved = helpers._transform_handle_bbox_for_final_joint(
        bbox,
        {"type": "revolute", "axis_root": np.array([0.0, 0.0, 1.0], dtype=np.float32)},
        initial_value=0.0,
        final_value=np.pi / 2.0,
        revolute_axis_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert np.allclose(moved[0], [0.0, 1.0, 0.0], atol=1e-6)


def test_select_bbox_for_requested_movable_part_maps_to_matching_fixed_handle():
    helpers = load_helpers()

    gapart_anno = [
        {"is_gapart": True, "link_name": "link_0", "category": "hinge_door"},
        {"is_gapart": True, "link_name": "link_1", "category": "slider_drawer"},
        {"is_gapart": True, "link_name": "link_2", "category": "slider_drawer"},
        {"is_gapart": False, "link_name": "link_3", "category": ""},
        {"is_gapart": True, "link_name": "link_4", "category": "line_fixed_handle"},
        {"is_gapart": True, "link_name": "link_5", "category": "line_fixed_handle"},
        {"is_gapart": True, "link_name": "link_6", "category": "line_fixed_handle"},
    ]
    gapart_raw_valid_anno = [anno for anno in gapart_anno if anno["is_gapart"]]
    fixed_parent = {
        "link_4": "link_0",
        "link_5": "link_1",
        "link_6": "link_2",
    }
    fixed_rotation = {}
    movable_joints_by_child = {
        "link_0": {"name": "joint_0", "type": "revolute", "parent": "base", "child": "link_0", "axis": np.array([0.0, 0.0, 1.0], dtype=np.float32)},
        "link_1": {"name": "joint_1", "type": "prismatic", "parent": "base", "child": "link_1", "axis": np.array([1.0, 0.0, 0.0], dtype=np.float32)},
        "link_2": {"name": "joint_2", "type": "prismatic", "parent": "base", "child": "link_2", "axis": np.array([1.0, 0.0, 0.0], dtype=np.float32)},
    }

    bbox_id, source, joint = helpers._select_bbox_for_requested_part(
        1,
        gapart_anno,
        gapart_raw_valid_anno,
        fixed_parent,
        fixed_rotation,
        movable_joints_by_child,
    )

    assert bbox_id == 4
    assert source == "raw_part_id_mapped_to_handle"
    assert joint["name"] == "joint_1"
    assert gapart_raw_valid_anno[bbox_id]["link_name"] == "link_5"


def test_legacy_open_demo_targets_match_c8d4ad2_offsets():
    helpers = load_helpers()
    init = np.array([0.6, 0.1, 0.3], dtype=np.float32)
    handle_out = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    pre_grasp, grasp, pull_targets = helpers._legacy_open_demo_targets(init, handle_out)

    assert np.allclose(pre_grasp, [0.4, 0.1, 0.3])
    assert np.allclose(grasp, [0.5, 0.1, 0.3])
    assert pull_targets.shape == (30, 3)
    assert np.allclose(pull_targets[0], [0.5, 0.1, 0.3])
    assert np.allclose(pull_targets[-1], [0.21, 0.1, 0.3])


def test_pose_z_for_support_surface_height_places_scaled_bottom_on_surface():
    helpers = load_helpers()

    pose_z = helpers._pose_z_for_support_surface_height(
        min_z=0.25,
        scale=0.4,
        support_surface_z=0.43,
    )

    assert np.isclose(pose_z, 0.33)
    assert np.isclose(pose_z + 0.4 * 0.25, 0.43)


def test_support_surface_z_for_asset_only_uses_tabletop_for_27044():
    helpers = load_helpers()

    assert np.isclose(helpers._support_surface_z_for_asset("27044"), 0.43)
    assert np.isclose(helpers._support_surface_z_for_asset("40147"), 0.0)
    assert np.isclose(helpers._support_surface_z_for_asset("45661"), 0.0)


def test_frame_sequence_metadata_counts_jpg_frames(tmp_path):
    helpers = load_helpers()
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    import cv2
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    cv2.imwrite(str(video_dir / "step-0000.jpg"), image)
    cv2.imwrite(str(video_dir / "step-0001.jpg"), image)

    metadata = helpers._frame_sequence_metadata(str(tmp_path), extension="jpg")

    assert metadata["extension"] == "jpg"
    assert metadata["count"] == 2
    assert metadata["width"] == 6
    assert metadata["height"] == 4


def test_frame_sequence_metadata_defaults_to_png_frames(tmp_path):
    helpers = load_helpers()
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    import cv2
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    cv2.imwrite(str(video_dir / "step-0000.png"), image)
    cv2.imwrite(str(video_dir / "step-0001.png"), image)

    metadata = helpers._frame_sequence_metadata(str(tmp_path))

    assert metadata["extension"] == "png"
    assert metadata["count"] == 2
    assert metadata["width"] == 6
    assert metadata["height"] == 4


def test_write_video_from_frame_list_does_not_create_png_sidecars_by_default(tmp_path):
    helpers = load_helpers()
    frames = [
        np.zeros((8, 10, 3), dtype=np.uint8),
        np.full((8, 10, 3), 255, dtype=np.uint8),
    ]

    result = helpers._write_video_mp4_from_frame_list(str(tmp_path), frames, output_name="manipulation.mp4", fps=5)

    assert result["path"].endswith("manipulation.mp4")
    assert result["writer"] in {"ffmpeg_rawvideo", "opencv_mp4v"}
    assert result["frame_metadata"]["count"] == 2
    assert result["video_metadata"]["exists"] is True
    assert result["video_metadata"]["width"] == 10
    assert result["video_metadata"]["height"] == 8
    assert pathlib.Path(result["path"]).exists()
    assert not (tmp_path / "video").exists()


def test_legacy_output_without_result_is_visual_baseline_not_joint_identity(tmp_path):
    helpers = load_helpers()
    output = tmp_path / "45661"
    (output / "video").mkdir(parents=True)
    (output / "manipulation.mp4").write_bytes(b"not a real video")

    metadata = helpers._legacy_output_baseline_metadata(str(output))

    assert metadata["baseline_success_standard"] == "old_video"
    assert metadata["baseline_result_json_exists"] is False
    assert metadata["joint_identity_inferred"] is False
    assert metadata["baseline_video_metadata"]["exists"] is True


def test_joint_type_request_skips_legacy_open_demo():
    helpers = load_helpers()

    assert helpers._should_run_legacy_open_demo(None) is True
    assert helpers._should_run_legacy_open_demo("prismatic") is False
    assert helpers._should_run_legacy_open_demo("revolute") is False


def test_control_profile_metadata_marks_joint_aware_without_legacy_demo():
    helpers = load_helpers()

    metadata = helpers._make_control_profile_metadata(
        requested_joint_type="revolute",
        legacy_demo_executed=False,
    )

    assert metadata == {
        "control_profile": "joint_aware_run_arti_open",
        "legacy_demo_executed": False,
    }


def test_reset_video_capture_buffer_clears_memory_and_disk_frames(tmp_path):
    helpers = load_helpers()
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    (video_dir / "step-0000.png").write_bytes(b"legacy")
    (video_dir / "step-0001.png").write_bytes(b"legacy")

    class DummyGym:
        save_root = str(tmp_path)

        def __init__(self):
            self.video_frames = ["legacy-frame-0", "legacy-frame-1"]

    gym = DummyGym()

    removed = helpers._reset_video_capture_buffer(gym, remove_disk_frames=True)

    assert removed == 2
    assert gym.video_frames == []
    assert list(video_dir.glob("step-*.png")) == []


def test_json_safe_converts_numpy_values():
    helpers = load_helpers()
    value = {
        "axis": np.array([1.0, 2.0], dtype=np.float32),
        "index": np.int64(3),
        "items": [np.float32(0.5)],
    }

    converted = helpers._json_safe(value)

    assert converted == {"axis": [1.0, 2.0], "index": 3, "items": [0.5]}

def test_normalize_joint_type_accepts_prismatic_revolute_and_continuous_alias():
    helpers = load_helpers()

    assert helpers._normalize_requested_joint_type(None) is None
    assert helpers._normalize_requested_joint_type("prismatic") == "prismatic"
    assert helpers._normalize_requested_joint_type("revolute") == "revolute"
    assert helpers._normalize_resolved_joint_type("continuous") == "revolute"


def test_normalize_requested_joint_type_rejects_unknown_type():
    helpers = load_helpers()

    try:
        helpers._normalize_requested_joint_type("fixed")
    except ValueError as exc:
        assert "Unsupported --joint_type" in str(exc)
    else:
        raise AssertionError("expected invalid requested joint type to raise")


def test_joint_type_mismatch_result_stops_before_operation():
    helpers = load_helpers()

    result = helpers._joint_type_mismatch_result(
        gapart_id="12345",
        object_path="/tmp/12345",
        task_root="output",
        save_root="output/12345",
        requested_joint_type="revolute",
        resolved_joint_type="prismatic",
        tested_part_id=2,
        selected_bbox_id=4,
        selected_source="raw_part_id_mapped_to_handle",
        selected_link="link_5",
        selected_category="line_fixed_handle",
        resolved_joint={"name": "joint_1", "type": "prismatic"},
    )

    assert result["status"] == "failure"
    assert result["failure_reason"] == "requested_joint_type_mismatch"
    assert result["requested_joint_type"] == "revolute"
    assert result["resolved_joint_type"] == "prismatic"
    assert result["control_profile"] == "joint_aware_run_arti_open"
    assert result["legacy_demo_executed"] is False
    assert result["tested_part_id"] == 2
    assert result["selected_joint"] == {"name": "joint_1", "type": "prismatic"}


def test_orthonormal_frame_from_z_uses_requested_z_axis():
    helpers = load_helpers()

    frame = helpers._orthonormal_frame_from_z(
        np.array([0.0, 0.0, 2.0], dtype=np.float32),
        x_hint=np.array([1.0, 1.0, 0.0], dtype=np.float32),
    )

    assert frame.shape == (3, 3)
    assert np.allclose(frame[:, 2], [0.0, 0.0, 1.0])
    assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-6)
