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
    "_estimate_revolute_axis_point_from_bbox",
}


def load_helpers():
    tree = ast.parse(RUN_PATH.read_text())
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            any(alias.name in {"numpy", "torch", "scipy.spatial.transform"} for alias in node.names)
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
