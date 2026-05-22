import ast
import pathlib
import types
import numpy as np

OBJECT_GYM_PATH = pathlib.Path(__file__).resolve().parents[1] / "manipulation" / "object_gym.py"


def load_helpers():
    tree = ast.parse(OBJECT_GYM_PATH.read_text())
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(alias.name == "numpy" for alias in node.names)
    ]
    selected += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_closed_dof_positions"]
    module = types.ModuleType("object_gym_helpers_for_test")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(OBJECT_GYM_PATH), "exec"), module.__dict__)
    return module


def test_closed_dof_positions_uses_lower_limit_to_match_legacy_initial_pose():
    helpers = load_helpers()
    lower = np.array([-1.0, 0.0, -0.5], dtype=np.float32)
    upper = np.array([1.0, 0.8, 0.5], dtype=np.float32)
    assert np.allclose(helpers._closed_dof_positions(lower, upper), [-1.0, 0.0, -0.5])


def test_closed_dof_positions_ignores_upper_limit_and_uses_lower_limit():
    helpers = load_helpers()
    lower = np.array([0.2, -2.0], dtype=np.float32)
    upper = np.array([1.0, -0.1], dtype=np.float32)
    assert np.allclose(helpers._closed_dof_positions(lower, upper), [0.2, -2.0])


def test_articulated_asset_defaults_match_legacy_commit_parameters():
    source = OBJECT_GYM_PATH.read_text()
    assert 'arti_obj_asset_options.disable_gravity = self.cfgs["asset"].get("arti_disable_gravity", False)' in source


def test_articulated_scale_is_applied_after_initial_dof_state_like_legacy_commit():
    source = OBJECT_GYM_PATH.read_text()
    create_idx = source.index("arti_obj_actor_handle = self.gym.create_actor")
    dof_state_idx = source.index("self.gym.set_actor_dof_states(env, arti_obj_actor_handle", create_idx)
    scale_idx = source.index("self.gym.set_actor_scale(env, arti_obj_actor_handle", create_idx)
    shape_props_idx = source.index("agent_shape_props = self.gym.get_actor_rigid_shape_properties", create_idx)

    assert dof_state_idx < scale_idx < shape_props_idx
