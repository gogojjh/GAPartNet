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


def test_closed_dof_positions_prefers_zero_when_inside_limits():
    helpers = load_helpers()
    lower = np.array([-1.0, 0.0, -0.5], dtype=np.float32)
    upper = np.array([1.0, 0.8, 0.5], dtype=np.float32)
    assert np.allclose(helpers._closed_dof_positions(lower, upper), [0.0, 0.0, 0.0])


def test_closed_dof_positions_clips_zero_to_limits_when_outside():
    helpers = load_helpers()
    lower = np.array([0.2, -2.0], dtype=np.float32)
    upper = np.array([1.0, -0.1], dtype=np.float32)
    assert np.allclose(helpers._closed_dof_positions(lower, upper), [0.2, -0.1])
