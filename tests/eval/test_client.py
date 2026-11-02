import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import concertina_eval_client
from concertina_eval_client import calvin

CLIENT_DIR = Path(concertina_eval_client.__file__).parent
CLIENT_MODULES = sorted(CLIENT_DIR.rglob("*.py"))
FORBIDDEN = ("torch", "lerobot", "concertina", "transformers")


def test_client_modules_import_with_numpy_only(tmp_path):
    """Simulation environments get only `concertina_eval_client` on their path; importing it must not pull in the
    policy stack."""
    (tmp_path / "concertina_eval_client").symlink_to(CLIENT_DIR)
    modules = [
        "concertina_eval_client." + ".".join(path.relative_to(CLIENT_DIR).with_suffix("").parts)
        for path in CLIENT_MODULES
        if path.name != "__init__.py"
    ]
    code = (
        "import importlib, sys, json\n"
        f"for name in {modules!r}: importlib.import_module(name)\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(tmp_path)!r})\n{code}"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    loaded = set(json.loads(result.stdout))
    assert not loaded & set(FORBIDDEN), sorted(loaded & set(FORBIDDEN))


@pytest.mark.parametrize("path", CLIENT_MODULES, ids=lambda path: path.name)
def test_client_modules_parse_as_python_38(path):
    ast.parse(path.read_text(), filename=str(path), feature_version=(3, 8))


def test_summary_counts_chains_and_tasks():
    rows = [
        {"sequence_index": 1, "sequence": ["a", "b", "c", "d", "e"], "success_len": 5},
        {"sequence_index": 0, "sequence": ["a", "c", "b", "d", "e"], "success_len": 1},
        {"sequence_index": 2, "sequence": ["b", "a", "c", "d", "e"], "success_len": 0},
    ]
    summary = calvin.summarize(rows)
    assert summary["num_sequences"] == 3
    assert summary["avg_seq_len"] == pytest.approx(2.0)
    assert summary["chain_sr"] == {"1": 2 / 3, "2": 1 / 3, "3": 1 / 3, "4": 1 / 3, "5": 1 / 3}
    assert summary["tasks"]["a"] == {"success": 2, "total": 2, "success_rate": 1.0}
    assert summary["tasks"]["b"] == {"success": 1, "total": 2, "success_rate": 0.5}
    assert summary["tasks"]["c"] == {"success": 1, "total": 2, "success_rate": 0.5}


def test_env_action_splits_absolute_target_and_binarizes_gripper():
    action = np.array([0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 0.0], np.float32)
    position, orientation, gripper = calvin.env_action(action)
    assert position.tobytes() == action[:3].tobytes() and orientation.tobytes() == action[3:6].tobytes()
    assert gripper.tolist() == [-1.0]
    assert calvin.env_action(np.array([0, 0, 0, 0, 0, 0, 0.2], np.float32))[2].tolist() == [1.0]
    with pytest.raises(ValueError):
        calvin.env_action(np.array([np.nan] * 7, np.float32))


def test_manifest_preserves_initial_state_key_order(tmp_path):
    rows = [
        {"sequence_index": 0, "initial_state": {"led": 1, "drawer": "open", "blue_block": "table"},
         "sequence": ["a", "b", "c", "d", "e"]},
    ]  # fmt: skip
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"sequences": rows}))
    assert list(calvin.load_manifest(path)[0]["initial_state"]) == ["led", "drawer", "blue_block"]
    rows[0]["sequence_index"] = 3
    path.write_text(json.dumps({"sequences": rows}))
    with pytest.raises(ValueError):
        calvin.load_manifest(path)


def test_released_manifest_has_1000_sequences():
    manifest = Path(__file__).resolve().parents[2] / "configs" / "eval" / "calvin_sequences_1000.json"
    rows = calvin.load_manifest(manifest)
    assert len(rows) == 1000
    assert list(rows[0]["initial_state"]) == [
        "led", "lightbulb", "slider", "drawer", "red_block", "blue_block", "pink_block", "grasped"
    ]  # fmt: skip


def test_parse_indices():
    assert calvin.parse_indices("0-3,7, 9") == [0, 1, 2, 3, 7, 9]
