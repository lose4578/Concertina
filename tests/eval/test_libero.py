import json

import pytest

from concertina.eval.libero import merge_tasks


def test_merge_tasks_requires_every_listed_task(tmp_path):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks.json").write_text(
        json.dumps([["libero_goal", 0], ["libero_goal", 1], ["libero_10", 0]])
    )
    for (suite, task_id), successes in {
        ("libero_goal", 0): [True, False],
        ("libero_10", 0): [True, True],
    }.items():
        (tmp_path / "tasks" / f"{suite}_{task_id}.json").write_text(json.dumps({"successes": successes}))
    with pytest.raises(RuntimeError, match="1 LIBERO tasks"):
        merge_tasks(tmp_path)
    (tmp_path / "tasks" / "libero_goal_1.json").write_text(json.dumps({"successes": [False, False]}))
    summary = merge_tasks(tmp_path)
    assert summary["overall"]["n_episodes"] == 6 and summary["overall"]["n_success"] == 3
    assert summary["per_suite"]["libero_goal"]["pc_success"] == 25.0
    assert summary["per_suite"]["libero_10"]["pc_success"] == 100.0
