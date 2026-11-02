import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from concertina.eval import runner
from concertina.eval.runner import CalvinBenchmark, ClientEnvConfig, EvalRunConfig, shard

FAKE_SERVER = """
import sys, threading
import numpy as np
from concertina.eval.server import PolicyServer

class Model:
    def reset_model(self):
        pass
    def predict(self, obs):
        return np.zeros(7, dtype=np.float32)

port = int(open(sys.argv[1]).read().split("port:")[1].split()[0])
PolicyServer(Model(), "127.0.0.1", port).serve_forever()
"""

# Checks that the client resolves `concertina_eval_client` from the runner's shim directory and sees only its GPU.
FAKE_CLIENT = """
import json, os, sys
from pathlib import Path
from concertina_eval_client.protocol import PolicyClient
import concertina_eval_client
port, out, units = int(sys.argv[1]), Path(sys.argv[2]), [int(u) for u in sys.argv[3].split(",")]
assert Path(concertina_eval_client.__file__).parent.parent.name == "_client_path", concertina_eval_client.__file__
with PolicyClient(port=port, timeout=30, connect_timeout=30) as client, open(out / "results.jsonl", "a") as f:
    for unit in units:
        client.reset()
        client.predict({})
        f.write(json.dumps({"unit": unit, "gpu": os.environ["CUDA_VISIBLE_DEVICES"]}) + "\\n")
"""


class FakeBenchmark:
    contiguous = False
    needs_server = True

    def __init__(self, config):
        self.units = list(range(7))
        self.script = Path(config.output_dir) / "fake_client.py"

    def worker_command(self, units, port, gpu, worker_dir):
        return [sys.executable, str(self.script), str(port), str(worker_dir), ",".join(map(str, units))]

    def done(self, worker_dirs):
        return set()

    def merge(self, output_dir):
        rows = [
            json.loads(line)
            for d in sorted(output_dir.glob("worker_*"))
            for line in (d / "results.jsonl").read_text().splitlines()
        ]
        return {"units": sorted(row["unit"] for row in rows), "rows": rows}


def test_shard_is_strided_or_contiguous_and_complete():
    assert shard(list(range(10)), 3) == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    assert shard(list(range(10)), 3, contiguous=True) == [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
    assert shard([1], 3) == [[1], [], []]
    assert shard([1], 3, contiguous=True) == [[], [1], []]
    assert sorted(sum(shard(list(range(1000)), 7, contiguous=True), [])) == list(range(1000))


@pytest.mark.parametrize("hub_id", [False, True], ids=["local", "hub"])
def test_run_launches_one_server_and_client_per_gpu(tmp_path, monkeypatch, hub_id):
    (tmp_path / "fake_server.py").write_text(FAKE_SERVER)
    (tmp_path / "fake_client.py").write_text(textwrap.dedent(FAKE_CLIENT))
    monkeypatch.setitem(runner.BENCHMARKS, "fake", FakeBenchmark)
    monkeypatch.setattr(
        runner, "_server_command", lambda path: [sys.executable, str(tmp_path / "fake_server.py"), str(path)]
    )
    monkeypatch.chdir(tmp_path)
    # A Hub repository id reaches the servers unchanged; a relative local directory becomes absolute.
    (tmp_path / "policy").mkdir()
    policy, expected = (
        ("org/fake-policy", "org/fake-policy") if hub_id else ("policy", str(tmp_path / "policy"))
    )
    config = EvalRunConfig(
        benchmark="fake",
        policy_path=policy,
        output_dir=str(tmp_path),
        gpus=[5, 2],
        server_startup_timeout_s=60,
    )
    config.server.image_keys = {"cam": "observation.images.cam"}
    summary = runner.run(config)
    assert summary["units"] == list(range(7))
    assert {row["unit"]: row["gpu"] for row in summary["rows"]} == {
        u: "5" if u % 2 == 0 else "2" for u in range(7)
    }
    assert json.loads((tmp_path / "summary.json").read_text())["policy_path"] == expected
    server_config = yaml.safe_load((tmp_path / "worker_00_gpu5" / "server_config.yaml").read_text())
    assert server_config["policy_path"] == expected


def test_calvin_merge_requires_every_sequence(tmp_path):
    config = EvalRunConfig(benchmark="calvin", policy_path="p", output_dir=str(tmp_path), gpus=[0])
    config.calvin.manifest = str(
        Path(__file__).resolve().parents[2] / "configs" / "eval" / "calvin_sequences_1000.json"
    )
    config.calvin.num_sequences = 3
    benchmark = CalvinBenchmark(config)
    worker = tmp_path / "worker_00_gpu0"
    worker.mkdir()
    rows = [
        {"sequence_index": i, "sequence": ["a", "b", "c", "d", "e"], "success_len": i, "num_steps": 1}
        for i in (0, 2)
    ]
    (worker / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="1 CALVIN sequences"):
        benchmark.merge(tmp_path)
    with open(worker / "results.jsonl", "a") as file:
        file.write(json.dumps({**rows[0], "sequence_index": 1, "success_len": 4}) + "\n")
    summary = benchmark.merge(tmp_path)
    assert summary["avg_seq_len"] == pytest.approx(2.0)
    assert [row["sequence_index"] for row in summary["sequences"]] == [0, 1, 2]


def test_calvin_sharding_is_opt_in(tmp_path):
    config = EvalRunConfig(benchmark="calvin", policy_path="p", output_dir=str(tmp_path), gpus=[0, 1])
    config.calvin.manifest = str(
        Path(__file__).resolve().parents[2] / "configs" / "eval" / "calvin_sequences_1000.json"
    )
    with pytest.raises(ValueError, match="calvin.sharded"):
        CalvinBenchmark(config)
    config.calvin.sharded = True
    assert CalvinBenchmark(config).contiguous


def test_conda_client_runs_the_environment_interpreter_by_absolute_path(monkeypatch):
    listing = {
        "root_prefix": "/opt/conda",
        "envs": ["/opt/conda", "/opt/conda/envs/concertina", "/opt/conda/envs/calvin"],
    }
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(listing))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner._conda_env_prefix.cache_clear()
    try:
        assert ClientEnvConfig(conda_env="calvin").command() == [
            "conda", "run", "--no-capture-output", "-n", "calvin", "/opt/conda/envs/calvin/bin/python"
        ]  # fmt: skip
        assert ClientEnvConfig(conda_env="base").command()[-1] == "/opt/conda/bin/python"
        with pytest.raises(ValueError, match="missing"):
            ClientEnvConfig(conda_env="missing").command()
        assert ClientEnvConfig(python="/usr/bin/python3").command() == ["/usr/bin/python3"]
    finally:
        runner._conda_env_prefix.cache_clear()
    assert calls == [["conda", "env", "list", "--json"]] * 3
