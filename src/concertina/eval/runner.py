"""Parallel evaluation runner: CALVIN (simulator in its own environment, policy server) and LIBERO (in process).

Each selected GPU hosts one worker: for CALVIN a `concertina-eval-server` process and the simulation client, both
pinned to that GPU (one worker unless `calvin.sharded`, then contiguous blocks of sequences); for LIBERO one
`concertina.eval.libero` process evaluating a shard of the tasks. Every worker writes per-unit results as units
finish, and the runner merges them into ``summary.json``. Re-running with the same output directory skips finished units.

    concertina-eval --config_path configs/eval/calvin_abc_d.yaml --policy_path <policy dir> \
        --output_dir <run dir> --gpus '[0,1]'
"""

import functools
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus

import concertina_eval_client
from concertina.eval.server import ServerConfig, resolve_policy_path
from concertina_eval_client import calvin as calvin_client

logger = logging.getLogger(__name__)


@dataclass
class ClientEnvConfig:
    # Interpreter of the simulation environment, or a conda environment run through `conda run`.
    python: str | None = None
    conda_env: str | None = None
    # Extra environment variables of the client processes; values may reference ${VARS}. A PYTHONPATH entry is
    # appended to the directory providing `concertina_eval_client`.
    env: dict[str, str] = field(default_factory=dict)

    def command(self) -> list[str]:
        if (self.python is None) == (self.conda_env is None):
            raise ValueError("Set exactly one of client.python and client.conda_env")
        if self.python is not None:
            return [os.path.expandvars(self.python)]
        # The environment's own interpreter by absolute path: a bare `python` can resolve to another environment's
        # interpreter when that environment's bin/ was put on PATH without `conda activate`.
        python = str(_conda_env_prefix(self.conda_env) / "bin" / "python")
        return ["conda", "run", "--no-capture-output", "-n", self.conda_env, python]


@functools.cache
def _conda_env_prefix(name: str) -> Path:
    listing = json.loads(
        subprocess.run(["conda", "env", "list", "--json"], capture_output=True, text=True, check=True).stdout
    )
    if name == "base" and listing.get("root_prefix"):
        return Path(listing["root_prefix"])
    for prefix in map(Path, listing["envs"]):
        if prefix.name == name:
            return prefix
    raise ValueError(f"No conda environment named {name!r}")


@dataclass
class CalvinConfig:
    calvin_root: str = "${CALVIN_ROOT}"
    # Split directory holding `validation/` (e.g. task_ABC_D).
    dataset_path: str = "${CALVIN_DATASET}"
    # Sequence manifest (relative paths are resolved against the working directory).
    manifest: str = "configs/eval/calvin_sequences_1000.json"
    # Evaluate the first `num_sequences` manifest sequences (None: all).
    num_sequences: int | None = None
    ep_len: int = 360
    seed: int = 0
    # "egl" (GPU rendering on the worker's GPU) or "tiny" (PyBullet software renderer); scores differ slightly.
    renderer: str = "egl"
    video: bool = True
    video_fps: int = 12
    # CALVIN's protocol evaluates all sequences in order in one environment, which carries state from one sequence
    # into the next. Splitting them over several GPUs (contiguous blocks) is faster but changes the outcome of the
    # sequences at block starts and after them; it must be enabled explicitly.
    sharded: bool = False


@dataclass
class LiberoRunConfig:
    # `concertina.eval.libero` configuration (e.g. configs/eval/libero.yaml); run with `client.python` or
    # `client.conda_env`, the environment holding LIBERO and this package.
    config_path: str = "configs/eval/libero.yaml"


@dataclass
class EvalRunConfig:
    benchmark: str = "calvin"
    policy_path: str = ""
    output_dir: str = ""
    # Physical GPU indices (nvidia-smi order); one worker per entry.
    gpus: list[int] = field(default_factory=list)
    server: ServerConfig = field(default_factory=ServerConfig)
    client: ClientEnvConfig = field(default_factory=ClientEnvConfig)
    calvin: CalvinConfig = field(default_factory=CalvinConfig)
    libero: LiberoRunConfig = field(default_factory=LiberoRunConfig)
    server_startup_timeout_s: float = 900.0


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    if "$" in str(path):
        raise ValueError(f"Unset environment variable in path {value!r}")
    return path if path.is_absolute() else base / path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _client_pythonpath(output_dir: Path) -> Path:
    """A directory exposing only `concertina_eval_client`, for interpreters that cannot import the full package."""
    root = output_dir / "_client_path"
    root.mkdir(parents=True, exist_ok=True)
    link = root / "concertina_eval_client"
    target = Path(concertina_eval_client.__file__).resolve().parent
    if link.is_symlink() and link.resolve() != target:
        link.unlink()
    if not link.exists():
        link.symlink_to(target)
    return root


def _launch(
    command: list[str], log_path: Path, env: dict[str, str], cwd: Path | None = None
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        log.write(f"$ {' '.join(command)}\n".encode())
        log.flush()
        # The child keeps its own copy of the log descriptor.
        return subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True
        )


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def _wait_for_port(port: int, process: subprocess.Popen, timeout_s: float, log_path: Path) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited with code {process.returncode}; see {log_path}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"Policy server on port {port} not ready after {timeout_s:.0f}s; see {log_path}")


def _server_command(config_path: Path) -> list[str]:
    return [sys.executable, "-m", "concertina.eval.server", "--config_path", str(config_path)]


def shard(units: list, num_workers: int, contiguous: bool = False) -> list[list]:
    """Strided (unit ``i`` to worker ``i % num_workers``) or contiguous blocks of near-equal size."""
    if not contiguous:
        return [units[worker::num_workers] for worker in range(num_workers)]
    bounds = [round(worker * len(units) / num_workers) for worker in range(num_workers + 1)]
    return [units[bounds[worker] : bounds[worker + 1]] for worker in range(num_workers)]


def _env_for_gpu(gpu: int, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update({key: os.path.expandvars(value) for key, value in (extra or {}).items()})
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


class CalvinBenchmark:
    needs_server = True
    # A CALVIN environment carries state from one sequence into the next, so each worker evaluates a contiguous
    # block in manifest order: with one worker this is CALVIN's own sequential protocol.
    contiguous = True

    def __init__(self, config: EvalRunConfig) -> None:
        self.config = config
        self.calvin = config.calvin
        self.manifest = _resolve_path(self.calvin.manifest, Path.cwd())
        rows = calvin_client.load_manifest(self.manifest)
        count = len(rows) if self.calvin.num_sequences is None else self.calvin.num_sequences
        if not 0 < count <= len(rows):
            raise ValueError(f"num_sequences={count} is outside the {len(rows)} manifest sequences")
        self.units = list(range(count))
        if len(config.gpus) > 1 and not self.calvin.sharded:
            raise ValueError(
                "CALVIN's protocol runs the sequences in order in one environment; use one GPU, or set "
                "calvin.sharded=true to split them (per-sequence outcomes then differ from the sequential run)"
            )
        if len(config.gpus) > 1:
            logger.warning(
                "Sharded CALVIN evaluation: results are not those of CALVIN's sequential protocol "
                "(the environment carries state across sequences)"
            )

    def worker_command(self, units: list[int], port: int, gpu: int, worker_dir: Path) -> list[str]:
        calvin = self.calvin
        command = [
            *self.config.client.command(),
            "-m",
            "concertina_eval_client.calvin",
            "--calvin-root",
            str(_resolve_path(calvin.calvin_root, Path.cwd())),
            "--dataset-path",
            str(_resolve_path(calvin.dataset_path, Path.cwd())),
            "--manifest",
            str(self.manifest),
            "--indices",
            ",".join(map(str, units)),
            "--port",
            str(port),
            "--output-dir",
            str(worker_dir),
            "--ep-len",
            str(calvin.ep_len),
            "--seed",
            str(calvin.seed),
            "--renderer",
            calvin.renderer,
            "--egl-device",
            str(gpu),
            "--video-fps",
            str(calvin.video_fps),
        ]
        if not calvin.video:
            command.append("--no-video")
        return command

    def done(self, worker_dirs: list[Path]) -> set[int]:
        return {
            int(row["sequence_index"])
            for worker_dir in worker_dirs
            for row in calvin_client.read_results(worker_dir / "results.jsonl")
        }

    def merge(self, output_dir: Path) -> dict[str, Any]:
        rows: dict[int, dict] = {}
        for worker_dir in sorted(output_dir.glob("worker_*")):
            for row in calvin_client.read_results(worker_dir / "results.jsonl"):
                rows[int(row["sequence_index"])] = {**row, "worker": worker_dir.name}
        missing = sorted(set(self.units) - set(rows))
        if missing:
            raise RuntimeError(f"{len(missing)} CALVIN sequences have no result, e.g. {missing[:10]}")
        selected = [rows[index] for index in self.units]
        summary = calvin_client.summarize(selected)
        summary["sequences"] = [
            {
                key: row[key]
                for key in ("sequence_index", "success_len", "num_steps", "worker", "video")
                if key in row
            }
            for row in selected
        ]
        return summary


class LiberoBenchmark:
    """LIBERO: each worker is one in-process `concertina.eval.libero` run over a shard of the tasks.

    The workers share the run directory: task results go to ``tasks/`` (finished tasks are skipped on a re-run,
    whatever the sharding) and videos to ``videos/``.
    """

    needs_server = False
    contiguous = True

    def __init__(self, config: EvalRunConfig) -> None:
        self.config = config
        self.config_path = _resolve_path(config.libero.config_path, Path.cwd())
        self.units = list(range(len(config.gpus)))

    def worker_command(self, units: list[int], port: int, gpu: int, worker_dir: Path) -> list[str]:
        (shard_index,) = units
        return [
            *self.config.client.command(),
            "-m",
            "concertina.eval.libero",
            "--config_path",
            str(self.config_path),
            "--policy_path",
            self.config.policy_path,
            "--output_dir",
            str(worker_dir.parent),
            "--shard_index",
            str(shard_index),
            "--num_shards",
            str(len(self.units)),
        ]

    def done(self, worker_dirs: list[Path]) -> set[int]:
        return set()

    def merge(self, output_dir: Path) -> dict[str, Any]:
        from concertina.eval.libero import merge_tasks

        return merge_tasks(output_dir)


BENCHMARKS = {"calvin": CalvinBenchmark, "libero": LiberoBenchmark}


def run(config: EvalRunConfig) -> dict[str, Any]:
    if config.benchmark not in BENCHMARKS:
        raise ValueError(f"benchmark must be one of {sorted(BENCHMARKS)}, got {config.benchmark!r}")
    if not config.policy_path or not config.output_dir or not config.gpus:
        raise ValueError("policy_path, output_dir and gpus are required")
    if len(set(config.gpus)) != len(config.gpus):
        raise ValueError(f"Each GPU hosts one worker; got duplicates in {config.gpus}")
    benchmark = BENCHMARKS[config.benchmark](config)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.policy_path = resolve_policy_path(config.policy_path)
    with open(output_dir / "eval_config.yaml", "w") as file:
        draccus.dump(config, file)
    client_path = _client_pythonpath(output_dir)

    # Units finished by earlier runs into this directory (with any GPU assignment) are not repeated.
    done = benchmark.done(sorted(output_dir.glob("worker_*")))
    todo = [unit for unit in benchmark.units if unit not in done]
    logger.info("%d of %d units to evaluate", len(todo), len(benchmark.units))
    workers = [
        (worker, gpu, units, output_dir / f"worker_{worker:02d}_gpu{gpu}")
        for (worker, gpu), units in zip(
            enumerate(config.gpus), shard(todo, len(config.gpus), benchmark.contiguous), strict=True
        )
        if units
    ]
    servers: list[subprocess.Popen] = []
    clients: list[tuple[subprocess.Popen, Path]] = []
    try:
        ports = []
        for worker, gpu, units, worker_dir in workers:
            worker_dir.mkdir(parents=True, exist_ok=True)
            if not benchmark.needs_server:
                ports.append(0)
                continue
            port = _free_port()
            server_config = ServerConfig(
                **{**vars(config.server), "policy_path": config.policy_path, "port": port}
            )
            with open(worker_dir / "server_config.yaml", "w") as file:
                draccus.dump(server_config, file)
            command = _server_command(worker_dir / "server_config.yaml")
            servers.append(_launch(command, worker_dir / "server.log", _env_for_gpu(gpu)))
            ports.append(port)
            logger.info("worker %d: GPU %d, port %d, %d units", worker, gpu, port, len(units))
        for process, port, (_, _, _, worker_dir) in zip(servers, ports, workers, strict=False):
            _wait_for_port(port, process, config.server_startup_timeout_s, worker_dir / "server.log")

        for port, (_, gpu, units, worker_dir) in zip(ports, workers, strict=True):
            env = _env_for_gpu(gpu, config.client.env)
            # Only the client shim and explicitly configured entries: the runner's own path must not leak into
            # another interpreter.
            extra = os.path.expandvars(config.client.env.get("PYTHONPATH", ""))
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(client_path), extra]))
            command = benchmark.worker_command(units, port, gpu, worker_dir)
            clients.append((_launch(command, worker_dir / "client.log", env), worker_dir))
        failures = []
        while clients:
            time.sleep(5)
            for process, worker_dir in list(clients):
                if process.poll() is None:
                    continue
                clients.remove((process, worker_dir))
                if process.returncode != 0:
                    failures.append(f"{worker_dir.name} (exit {process.returncode})")
        if failures:
            raise RuntimeError(f"Workers failed: {failures}; see their client.log")
    finally:
        for process, *_ in clients:
            _terminate(process)
        for process in servers:
            _terminate(process)

    summary = benchmark.merge(output_dir)
    summary = {"benchmark": config.benchmark, "policy_path": config.policy_path, **summary}
    with tempfile.NamedTemporaryFile("w", dir=output_dir, delete=False, suffix=".json") as file:
        json.dump(summary, file, indent=2)
    os.replace(file.name, output_dir / "summary.json")
    return summary


def _exit_on_signal(signum: int, frame: Any) -> None:
    # Raise in the main thread so that `run` terminates its servers and clients.
    raise SystemExit(128 + signum)


@draccus.wrap()
def main(config: EvalRunConfig) -> None:
    signal.signal(signal.SIGTERM, _exit_on_signal)
    signal.signal(signal.SIGINT, _exit_on_signal)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
    )
    summary = run(config)
    headline = {key: value for key, value in summary.items() if not isinstance(value, (list, dict))}
    logger.info("Summary: %s", json.dumps(headline))


if __name__ == "__main__":
    main()
