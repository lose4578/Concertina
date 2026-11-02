# Evaluation

Concertina is evaluated on LIBERO and CALVIN (ABC -> D). This document describes the evaluation stack, how to run both
benchmarks, and what has to match for a result to reproduce the reported numbers.

## Design

| Benchmark | Simulator process | Policy process | Entry point |
|---|---|---|---|
| CALVIN | `concertina_eval_client.calvin` in CALVIN's environment (Python 3.8) | `concertina-eval-server` | `concertina-eval` with `configs/eval/calvin_abc_d.yaml` |
| LIBERO | same process (LeRobot's LIBERO environment and rollout loop) | same process | `concertina-eval-libero` with `configs/eval/libero.yaml`; multi-GPU through `concertina-eval` |

**Policy server** (`concertina.eval.server`). Loads a LeRobot policy directory as written by `concertina-train` or released
(`config.json`, `model.safetensors`, the saved pre/post-processors), given as a local path or a Hugging Face Hub
repository id (downloaded once into the Hugging Face cache), and answers two commands, `reset_model` and `predict`;
any other command is rejected without being dispatched. Observations are benchmark-neutral (`{"images": {camera:
uint8 HxWx3}, "state": {name: float vector}, "instruction": str}`); the server config renames them to the keys of
the dataset the policy was trained on (`image_keys`, `state_keys`), and the checkpoint's preprocessor converts them
to the canonical 20D layout exactly as in training (`config.data.inference_dataset_index` selects the dataset).
`predict` returns one native environment action and follows LeRobot's `select_action` contract: a chunk of
`n_action_steps` actions is decoded whenever the action queue is empty. The queue lives in the server, so a server
serves one client at a time, and the CALVIN client calls `reset_model` at every subtask start, as in the reference
runs. The server binds `127.0.0.1` by default, rejects messages above 64 MiB, and drops connections that do not
send a request within 60 s (port scanners would otherwise hold the single client slot; on shared hosts, tools that
auto-detect listening ports do probe them with HTTP requests).

**Wire protocol** (`concertina_eval_client.protocol`). A 4-byte big-endian length followed by UTF-8 JSON; a NumPy array
travels as `{"__ndarray__": <base64 of the array in NumPy's .npy format>}`, read back without pickle support and
accepted only with a boolean, integer or floating-point dtype.

**Client** (`concertina_eval_client`). NumPy and the standard library only, Python >= 3.8, so it runs inside CALVIN's own
environment; a test checks that importing it loads neither torch nor LeRobot nor `concertina`. The runner exposes only
this package to the client interpreter (a directory holding a link to it on `PYTHONPATH`).

**Runner** (`concertina.eval.runner`, `concertina-eval`). One worker per GPU in `--gpus`. For CALVIN a worker is a policy server
and the simulation client on the same physical GPU (`CUDA_VISIBLE_DEVICES`; CALVIN's EGL renderer is mapped to the
same GPU); CALVIN runs on one GPU unless `calvin.sharded` is set (see *CALVIN* below). For LIBERO a worker is one
`concertina.eval.libero` process evaluating every `n`-th task. Results are written per unit as soon as a unit finishes;
re-running with the same `--output_dir` skips finished units. The merged result is `summary.json`.

**LIBERO** (`concertina.eval.libero`). Uses LeRobot's `make_env`, LIBERO environment processors and per-task `run_one`
loop (the code path of `lerobot-eval`), with three differences: every episode is rendered (`lerobot-eval` renders at
most 10 per task), the decoding recipe comes from the YAML, and each task's result is saved as soon as it finishes
(`tasks/<suite>_<task_id>.json`), with the global seed reset to `seed` before every task so that results do not
depend on the sharding.

**Decoding recipe.** The recipe is not part of a checkpoint's evaluation contract, so each YAML spells out the
complete `ConcertinaInferenceConfig`. LIBERO and CALVIN use: 10 executed actions per decoded chunk, 4 draft steps,
token-risk confidence, dynamic commits (threshold 0, at most half of a block per round), no visible revision, and 5
remask rounds of 15% (no cooldown, no budget decay, every round accepted, last state kept). The recipe saved with the
checkpoints (visible revision, 2 remask rounds of 20%) is not the one the reported results used.

## Environments

| Process | Environment |
|---|---|
| policy server, runner | this package (`pip install -e .`) |
| CALVIN client | CALVIN's environment (`calvin_venv`, Python 3.8, `calvin_env`/`calvin_agent` from a CALVIN checkout, PyBullet with the EGL plugin, imageio-ffmpeg) |
| LIBERO | this package with the `libero` extra (`pip install -e ".[libero]"`: hf-libero, robosuite 1.4, MuJoCo); `MUJOCO_GL=egl` |

The runner starts CALVIN clients with `client.conda_env` or `client.python` (an interpreter path). With
`client.conda_env` it runs the environment's own interpreter by absolute path through `conda run`: a bare `python`
under `conda run` can resolve to the Concertina environment's interpreter when that environment's `bin/` was put on
`PATH` by hand instead of `conda activate`, and the client then fails to import CALVIN's dependencies.

## Running

```bash
# CALVIN ABC -> D (1000 sequences, CALVIN's sequential protocol on one GPU)
export CALVIN_ROOT=/path/to/calvin CALVIN_DATASET=/path/to/calvin/dataset/task_ABC_D
concertina-eval --config_path configs/eval/calvin_abc_d.yaml --policy_path <policy> --output_dir <run> --gpus '[0]'
# faster, not CALVIN's protocol (see Reproduction notes)
concertina-eval --config_path configs/eval/calvin_abc_d.yaml --policy_path <policy> --output_dir <run> \
    --gpus '[0,1,2]' --calvin.sharded true

# LIBERO (4 suites x 10 tasks x 50 episodes), one GPU in process ...
MUJOCO_GL=egl concertina-eval-libero --config_path configs/eval/libero.yaml --policy_path <policy> --output_dir <run>
# ... or sharded over GPUs
concertina-eval --benchmark libero --libero.config_path configs/eval/libero.yaml --policy_path <policy> \
    --output_dir <run> --gpus '[0,1]' --client.python "$(which python)" --client.env '{MUJOCO_GL: egl}'
```

Every evaluated episode is recorded by default (CALVIN: one static-camera MP4 per sequence at 12 fps; LIBERO:
LeRobot's recorder, one MP4 per episode). `calvin.video: false` and `max_episodes_rendered: 0` turn recording off.

Outputs (under `--output_dir`): `eval_config.yaml` (the resolved configuration) and `summary.json`. CALVIN: per
worker `worker_<i>_gpu<g>/` with `server_config.yaml`, `server.log`, `client.log`, `results.jsonl` (one row per
sequence) and `videos/`. LIBERO: `tasks/` (one JSON per task), `tasks.json` and `videos/`.

## Reproduction notes

### CALVIN

- **Sequences.** `configs/eval/calvin_sequences_1000.json` holds CALVIN's evaluation sequences
  (`get_sequences(1000)`); `concertina_eval_client.calvin --write-manifest 1000` regenerates them. The key order of every
  `initial_state` matters: CALVIN seeds the block placement of an initial condition with a hash of
  `str(initial_state.values())`, so a manifest re-serialized with sorted keys yields different scenes.
- **The environment carries state across sequences.** One environment instance evaluates a worker's sequences in
  turn (as CALVIN's own evaluation loop does), and a sequence's outcome depends on the sequences evaluated before it
  in that environment. Per-sequence outcomes are therefore reproducible only for the same per-worker order, and the
  average length of a run depends on how the sequences are distributed. The release protocol is CALVIN's own: all
  1000 sequences in manifest order in one environment (`concertina-eval` with one GPU). `calvin.sharded: true` splits them
  into contiguous blocks over several GPUs for speed, and the runner warns that the result is then not the
  sequential protocol's. Resuming an interrupted run starts a fresh environment at the first unfinished sequence.
- **Reported numbers and noise.** The CALVIN numbers reported for the original code were obtained with several
  workers claiming sequences dynamically (four for the released CALVIN checkpoint's 4.142, three for the reference
  run in *Validation*), an order that cannot be fixed in advance. A sequential run of the same checkpoint is
  statistically comparable to them but not identical. Measured on one checkpoint (see *Validation* below): changing the order changes the outcome of 63 of 1000
  sequences (31 longer, 32 shorter), and the average length moved by 0.002 (4.161 vs 4.159); the per-sequence
  differences imply a standard deviation of about 0.017 for the average length between two orders. Comparisons
  across checkpoints or trainings are dominated by the sampling error of 1000 sequences, about 0.04.
- **Reset.** CALVIN's `Robot.reset` restores the joints but not the internal gripper command reported in
  `robot_obs[14]`; the client restores it before every sequence reset, as the reference runs did.
- **Actions.** Absolute end-effector targets `[xyz, Euler xyz, gripper]` passed to `env.step` as
  `[position, orientation, gripper]` (a flat array would be read as a relative action); the gripper command is
  `+1` if the policy's value is positive, else `-1`. No action clipping.
- **Rendering.** GPU EGL rendering on the worker's GPU (the reference). PyBullet's software renderer
  (`calvin.renderer: tiny`) gives slightly different images and scores (reference runs: 4.017 vs 4.048 on one
  checkpoint).
- **Budget.** 360 steps per subtask; the model is reset at every subtask start; seed 0.

### LIBERO

- **Protocol.** Suites spatial, object, goal and 10; 50 episodes per task with init states 0-49 and seeds from
  1000; 10 settle steps; step limits 280/280/300/520; 360x360 cameras flipped by 180 degrees (LeRobot's
  `LiberoProcessorStep`); 8D state; TF32 matmuls on (as in the reference runs).
- **Init states.** The reference runs used a LeRobot version whose LIBERO environment reset itself after a
  successful episode, advancing the init state an extra time: after a success, episode `k` started from a later init
  state (modulo 50) than in current LeRobot, which uses init state `k` for episode `k`. Per-episode outcomes are
  therefore not comparable; suite success rates are, within sampling noise.
- **Simulator versions.** The reference environment had hf-libero 0.1.3 with MuJoCo 3.3.2; LeRobot's `libero`
  extra installs hf-libero 0.1.4 with a newer MuJoCo. The BDDL task files and init states are identical.

## Validation against the original implementation

On a CALVIN checkpoint of the original code (converted to a Concertina policy directory, CALVIN recipe), the release
stack reproduces the original evaluation exactly:

- per-step actions of the two policy servers on identical observations are bitwise equal (2528 `predict` calls);
- replaying the per-worker sequence orders of the original 1000-sequence run (three workers) gives identical
  outcomes for all 1000 sequences (average length 4.161, success rates 99.1 / 93.9 / 83.9 / 74.4 / 64.8% for 1-5
  chained tasks);
- the original and the release stack, run on sequences 0-99 in the same order, agree on every sequence, and two
  release runs on different GPUs agree on every sequence.

Under CALVIN's sequential protocol the same checkpoint scores an average length of 4.159 (success rates 99.0 /
94.1 / 83.8 / 74.3 / 64.7%), with 937 of 1000 sequences identical to the original run.
