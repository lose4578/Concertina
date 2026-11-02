# Data quality tools (`concertina-data-quality`)

Offline quality control for local LeRobot v3 robot datasets. The tool scans a dataset without modifying it,
compiles a **quality package** (source-grain facts and decisions), and can write a **cleaned copy** of the
dataset. Training can also consume a package directly to filter samples of the unmodified dataset.

Evidence and decisions are kept apart. Schema, index and non-finite violations quarantine an episode
automatically. Pauses, jumps, residuals and video anomalies are only *evidence*: they change retention through an
explicit cleaning policy or a reviewed decision, never by themselves.

## Stages

```bash
concertina-data-quality scan --dataset DATA --output RUN --contract droid [--config scan.yaml] [--workers 8]
concertina-data-quality compile --run RUN [--cleaning-policy] [--decisions reviewed.json]
concertina-data-quality materialize --package RUN/package --dataset DATA --output CLEAN [--media symlink]
```

1. **`scan`** reads every data file once (only the contract's columns) and writes `RUN/scan/`:
   - `episode_metrics`, `interval_findings`, `episode_decisions`, `field_masks` and `video_findings` (JSONL);
   - `scan.json`, a summary with the resolved configuration.

   It runs three kinds of checks:
   - per episode: structure, trajectory jumps, near-static runs (pause candidates) of the canonical action and
     state, and, for CALVIN, the action vs next-state residual and lag;
   - a bounded decode probe of a sample of the MP4 files;
   - camera-motion evidence for each pause candidate.

   `--max-files` limits the scan (a partial scan is never release-ready).
2. **`compile`** builds the package at `RUN/package` (`quality_package.json` plus JSONL tables:
   `episode_metrics`, `interval_findings`, `episode_decisions`, `frame_decisions`, `field_masks`,
   `group_distribution`, `video_findings`, and `policy_decisions` with `--cleaning-policy`). Its inputs:
   - the contract quarantines from the scan;
   - optional reviewed decisions, `--decisions` (see below);
   - the pause-cleaning policy, `--cleaning-policy`.

   The package is `release_ready` unless a blocker is listed in `release_blockers`:
   - a partial scan, or missing data or video files;
   - critical/high video findings;
   - more than `--max-coverage-loss` (default 0.10) of the frames removed by quarantines and reviewed decisions;
   - a blocked contract.

   Per-task coverage is reported in `group_distribution`.
3. **`materialize`** writes a new LeRobot v3 dataset:
   - quarantined episodes are dropped, and every excluded frame interval splits its episode (each retained
     interval becomes one output episode);
   - episode, frame and global indices are dense and timestamps restart at 0;
   - video `from_timestamp`/`to_timestamp` point into the unchanged MP4 files, which are symlinked (default),
     hardlinked or copied;
   - `action_config` phases (AgiBot lists, RoboMIND `steps` mappings) are clipped to each output episode;
   - numeric statistics are recomputed into `meta/stats.json`.

   `--rotation-stats-source droid|agibot|robomind|libero|calvin` also writes `meta/rotation_stats.json`;
   without it, run `concertina-compute-rotation-stats` on the output. It refuses packages that are not release-ready
   or that contain field masks, since masks cannot be expressed by rewriting rows. `meta/cleaning/` records the
   segments and the package manifest.

### Reviewed decisions

`--decisions` takes a JSON list (or `{"decisions": [...]}`). Each entry cites a finding by `candidate_id` (the
`finding_id` of `interval_findings`) and needs a `reason`:

```json
[
  {"candidate_id": "8dcc6e96bf348303dbb8dfe0", "action": "exclude_frame_interval",
   "start_frame": 80, "end_frame": 146, "reason": "idle tail after task completion"},
  {"candidate_id": "...", "action": "mask_field_interval", "field": "action.cartesian_position",
   "start_frame": 9, "end_frame": 11, "reason": "corrupted command values"},
  {"candidate_id": "...", "action": "quarantine_episode", "reason": "wrong task"},
  {"candidate_id": "...", "action": "keep", "reason": "valid behaviour"}
]
```

Each action has containment rules:
- intervals must lie inside the finding;
- an exclusion must not overlap camera-protected motion;
- a mask must name an exact source column with evidence in the finding.

`reject_finding` is accepted as a synonym of `keep`.

## Contracts and presets

`--contract` selects the source contract: which columns hold the action and state, their rotation
representation, and the thresholds. The contracts are `droid`, `agibot`, `robomind_agilex`, `robomind_franka`,
`robomind_fr3`, `robomind_ur` and `calvin`; aliases such as `agilex_3rgb` also work. The scan preset
follows the contract:

| Preset | Contracts | Checks |
|---|---|---|
| pretrain | DROID, AgiBot, RoboMIND | pause runs reported from 1 s; physical jump candidates; video probe (1 % of files per camera, 3 bursts of 2 frames); camera-motion evidence for pauses ≥ 3 s |
| audit | CALVIN | pause runs from 0.2 s; robust velocity/acceleration/jerk candidates; action vs next-state residual and lag; video probe; no camera-motion stage |

A YAML file given with `--config` overrides any field of `concertina.data_quality.config.ScanConfig`:

```yaml
contract: droid
trajectory:
  pause_duration_seconds: [1.0, 2.0, 3.0, 5.0]
video:
  sample_fraction: 0.05
```

## Requirements

- Python dependencies come with LeRobot (PyArrow, PyAV, OpenCV).
- The camera-motion stage runs FFmpeg's `freezedetect` filter through an `ffmpeg` executable on `PATH`
  (`visual_motion.ffmpeg_binary`); the FFmpeg build must decode the dataset's video codec (LeRobot encodes
  AV1 by default).
- If the camera stage is disabled or fails, pauses have no camera evidence and the cleaning policy keeps them,
  unless `--allow-missing-visual-evidence` is given. A missing `ffmpeg` is logged as a warning and recorded in
  `scan.json` and the package manifest (`scan.warnings`); `scan --require-ffmpeg` makes it an error instead.

## Using a package in training

Training reads a package per dataset. The option is off by default and no paper run used it: the paper's
pretraining read cleaned copies written by the materializer.

```yaml
policy:
  data:
    datasets:
      - path: droid
        source: droid
        embodiment: droid
        valid_action_dim: 10
        quality_package: quality/droid/package   # relative to dataset.root, like `path`
    quality:
      require_release_ready: true
```

The package must have been compiled from exactly this dataset: every episode and its length are checked.

- **Samples.** A frame is a training sample iff:
  - its episode is kept and it lies inside the drop window;
  - every observation row it reads is valid;
  - at least one future action target (offset ≥ 0) is valid.

  The sampler restricts each dataset's valid intervals accordingly. It keeps its seeded, resumable order, and
  with no decisions the order is unchanged.
- **Targets.** Invalid action targets of a sample are marked in `action_is_pad`, so the loss ignores them.
- **Masks.** A mask on a column the sample reader consumes (an action column or an observation input) applies.
  Masks on other columns of the dataset are inert.
- **Order.** Changing the package changes the epoch length, so resuming with another package is not
  sample-exact.

## How the pretraining data was cleaned (`cleaned-v4r6`)

The pretraining mixture used cleaned copies of DROID 1.0.1, the AgiBot World Alpha tasks and RoboMIND (Agilex,
Franka, FR3), each scanned with the pretrain preset (all stages) and cleaned with pause policy v5. The released
port of that policy is `concertina.data_quality.policy` with its default `PausePolicyConfig`:

1. **AgiBot useful intervals.** The ordered `action_config` phases of an episode are its useful frames:
   - everything outside their union is excluded;
   - an episode without phases is kept whole.
2. **Pauses.** Canonical-action pause candidates (restricted to the useful frames) at least 3 s long are
   excluded, except frames that are vetoed:
   - the separately recorded joint action moves (per-step norm > 1e-3, active for ≥ 0.4 s in total, gaps of
     ≤ 0.15 s closed);
   - cameras show motion (optical flow and FFmpeg `freezedetect` agreeing) on one ego camera or on two
     external cameras;
   - the camera evidence is missing or uncertain;
   - the frames lie inside an AgiBot pour phase (dwelling while pouring is the task).

   Remaining pieces of at least 3 s are excluded; a piece covering a whole episode quarantines the episode.
3. **Blockers.**
   - RoboMIND UR is blocked: its end-effector action freezes while its joints and video move.
   - An AgiBot task whose episode catalog contains duplicate episodes cannot be cleaned (the scan fails).

The cleaned roots were then materialized with symlinked videos. DROID appears in two training slots, with two
different camera maps.
