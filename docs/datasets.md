# Datasets

Concertina trains on local [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) datasets. The fine-tuning
datasets of the paper are public on the Hugging Face Hub; the cleaned pretraining datasets are not released.

| Dataset | Hub id | Episodes | Frames | fps | Cameras | Configurations |
|---|---|---:|---:|---:|---|---|
| CALVIN ABC -> D (training split) | `Traly/calvin_abc_d-lerobot` | 17,870 | 1,071,743 | 30 | `top` 200x200, `wrist` 84x84 | `calvin.yaml` |
| LIBERO (4 suites) | `lerobot/libero` | 1,693 | 273,465 | 10 | `image`, `image2` 256x256 | `libero.yaml` |

Download them into the directory given as `dataset.root`, under the names the configurations use:

```bash
export DATA=/path/to/datasets
hf download Traly/calvin_abc_d-lerobot --repo-type dataset --revision 60371339eb9f0325922d8e61877dcc5cf08260ae \
    --local-dir $DATA/calvin_abc_d-lerobot
hf download lerobot/libero --repo-type dataset --revision a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
    --local-dir $DATA/libero
```

## Requirements

Every dataset given to `concertina-train` must

- be complete for the cameras the model reads: every data file, and in every episode the videos of each camera that
  `model.camera_keys` maps to through the dataset's `camera_map` (LeRobot's reader otherwise tries to download the
  dataset from the Hub). Other cameras may be missing or have null video references: `concertina-train` then reads the
  dataset through a view whose `meta/info.json` declares only the cameras it reads. Views are created under
  `$CONCERTINA_CACHE_DIR/dataset_views` (default `$XDG_CACHE_HOME/concertina`, i.e. `~/.cache/concertina`); they link every other
  file to the dataset, which is not modified, so the cache directory must support symbolic links;
- have a dense episode layout: `concertina-normalize-episode-metadata <dataset>` reports `episode layout is dense`
  (`--write` repairs it in place). Both datasets above are dense.

Concertina normalizes actions and states with statistics computed after converting rotations to the 6D representation,
read from `meta/rotation_stats.json`. When the dataset has no such file, training computes them once per machine
(under a file lock shared by the ranks) into `$CONCERTINA_CACHE_DIR/rotation_stats/<dataset>-<fingerprint>.json`, keyed by
the episode metadata and data files, and leaves the dataset untouched. `concertina-compute-rotation-stats` stores them with
the dataset beforehand, which also saves the computation at the start of training:

```bash
concertina-compute-rotation-stats $DATA/libero --source libero --num-workers 8
```

The CALVIN dataset on the Hub ships the statistics the paper trained with.

## CALVIN (ABC -> D)

`Traly/calvin_abc_d-lerobot` is the training split of the official `task_ABC_D` release converted to LeRobot v3.
Each language-annotated segment is one episode whose task is the annotation; `action` and `action.absolute` hold
CALVIN's absolute `actions` (Concertina reads `action.absolute`), `action.relative` holds `rel_actions`,
`observation.state` is `robot_obs` and `observation.environment_state` is `scene_obs`; the static and gripper
cameras are `observation.images.top` and `observation.images.wrist` (H.264).

### Rebuilding it from the official release

Download and check the official split (517 GB; the archive's SHA-256 is listed in the official
`http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt`):

```bash
export SOURCES=/path/to/sources
cd "$SOURCES"
wget http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip
echo "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74  task_ABC_D.zip" | sha256sum --check
unzip -q task_ABC_D.zip
```

Convert the training split and compute its statistics:

```bash
concertina-convert-calvin --calvin-root $SOURCES/task_ABC_D --output $DATA/calvin_abc_d-lerobot --workers 32
concertina-compute-rotation-stats $DATA/calvin_abc_d-lerobot --source calvin --num-workers 8
```

`--workers` runs parallel converters on contiguous parts of the episode list and aggregates them; it changes neither
the episode order nor any frame. The default `--episode-interleave 192` reproduces the episode order of the published
dataset (its segments were dealt round-robin to 192 converters); `--episode-interleave 1` keeps the annotation order.
`--max-episodes N` converts only the first episodes, for a quick check.

The converter reproduces the published dataset frame for frame in every numeric column, task, episode order and
metadata, but not its video files:

- **Video encoding is not bit-reproducible.** Re-encoding the same frames with the same encoder build and settings
  (H.264, CRF 30) gives slightly different bitstreams from run to run: on 290 real CALVIN frames, two conversions
  on the same machine decode to identical images for 95% of the static-camera frames, a conversion and the
  published files for 79-83% (99-100% of the wrist-camera frames); the differing frames are within 35 dB PSNR of
  each other, against about 28 dB between either and the source images. Use the published dataset to reproduce the
  paper exactly.
- **File layout and statistics.** The published dataset was written by 192 parallel converters, so its files are
  split differently; the per-episode statistics in the episode metadata, computed by the pinned LeRobot, differ from
  the published ones in the last digits. Neither is read by Concertina training. Quantiles in `rotation_stats.json` are
  pooled per data file and therefore depend on the layout; Concertina normalizes with the mean and standard deviation,
  which are equal.

## LIBERO

`lerobot/libero` is LeRobot's LIBERO dataset (the demonstrations of the spatial, object, goal and 10 suites, AV1
videos) and is used as published: the paper's training copy has identical data and video files; it differs only in
its `meta/episodes` file, rewritten by the original pipeline's metadata repair, and in the `rotation_stats.json`
that training generated. The cameras `observation.images.image` and `observation.images.image2` are the
agent-view and wrist cameras; `observation.state` is 8D (end-effector position, axis-angle orientation and both
finger joints) and `action` is LIBERO's 7D relative end-effector command.

## Pretraining data

The pretraining datasets are not released. Pretraining (`configs/train/pretrain_60k.yaml`, then
`pretrain_90k.yaml`) samples 39 dataset entries by weight, normalized per embodiment:

| Source | Datasets | Embodiments | Relative weight |
|---|---|---|---:|
| DROID | DROID (listed twice, 0.15 each) | Franka, single arm | 0.30 |
| AgiBot World | 34 tasks | AgiBot dual arm | 0.40 |
| RoboMIND | AgileX dual arm, Franka single arm, FR3 dual arm | per robot | 0.20 |

All were converted to LeRobot v3, resampled to 30 fps (`fps_align_target_fps`), and cleaned with the offline
data-quality tool (`concertina-data-quality`, see [src/concertina/data_quality/README.md](../src/concertina/data_quality/README.md)):
a scan against the source's contract (`droid`, `agibot`, `robomind_agilex`, `robomind_franka`, `robomind_fr3`), a
quality package compiled with the pause-cleaning policy, and a cleaned copy written by `materialize`, which also
computes the rotation statistics (`--rotation-stats-source`).
