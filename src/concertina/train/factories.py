# Copyright 2026 The Concertina Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions adapted from LeRobot (lerobot/scripts/lerobot_train.py, `make_dataloaders`),
# Copyright 2024 The HuggingFace Inc. team, Apache License 2.0. Modified: the Concertina weighted mixture sampler
# replaces the episode-aware sampler, and a resumed run restores the sampler position. See NOTICE.
"""Concertina replacements for `lerobot_train.make_train_eval_datasets` and `lerobot_train.make_dataloaders`.

They keep the upstream parameter layout (checked by `concertina.train.overrides`) and are installed only by
`concertina-train`. The dataset mixture and its sampler are configured by ``cfg.policy.data``; ``cfg.dataset``
supplies the dataset root, video backend, image transforms and ImageNet statistics.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import torch
from lerobot.common.train_utils import load_training_metadata
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import compute_sampler_state
from lerobot.distributed import is_main_process
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import TRAINING_STATE_DIR

from concertina.data.mixture import ConcertinaMixtureDataset
from concertina.data.sampler import WeightedMixtureSampler, episode_valid_intervals, round_up_epoch_length
from concertina.model.constants import IMAGE_SIZE
from concertina.policy.configuration_concertina import ConcertinaConfig

logger = logging.getLogger(__name__)

# `cfg.dataset` options the mixture does not implement, with their required values.
_UNSUPPORTED_DATASET_OPTIONS = {
    "streaming": False,
    "eval_split": 0.0,
    "episodes": None,
    "exclude_episodes": None,
    "repo_type": "dataset",
    "return_uint8": False,
}


def _concertina_policy(cfg: TrainPipelineConfig):
    policy = cfg.policy
    if not isinstance(policy, ConcertinaConfig):
        raise ValueError(f"concertina-train trains the Concertina policy, got {type(policy).__name__}")
    return policy


def make_train_eval_datasets(cfg):
    """Build the Concertina mixture dataset; held-out evaluation splits are not supported.

    Also binds the policy's sample-keyed training randomness to ``cfg.seed``: `train()` calls this right after
    `validate()` resolved the policy config (``--policy.path``, resume) and before it creates the policy.
    """
    policy = _concertina_policy(cfg)
    policy.model.rng_seed = cfg.seed if cfg.seed is not None else 0
    for name, required in _UNSUPPORTED_DATASET_OPTIONS.items():
        if getattr(cfg.dataset, name) != required:
            raise ValueError(
                f"concertina-train requires dataset.{name}={required!r}, got {getattr(cfg.dataset, name)!r}"
            )
    data = policy.data
    transforms_cfg = cfg.dataset.image_transforms
    image_transforms = (
        ImageTransforms(transforms_cfg)
        if transforms_cfg.enable and transforms_cfg.backend == "dataloader"
        else None
    )
    dataset = ConcertinaMixtureDataset(
        data.datasets,
        root=Path(cfg.dataset.root) if cfg.dataset.root else None,
        camera_keys=policy.model.camera_keys,
        action_delta_indices=policy.action_delta_indices,
        observation_delta_indices=policy.observation_delta_indices,
        image_hw=IMAGE_SIZE,
        resize_images=data.resize_images,
        fps_align_target_fps=data.fps_align_target_fps,
        normalization=data.normalization,
        use_imagenet_stats=cfg.dataset.use_imagenet_stats,
        image_transforms=image_transforms,
        video_backend=cfg.dataset.video_backend,
        tolerance_s=cfg.tolerance_s,
        repo_id=cfg.dataset.repo_id,
        quality_config=data.quality,
    )
    return dataset, None


def make_mixture_sampler(
    dataset: ConcertinaMixtureDataset, data, *, batch_size: int, dp_world_size: int, seed: int
) -> WeightedMixtureSampler:
    """Sampler over the valid frames of every dataset; an epoch is a whole number of global batches.

    A single dataset is visited as one seeded permutation of its valid frames per epoch (the order the original
    single-dataset trainer used); a mixture draws datasets by weight, with replacement.
    """
    intervals = []
    for spec, quality, episodes, (starts, ends), offset in zip(
        dataset.specs,
        dataset.quality,
        dataset.episode_ids,
        dataset.episode_bounds,
        dataset.dataset_offsets,
        strict=True,
    ):
        drops = {
            "drop_n_first_frames": data.drop_n_first_frames,
            "drop_n_last_frames": data.drop_n_last_frames,
        }
        if quality is None:
            repo_intervals = episode_valid_intervals(starts, ends, offset=offset, **drops)
        else:
            repo_intervals = quality.valid_intervals(episodes, starts, ends, offset=offset, **drops)
        if not repo_intervals:
            logger.warning(
                "%s has no valid frames left after the drop bounds and quality filter and is never sampled",
                spec.path,
            )
        intervals.append(repo_intervals)
    total_valid = sum(end - start for repo in intervals for start, end in repo)
    mixture = len(dataset.specs) > 1
    weights = (
        None
        if not mixture or all(spec.weight is None for spec in dataset.specs)
        else [spec.weight for spec in dataset.specs]
    )
    sampler = WeightedMixtureSampler(
        repo_valid_intervals=intervals,
        repo_weights=weights,
        samples_per_epoch=round_up_epoch_length(total_valid, batch_size, dp_world_size),
        seed=seed,
        replacement=mixture,
    )
    if is_main_process():
        for spec, probability, valid in zip(
            dataset.specs, sampler.repo_probs, sampler.repo_valid_lengths, strict=True
        ):
            logger.info(
                "dataset %s: %d valid frames, sampling probability %.6f", spec.path, valid, probability
            )
    return sampler


def make_dataloaders(cfg, dataset, eval_dataset, step, parallel_dims):
    """Train dataloader over the Concertina mixture, resumed at the sample the checkpointed step reached."""
    if eval_dataset is not None:
        raise ValueError("concertina-train does not support a held-out evaluation dataset")
    policy = _concertina_policy(cfg)
    sampler = make_mixture_sampler(
        dataset,
        policy.data,
        batch_size=cfg.batch_size,
        dp_world_size=parallel_dims.dp_world_size,
        seed=cfg.seed if cfg.seed is not None else 0,
    )
    if cfg.resume and step > 0:
        if cfg.checkpoint_path is None:
            raise ValueError(
                "Resuming requires `checkpoint_path`; TrainPipelineConfig resolves it on --resume."
            )
        metadata = load_training_metadata(cfg.checkpoint_path / TRAINING_STATE_DIR)
        saved_dp_world, saved_batch_size = metadata["dp_world_size"], metadata["batch_size"]
        if is_main_process() and (
            saved_dp_world not in (None, parallel_dims.dp_world_size)
            or saved_batch_size not in (None, cfg.batch_size)
        ):
            logger.warning(
                "Resuming with dp_world_size=%d, batch_size=%d; the checkpoint used %s, %s. The epoch length "
                "depends on both, so the resumed data order is not sample-exact.",
                parallel_dims.dp_world_size,
                cfg.batch_size,
                saved_dp_world,
                saved_batch_size,
            )
        state = compute_sampler_state(
            step,
            len(sampler),
            saved_batch_size or cfg.batch_size,
            saved_dp_world or parallel_dims.dp_world_size,
        )
        sampler.load_state_dict(state)
        if is_main_process():
            logger.info("Resuming data order at epoch %d, sample %d", state["epoch"], state["start_index"])

    workers = cfg.num_workers > 0
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        generator=torch.Generator().manual_seed(cfg.seed) if cfg.seed is not None else None,
        pin_memory=parallel_dims.device_type == "cuda",
        drop_last=False,
        prefetch_factor=cfg.prefetch_factor if workers else None,
        persistent_workers=cfg.persistent_workers and workers,
        multiprocessing_context=cfg.dataloader_multiprocessing_context if workers else None,
    )
    return dataloader, None


class TrainingRunHooks:
    """The `lerobot_train` replacements of one concertina-train run.

    `train()` calls `make_train_eval_datasets` after `validate()` and before `make_policy`, on every rank, so the
    dataset hook records whether the run resumes and the policy hook uses it: a run that does not resume (from
    scratch, or fine-tuning a checkpoint) restarts the policy's training-step counter, which keys its sample-keyed
    training randomness, while a resumed run keeps the checkpointed counter.
    """

    def __init__(self, make_policy: Callable) -> None:
        self._make_policy = make_policy
        self.resume: bool | None = None

    def make_train_eval_datasets(self, cfg):
        self.resume = bool(cfg.resume)
        return make_train_eval_datasets(cfg)

    def make_policy(
        self, cfg, ds_meta=None, env_cfg=None, rename_map=None, defer_weight_load=False, pretrained_path=None
    ):
        if self.resume is None:
            raise RuntimeError("make_policy ran before make_train_eval_datasets; the training loop changed")
        policy = self._make_policy(
            cfg,
            ds_meta=ds_meta,
            env_cfg=env_cfg,
            rename_map=rename_map,
            defer_weight_load=defer_weight_load,
            pretrained_path=pretrained_path,
        )
        if not self.resume:
            policy.model.training_step.zero_()
        return policy
