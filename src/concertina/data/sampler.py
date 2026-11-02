"""Weighted, drop-aware, resumable frame sampler for the Concertina mixture dataset.

The order is a pure function of ``(seed, epoch, position)``: every rank regenerates the same
sequence and accelerate's ``BatchSamplerShard`` slices it, so resume is sample-exact through
``lerobot.datasets.compute_sampler_state`` -> ``load_state_dict``.

Two modes share the same epoch/resume protocol:

- ``replacement=True`` (mixtures): a repo is drawn by weight, then a frame uniformly over that
  repo's valid intervals. Indices come from a counter-based hash (splitmix64), so any position is
  addressable in O(1) and the stream does not depend on chunking or on library RNG internals.
- ``replacement=False`` (single source): one permutation of every valid frame per epoch, drawn
  with ``torch.randperm`` seeded by ``seed + epoch`` — the seeded form of lerobot's
  ``EpisodeAwareSampler`` order. Positions past the number of valid frames wrap around.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

Interval = tuple[int, int]

_UINT64_MASK = (1 << 64) - 1
_STREAM_REPO = 0x7265706F5F696478  # "repo_idx"
_STREAM_FRAME = 0x6672616D655F6978  # "frame_ix"
_UNIT_SCALE = 2.0**-53


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64 finalizer over a uint64 array (wraps modulo 2**64)."""
    z = values + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def hash_to_unit(values: np.ndarray) -> np.ndarray:
    """Map uint64 hashes to float64 in [0, 1) using the top 53 bits."""
    return (values >> np.uint64(11)).astype(np.float64) * _UNIT_SCALE


def episode_valid_intervals(
    dataset_from_indices: Sequence[int],
    dataset_to_indices: Sequence[int],
    *,
    drop_n_first_frames: int = 0,
    drop_n_last_frames: int = 0,
    offset: int = 0,
) -> list[Interval]:
    """Per-episode ``[from + drop_first, to - drop_last)`` windows, shifted by ``offset``.

    Episodes left without frames are skipped. ``offset`` moves the intervals into the mixture
    index space (the repo's ``dataset_offset``).
    """
    if drop_n_first_frames < 0 or drop_n_last_frames < 0:
        raise ValueError("drop_n_first_frames and drop_n_last_frames must be >= 0")
    if len(dataset_from_indices) != len(dataset_to_indices):
        raise ValueError("dataset_from_indices and dataset_to_indices must have the same length")
    intervals: list[Interval] = []
    skipped = 0
    for start, end in zip(dataset_from_indices, dataset_to_indices, strict=True):
        valid_start = int(start) + drop_n_first_frames
        valid_end = int(end) - drop_n_last_frames
        if valid_start >= valid_end:
            skipped += 1
            continue
        intervals.append((valid_start + offset, valid_end + offset))
    if skipped:
        logger.warning(
            "%d of %d episodes have no frames left after drop_n_first_frames=%d / drop_n_last_frames=%d",
            skipped,
            len(dataset_from_indices),
            drop_n_first_frames,
            drop_n_last_frames,
        )
    return intervals


def round_up_epoch_length(total_valid_frames: int, batch_size: int, dp_world_size: int) -> int:
    """Smallest multiple of ``batch_size * dp_world_size`` that is >= ``total_valid_frames``.

    An epoch that is an exact number of global batches never triggers accelerate's even-batch
    padding, which is what makes a resumed run reproduce the uninterrupted stream exactly.
    """
    if total_valid_frames <= 0:
        raise ValueError("total_valid_frames must be > 0")
    if batch_size <= 0 or dp_world_size <= 0:
        raise ValueError("batch_size and dp_world_size must be > 0")
    global_batch = batch_size * dp_world_size
    return math.ceil(total_valid_frames / global_batch) * global_batch


class WeightedMixtureSampler(torch.utils.data.Sampler[int]):
    """Sample frames from a mixture of repos, each restricted to its valid intervals.

    Args:
        repo_valid_intervals: Per repo, the ``[start, end)`` windows of sampleable frames in the
            mixture index space (see :func:`episode_valid_intervals`).
        repo_weights: Relative sampling weight per repo. ``None`` weights every repo by its number
            of valid frames, i.e. uniform over all valid frames of the mixture. Repos without valid
            frames always get probability 0.
        samples_per_epoch: Epoch length reported by ``__len__`` (see :func:`round_up_epoch_length`).
        seed: Base seed; combined with the epoch and the position inside the epoch.
        chunk_size: Positions materialized per vectorized step while iterating.
        replacement: ``True`` draws independently per position (weighted mixture). ``False`` emits
            one permutation of all valid frames per epoch (requires ``repo_weights=None``); when
            ``samples_per_epoch`` exceeds the valid frame count, positions wrap to the permutation's
            start, so an epoch rounded up to the global batch repeats at most that many frames.

    Epoch state: ``load_state_dict({"epoch": E, "start_index": S})`` makes the *next* iteration
    yield epoch ``E`` from position ``S`` and the one after it epoch ``E + 1`` from 0. accelerate's
    ``DataLoaderShard`` calls ``set_epoch(iteration)`` with ``iteration = 0, 1, ...`` counted from
    the dataloader's creation, so ``set_epoch`` sets the number of passes since the last load and
    the effective epoch is ``E + passes``; iterating without ``set_epoch`` advances the same counter.
    """

    def __init__(
        self,
        *,
        repo_valid_intervals: Sequence[Sequence[Interval]],
        repo_weights: Sequence[float] | None,
        samples_per_epoch: int,
        seed: int = 0,
        chunk_size: int = 65536,
        replacement: bool = True,
    ) -> None:
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be > 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if repo_weights is not None and len(repo_weights) != len(repo_valid_intervals):
            raise ValueError("repo_weights must have one entry per repo")
        if not replacement and repo_weights is not None:
            raise ValueError(
                "replacement=False visits every valid frame once per epoch and is incompatible with "
                "explicit repo_weights; pass repo_weights=None"
            )

        starts: list[int] = []
        lengths: list[int] = []
        repo_of_interval: list[int] = []
        valid_lengths: list[int] = []
        for repo_idx, intervals in enumerate(repo_valid_intervals):
            total = 0
            for start, end in intervals:
                start, end = int(start), int(end)
                if start >= end:
                    raise ValueError(f"empty interval ({start}, {end}) in repo {repo_idx}")
                starts.append(start)
                lengths.append(end - start)
                repo_of_interval.append(repo_idx)
                total += end - start
            valid_lengths.append(total)

        self.num_repos = len(repo_valid_intervals)
        self.repo_valid_lengths = valid_lengths
        self.total_valid_frames = int(sum(valid_lengths))
        if self.total_valid_frames == 0:
            raise ValueError("no valid frames in any repo")

        weights = (
            np.asarray(valid_lengths, dtype=np.float64)
            if repo_weights is None
            else np.asarray([float(w) for w in repo_weights], dtype=np.float64)
        )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("repo_weights must be finite and >= 0")
        weights = np.where(np.asarray(valid_lengths) > 0, weights, 0.0)
        if weights.sum() <= 0:
            raise ValueError("all repos have zero weight or no valid frames")
        self.repo_probs = weights / weights.sum()
        self._cum_probs = np.cumsum(self.repo_probs)
        self._cum_probs[-1] = 1.0

        # Global valid-position space: repo r owns [valid_offset[r], valid_offset[r] + valid_len[r]).
        self._valid_lengths_u64 = np.asarray(valid_lengths, dtype=np.uint64)
        self._valid_offsets = np.concatenate([[0], np.cumsum(valid_lengths)[:-1]]).astype(np.int64)
        self._interval_starts = np.asarray(starts, dtype=np.int64)
        self._interval_lengths = np.asarray(lengths, dtype=np.int64)
        self._interval_cum = np.cumsum(self._interval_lengths)
        self._interval_cum_prev = self._interval_cum - self._interval_lengths
        self._interval_repo = np.asarray(repo_of_interval, dtype=np.int64)

        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.replacement = bool(replacement)
        self._all_valid: np.ndarray | None = None
        self._permutation: tuple[int, np.ndarray] | None = None
        self._base_epoch = 0
        self._start_index = 0
        self._passes = 0

    # ── epoch / resume protocol ────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        self._passes = int(epoch)

    def state_dict(self) -> dict:
        if self._passes == 0:
            return {"epoch": self._base_epoch, "start_index": self._start_index}
        return {"epoch": self._base_epoch + self._passes, "start_index": 0}

    def load_state_dict(self, state: dict) -> None:
        epoch, start = int(state["epoch"]), int(state["start_index"])
        if epoch < 0 or not 0 <= start <= self.samples_per_epoch:
            raise ValueError(f"invalid sampler state {state!r} for epoch length {self.samples_per_epoch}")
        self._base_epoch = epoch
        self._start_index = start
        self._passes = 0

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        epoch = self._base_epoch + self._passes
        start = self._start_index if self._passes == 0 else 0
        self._passes += 1
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        for chunk_start in range(start, self.samples_per_epoch, self.chunk_size):
            chunk_stop = min(chunk_start + self.chunk_size, self.samples_per_epoch)
            yield from self.indices(epoch, chunk_start, chunk_stop).tolist()

    # ── pure index generation ──────────────────────────────────────────

    def indices(self, epoch: int, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Mixture indices at positions ``[start, stop)`` of ``epoch`` (int64 array)."""
        stop = self.samples_per_epoch if stop is None else stop
        if not 0 <= start <= stop <= self.samples_per_epoch:
            raise ValueError(f"positions [{start}, {stop}) outside epoch length {self.samples_per_epoch}")
        if start == stop:
            return np.empty(0, dtype=np.int64)
        if not self.replacement:
            positions = np.arange(start, stop, dtype=np.int64) % self.total_valid_frames
            return self.valid_indices[self._epoch_permutation(epoch)[positions]]

        positions = np.arange(start, stop, dtype=np.uint64)
        base = splitmix64(np.asarray([self.seed & _UINT64_MASK], dtype=np.uint64))
        base = splitmix64(base ^ np.uint64(epoch & _UINT64_MASK))
        keys = splitmix64(base ^ positions)
        repo_draw = splitmix64(keys ^ np.uint64(_STREAM_REPO))
        frame_draw = splitmix64(keys ^ np.uint64(_STREAM_FRAME))

        repos = np.searchsorted(self._cum_probs, hash_to_unit(repo_draw), side="right")
        repos = np.minimum(repos, self.num_repos - 1).astype(np.int64)
        local = (frame_draw % self._valid_lengths_u64[repos]).astype(np.int64)
        return self._valid_position_to_index(self._valid_offsets[repos] + local)

    @property
    def valid_indices(self) -> np.ndarray:
        """Every sampleable mixture index, in interval order (materialized once, O(total_valid_frames))."""
        if self._all_valid is None:
            self._all_valid = self._valid_position_to_index(
                np.arange(self.total_valid_frames, dtype=np.int64)
            )
        return self._all_valid

    def _epoch_permutation(self, epoch: int) -> np.ndarray:
        if self._permutation is None or self._permutation[0] != epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(self.total_valid_frames, generator=generator).numpy()
            self._permutation = (epoch, order)
        return self._permutation[1]

    def _valid_position_to_index(self, global_valid: np.ndarray) -> np.ndarray:
        interval = np.searchsorted(self._interval_cum, global_valid, side="right")
        return self._interval_starts[interval] + (global_valid - self._interval_cum_prev[interval])

    def repo_of(self, indices: np.ndarray | Sequence[int]) -> np.ndarray:
        """Repo index of each mixture index; -1 for indices outside every valid interval."""
        values = np.asarray(indices, dtype=np.int64)
        order = np.argsort(self._interval_starts, kind="stable")
        starts = self._interval_starts[order]
        ends = starts + self._interval_lengths[order]
        pos = np.searchsorted(starts, values, side="right") - 1
        inside = (pos >= 0) & (values < ends[np.maximum(pos, 0)])
        return np.where(inside, self._interval_repo[order][np.maximum(pos, 0)], -1)
