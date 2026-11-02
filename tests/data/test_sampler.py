import math

import numpy as np
import pytest
import torch
from torch.utils.data import BatchSampler

from concertina.data.sampler import (
    WeightedMixtureSampler,
    episode_valid_intervals,
    round_up_epoch_length,
)

accelerate_data_loader = pytest.importorskip("accelerate.data_loader", reason="accelerate is optional here")
BatchSamplerShard = accelerate_data_loader.BatchSamplerShard

# Three repos with episodes of varying length; repo 1 holds a tiny episode that drop bounds remove.
REPOS = [
    ([0, 30, 70], [30, 70, 100]),
    ([0, 40, 45], [40, 45, 90]),
    ([0, 60], [60, 150]),
]
DROP_FIRST, DROP_LAST = 1, 4


def compute_sampler_state(step: int, num_frames: int, batch_size: int, num_processes: int) -> dict:
    # Local copy of lerobot.datasets.sampler.compute_sampler_state so the test is pure torch.
    batches_per_epoch = math.ceil(math.ceil(num_frames / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    return {"epoch": epoch, "start_index": min(batches_into_epoch * batch_size * num_processes, num_frames)}


def build_mixture(drop_first=DROP_FIRST, drop_last=DROP_LAST):
    offsets, intervals, lengths = [], [], []
    cursor = 0
    for from_idx, to_idx in REPOS:
        offsets.append(cursor)
        intervals.append(
            episode_valid_intervals(
                from_idx, to_idx, drop_n_first_frames=drop_first, drop_n_last_frames=drop_last, offset=cursor
            )
        )
        lengths.append(to_idx[-1])
        cursor += to_idx[-1]
    return offsets, lengths, intervals


def make_sampler(weights=(0.5, 0.3, 0.2), samples_per_epoch=1000, seed=7, **kwargs):
    _, _, intervals = build_mixture()
    return WeightedMixtureSampler(
        repo_valid_intervals=intervals,
        repo_weights=weights,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
        **kwargs,
    )


def valid_list(intervals_per_repo):
    return [idx for intervals in intervals_per_repo for start, end in intervals for idx in range(start, end)]


def test_episode_valid_intervals_apply_drop_bounds_and_skip_empty_episodes():
    intervals = episode_valid_intervals(
        [0, 40, 45], [40, 45, 90], drop_n_first_frames=1, drop_n_last_frames=4, offset=100
    )
    assert intervals == [(101, 136), (146, 186)]
    with pytest.raises(ValueError):
        episode_valid_intervals([0], [10], drop_n_first_frames=-1)


def test_round_up_epoch_length_is_a_multiple_of_the_global_batch():
    assert round_up_epoch_length(1000, 8, 3) == 1008
    assert round_up_epoch_length(1008, 8, 3) == 1008
    with pytest.raises(ValueError):
        round_up_epoch_length(0, 8, 3)


def test_indices_stay_inside_valid_intervals():
    offsets, lengths, intervals = build_mixture()
    sampler = make_sampler(samples_per_epoch=20000)
    indices = sampler.indices(0)
    assert indices.dtype == np.int64
    assert set(indices.tolist()) <= set(valid_list(intervals))
    repos = sampler.repo_of(indices)
    assert (repos >= 0).all()
    for repo_idx, (offset, length) in enumerate(zip(offsets, lengths, strict=True)):
        mask = (indices >= offset) & (indices < offset + length)
        assert (repos[mask] == repo_idx).all()
    assert sampler.valid_indices.tolist() == valid_list(intervals)


def test_order_is_a_pure_function_of_seed_and_epoch():
    first, second = make_sampler(), make_sampler()
    epochs = [list(first) for _ in range(3)]
    assert epochs == [list(second) for _ in range(3)]
    assert epochs[0] != epochs[1]
    positioned = make_sampler()
    positioned.set_epoch(2)
    assert list(positioned) == epochs[2]
    assert list(make_sampler(seed=8)) != epochs[0]
    assert epochs[1] == make_sampler().indices(1).tolist()


def test_chunking_is_transparent():
    sampler = make_sampler(samples_per_epoch=1000, chunk_size=7)
    full = sampler.indices(3)
    assert list(sampler._iter_epoch(3, 0)) == full.tolist()
    assert sampler.indices(3, 123, 456).tolist() == full[123:456].tolist()
    assert make_sampler(samples_per_epoch=1000, chunk_size=1000).indices(3).tolist() == full.tolist()


def test_repo_frequencies_follow_weights():
    sampler = make_sampler(weights=(0.5, 0.3, 0.2), samples_per_epoch=200_000)
    counts = np.bincount(sampler.repo_of(sampler.indices(0)), minlength=3) / 200_000
    assert np.allclose(counts, [0.5, 0.3, 0.2], atol=0.01)


def test_none_weights_are_proportional_to_valid_frames():
    sampler = make_sampler(weights=None, samples_per_epoch=200_000)
    valid = np.asarray(sampler.repo_valid_lengths, dtype=np.float64)
    assert np.allclose(sampler.repo_probs, valid / valid.sum())
    indices = sampler.indices(0)
    # Uniform over all valid frames: every frame is hit about equally often.
    hits = np.bincount(indices, minlength=max(indices) + 1)[sampler.valid_indices]
    expected = 200_000 / valid.sum()
    assert abs(hits.mean() - expected) < 1e-9
    assert hits.std() < 3 * math.sqrt(expected)


def test_frames_within_a_repo_are_uniform_across_episodes():
    _, _, intervals = build_mixture()
    sampler = make_sampler(weights=(0.0, 0.0, 1.0), samples_per_epoch=100_000)
    indices = sampler.indices(0)
    for start, end in intervals[2]:
        share = ((indices >= start) & (indices < end)).mean()
        assert abs(share - (end - start) / sampler.repo_valid_lengths[2]) < 0.01


def test_zero_weight_and_empty_repos_are_never_sampled():
    sampler = make_sampler(weights=(0.0, 1.0, 1.0), samples_per_epoch=50_000)
    assert 0 not in set(sampler.repo_of(sampler.indices(0)).tolist())
    empty = WeightedMixtureSampler(
        repo_valid_intervals=[[], [(0, 10)]], repo_weights=[5.0, 1.0], samples_per_epoch=100, seed=1
    )
    assert empty.repo_probs.tolist() == [0.0, 1.0]
    with pytest.raises(ValueError):
        WeightedMixtureSampler(repo_valid_intervals=[[], []], repo_weights=None, samples_per_epoch=10)
    with pytest.raises(ValueError):
        WeightedMixtureSampler(repo_valid_intervals=[[(0, 5)]], repo_weights=[0.0], samples_per_epoch=10)


def test_without_replacement_visits_every_valid_frame_once_per_epoch():
    _, _, intervals = build_mixture()
    valid = valid_list(intervals)
    sampler = make_sampler(weights=None, samples_per_epoch=len(valid), seed=9, replacement=False)
    epoch0, epoch1 = sampler.indices(0), sampler.indices(1)
    assert sorted(epoch0.tolist()) == valid
    assert sorted(epoch1.tolist()) == valid
    assert epoch0.tolist() != epoch1.tolist()
    # Seeded EpisodeAwareSampler semantics: valid_indices[randperm(n, manual_seed(seed + epoch))].
    for epoch in (0, 1, 5):
        order = torch.randperm(len(valid), generator=torch.Generator().manual_seed(9 + epoch)).numpy()
        assert sampler.indices(epoch).tolist() == np.asarray(valid)[order].tolist()
    assert list(sampler) == epoch0.tolist()
    with pytest.raises(ValueError, match="replacement=False"):
        make_sampler(weights=(1.0, 1.0, 1.0), replacement=False)


def test_without_replacement_wraps_when_the_epoch_is_rounded_up():
    _, _, intervals = build_mixture()
    n_valid = len(valid_list(intervals))
    length = round_up_epoch_length(n_valid, 8, 3)
    sampler = make_sampler(weights=None, samples_per_epoch=length, seed=9, replacement=False)
    indices = sampler.indices(2)
    assert len(indices) == length
    assert indices[n_valid:].tolist() == indices[: length - n_valid].tolist()
    assert sampler.indices(2, 100, 140).tolist() == indices[100:140].tolist()


def test_state_dict_roundtrip_and_len_during_resume():
    reference = make_sampler(seed=3)
    reference.set_epoch(1)
    epoch1, epoch2 = list(reference), list(reference)
    assert reference.state_dict() == {"epoch": 3, "start_index": 0}

    resumed = make_sampler(seed=3)
    resumed.load_state_dict({"epoch": 1, "start_index": 5})
    assert resumed.state_dict() == {"epoch": 1, "start_index": 5}
    assert len(resumed) == len(epoch1)
    assert list(resumed) == epoch1[5:]
    assert resumed.state_dict() == {"epoch": 2, "start_index": 0}
    assert list(resumed) == epoch2
    with pytest.raises(ValueError):
        resumed.load_state_dict({"epoch": 0, "start_index": len(resumed) + 1})


def test_accelerate_set_epoch_after_load_state_dict_does_not_reset_the_epoch():
    # accelerate's DataLoaderShard.__iter__ calls sampler.set_epoch(iteration) with iteration=0,1,...
    reference = make_sampler(seed=3)
    epoch2 = reference.indices(2).tolist()
    epoch3 = reference.indices(3).tolist()
    resumed = make_sampler(seed=3)
    resumed.load_state_dict({"epoch": 2, "start_index": 10})
    resumed.set_epoch(0)
    assert list(resumed) == epoch2[10:]
    resumed.set_epoch(1)
    assert list(resumed) == epoch3


def run_ranks(sampler_factory, *, batch_size, num_processes, steps, resume_step=0):
    """Per-rank batches for loop steps [resume_step, steps) under accelerate's sharding."""
    outputs = {rank: [] for rank in range(num_processes)}
    for rank in range(num_processes):
        sampler = sampler_factory()
        if resume_step > 0:
            sampler.load_state_dict(
                compute_sampler_state(resume_step, len(sampler), batch_size, num_processes)
            )
        shard = BatchSamplerShard(
            BatchSampler(sampler, batch_size=batch_size, drop_last=False),
            num_processes=num_processes,
            process_index=rank,
            split_batches=False,
            even_batches=True,
        )
        iteration = 0
        step = resume_step
        while step < steps:
            sampler.set_epoch(iteration)  # what DataLoaderShard.__iter__ does
            for batch in shard:
                outputs[rank].append((step, batch))
                step += 1
                if step >= steps:
                    break
            iteration += 1
    return outputs


@pytest.mark.parametrize("replacement", [True, False])
@pytest.mark.parametrize(
    ("batch_size", "num_processes", "resume_step"),
    [(4, 1, 7), (4, 2, 5), (4, 2, 12), (3, 3, 10), (8, 4, 9), (5, 2, 0)],
)
def test_resume_equivalence_across_ranks(batch_size, num_processes, resume_step, replacement):
    _, _, intervals = build_mixture()
    total_valid = sum(end - start for repo in intervals for start, end in repo)
    length = round_up_epoch_length(total_valid, batch_size, num_processes)
    bpe = length // (batch_size * num_processes)
    steps = 3 * bpe + 2  # spans several epoch boundaries

    def factory():
        return WeightedMixtureSampler(
            repo_valid_intervals=intervals,
            repo_weights=(0.5, 0.3, 0.2) if replacement else None,
            samples_per_epoch=length,
            seed=11,
            replacement=replacement,
        )

    uninterrupted = run_ranks(factory, batch_size=batch_size, num_processes=num_processes, steps=steps)
    resumed = run_ranks(
        factory, batch_size=batch_size, num_processes=num_processes, steps=steps, resume_step=resume_step
    )
    for rank in range(num_processes):
        assert resumed[rank] == uninterrupted[rank][resume_step:]
    # Ranks never overlap within a step and together cover the whole epoch stream.
    flat = {}
    for rank in range(num_processes):
        for step, batch in uninterrupted[rank]:
            assert len(batch) == batch_size
            flat.setdefault(step, []).extend(batch)
    assert flat[0] == factory().indices(0, 0, batch_size * num_processes).tolist()
    assert flat[bpe] == factory().indices(1, 0, batch_size * num_processes).tolist()


def test_resume_is_inexact_when_epoch_length_is_not_a_multiple_of_the_global_batch():
    # Documents why make_dataloaders rounds the epoch length up: accelerate pads the final step of an
    # epoch from the first batches of the *current* iteration, which differ after a resume.
    _, _, intervals = build_mixture()
    batch_size, num_processes = 4, 2
    length = round_up_epoch_length(100, batch_size, num_processes) + batch_size  # one stray batch

    def factory():
        return WeightedMixtureSampler(
            repo_valid_intervals=intervals, repo_weights=None, samples_per_epoch=length, seed=5
        )

    bpe = math.ceil(math.ceil(length / batch_size) / num_processes)
    uninterrupted = run_ranks(factory, batch_size=batch_size, num_processes=num_processes, steps=bpe)
    resumed = run_ranks(factory, batch_size=batch_size, num_processes=num_processes, steps=bpe, resume_step=1)
    assert any(resumed[rank] != uninterrupted[rank][1:] for rank in range(num_processes))


def test_dataloader_roundtrip_matches_indices():
    sampler = make_sampler(samples_per_epoch=64, seed=2)
    loader = torch.utils.data.DataLoader(list(range(400)), batch_size=8, sampler=sampler, shuffle=False)
    seen = [int(v) for batch in loader for v in batch]
    assert seen == sampler.indices(0).tolist()
