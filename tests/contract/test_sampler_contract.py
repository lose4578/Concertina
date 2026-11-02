import lerobot.datasets as lerobot_datasets
import pytest
from lerobot.datasets.sampler import EpisodeAwareSampler, compute_sampler_state

SAMPLER_CONSEQUENCE = (
    "The Concertina weighted sampler mirrors this API so that lerobot's resume path "
    "(compute_sampler_state -> sampler.load_state_dict) stays sample-exact; re-audit concertina/data before "
    "updating the expected value here."
)

EXPECTED_SAMPLER_INIT = (
    "(self, dataset_from_indices: list[int], dataset_to_indices: list[int], "
    "episode_indices_to_use: list | None = None, drop_n_first_frames: int = 0, "
    "drop_n_last_frames: int = 0, shuffle: bool = False, seed: int = 0, "
    "absolute_to_relative_idx: dict[int, int] | None = None)"
)
EXPECTED_COMPUTE_SAMPLER_STATE = "(step: int, num_frames: int, batch_size: int, num_processes: int) -> dict"
STATE_KEYS = {"epoch", "start_index"}

# (step, num_frames, batch_size, num_processes) -> state. One step consumes batch_size * num_processes
# positions; an epoch holds ceil(ceil(num_frames / batch_size) / num_processes) steps.
COMPUTE_SAMPLER_STATE_CASES = [
    ((0, 100, 8, 2), {"epoch": 0, "start_index": 0}),
    ((3, 100, 8, 2), {"epoch": 0, "start_index": 48}),
    ((6, 100, 8, 2), {"epoch": 0, "start_index": 96}),
    ((7, 100, 8, 2), {"epoch": 1, "start_index": 0}),
    ((10, 100, 8, 2), {"epoch": 1, "start_index": 48}),
    ((5, 10, 3, 1), {"epoch": 1, "start_index": 3}),
]

EPISODE_FROM = [0, 10, 15]
EPISODE_TO = [10, 15, 30]


def _sampler(**kwargs) -> EpisodeAwareSampler:
    return EpisodeAwareSampler(EPISODE_FROM, EPISODE_TO, **kwargs)


def test_sampler_is_exported_where_lerobot_train_imports_it():
    assert lerobot_datasets.EpisodeAwareSampler is EpisodeAwareSampler
    assert lerobot_datasets.compute_sampler_state is compute_sampler_state


def test_sampler_signatures(assert_signature):
    assert_signature(EpisodeAwareSampler.__init__, EXPECTED_SAMPLER_INIT, SAMPLER_CONSEQUENCE)
    assert_signature(compute_sampler_state, EXPECTED_COMPUTE_SAMPLER_STATE, SAMPLER_CONSEQUENCE)


@pytest.mark.parametrize("method", ["set_epoch", "state_dict", "load_state_dict", "__len__", "__iter__"])
def test_sampler_protocol_methods_exist(method):
    assert callable(getattr(EpisodeAwareSampler, method, None)), (
        f"EpisodeAwareSampler.{method} is gone. {SAMPLER_CONSEQUENCE}"
    )


@pytest.mark.parametrize(("args", "expected"), COMPUTE_SAMPLER_STATE_CASES)
def test_compute_sampler_state_values(args, expected):
    assert compute_sampler_state(*args) == expected, (
        f"compute_sampler_state{args} changed its step -> (epoch, start_index) mapping. {SAMPLER_CONSEQUENCE}"
    )


def test_state_dict_keys_match_compute_sampler_state():
    assert set(_sampler().state_dict()) == STATE_KEYS
    assert set(compute_sampler_state(3, 100, 8, 2)) == STATE_KEYS


def test_order_is_a_pure_function_of_seed_and_epoch():
    first, second = _sampler(shuffle=True, seed=7), _sampler(shuffle=True, seed=7)
    epochs = [list(first) for _ in range(3)]
    assert epochs == [list(second) for _ in range(3)]
    assert epochs[0] != epochs[1], "consecutive epochs produced the same permutation"
    positioned = _sampler(shuffle=True, seed=7)
    positioned.set_epoch(2)
    assert list(positioned) == epochs[2]
    assert list(_sampler(shuffle=True, seed=8)) != epochs[0]


def test_load_state_dict_resumes_sample_exactly():
    reference = _sampler(shuffle=True, seed=3)
    reference.set_epoch(1)
    epoch1, epoch2 = list(reference), list(reference)

    resumed = _sampler(shuffle=True, seed=3)
    resumed.load_state_dict({"epoch": 1, "start_index": 5})
    assert len(resumed) == len(epoch1), "__len__ must report the full epoch length while resuming"
    assert list(resumed) == epoch1[5:]
    assert list(resumed) == epoch2, "the epoch after a resumed one must start from index 0"


def test_drop_first_and_last_frames_per_episode():
    sampler = _sampler(drop_n_first_frames=1, drop_n_last_frames=2)
    expected = [*range(1, 8), *range(11, 13), *range(16, 28)]
    assert list(sampler) == expected
    assert len(sampler) == len(expected)
