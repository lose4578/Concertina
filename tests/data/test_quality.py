"""Training-time data-quality filter: package files, interval projection, field binding and target masking."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from concertina.data.quality import (
    ACTION_PAD_KEY,
    PACKAGE_TABLES,
    QualityConfig,
    QualityIndex,
    QualityPackage,
    complement_intervals,
    intersect_intervals,
    load_quality_index,
    union_intervals,
)
from concertina.data.sampler import WeightedMixtureSampler, episode_valid_intervals, round_up_epoch_length

REFERENCE = (
    Path(os.environ["CONCERTINA_QUALITY_REFERENCE_DIR"])
    if os.environ.get("CONCERTINA_QUALITY_REFERENCE_DIR")
    else None
)
FEATURES = ["action", "action.relative", "observation.state", "observation.images.top", "index", "timestamp"]


def _frames(intervals) -> set[int]:
    return {frame for start, end in intervals for frame in range(start, end)}


def write_package(root: Path, lengths, *, quarantine=(), exclude=(), masks=(), blockers=()) -> QualityPackage:
    """A package over episodes ``0..len(lengths)-1``; exclusion/mask rows: (episode, start, end[, field])."""
    tables = {name: [] for name in PACKAGE_TABLES}
    tables["episode_metrics"] = [
        {"episode_index": episode, "frame_count": length} for episode, length in enumerate(lengths)
    ]
    tables["episode_decisions"] = [
        {"episode_index": episode, "decision": "quarantine" if episode in quarantine else "keep"}
        for episode in range(len(lengths))
    ]
    tables["frame_decisions"] = [
        {"episode_index": e, "start_frame": s, "end_frame": t, "decision": "exclude_source_frames"}
        for e, s, t in exclude
    ]
    tables["field_masks"] = [
        {"episode_index": e, "start_frame": s, "end_frame": t, "field": f, "decision": "mask_field"}
        for e, s, t, f in masks
    ]
    return QualityPackage.write(
        root, dataset={"total_episodes": len(lengths)}, tables=tables, release_blockers=blockers
    )


def make_index(
    package, lengths, *, action_offsets=range(4), observation_offsets=(0,), **binding
) -> QualityIndex:
    binding = {"action_fields": ["action"], "observation_fields": ["observation.state"], **binding}
    return QualityIndex(
        package,
        episode_lengths=dict(enumerate(lengths)),
        features=FEATURES,
        action_offsets=list(action_offsets),
        observation_offsets=list(observation_offsets),
        **binding,
    )


def _random_intervals(rng, limit: int, count: int):
    starts = rng.integers(-3, limit + 3, count)
    return [(int(s), int(s + rng.integers(0, 6))) for s in starts]


def test_interval_algebra_matches_frame_sets():
    rng = np.random.default_rng(0)
    for _ in range(300):
        left, right = _random_intervals(rng, 30, 5), _random_intervals(rng, 30, 5)
        union = union_intervals(left)
        assert _frames(union) == _frames(left)
        assert all(a[1] < b[0] for a, b in zip(union, union[1:], strict=False)), "unions must not touch"
        assert _frames(intersect_intervals(left, right)) == _frames(left) & _frames(right)
        assert _frames(complement_intervals(left, 2, 25)) == set(range(2, 25)) - _frames(left)


def test_package_roundtrip_and_integrity(tmp_path):
    package = write_package(tmp_path / "pkg", [5, 7], exclude=[(1, 2, 4)], blockers=["partial_scan"])
    loaded = QualityPackage.load(tmp_path / "pkg")
    assert not loaded.release_ready and loaded.release_blockers == ["partial_scan"]
    assert loaded.records("frame_decisions") == package.records("frame_decisions")
    assert QualityPackage.load(tmp_path / "pkg" / "quality_package.json").root == loaded.root

    table = tmp_path / "pkg" / "frame_decisions.jsonl"
    table.write_text(table.read_text().replace('"start_frame": 2', '"start_frame": 1'))
    with pytest.raises(ValueError, match="SHA-256"):
        loaded.records("frame_decisions")

    manifest = json.loads((tmp_path / "pkg" / "quality_package.json").read_text())
    manifest["tables"]["field_masks"]["path"] = "../escape.jsonl"
    (tmp_path / "pkg" / "quality_package.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside the package"):
        QualityPackage.load(tmp_path / "pkg")
    manifest["schema_version"] = "another-package-schema.v2"
    (tmp_path / "pkg" / "quality_package.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported quality package schema"):
        QualityPackage.load(tmp_path / "pkg")


def test_without_decisions_the_intervals_equal_the_drop_windows(tmp_path):
    lengths = [12, 3, 30, 9]
    index = make_index(write_package(tmp_path / "pkg", lengths), lengths, action_offsets=range(10))
    ends = np.cumsum(lengths)
    starts = ends - lengths
    for drop_first, drop_last in [(0, 0), (0, 9), (2, 5), (4, 20)]:
        expected = episode_valid_intervals(
            starts, ends, drop_n_first_frames=drop_first, drop_n_last_frames=drop_last, offset=100
        )
        got = index.valid_intervals(
            range(len(lengths)),
            starts,
            ends,
            drop_n_first_frames=drop_first,
            drop_n_last_frames=drop_last,
            offset=100,
        )
        assert got == expected


def test_projection_matches_the_per_frame_definition(tmp_path):
    rng = np.random.default_rng(1)
    for case in range(40):
        lengths = [int(v) for v in rng.integers(1, 25, 4)]
        exclude, masks = [], []
        for episode, length in enumerate(lengths):
            for _ in range(int(rng.integers(0, 3))):
                start = int(rng.integers(0, length))
                span = (episode, start, int(rng.integers(start + 1, length + 1)))
                choice = rng.integers(0, 4)
                if choice == 0:
                    exclude.append(span)
                else:
                    masks.append((*span, ["action", "observation.state", "action.relative"][choice - 1]))
        quarantine = {int(rng.integers(0, 4))} if case % 3 == 0 else set()
        action_offsets = list(range(int(rng.integers(1, 6))))
        observation_offsets = [0] if case % 2 else [-1, 0]
        package = write_package(
            tmp_path / f"pkg{case}", lengths, quarantine=quarantine, exclude=exclude, masks=masks
        )
        index = make_index(
            package, lengths, action_offsets=action_offsets, observation_offsets=observation_offsets
        )
        drop_first, drop_last = int(rng.integers(0, 3)), int(rng.integers(0, 4))
        for episode, length in enumerate(lengths):
            got = _frames(
                index.episode_sample_frames(
                    episode, drop_n_first_frames=drop_first, drop_n_last_frames=drop_last
                )
            )
            expected = {
                frame
                for frame in range(drop_first, length - drop_last)
                if episode not in quarantine
                and index.observation_valid(episode, frame)
                and index.target_mask(episode, frame).any()
            }
            assert got == expected, (case, episode)
            invalid_targets = _frames(index.invalid_targets.get(episode, ()))
            for frame in sorted(expected):
                mask = index.target_mask(episode, frame)
                for offset, valid in zip(action_offsets, mask, strict=True):
                    target = frame + offset
                    assert valid == (target < length and target not in invalid_targets)


def test_masks_bind_to_the_consumed_columns(tmp_path):
    lengths = [20]
    masks = [(0, 2, 4, "action"), (0, 10, 12, "observation.state"), (0, 15, 18, "action.relative")]
    index = make_index(write_package(tmp_path / "a", lengths, masks=masks), lengths, action_offsets=range(3))
    assert index.inert_mask_count == 1
    assert index.target_mask(0, 1).tolist() == [True, False, False]
    assert not index.observation_valid(0, 11) and index.observation_valid(0, 12)
    assert index.target_mask(0, 15).all(), "the unused action representation is inert"
    frames = _frames(index.episode_sample_frames(0))
    assert {2, 3} <= frames, "a sample keeps valid targets after its masked rows"
    assert not {10, 11} & frames and 18 in frames

    index = make_index(
        write_package(tmp_path / "b", lengths, masks=[(0, 0, 4, "observation.images.top")]),
        lengths,
        observation_fields=["observation.state", "observation.images.top"],
    )
    assert not index.observation_valid(0, 3)
    with pytest.raises(ValueError, match="not a column of the dataset"):
        make_index(write_package(tmp_path / "c", lengths, masks=[(0, 0, 4, "action.joints")]), lengths)
    with pytest.raises(ValueError, match="both as action and as observation"):
        make_index(write_package(tmp_path / "d", lengths), lengths, observation_fields=["action"])
    with pytest.raises(ValueError, match="not features of the dataset"):
        make_index(write_package(tmp_path / "e", lengths), lengths, action_fields=["action.absolute"])


def test_package_must_describe_the_dataset(tmp_path):
    package = write_package(tmp_path / "pkg", [5, 7])
    with pytest.raises(ValueError, match="another dataset"):
        make_index(package, [5, 8])
    with pytest.raises(ValueError, match="another dataset"):
        make_index(package, [5, 7, 2])
    index = make_index(package, [5, 7])
    with pytest.raises(ValueError, match="disagree"):
        index.valid_intervals([1], [0], [6])
    with pytest.raises(ValueError, match="outside episode"):
        make_index(write_package(tmp_path / "bad", [5], exclude=[(0, 3, 6)]), [5])


def test_mark_invalid_targets(tmp_path):
    lengths = [10]
    index = make_index(write_package(tmp_path / "pkg", lengths, exclude=[(0, 4, 6)]), lengths)
    item = {"episode_index": torch.tensor(0), "frame_index": torch.tensor(2), ACTION_PAD_KEY: torch.zeros(4)}
    marked = index.mark_invalid_targets(dict(item))
    assert marked[ACTION_PAD_KEY].tolist() == [False, False, True, True]
    item["frame_index"] = torch.tensor(7)
    item[ACTION_PAD_KEY] = torch.tensor([False, False, False, True])
    assert index.mark_invalid_targets(dict(item))[ACTION_PAD_KEY].tolist() == [False, False, False, True]
    clean = {"episode_index": torch.tensor(0), "frame_index": torch.tensor(0)}
    assert ACTION_PAD_KEY not in index.mark_invalid_targets(dict(clean))
    with pytest.raises(ValueError, match="not a valid sample"):
        index.mark_invalid_targets({"episode_index": torch.tensor(0), "frame_index": torch.tensor(5)})
    with pytest.raises(ValueError, match="shape"):
        index.mark_invalid_targets({**item, ACTION_PAD_KEY: torch.zeros(3)})


def test_quality_restricted_sampler_draws_only_valid_samples(tmp_path):
    lengths = [30, 12, 25]
    package = write_package(
        tmp_path / "pkg",
        lengths,
        quarantine={1},
        exclude=[(0, 5, 9), (2, 0, 4)],
        masks=[(2, 10, 20, "observation.state")],
    )
    index = make_index(package, lengths, action_offsets=range(5))
    ends = np.cumsum(lengths)
    starts = ends - lengths
    intervals = index.valid_intervals(range(3), starts, ends, drop_n_last_frames=2, offset=7)
    valid = _frames(intervals)
    total = len(valid)
    for replacement in (False, True):
        sampler = WeightedMixtureSampler(
            repo_valid_intervals=[intervals],
            repo_weights=None,
            samples_per_epoch=round_up_epoch_length(total, 4, 2),
            seed=3,
            replacement=replacement,
        )
        epoch = sampler.indices(0)
        assert set(epoch.tolist()) <= valid
        if not replacement:
            assert set(epoch[:total].tolist()) == valid
        resumed = WeightedMixtureSampler(
            repo_valid_intervals=[intervals],
            repo_weights=None,
            samples_per_epoch=len(sampler),
            seed=3,
            replacement=replacement,
        )
        resumed.load_state_dict({"epoch": 1, "start_index": 8})
        assert list(resumed) == sampler.indices(1)[8:].tolist()


def test_load_quality_index_on_a_lerobot_dataset(tmp_path, dataset_dir):
    root = dataset_dir / "libero"
    meta = LeRobotDatasetMetadata("local/libero", root=root)
    lengths = [int(v) for v in meta.episodes.with_format("numpy")["length"]]
    blocked = write_package(tmp_path / "pkg", lengths, exclude=[(1, 2, 5)], blockers=["review_pending"])
    binding = {
        "meta": meta,
        "action_fields": ["action"],
        "observation_fields": ["observation.state", "observation.images.image"],
        "action_offsets": list(range(4)),
        "observation_offsets": [0],
    }
    with pytest.raises(ValueError, match="not release-ready"):
        load_quality_index(blocked.root, **binding)
    index = load_quality_index(blocked.root, config=QualityConfig(require_release_ready=False), **binding)

    dataset = LeRobotDataset(
        "local/libero", root=root, delta_timestamps={"action": [d / meta.fps for d in range(4)]}
    )
    offset = int(meta.episodes[1]["dataset_from_index"])
    item = index.mark_invalid_targets(dataset[offset])
    assert item[ACTION_PAD_KEY].tolist() == [False, False, True, True]
    last = dataset[offset + lengths[1] - 2]
    assert index.mark_invalid_targets(last)[ACTION_PAD_KEY].tolist() == [False, False, True, True]


@pytest.mark.skipif(
    REFERENCE is None or not (REFERENCE / "runtime_projection.json").is_file(),
    reason="CONCERTINA_QUALITY_REFERENCE_DIR with legacy reference outputs is not set",
)
def test_projection_matches_the_legacy_runtime_sampler(tmp_path):
    """Legacy `runtime_valid_intervals_to_global` / `QualityMaskIndex.target_mask` on the same decisions."""
    cases = json.loads((REFERENCE / "runtime_projection.json").read_text())
    assert cases
    for number, case in enumerate(cases):
        lengths = case["lengths"]
        package = write_package(
            tmp_path / f"pkg{number}",
            lengths,
            quarantine=set(case["quarantine"]),
            exclude=[tuple(v) for v in case["exclude"]],
            masks=[tuple(v) for v in case["masks"]],
        )
        index = make_index(
            package,
            lengths,
            action_offsets=case["action_offsets"],
            observation_offsets=case["observation_offsets"],
        )
        ends = np.cumsum(lengths)
        got = index.valid_intervals(
            range(len(lengths)),
            ends - lengths,
            ends,
            drop_n_first_frames=case["drop_first"],
            drop_n_last_frames=case["drop_last"],
        )
        assert union_intervals(got) == [tuple(v) for v in case["valid_intervals"]], number
        for episode, frame, mask in case["target_masks"]:
            # The legacy mask reports out-of-episode targets as valid; they are reader padding either way.
            in_episode = [frame + d < lengths[episode] for d in case["action_offsets"]]
            expected = np.asarray(mask) & np.asarray(in_episode)
            assert index.target_mask(episode, frame).tolist() == expected.tolist(), (number, episode, frame)


def test_history_targets_alone_do_not_make_a_sample(tmp_path):
    lengths = [20]
    index = make_index(
        write_package(tmp_path / "pkg", lengths, exclude=[(0, 8, 12)]), lengths, action_offsets=range(-3, 4)
    )
    frames = _frames(index.episode_sample_frames(0))
    assert 7 in frames and 12 in frames
    assert not {8, 9, 10, 11} & frames, "only history targets would remain valid"
    item = {"episode_index": torch.tensor(0), "frame_index": torch.tensor(13)}
    assert index.mark_invalid_targets(item)[ACTION_PAD_KEY].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]


HEAD, WRIST = "observation.images.head", "observation.images.wrist_left"
CHUNK = 6
DROP_LAST = 2


def test_mixture_training_with_quality_packages(tmp_path, dataset_dir):
    from concertina.data.embodiments import interpolate_padding
    from concertina.data.mixture import ConcertinaMixtureDataset
    from concertina.policy.config_data import ConcertinaDataConfig, DatasetSpec
    from concertina.train.factories import make_mixture_sampler

    def lengths(name):
        meta = LeRobotDatasetMetadata(f"local/{name}", root=dataset_dir / name)
        return [int(v) for v in meta.episodes.with_format("numpy")["length"]]

    calvin, droid = lengths("calvin"), lengths("droid")
    # CALVIN trains on action.absolute: the action.relative mask is inert. DROID masks a consumed column.
    calvin_package = write_package(
        tmp_path / "calvin", calvin, exclude=[(0, 4, 7)], masks=[(1, 0, 5, "action.relative")]
    )
    droid_package = write_package(
        tmp_path / "droid", droid, exclude=[(1, 0, 3)], masks=[(0, 8, 10, "action.gripper_position")]
    )
    calvin_spec = DatasetSpec(
        path="calvin",
        source="calvin",
        embodiment="calvin",
        valid_action_dim=10,
        camera_map={HEAD: ["observation.images.top"], WRIST: ["observation.images.wrist"]},
        quality_package=str(calvin_package.root),
    )
    droid_spec = DatasetSpec(
        path="droid",
        source="droid",
        embodiment="droid",
        valid_action_dim=10,
        camera_map={HEAD: ["observation.images.exterior_1_left"]},
        quality_package=str(droid_package.root),
    )
    specs = [calvin_spec, droid_spec, DatasetSpec(**{**droid_spec.__dict__, "episodes": [1]})]
    data = ConcertinaDataConfig(
        datasets=specs, drop_n_last_frames=DROP_LAST, quality={"require_release_ready": True}
    )
    assert isinstance(data.quality, QualityConfig)
    dataset = ConcertinaMixtureDataset(
        data.datasets,
        root=dataset_dir,
        camera_keys=[HEAD, WRIST],
        action_delta_indices=list(range(CHUNK)),
        observation_delta_indices=[0],
        image_hw=(24, 32),
        resize_images=True,
        fps_align_target_fps=30.0,
        quality_config=data.quality,
    )

    def valid_frames(lengths, episodes, excluded, masked):
        """Samples by the package definition: frame kept, one future target kept, inside the drop window."""
        output = set()
        cursor = 0
        for episode in episodes:
            length = lengths[episode]
            bad = excluded.get(episode, set())
            invalid = bad | masked.get(episode, set())
            for frame in range(length - DROP_LAST):
                targets = [frame + d for d in range(CHUNK) if frame + d < length]
                if frame not in bad and any(t not in invalid for t in targets):
                    output.add(cursor + frame)
            cursor += length
        return output

    expected = [
        valid_frames(calvin, range(len(calvin)), {0: set(range(4, 7))}, {}),
        valid_frames(droid, range(len(droid)), {1: set(range(3))}, {0: {8, 9}}),
        valid_frames(droid, [1], {1: set(range(3))}, {}),
    ]
    sampler = make_mixture_sampler(dataset, data, batch_size=4, dp_world_size=1, seed=0)
    assert sampler.repo_valid_lengths == [len(frames) for frames in expected]
    drawn = sampler.indices(0)
    for repo, frames in enumerate(expected):
        local = {int(i) - dataset.dataset_offsets[repo] for i in drawn if dataset.locate(int(i))[0] == repo}
        assert local and local <= frames, repo

    def padding(repo, local_index, invalid_offsets):
        sample = dataset[dataset.dataset_offsets[repo] + local_index]
        source = torch.tensor([d in invalid_offsets for d in range(CHUNK)])
        converter = dataset.converters[repo]
        expected = interpolate_padding(source, converter.source_fps, converter.target_fps)
        assert converter.resamples and sample["action_is_pad"].tolist() == expected.tolist(), (
            repo,
            local_index,
        )

    padding(0, 2, {2, 3, 4})  # targets 4..6 excluded
    padding(0, calvin[0], set())  # the inert action.relative mask
    padding(1, 6, {2, 3})  # targets 8, 9 masked
    padding(2, 4, set())  # first kept frame of the subset's episode 1
    with pytest.raises(ValueError, match="not a valid sample"):
        dataset[dataset.dataset_offsets[2] + 1]
