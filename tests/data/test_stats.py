import json
import shutil

import numpy as np
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from concertina.data.embodiments import make_source_adapter
from concertina.data.stats import (
    aggregate_feature_stats,
    build_normalization_stats,
    embodiment_stats,
    embodiment_stats_key,
    load_rotation_stats,
    merge_canonical_stats,
    pool_raw_stats,
)
from concertina.data.tools.rotation_stats import compute_rotation_stats, write_rotation_stats
from concertina.policy.config_data import DatasetSpec


def _stats(dim: int, count: int = 10, offset: float = 0.0) -> dict:
    base = np.arange(dim, dtype=np.float64) + offset
    return {
        "mean": base,
        "std": base + 1,
        "min": base - 1,
        "max": base + 2,
        "q01": base - 0.5,
        "q99": base + 1.5,
        "count": np.array([count]),
    }


def test_libero_merger_drops_the_second_finger_and_pads_the_right_arm():
    merged = merge_canonical_stats({"action": _stats(10), "observation.state": _stats(11)}, "libero")
    state = merged["observation.state"]
    np.testing.assert_array_equal(state["mean"][:10], np.arange(10))
    np.testing.assert_array_equal(state["mean"][10:], np.zeros(10))
    np.testing.assert_array_equal(state["std"][10:], np.ones(10))
    np.testing.assert_array_equal(state["min"][10:], np.zeros(10))
    np.testing.assert_array_equal(state["max"][10:], np.ones(10))
    np.testing.assert_array_equal(state["count"], [10])
    with pytest.raises(ValueError, match="LIBERO"):
        merge_canonical_stats({"action": _stats(12), "observation.state": _stats(11)}, "libero")


def test_calvin_merger_reads_the_selected_action_column():
    raw = {"action": _stats(10, offset=5), "action.absolute": _stats(10), "observation.state": _stats(10)}
    merged = merge_canonical_stats(raw, "calvin", "action.absolute")
    np.testing.assert_array_equal(merged["action"]["mean"][:10], np.arange(10))
    np.testing.assert_array_equal(merged["action"]["q01"][10:], np.zeros(10))
    np.testing.assert_array_equal(merged["action"]["q99"][10:], np.ones(10))
    with pytest.raises(KeyError, match="action.absolute"):
        merge_canonical_stats(
            {"action": _stats(10), "observation.state": _stats(10)}, "calvin", "action.absolute"
        )


def test_agibot_merger_interleaves_the_two_arms():
    def fields(prefix_pos, prefix_ori, prefix_grip, offset):
        return {
            prefix_pos: _stats(6, offset=offset),
            prefix_ori: _stats(12, offset=offset + 100),
            prefix_grip: _stats(2, offset=offset + 200),
        }

    raw = {
        **fields("actions.end.position", "actions.end.orientation", "actions.effector.position", 0),
        **fields(
            "observation.states.end.position",
            "observation.states.end.orientation",
            "observation.states.effector.position",
            1000,
        ),
    }
    merged = merge_canonical_stats(raw, "agibot")
    expected = np.concatenate([[0, 1, 2], np.arange(100, 106), [200], [3, 4, 5], np.arange(106, 112), [201]])
    np.testing.assert_array_equal(merged["action"]["mean"], expected)
    assert set(merged["action"]) == {"mean", "std", "min", "max"}


def test_robomind_merger_pads_single_arm_statistics():
    merged = merge_canonical_stats({"action": _stats(10), "observation.state": _stats(10)}, "robomind")
    np.testing.assert_array_equal(merged["action"]["q99"][10:], np.zeros(10))
    np.testing.assert_array_equal(merged["action"]["std"][10:], np.ones(10))


def test_pooling_matches_the_statistics_of_the_concatenated_data():
    rng = np.random.default_rng(0)
    parts = [rng.normal(i, 1 + i, (n, 4)) for i, n in enumerate((50, 80, 30))]

    def stats(data):
        return {
            "mean": data.mean(0),
            "std": data.std(0),
            "min": data.min(0),
            "max": data.max(0),
            "count": np.array([len(data)]),
        }

    pooled = aggregate_feature_stats([stats(part) for part in parts])
    reference = stats(np.concatenate(parts))
    for name in ("mean", "std", "min", "max", "count"):
        np.testing.assert_allclose(pooled[name], reference[name], rtol=1e-12)


def test_pooling_requires_counts():
    raw = {"action": _stats(10), "observation.state": _stats(10)}
    without_count = {key: {k: v for k, v in value.items() if k != "count"} for key, value in raw.items()}
    assert pool_raw_stats([without_count], ["action", "observation.state"], ["a"]) is without_count
    with pytest.raises(ValueError, match="count"):
        pool_raw_stats([raw, without_count], ["action", "observation.state"], ["a", "b"])


@pytest.mark.parametrize("source", ["libero", "calvin", "droid"])
def test_rotation_stats_tool_matches_the_canonical_data(dataset_dir, source):
    """The offline statistics describe exactly what the adapters produce (first arm for single-arm sources)."""
    dataset = LeRobotDataset(str(dataset_dir / source), root=dataset_dir / source)
    adapter = make_source_adapter(source, None, dataset.meta.features)
    rows = dataset.hf_dataset.with_format("torch")[:]
    action, state = adapter.action(rows).double(), adapter.state(rows).double()
    stats = merge_canonical_stats(
        load_rotation_stats(dataset_dir / source / "meta" / "rotation_stats.json"),
        source,
        adapter.action_keys[0] if source == "calvin" else "action",
    )
    width = 10
    for key, values in (("action", action), ("observation.state", state)):
        np.testing.assert_allclose(stats[key]["mean"][:width], values[:, :width].mean(0).numpy(), atol=1e-6)
        np.testing.assert_allclose(
            stats[key]["std"][:width], values[:, :width].std(0, correction=0).numpy(), atol=1e-6
        )
        np.testing.assert_array_equal(stats[key]["count"], [len(values)])


def test_rotation_stats_tool_pools_files(dataset_dir, tmp_path):
    single = compute_rotation_stats(dataset_dir / "libero", "libero")
    write_rotation_stats(single, tmp_path / "rotation_stats.json", "libero")
    payload = json.loads((tmp_path / "rotation_stats.json").read_text())
    assert set(payload["modes"]) == {"rotate6d"} and payload["dataset_type"] == "libero"
    assert set(payload["modes"]["rotate6d"]) == {"action", "observation.state"}
    assert len(payload["modes"]["rotate6d"]["observation.state"]["mean"]) == 11


def _spec(path, **kwargs):
    defaults = {"source": "libero", "embodiment": "libero", "valid_action_dim": 10}
    return DatasetSpec(path=str(path), **{**defaults, **kwargs})


def test_global_statistics_read_a_shared_file_once(dataset_dir, tmp_path):
    copy = tmp_path / "libero_copy"
    shutil.copytree(dataset_dir / "libero", copy)
    stats_file = str(dataset_dir / "libero" / "meta" / "rotation_stats.json")
    specs = [_spec(dataset_dir / "libero"), _spec(copy, rotation_stats_path=stats_file)]
    roots = [dataset_dir / "libero", copy]
    shared = build_normalization_stats(specs, roots, ["action"] * 2, normalization="global")
    single = build_normalization_stats(specs[:1], roots[:1], ["action"], normalization="global")
    for key in ("action", "observation.state"):
        for name, value in single[key].items():
            torch.testing.assert_close(shared[key][name], value)
        assert shared[key]["mean"].dtype == torch.float32 and shared[key]["mean"].shape == (20,)

    pooled = build_normalization_stats([specs[0], _spec(copy)], roots, ["action"] * 2, normalization="global")
    torch.testing.assert_close(pooled["action"]["mean"], single["action"]["mean"])
    torch.testing.assert_close(pooled["action"]["count"], 2 * single["action"]["count"])


def test_per_embodiment_statistics_and_imagenet_entries(dataset_dir):
    specs = [
        _spec(dataset_dir / "libero"),
        _spec(dataset_dir / "agibot", source="agibot", embodiment="agibot", valid_action_dim=20),
    ]
    stats = build_normalization_stats(
        specs,
        [dataset_dir / "libero", dataset_dir / "agibot"],
        ["action", "action"],
        normalization="per_embodiment",
        camera_keys=["observation.images.head"],
        use_imagenet_stats=True,
    )
    assert set(stats) == {
        embodiment_stats_key("libero", "action"),
        embodiment_stats_key("libero", "observation.state"),
        embodiment_stats_key("agibot", "action"),
        embodiment_stats_key("agibot", "observation.state"),
        "observation.images.head",
    }
    assert stats["observation.images.head"]["mean"].shape == (3, 1, 1)
    unpacked = embodiment_stats(stats, ["libero", "agibot", "libero"])
    assert set(unpacked) == {"libero", "agibot"}
    with pytest.raises(KeyError, match="per_embodiment"):
        embodiment_stats(stats, ["droid"])
    assert (
        build_normalization_stats(specs, [dataset_dir / "libero"] * 2, ["action"] * 2, normalization="none")
        == {}
    )


def test_missing_or_incomplete_statistics_point_to_the_tool(tmp_path):
    with pytest.raises(FileNotFoundError, match="concertina-compute-rotation-stats"):
        load_rotation_stats(tmp_path / "rotation_stats.json")
    (tmp_path / "rotation_stats.json").write_text(json.dumps({"modes": {"quaternion": {}}}))
    with pytest.raises(ValueError, match="rotate6d"):
        load_rotation_stats(tmp_path / "rotation_stats.json")


def test_missing_statistics_are_computed_once_into_the_cache(dataset_dir, tmp_path, monkeypatch):
    import threading

    from concertina.data import stats as stats_module
    from concertina.data.tools import rotation_stats

    dataset = tmp_path / "libero"
    shutil.copytree(dataset_dir / "libero", dataset)
    (dataset / "meta" / "rotation_stats.json").unlink()
    monkeypatch.setenv("CONCERTINA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CONCERTINA_STATS_WORKERS", "1")
    calls = []
    compute = rotation_stats.compute_rotation_stats

    def counting(*args, **kwargs):
        calls.append(args)
        return compute(*args, **kwargs)

    monkeypatch.setattr(rotation_stats, "compute_rotation_stats", counting)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(stats_module.cached_rotation_stats(dataset, "libero")))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 1 and len(set(results)) == 1
    (path,) = set(results)
    assert path.parent == tmp_path / "cache" / "rotation_stats" and path.name.startswith("libero-")
    expected = load_rotation_stats(dataset_dir / "libero" / "meta" / "rotation_stats.json")
    for feature, values in load_rotation_stats(path).items():
        for name, value in values.items():
            np.testing.assert_array_equal(value, expected[feature][name])
    assert sorted(p.name for p in (dataset / "meta").iterdir()) == [
        "episodes",
        "info.json",
        "stats.json",
        "tasks.parquet",
    ]
    # Another dataset (different episode metadata) gets its own statistics.
    assert stats_module.dataset_fingerprint(dataset_dir / "calvin", "calvin") != path.stem.split("-", 1)[1]
