from pathlib import Path

import pytest
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import DeviceProcessorStep, NormalizerProcessorStep
from lerobot.utils.feature_utils import dataset_to_policy_features
from torch.utils.data import default_collate

from concertina.data.mixture import ConcertinaMixtureDataset
from concertina.data.stats import embodiment_stats_key
from concertina.policy.config_data import ConcertinaDataConfig, DatasetSpec
from concertina.policy.config_model import ConcertinaModelConfig
from concertina.policy.configuration_concertina import ConcertinaConfig
from concertina.policy.processor_concertina import (
    EmbodimentNormalizeStep,
    GripperProbabilityRestoreStep,
    SourceBindingStep,
    ValidDimMaskStep,
    make_concertina_pre_post_processors,
    make_concertina_pre_post_processors_from_pretrained,
)

HEAD, WRIST_LEFT, WRIST_RIGHT = (
    "observation.images.head",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)
CHUNK = 10
EPS = 1e-8
CALVIN = DatasetSpec(
    path="calvin",
    source="calvin",
    embodiment="calvin",
    valid_action_dim=10,
    camera_map={HEAD: ["observation.images.top"], WRIST_LEFT: ["observation.images.wrist"]},
)
LIBERO = DatasetSpec(
    path="libero",
    source="libero",
    embodiment="libero",
    valid_action_dim=10,
    camera_map={HEAD: ["observation.images.image"], WRIST_LEFT: ["observation.images.image2"]},
)
AGIBOT = DatasetSpec(
    path="agibot",
    source="agibot",
    embodiment="agibot",
    camera_map={
        WRIST_LEFT: ["observation.images.hand_left"],
        WRIST_RIGHT: ["observation.images.hand_right"],
    },
)


def setup(
    dataset_dir: Path, specs, *, cameras, gripper_indices, gripper_target="probability", **data_options
):
    data = ConcertinaDataConfig(datasets=specs, **data_options)
    model = ConcertinaModelConfig(
        camera_keys=cameras, chunk_size=CHUNK, gripper_indices=gripper_indices, gripper_target=gripper_target
    )
    config = ConcertinaConfig(device="cpu", model=model, data=data)
    mixture = ConcertinaMixtureDataset(
        specs,
        root=dataset_dir,
        camera_keys=cameras,
        action_delta_indices=config.action_delta_indices,
        observation_delta_indices=config.observation_delta_indices,
        image_hw=(24, 32),
        resize_images=data.resize_images,
        normalization=data.normalization,
    )
    features = dataset_to_policy_features(mixture.meta.features)
    config.output_features = {"action": features.pop("action")}
    config.input_features = features
    return config, mixture


def normalize(value, stats):
    return (value - stats["mean"]) / (stats["std"] + EPS)


def closure(value, opened, closed):
    return ((value.double() - opened) / (closed - opened)).float()


@pytest.fixture(scope="module")
def calvin_setup(dataset_dir):
    config, mixture = setup(
        dataset_dir,
        [CALVIN],
        cameras=[HEAD, WRIST_LEFT],
        gripper_indices=[9],
        normalization="global",
    )
    batch = default_collate(mixture.__getitems__([0, 5, 9, 11, 20]))
    return config, mixture, batch


def test_global_pipeline_calibrates_normalizes_and_writes_probability_tokens(calvin_setup):
    config, mixture, batch = calvin_setup
    pre, _ = make_concertina_pre_post_processors(config, mixture.meta.stats)
    out = pre(dict(batch))

    stats = mixture.meta.stats
    valid = ~batch["action_is_pad"]
    calibrated = batch["action"].clone()
    calibrated[..., 9] = torch.where(valid, closure(batch["action"][..., 9], 1.0, -1.0), 0.0)
    expected = normalize(calibrated, stats["action"])
    expected[..., 9] = calibrated[..., 9]
    torch.testing.assert_close(out["action"], expected)
    torch.testing.assert_close(out["action_gripper_binary"], calibrated[..., 9:10])
    torch.testing.assert_close(
        out["observation.state"], normalize(batch["observation.state"], stats["observation.state"])
    )
    assert out["action_valid_dim_mask"].tolist() == [[True] * 10 + [False] * 10] * 5
    assert out["state_valid_dim_mask"].shape == (5, 20)
    assert out["dataset_index"].tolist() == [0] * 5 and out["task"] == batch["task"]
    torch.testing.assert_close(out[HEAD], batch[HEAD])
    assert out["action_is_pad"][3].any() and set(out["action_gripper_binary"].unique().tolist()) <= {0.0, 1.0}


def test_postprocessor_inverts_to_native_calvin_actions(calvin_setup):
    config, mixture, batch = calvin_setup
    pre, post = make_concertina_pre_post_processors(config, mixture.meta.stats)
    normalized = pre(dict(batch))["action"]
    native = post(normalized)
    raw = default_collate(mixture.children[0].__getitems__(batch["index"].tolist()))["action.absolute"]
    valid = ~batch["action_is_pad"]
    torch.testing.assert_close(native[..., :3][valid], raw[..., :3][valid], atol=1e-5, rtol=0)
    torch.testing.assert_close(native[..., 6][valid], raw[..., 6][valid])
    canonical = mixture.converters[0].adapter.action({"action.absolute": native})
    torch.testing.assert_close(
        canonical[..., 3:9][valid], batch["action"][..., 3:9][valid], atol=1e-4, rtol=0
    )


def test_binding_converts_raw_environment_observations(calvin_setup):
    config, mixture, batch = calvin_setup
    pre, _ = make_concertina_pre_post_processors(config, mixture.meta.stats)
    raw = default_collate(mixture.children[0].__getitems__(batch["index"].tolist()))
    environment = {
        "observation.state": raw["observation.state"][:, 0],
        "observation.images.top": raw["observation.images.top"],
        "observation.images.wrist": raw["observation.images.wrist"],
        "task": raw["task"],
    }
    from_environment = pre(environment)
    from_dataset = pre(dict(batch))
    assert from_environment["dataset_index"].tolist() == [0] * 5
    torch.testing.assert_close(from_environment["observation.state"], from_dataset["observation.state"][:, 0])
    torch.testing.assert_close(from_environment[HEAD], from_dataset[HEAD])
    torch.testing.assert_close(from_environment[WRIST_LEFT], from_dataset[WRIST_LEFT])
    assert not from_environment[f"{HEAD}_is_missing"].any()
    assert "observation.images.top" not in from_environment
    torch.testing.assert_close(
        from_environment["action_valid_dim_mask"], from_dataset["action_valid_dim_mask"]
    )
    as_uint8 = {
        **environment,
        "observation.images.top": (environment["observation.images.top"] * 255).round().byte(),
    }
    torch.testing.assert_close(pre(as_uint8)[HEAD], from_dataset[HEAD], atol=1e-6, rtol=0)


def test_pipeline_without_gripper_channels_has_no_gripper_objective(dataset_dir):
    """60k pretraining regressed the gripper like any other dimension: no calibration, target or tokens."""
    config, mixture = setup(
        dataset_dir,
        [CALVIN],
        cameras=[HEAD, WRIST_LEFT],
        gripper_indices=[],
        normalization="global",
    )
    batch = default_collate(mixture.__getitems__([0, 5, 20]))
    pre, post = make_concertina_pre_post_processors(config, mixture.meta.stats)
    names = [type(step).__name__ for step in (*pre.steps, *post.steps)]
    assert not [name for name in names if "Gripper" in name]
    out = pre(dict(batch))
    assert "action_gripper_binary" not in out
    torch.testing.assert_close(out["action"], normalize(batch["action"], mixture.meta.stats["action"]))
    native = post(out["action"])
    torch.testing.assert_close(native[..., 6], torch.where(batch["action"][..., 9] > 0, 1.0, -1.0))


@pytest.fixture(scope="module")
def mixed_setup(dataset_dir):
    config, mixture = setup(
        dataset_dir,
        [LIBERO, AGIBOT],
        cameras=[HEAD, WRIST_LEFT, WRIST_RIGHT],
        gripper_indices=[9, 19],
        normalization="per_embodiment",
        resize_images=True,
        inference_dataset_index=1,
    )
    offset = mixture.dataset_offsets[1]
    batch = default_collate(mixture.__getitems__([0, offset + 2, 13, offset + 9, offset + 22]))
    return config, mixture, batch


def test_per_embodiment_pipeline_uses_each_samples_statistics(mixed_setup):
    config, mixture, batch = mixed_setup
    pre, _ = make_concertina_pre_post_processors(config, mixture.meta.stats)
    assert not any(isinstance(step, NormalizerProcessorStep) for step in pre.steps)
    out = pre(dict(batch))
    valid = ~batch["action_is_pad"]
    for row, dataset_index in enumerate(batch["dataset_index"].tolist()):
        embodiment, source = ("libero", (-1.0, 1.0)) if dataset_index == 0 else ("agibot", (0.0, 1.0))
        stats = {
            f: mixture.meta.stats[embodiment_stats_key(embodiment, f)]
            for f in ("action", "observation.state")
        }
        calibrated = batch["action"][row].clone()
        for index in (9, 19):
            dim_valid = index < (10 if dataset_index == 0 else 20)
            mapped = closure(batch["action"][row][:, index], *source)
            calibrated[:, index] = torch.where(valid[row] & dim_valid, mapped, 0.0)
        expected = normalize(calibrated, stats["action"])
        expected[:, [9, 19]] = calibrated[:, [9, 19]]
        torch.testing.assert_close(out["action"][row], expected)
        torch.testing.assert_close(out["action_gripper_binary"][row], calibrated[:, [9, 19]])
        torch.testing.assert_close(
            out["observation.state"][row],
            normalize(batch["observation.state"][row], stats["observation.state"]),
        )
        assert out["action_valid_dim_mask"][row].sum().item() == (10 if dataset_index == 0 else 20)


def test_per_embodiment_postprocessor_binds_the_inference_dataset(mixed_setup):
    config, mixture, batch = mixed_setup
    pre, post = make_concertina_pre_post_processors(config, mixture.meta.stats)
    bound_rows = batch["dataset_index"] == 1
    restored = post(pre(dict(batch))["action"][bound_rows])
    # Without a native action conversion for this source, the postprocessor returns its canonical action.
    valid = ~batch["action_is_pad"][bound_rows]
    torch.testing.assert_close(restored[valid], batch["action"][bound_rows][valid], atol=1e-5, rtol=0)


def test_saved_pipelines_reload_for_resume_and_evaluation(mixed_setup, tmp_path):
    config, mixture, batch = mixed_setup
    pre, post = make_concertina_pre_post_processors(config, mixture.meta.stats)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded_pre, loaded_post = make_concertina_pre_post_processors_from_pretrained(
        config,
        tmp_path,
        revision=None,
        dataset_stats=None,
        dataset_meta=None,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "normalizer_processor": {"norm_map": config.normalization_mapping},
        },
        postprocessor_overrides={"unnormalizer_processor": {}},
        preprocessor_config_filename="policy_preprocessor.json",
        postprocessor_config_filename="policy_postprocessor.json",
    )
    expected, out = pre(dict(batch)), loaded_pre(dict(batch))
    for key in ("action", "observation.state", "action_gripper_binary", "action_valid_dim_mask", HEAD):
        torch.testing.assert_close(out[key], expected[key])
    bound = pre(dict(batch))["action"][batch["dataset_index"] == 1]
    torch.testing.assert_close(loaded_post(bound), post(bound))

    # The upstream factory resolves the same loader by naming convention.
    factory_pre, _ = make_pre_post_processors(config, pretrained_path=tmp_path)
    torch.testing.assert_close(factory_pre(dict(batch))["action"], expected["action"])


def test_fine_tuning_rebuilds_pipelines_for_the_current_data(mixed_setup, tmp_path):
    config, mixture, batch = mixed_setup
    pre, post = make_concertina_pre_post_processors(config, mixture.meta.stats)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    shifted = {
        key: {name: value + 1 if name == "mean" else value for name, value in stats.items()}
        for key, stats in mixture.meta.stats.items()
    }
    rebuilt, _ = make_concertina_pre_post_processors_from_pretrained(
        config,
        tmp_path,
        revision=None,
        dataset_stats=shifted,
        dataset_meta=None,
        preprocessor_overrides={"device_processor": {"device": "cpu"}, "normalizer_processor": {"stats": {}}},
        postprocessor_overrides=None,
        preprocessor_config_filename="policy_preprocessor.json",
        postprocessor_config_filename="policy_postprocessor.json",
    )
    original = pre(dict(batch))["observation.state"]
    torch.testing.assert_close(
        rebuilt(dict(batch))["observation.state"], original - 1 / (_stds(mixture, batch) + EPS)
    )
    assert any(isinstance(step, DeviceProcessorStep) and step.device == "cpu" for step in rebuilt.steps)


def _stds(mixture, batch):
    rows = []
    for dataset_index in batch["dataset_index"].tolist():
        embodiment = "libero" if dataset_index == 0 else "agibot"
        rows.append(mixture.meta.stats[embodiment_stats_key(embodiment, "observation.state")]["std"])
    return torch.stack(rows)[:, None]


def test_steps_reject_inconsistent_inputs(calvin_setup):
    config, mixture, batch = calvin_setup
    pre, _ = make_concertina_pre_post_processors(config, mixture.meta.stats)
    with pytest.raises(ValueError, match="dataset_index"):
        pre({**batch, "dataset_index": torch.full((5,), 3)})
    broken = dict(batch)
    broken["action"] = batch["action"].clone()
    broken["action"][0, 0, 9] = 3.0
    with pytest.raises(ValueError, match="endpoints"):
        pre(broken)
    with pytest.raises(ValueError, match="statistics"):
        make_concertina_pre_post_processors(config, None)
    with pytest.raises(ValueError, match="20D canonical"):
        pre({**batch, "action": batch["action"][..., :19]})


def test_embodiment_normalization_state_round_trip():
    step = EmbodimentNormalizeStep(dataset_embodiments=["a", "b"], preserve_gripper_indices=[9])
    stats = {
        name: {
            feature: {"mean": torch.full((20,), i + 1.0), "std": torch.full((20,), 2.0)}
            for feature in ("action", "observation.state")
        }
        for i, name in enumerate(["a", "b"])
    }
    step.set_stats(stats)
    restored = EmbodimentNormalizeStep(**step.get_config())
    restored.load_state_dict(step.state_dict())
    transition = {
        "observation": {"observation.state": torch.ones(2, 1, 20)},
        "action": torch.ones(2, 3, 20),
        "complementary_data": {"dataset_index": torch.tensor([0, 1])},
    }
    out = restored(transition)
    torch.testing.assert_close(out["action"][0, 0, 0], torch.tensor(0.0))
    torch.testing.assert_close(out["action"][1, 0, 0], torch.tensor(-0.5))
    torch.testing.assert_close(
        out["complementary_data"]["action_gripper_raw_before_normalization"], torch.ones(2, 3, 1)
    )
    inverse = EmbodimentNormalizeStep(**{**step.get_config(), "inverse": True})
    inverse.set_stats(stats)
    back = inverse({**transition, "action": out["action"]})["action"]
    torch.testing.assert_close(back[..., :9], torch.ones(2, 3, 9))
    torch.testing.assert_close(back[..., 9], out["action"][..., 9])


def test_restore_step_keeps_constant_channels():
    step = GripperProbabilityRestoreStep(gripper_indices=[1])
    step.set_action_stats({"mean": torch.tensor([0.0, 0.3]), "std": torch.tensor([1.0, 0.0])})
    out = step({"action": torch.tensor([[5.0, 7.0]])})
    torch.testing.assert_close(out["action"], torch.tensor([[5.0, 0.3]]))


def test_masks_and_binding_for_unbatched_policy_actions():
    masks = ValidDimMaskStep(valid_action_dims=[10, 20])
    out = masks({"complementary_data": {"dataset_index": torch.tensor([1, 0])}, "observation": None})
    assert out["complementary_data"]["action_valid_dim_mask"].sum(-1).tolist() == [20, 10]
    binding = SourceBindingStep(dataset_index=1, source="agibot")
    bound = binding({"action": torch.zeros(3, 20), "complementary_data": {}, "observation": None})
    assert bound["complementary_data"]["dataset_index"].tolist() == [1, 1, 1]


def test_binding_accepts_cameras_already_named_like_the_model():
    binding = SourceBindingStep(
        source="libero", camera_keys=[HEAD], camera_map={HEAD: ["observation.images.front"]}
    )
    image = torch.rand(1, 3, 8, 8)
    observation = {HEAD: image, "observation.state": torch.zeros(1, 8)}
    bound = binding({"observation": observation, "complementary_data": {}})["observation"]
    assert torch.equal(bound[HEAD], image) and not bound[f"{HEAD}_is_missing"].any()


def test_unnormalized_pipeline_only_calibrates(dataset_dir):
    config, mixture = setup(
        dataset_dir,
        [LIBERO],
        cameras=[HEAD, WRIST_LEFT],
        gripper_indices=[9],
        normalization="none",
    )
    batch = default_collate(mixture.__getitems__([0, 13, 30]))
    pre, post = make_concertina_pre_post_processors(config, None)
    out = pre(dict(batch))
    valid = ~batch["action_is_pad"]
    expected = batch["action"].clone()
    expected[..., 9] = torch.where(valid, closure(batch["action"][..., 9], -1.0, 1.0), 0.0)
    torch.testing.assert_close(out["action"], expected)
    torch.testing.assert_close(out["observation.state"], batch["observation.state"])
    native = post(out["action"])
    torch.testing.assert_close(native[..., :3][valid], batch["action"][..., :3][valid])


def test_model_batch_keys_survive_preprocessing(mixed_setup):
    config, mixture, batch = mixed_setup
    pre, _ = make_concertina_pre_post_processors(config, mixture.meta.stats)
    out = pre(dict(batch))
    required = {
        "index",
        "dataset_index",
        "task",
        "action",
        "action_is_pad",
        "action_gripper_binary",
        "action_valid_dim_mask",
        "state_valid_dim_mask",
        "observation.state",
        *(camera for camera in (HEAD, WRIST_LEFT, WRIST_RIGHT)),
        *(f"{camera}_is_missing" for camera in (HEAD, WRIST_LEFT, WRIST_RIGHT)),
    }
    assert required <= set(out), sorted(required - set(out))
    assert out["action_gripper_binary"].shape == (5, CHUNK, 2)
    assert "embodiment_id" not in out and not [key for key in out if key.startswith("rng_")]
    torch.testing.assert_close(out["index"], batch["index"])
    torch.testing.assert_close(out["dataset_index"], batch["dataset_index"])


def test_single_arm_datasets_get_black_placeholders_for_missing_cameras(dataset_dir):
    """Pretraining DROID has no right wrist camera: every model camera key is still in the batch."""
    droid = DatasetSpec(
        path="droid",
        source="droid",
        embodiment="droid_single",
        valid_action_dim=10,
        camera_map={
            HEAD: ["observation.images.exterior_1_left"],
            WRIST_LEFT: ["observation.images.wrist_left"],
        },
    )
    config, mixture = setup(
        dataset_dir,
        [droid],
        cameras=[HEAD, WRIST_LEFT, WRIST_RIGHT],
        gripper_indices=[9, 19],
        normalization="per_embodiment",
        resize_images=True,
    )
    batch = default_collate(mixture.__getitems__([0, 7, 20]))
    out = make_concertina_pre_post_processors(config, mixture.meta.stats)[0](dict(batch))
    assert out[WRIST_RIGHT].shape == (3, 3, 24, 32) and torch.count_nonzero(out[WRIST_RIGHT]) == 0
    assert out[f"{WRIST_RIGHT}_is_missing"].all()
    assert not out[f"{HEAD}_is_missing"].any() and not out[f"{WRIST_LEFT}_is_missing"].any()
    assert out["action_valid_dim_mask"].sum(-1).tolist() == [10, 10, 10]
    torch.testing.assert_close(out["action_gripper_binary"][..., 1], torch.zeros(3, CHUNK))


def test_binary_gripper_targets_skip_calibration(dataset_dir):
    """The legacy objective: BCE on the source gripper value thresholded at 0.5, no closure calibration."""
    config, mixture = setup(
        dataset_dir,
        [CALVIN],
        cameras=[HEAD, WRIST_LEFT],
        gripper_indices=[9],
        gripper_target="binary",
        normalization="global",
    )
    batch = default_collate(mixture.__getitems__([0, 5, 20]))
    pre, post = make_concertina_pre_post_processors(config, mixture.meta.stats)
    assert not [step for step in (*pre.steps, *post.steps) if "Calibration" in type(step).__name__]
    out = pre(dict(batch))
    target = (batch["action"][..., 9:10] > 0.5).float()
    torch.testing.assert_close(out["action_gripper_binary"], target)
    expected = normalize(batch["action"], mixture.meta.stats["action"])
    expected[..., 9:10] = target
    torch.testing.assert_close(out["action"], expected)
    native = post(out["action"])
    torch.testing.assert_close(native[..., 6], torch.where(target[..., 0] > 0.5, 1.0, -1.0))


def test_action_history_windows_are_processed_like_the_chunk(dataset_dir):
    config, mixture = setup(
        dataset_dir, [CALVIN], cameras=[HEAD, WRIST_LEFT], gripper_indices=[9], normalization="global"
    )
    history = ConcertinaMixtureDataset(
        [CALVIN],
        root=dataset_dir,
        camera_keys=[HEAD, WRIST_LEFT],
        action_delta_indices=list(range(-4, CHUNK)),
        observation_delta_indices=[0],
        image_hw=(24, 32),
        normalization="global",
    )
    batch = default_collate(history.__getitems__([0, 5, 20]))
    out = make_concertina_pre_post_processors(config, history.meta.stats)[0](dict(batch))
    window = 4 + CHUNK
    assert out["action"].shape == (3, window, 20) and out["action_is_pad"].shape == (3, window)
    assert out["action_gripper_binary"].shape == (3, window, 1)
    valid = ~batch["action_is_pad"]
    calibrated = torch.where(valid, closure(batch["action"][..., 9], 1.0, -1.0), 0.0)
    torch.testing.assert_close(out["action_gripper_binary"][..., 0], calibrated)
    expected = normalize(batch["action"], history.meta.stats["action"])
    torch.testing.assert_close(out["action"][..., :9], expected[..., :9])
    assert batch["action_is_pad"][0, :4].all()
