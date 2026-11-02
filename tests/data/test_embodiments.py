import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from concertina.data.embodiments import (
    AgiBotAdapter,
    CalvinAdapter,
    DroidAdapter,
    LiberoAdapter,
    RoboMindAgilexAdapter,
    RoboMindFr3DualAdapter,
    RoboMindSingleArmAdapter,
    SampleConverter,
    interpolate_padding,
    interpolate_sequence,
    make_source_adapter,
    merge_padding,
)
from concertina.data.rotation import axis_angle_to_rotate6d, euler_to_rotate6d, quat_to_rotate6d

T = 5
RNG = np.random.default_rng(0)


def _tensor(*shape, scale=1.0):
    return torch.tensor(RNG.normal(0, scale, shape), dtype=torch.float32)


def _quats(*shape):
    quats = Rotation.random(int(np.prod(shape)), random_state=1).as_quat().reshape(*shape, 4)
    return torch.tensor(quats, dtype=torch.float32)


def _assert_single_arm(canonical, pos, rot6d, gripper):
    torch.testing.assert_close(canonical[..., :3], pos)
    torch.testing.assert_close(canonical[..., 3:9], rot6d)
    torch.testing.assert_close(canonical[..., 9:10], gripper)
    assert torch.count_nonzero(canonical[..., 10:]) == 0


def test_libero_adapter_keeps_the_first_finger_joint():
    action, state = _tensor(T, 7), _tensor(1, 8)
    adapter = LiberoAdapter()
    _assert_single_arm(
        adapter.action({"action": action}),
        action[:, :3],
        axis_angle_to_rotate6d(action[:, 3:6]),
        action[:, 6:7],
    )
    _assert_single_arm(
        adapter.state({"observation.state": state}),
        state[:, :3],
        axis_angle_to_rotate6d(state[:, 3:6]),
        state[:, 6:7],
    )
    with pytest.raises(ValueError, match="LIBERO action"):
        adapter.action({"action": _tensor(T, 8)})


def test_calvin_adapter_uses_the_last_gripper_action_of_the_state():
    action = _tensor(T, 7)
    state = _tensor(1, 15)
    adapter = CalvinAdapter("action.absolute")
    canonical = adapter.action({"action.absolute": action})
    _assert_single_arm(canonical, action[:, :3], euler_to_rotate6d(action[:, 3:6]), action[:, 6:7])
    _assert_single_arm(
        adapter.state({"observation.state": state}),
        state[:, :3],
        euler_to_rotate6d(state[:, 3:6]),
        state[:, 14:15],
    )
    assert make_source_adapter("calvin", None, {"action": {}}).action_keys == ("action",)
    assert make_source_adapter("calvin", None, {"action": {}, "action.absolute": {}}).action_keys == (
        "action.absolute",
    )


def test_droid_and_robomind_single_arm_adapters():
    cartesian, gripper = _tensor(T, 6), torch.rand(T)
    canonical = DroidAdapter().action(
        {"action.cartesian_position": cartesian, "action.gripper_position": gripper}
    )
    _assert_single_arm(canonical, cartesian[:, :3], euler_to_rotate6d(cartesian[:, 3:6]), gripper[:, None])
    pose, grip = _tensor(T, 6), torch.rand(T, 1)
    canonical = RoboMindSingleArmAdapter().action({"actions.end_effector": pose, "actions.gripper": grip})
    _assert_single_arm(canonical, pose[:, :3], euler_to_rotate6d(pose[:, 3:6]), grip)


def test_dual_arm_adapters_concatenate_left_then_right():
    position, orientation, effector = _tensor(T, 2, 3), _quats(T, 2), torch.rand(T, 2)
    canonical = AgiBotAdapter().action(
        {
            "actions.end.position": position,
            "actions.end.orientation": orientation,
            "actions.effector.position": effector,
        }
    )
    assert canonical.shape == (T, 20)
    for arm in range(2):
        block = canonical[:, 10 * arm : 10 * (arm + 1)]
        torch.testing.assert_close(block[:, :3], position[:, arm])
        torch.testing.assert_close(block[:, 3:9], quat_to_rotate6d(orientation[:, arm]))
        torch.testing.assert_close(block[:, 9], effector[:, arm])

    left = torch.cat([_tensor(T, 3), _quats(T)], dim=-1)
    right = torch.cat([_tensor(T, 3), _quats(T)], dim=-1)
    item = {
        "actions.end_effector_left": left,
        "actions.end_effector_right": right,
        "actions.gripper_left": torch.rand(T),
        "actions.gripper_right": torch.rand(T),
    }
    canonical = RoboMindAgilexAdapter().action(item)
    torch.testing.assert_close(canonical[:, 13:19], quat_to_rotate6d(right[:, 3:7]))
    torch.testing.assert_close(canonical[:, 19], item["actions.gripper_right"])

    pose, grip = _tensor(T, 12), torch.rand(T, 2)
    canonical = RoboMindFr3DualAdapter().action({"actions.end_effector": pose, "actions.gripper": grip})
    torch.testing.assert_close(canonical[:, 10:13], pose[:, 6:9])
    torch.testing.assert_close(canonical[:, 13:19], euler_to_rotate6d(pose[:, 9:12]))
    torch.testing.assert_close(canonical[:, 19], grip[:, 1])


@pytest.mark.parametrize("adapter_cls", [LiberoAdapter, CalvinAdapter])
def test_single_arm_source_actions_round_trip(adapter_cls):
    rotations = Rotation.random(T, random_state=4)
    if adapter_cls is LiberoAdapter:
        rotation, key = rotations.as_rotvec(), "action"
        adapter = LiberoAdapter()
    else:
        rotation, key = rotations.as_euler("xyz"), "action.absolute"
        adapter = CalvinAdapter(key)
    gripper = np.where(np.arange(T) % 2, 1.0, -1.0)[:, None]
    action = torch.tensor(
        np.concatenate([RNG.normal(0, 0.1, (T, 3)), rotation, gripper], axis=-1), dtype=torch.float32
    )
    canonical = adapter.action({key: action})
    torch.testing.assert_close(
        adapter.source_action(canonical, binarize=True, threshold=0.0), action, atol=2e-5, rtol=0
    )
    probabilities = canonical.clone()
    probabilities[:, 9] = torch.tensor([0.2, 0.7, 0.5, 0.9, 0.1])
    binarized = adapter.source_action(probabilities, binarize=True, threshold=0.5)
    torch.testing.assert_close(binarized[:, 6], torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0]))


def test_interpolation_keeps_length_and_endpoints():
    sequence = torch.arange(6, dtype=torch.float32)[:, None].repeat(1, 2)
    resampled = interpolate_sequence(sequence, source_fps=15.0, target_fps=30.0)
    torch.testing.assert_close(resampled[:, 0], torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]))
    torch.testing.assert_close(interpolate_sequence(sequence[:1], 15.0, 30.0), sequence[:1])
    # Target point k sits at source position k / 2: odd points interpolate two source frames.
    padding = torch.tensor([False, False, False, True, True, True])
    assert interpolate_padding(padding, 15.0, 30.0).tolist() == [False, False, False, False, False, True]
    padding = torch.tensor([False, False, True, True, True, True])
    assert interpolate_padding(padding, 15.0, 30.0).tolist() == [False, False, False, True, True, True]


def test_merge_padding_unions_the_source_masks():
    value = torch.zeros(4, 20)
    item = {
        "action.cartesian_position_is_pad": torch.tensor([False, False, True, True]),
        "action.gripper_position_is_pad": torch.tensor([False, True, False, False]),
    }
    merged = merge_padding(item, value, "action", DroidAdapter.action_keys)
    assert merged.tolist() == [False, True, True, True]
    with pytest.raises(ValueError, match="does not match"):
        merge_padding({"action_is_pad": torch.zeros(3, dtype=torch.bool)}, value, "action", ())


def test_sample_converter_maps_resizes_and_fills_missing_cameras():
    converter = SampleConverter(
        adapter=LiberoAdapter(),
        camera_sources={"cam.head": "src.front", "cam.wrist_right": None},
        resize_hw=(8, 6),
        placeholder_hw=(4, 4),
    )
    item = {
        "action": _tensor(T, 7),
        "observation.state": _tensor(1, 8),
        "action_is_pad": torch.tensor([False] * 3 + [True] * 2),
        "observation.state_is_pad": torch.tensor([False]),
        "src.front": torch.rand(3, 16, 12),
    }
    sample = converter(item)
    assert sample["cam.head"].shape == (3, 8, 6) and not sample["cam.head_is_missing"]
    assert sample["cam.wrist_right"].shape == (3, 8, 6) and sample["cam.wrist_right_is_missing"]
    assert torch.count_nonzero(sample["cam.wrist_right"]) == 0
    assert sample["action_is_pad"].tolist() == [False] * 3 + [True] * 2
    assert sample["observation.state"].shape == (1, 20) and sample["observation.state_is_pad"].shape == (1,)

    native = SampleConverter(LiberoAdapter(), {"cam.head": "src.front", "cam.wrist": None}, None, (4, 4))(
        item
    )
    assert native["cam.head"].shape == (3, 16, 12) and native["cam.wrist"].shape == (3, 4, 4)


def test_sample_converter_resamples_actions_between_frame_rates():
    item = {
        "action.cartesian_position": torch.arange(T, dtype=torch.float32)[:, None].repeat(1, 6) * 0.1,
        "action.gripper_position": torch.linspace(0, 1, T),
        "observation.state.cartesian_position": torch.zeros(1, 6),
        "observation.state.gripper_position": torch.zeros(1),
        "action.cartesian_position_is_pad": torch.tensor([False, False, False, True, True]),
    }
    plain = SampleConverter(DroidAdapter(), {}, None, (4, 4))(item)
    aligned = SampleConverter(DroidAdapter(), {}, None, (4, 4), source_fps=15.0, target_fps=30.0)(item)
    torch.testing.assert_close(aligned["action"], interpolate_sequence(plain["action"], 15.0, 30.0))
    torch.testing.assert_close(aligned["observation.state"], plain["observation.state"])
    assert aligned["action_is_pad"].tolist() == [False, False, False, False, False]
    same_rate = SampleConverter(DroidAdapter(), {}, None, (4, 4), source_fps=30.0, target_fps=30.0)
    assert not same_rate.resamples
