import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from concertina.data.rotation import (
    axis_angle_to_rotate6d,
    euler_to_rotate6d,
    normalize_quaternion,
    quat_to_rotate6d,
    rotate6d_to_axis_angle,
    rotate6d_to_euler,
)


def _rotations(n: int, seed: int = 0) -> Rotation:
    return Rotation.random(n, random_state=seed)


def _rotate6d(rotations: Rotation) -> np.ndarray:
    matrices = rotations.as_matrix()
    return np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=-1)


def test_euler_to_rotate6d_takes_the_first_two_matrix_columns():
    rotations = _rotations(32)
    euler = torch.tensor(rotations.as_euler("xyz"), dtype=torch.float32).reshape(4, 8, 3)
    rot6d = euler_to_rotate6d(euler)
    assert rot6d.shape == (4, 8, 6) and rot6d.dtype == torch.float32
    expected = _rotate6d(Rotation.from_euler("xyz", euler.reshape(-1, 3).double().numpy()))
    np.testing.assert_allclose(rot6d.reshape(-1, 6).numpy(), expected, atol=1e-6)


def test_axis_angle_and_quaternion_conversions_agree_with_scipy():
    rotations = _rotations(16, seed=1)
    expected = _rotate6d(rotations)
    axis_angle = torch.tensor(rotations.as_rotvec(), dtype=torch.float32)
    np.testing.assert_allclose(axis_angle_to_rotate6d(axis_angle).numpy(), expected, atol=1e-6)
    quat = torch.tensor(rotations.as_quat(), dtype=torch.float32) * 3.0  # normalized internally
    np.testing.assert_allclose(quat_to_rotate6d(quat).numpy(), expected, atol=1e-6)


def test_inverse_conversions_round_trip():
    rotations = _rotations(16, seed=2)
    rot6d = torch.tensor(_rotate6d(rotations), dtype=torch.float32)
    recovered = Rotation.from_rotvec(rotate6d_to_axis_angle(rot6d).double().numpy())
    np.testing.assert_allclose(_rotate6d(recovered), rot6d.numpy(), atol=2e-5)
    recovered = Rotation.from_euler("xyz", rotate6d_to_euler(rot6d).double().numpy())
    np.testing.assert_allclose(_rotate6d(recovered), rot6d.numpy(), atol=2e-5)


def test_inverse_conversions_keep_leading_dimensions():
    rot6d = torch.tensor(_rotate6d(_rotations(6, seed=3)), dtype=torch.float32).reshape(2, 3, 6)
    assert rotate6d_to_axis_angle(rot6d).shape == (2, 3, 3)
    assert rotate6d_to_euler(rot6d).shape == (2, 3, 3)
    assert rotate6d_to_axis_angle(rot6d[0, 0]).shape == (3,)


def test_normalize_quaternion_maps_degenerate_inputs_to_identity():
    quat = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]])
    torch.testing.assert_close(
        normalize_quaternion(quat), torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
    )


@pytest.mark.parametrize("fn", [euler_to_rotate6d, axis_angle_to_rotate6d])
def test_conversions_of_empty_batches(fn):
    assert fn(torch.zeros(0, 3)).shape == (0, 6)
