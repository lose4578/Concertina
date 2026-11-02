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
# Portions adapted from LeRobot (lerobot/policies/xvla/utils.py, `mat2quat`, `quat2axisangle`,
# `rotate6d_to_axis_angle`, `mat_to_rotate6d`; derived from X-VLA), Copyright 2024 The HuggingFace Inc. team,
# Apache License 2.0. `mat2quat` and `quat2axisangle` originate from robosuite (robosuite/utils/transform_utils.py),
# Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab, MIT License.
# Modified: batched NumPy/PyTorch wrappers around the per-element conversions, and the inverse conversion to
# Euler angles. See NOTICE.
"""Rotation conversions to and from the continuous 6D representation (first two rotation-matrix columns).

The per-element loops mirror the conversions the released checkpoints were trained with, so converted values are
bit-identical to them; quaternions are ``(x, y, z, w)``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from lerobot.utils.rotation import Rotation
from scipy.spatial.transform import Rotation as ScipyRotation


def matrix_to_rotate6d(matrix: np.ndarray) -> np.ndarray:
    """``[..., 3, 3]`` rotation matrices -> ``[..., 6]`` (first column, then second column)."""
    return np.concatenate([matrix[..., :3, 0], matrix[..., :3, 1]], axis=-1)


def _per_vector(values: np.ndarray, width: int, fn) -> np.ndarray:
    flat = values.reshape(-1, values.shape[-1])
    out = np.stack([fn(row) for row in flat]) if len(flat) else np.empty((0, width))
    return out.reshape(values.shape[:-1] + (width,))


def euler_to_rotate6d_np(euler: np.ndarray) -> np.ndarray:
    """Extrinsic ``xyz`` Euler angles ``[..., 3]`` -> ``[..., 6]``."""
    return _per_vector(euler, 6, lambda e: matrix_to_rotate6d(ScipyRotation.from_euler("xyz", e).as_matrix()))


def axis_angle_to_rotate6d_np(axis_angle: np.ndarray) -> np.ndarray:
    """Rotation vectors ``[..., 3]`` -> ``[..., 6]``."""
    return _per_vector(axis_angle, 6, lambda v: matrix_to_rotate6d(Rotation.from_rotvec(v).as_matrix()))


def quat_to_rotate6d_np(quat: np.ndarray) -> np.ndarray:
    """Quaternions ``[..., 4]`` (normalized internally) -> ``[..., 6]``."""
    return _per_vector(quat, 6, lambda q: matrix_to_rotate6d(Rotation.from_quat(q).as_matrix()))


def euler_to_rotate6d(euler: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(euler_to_rotate6d_np(euler.detach().cpu().numpy())).to(euler.device, euler.dtype)


def axis_angle_to_rotate6d(axis_angle: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(axis_angle_to_rotate6d_np(axis_angle.detach().cpu().numpy())).to(
        axis_angle.device, axis_angle.dtype
    )


def quat_to_rotate6d(quat: torch.Tensor) -> torch.Tensor:
    """Quaternions ``[..., 4]`` -> ``[..., 6]`` with torch ops (normalizes the quaternion first)."""
    if quat.shape[-1] != 4:
        raise ValueError(f"quaternion last dimension must be 4, got shape {tuple(quat.shape)}")
    q = quat.to(dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    col0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy + zw), 2.0 * (xz - yw)], dim=-1)
    col1 = torch.stack([2.0 * (xy - zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + xw)], dim=-1)
    return torch.cat([col0, col1], dim=-1).to(dtype=quat.dtype)


def normalize_quaternion(quat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unit quaternions; near-zero inputs become the identity ``(0, 0, 0, 1)``."""
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    identity = torch.zeros_like(quat)
    identity[..., -1] = 1.0
    return torch.where(norm > eps, quat / norm.clamp_min(eps), identity)


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> ``(x, y, z, w)`` via the eigenvector of the symmetric K matrix (float32)."""
    m = np.asarray(matrix).astype(np.float32)[:3, :3]
    m00, m01, m02 = m[0, 0], m[0, 1], m[0, 2]
    m10, m11, m12 = m[1, 0], m[1, 1], m[1, 2]
    m20, m21, m22 = m[2, 0], m[2, 1], m[2, 2]
    zero = np.float32(0.0)
    k = np.array(
        [
            [m00 - m11 - m22, zero, zero, zero],
            [m01 + m10, m11 - m00 - m22, zero, zero],
            [m02 + m20, m12 + m21, m22 - m00 - m11, zero],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    k /= 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(k)
    q = eigenvectors[np.array([3, 0, 1, 2]), np.argmax(eigenvalues)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q[np.array([1, 2, 3, 0])]


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def rotate6d_to_axis_angle_np(rot6d: np.ndarray) -> np.ndarray:
    """``[..., 6]`` -> rotation vectors ``[..., 3]`` (Gram-Schmidt, then matrix -> quaternion -> axis-angle)."""
    flat = rot6d.reshape(-1, 6)
    a1, a2 = flat[:, 0:3], flat[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-6)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-6)
    b3 = np.cross(b1, b2, axis=-1)
    matrices = np.stack([b1, b2, b3], axis=-1)
    out = np.stack([_quat_to_axis_angle(_matrix_to_quat(matrix)) for matrix in matrices], axis=0)
    return out.reshape(rot6d.shape[:-1] + (3,))


def rotate6d_to_axis_angle(rot6d: torch.Tensor) -> torch.Tensor:
    out = rotate6d_to_axis_angle_np(rot6d.detach().float().cpu().numpy())
    return torch.from_numpy(out).to(rot6d.device, rot6d.dtype)


def rotate6d_to_euler(rot6d: torch.Tensor) -> torch.Tensor:
    """``[..., 6]`` -> extrinsic ``xyz`` Euler angles ``[..., 3]``."""
    axis_angle = rotate6d_to_axis_angle_np(rot6d.detach().float().cpu().numpy())
    out = _per_vector(axis_angle, 3, lambda v: ScipyRotation.from_rotvec(v).as_euler("xyz"))
    return torch.from_numpy(out).to(rot6d.device, rot6d.dtype)
