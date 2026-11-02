"""Stateless, sample-keyed random numbers for training.

Every random draw of a training sample comes from a CPU generator seeded by
`(seed, step, dataset_index, sample_index, stream)`. A sample's augmentation and corruption are therefore
independent of batch composition, rank layout and of which other random draws are enabled.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

_UINT64_MASK = (1 << 64) - 1
_TORCH_SEED_MASK = (1 << 63) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def _stable_stream_id(stream: str) -> int:
    # FNV-1a: unlike `hash`, stable across processes.
    value = 0xCBF29CE484222325
    for byte in stream.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & _UINT64_MASK
    return value


@dataclass(frozen=True)
class SampleKeyedRNG:
    sample_ids: tuple[int, ...]
    dataset_ids: tuple[int, ...]
    seed: int
    step: int

    @classmethod
    def from_values(
        cls,
        sample_ids: Tensor | Sequence[int],
        *,
        dataset_ids: Tensor | Sequence[int] | None,
        seed: int,
        step: int,
    ) -> "SampleKeyedRNG":
        sample_ids = _as_int_tuple(sample_ids)
        dataset_ids = (0,) * len(sample_ids) if dataset_ids is None else _as_int_tuple(dataset_ids)
        if len(dataset_ids) != len(sample_ids):
            raise ValueError(
                f"Expected one dataset index per sample, got {len(dataset_ids)} for {len(sample_ids)} samples"
            )
        return cls(sample_ids=sample_ids, dataset_ids=dataset_ids, seed=int(seed), step=int(step))

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)

    def _generator(self, sample_index: int, stream: str) -> torch.Generator:
        value = _splitmix64(self.seed & _UINT64_MASK)
        value = _splitmix64(value ^ (self.step & _UINT64_MASK))
        value = _splitmix64(value ^ (self.dataset_ids[sample_index] & _UINT64_MASK))
        value = _splitmix64(value ^ (self.sample_ids[sample_index] & _UINT64_MASK))
        value = _splitmix64(value ^ _stable_stream_id(stream))
        value = _splitmix64(value)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(value & _TORCH_SEED_MASK)
        return generator

    def rand(
        self,
        tail_shape: Sequence[int] = (),
        *,
        stream: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Uniform [0, 1) draws of shape `(batch_size, *tail_shape)`."""
        return self._draw(torch.rand, tail_shape, stream=stream, device=device, dtype=dtype)

    def randn(
        self,
        tail_shape: Sequence[int] = (),
        *,
        stream: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Standard-normal draws of shape `(batch_size, *tail_shape)`, generated in `dtype`."""
        return self._draw(torch.randn, tail_shape, stream=stream, device=device, dtype=dtype)

    def randperm(self, sample_index: int, size: int, *, stream: str, device: torch.device | str) -> Tensor:
        """A random permutation of `range(size)` for one sample."""
        return torch.randperm(int(size), generator=self._generator(sample_index, stream)).to(device=device)

    def _draw(self, sampler, tail_shape: Sequence[int], *, stream: str, device, dtype: torch.dtype) -> Tensor:
        tail_shape = tuple(int(size) for size in tail_shape)
        rows = [
            sampler(tail_shape, generator=self._generator(index, stream), dtype=dtype)
            for index in range(self.batch_size)
        ]
        return torch.stack(rows, dim=0).to(device=device)


def _as_int_tuple(values: Tensor | Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, Tensor):
        values = values.detach().cpu().reshape(-1).tolist()
    return tuple(int(value) for value in values)
