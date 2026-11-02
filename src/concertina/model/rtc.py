"""Real-time chunking (RTC): the executable actions of one sample's current chunk.

Each chunk is decoded with actions of the previous chunk fixed as its prefix (see
`ConcertinaInferenceConfig.rtc_prefix_source`); the queue then skips the inference delay and that prefix and exposes
the next `execution_horizon` actions.
"""

import torch
from torch import Tensor


class RTCActionQueue:
    def __init__(self, execution_horizon: int) -> None:
        if execution_horizon <= 0:
            raise ValueError(f"execution_horizon must be > 0, got {execution_horizon}")
        self.execution_horizon = execution_horizon
        # The decoded chunk after the inference delay, and the executable actions taken from it.
        self.chunk: Tensor | None = None
        self.queue: Tensor | None = None
        # Executable actions consumed so far, and the position of the first executable action in `chunk`.
        self.index = 0
        self.start = 0

    def __len__(self) -> int:
        """Executable actions not yet taken."""
        return 0 if self.queue is None else max(len(self.queue) - self.index, 0)

    def empty(self) -> bool:
        return len(self) == 0

    def drop(self) -> None:
        """Discard the remaining executable actions; the chunk still provides the next prefix."""
        if self.queue is not None:
            self.queue = self.queue[: self.index]

    def get(self) -> Tensor:
        if self.empty():
            raise RuntimeError("The RTC action queue is empty")
        action = self.queue[self.index]
        self.index += 1
        return action.clone()

    def leftover(self) -> Tensor | None:
        """The not yet executed rest of the current chunk."""
        if self.chunk is None:
            return None
        return self.chunk[self.start + self.index :].clone()

    def executed_tail(self, length: int) -> Tensor | None:
        """The last `length` (at most) executed actions of the current chunk."""
        if self.chunk is None or self.index == 0:
            return None
        end = self.start + self.index
        return self.chunk[max(self.start, end - length) : end].clone()

    def replace(self, actions: Tensor, delay: int, prefix_length: int) -> None:
        """Install a new chunk `(T, D)`: drop the `delay` actions executed while decoding, then skip the
        `prefix_length` actions that were fixed from executed actions."""
        delay = max(0, min(delay, len(actions)))
        self.chunk = actions[delay:].clone()
        self.start = max(prefix_length, 0)
        begin = delay + self.start
        self.queue = actions[begin : begin + max(self.execution_horizon - delay, 0)].clone()
        self.index = 0


def stack_rtc_prefix(queues: list[RTCActionQueue], source: str, chunk_size: int) -> Tensor | None:
    """The fixed prefix `(B, P, D)` of the next chunk, or None for the first chunk."""
    if source == "leftover_tail":
        parts = [queue.leftover() for queue in queues]
    else:
        parts = [queue.executed_tail(queue.execution_horizon) for queue in queues]
    available = [part for part in parts if part is not None]
    if not available:
        return None
    if len(available) != len(parts):
        raise RuntimeError("RTC prefix availability differs across the batch")
    length = min(min(part.shape[0] for part in available), chunk_size)
    if length <= 0:
        return None
    if source == "leftover_tail":
        return torch.stack([part[:length] for part in available], dim=0)
    return torch.stack([part[-length:] for part in available], dim=0)
