"""Bounded LRU cache for derived integer layouts (tokenized prompts, MRoPE positions).

Entries are pure functions of their keys, are never checkpointed and may be evicted at any time.
"""

from collections import OrderedDict


class LayoutCache[KeyT, ValueT]:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = capacity
        self._entries: OrderedDict[KeyT, ValueT] = OrderedDict()

    def get(self, key: KeyT) -> ValueT | None:
        value = self._entries.pop(key, None)
        if value is not None:
            self._entries[key] = value
        return value

    def put(self, key: KeyT, value: ValueT) -> None:
        self._entries.pop(key, None)
        if len(self._entries) >= self.capacity:
            self._entries.popitem(last=False)
        self._entries[key] = value

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
