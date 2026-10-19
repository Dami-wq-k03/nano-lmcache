"""In-memory key-value cache, standing in for a real KV-tensor store."""


class MemoryKVCache:
    """Dict-backed cache keyed by chunk key, storing arbitrary payloads.

    Payloads are opaque `object` values here on purpose: this stage models
    the storage/lookup contract (put/get by key) without binding to any
    particular KV-tensor representation (e.g. PyTorch tensors).
    """

    def __init__(self) -> None:
        self._store: dict[str, object] = {}

    def put(self, key: str, value: object) -> None:
        self._store[key] = value

    def get(self, key: str) -> object | None:
        return self._store.get(key)
