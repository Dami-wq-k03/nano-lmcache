"""How a disk-backed cache turns payloads into bytes and back.

Why the default is JSON and not `pickle`: unpickling runs code chosen by
whoever wrote the file. A cache directory is easy to share between users,
machines, or containers, and anyone who can write a file into it could
then execute code in every process that reads the cache. JSON can only
describe data, so a malicious file can at worst fail to parse.

JSON limits payloads to strings, numbers, booleans, None, lists, and dicts
with string keys (tuples come back as lists). Tensor payloads need a
tensor serializer, which lives in the optional `tensor` package.
"""

import json
from typing import Protocol, runtime_checkable


@runtime_checkable
class Serializer(Protocol):
    """Converts one payload to bytes and back."""

    def dumps(self, value: object) -> bytes:
        """Encode `value`; raise TypeError if it cannot be represented."""
        ...

    def loads(self, data: bytes) -> object:
        """Decode bytes produced by `dumps`."""
        ...


class JsonSerializer:
    """Safe default for JSON-compatible payloads."""

    def dumps(self, value: object) -> bytes:
        return json.dumps(value, separators=(",", ":")).encode("utf-8")

    def loads(self, data: bytes) -> object:
        return json.loads(data.decode("utf-8"))
