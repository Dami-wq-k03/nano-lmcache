"""A small JSON-framed shared cache server for multi-process experiments.

The wire format carries serializer bytes as base64 inside length-prefixed JSON
frames. It is deliberately boring: no pickle, no executable payloads, and a
bounded frame size. The server reuses the existing prefix-safe in-memory
backend, so parent admission and eviction semantics remain unchanged.
"""

from __future__ import annotations

import base64
import json
import socket
import socketserver
import struct
import threading
from collections.abc import Sequence
from typing import Any

from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.serializers import JsonSerializer, Serializer

_DEFAULT_MAX_FRAME = 64 * 1024 * 1024


class RemoteCacheError(RuntimeError):
    """A remote cache request failed explicitly."""


class RemoteCacheProtocolError(RemoteCacheError):
    """A peer sent a malformed or oversized frame."""


def _send_frame(
    stream: socket.socket, message: dict[str, Any], *, max_frame_size: int
) -> None:
    data = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if len(data) > max_frame_size:
        raise RemoteCacheProtocolError("response exceeds maximum frame size")
    stream.sendall(struct.pack("!I", len(data)) + data)


def _receive_exact(stream: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive_frame(stream: socket.socket, *, max_frame_size: int) -> dict[str, Any]:
    header = _receive_exact(stream, 4)
    (size,) = struct.unpack("!I", header)
    if size > max_frame_size:
        raise RemoteCacheProtocolError("request exceeds maximum frame size")
    try:
        message = json.loads(_receive_exact(stream, size))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteCacheProtocolError("request is not valid JSON") from error
    if not isinstance(message, dict):
        raise RemoteCacheProtocolError("request must be a JSON object")
    return message


def _decode_payload(value: object) -> bytes:
    if not isinstance(value, str):
        raise RemoteCacheProtocolError("payload must be base64 text")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeError) as error:
        raise RemoteCacheProtocolError("payload is not valid base64") from error


def _encode_payload(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    cache_server: RemoteCacheServer


class _RequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        assert isinstance(server, _ThreadingTCPServer)
        cache_server = server.cache_server
        while True:
            try:
                request = _receive_frame(
                    self.request, max_frame_size=cache_server.max_frame_size
                )
            except EOFError:
                return
            except RemoteCacheProtocolError as error:
                _send_frame(
                    self.request,
                    {"ok": False, "error": str(error)},
                    max_frame_size=cache_server.max_frame_size,
                )
                return
            try:
                response = cache_server.handle(request)
            except (KeyError, TypeError, ValueError, RemoteCacheError) as error:
                response = {"ok": False, "error": str(error)}
            except Exception:
                response = {"ok": False, "error": "internal server error"}
            _send_frame(
                self.request,
                response,
                max_frame_size=cache_server.max_frame_size,
            )


class RemoteCacheServer:
    """Threaded TCP service exposing a prefix-safe byte cache."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        storage: MemoryKVCache | None = None,
        max_entries: int | None = None,
        max_bytes: int | None = None,
        max_frame_size: int = _DEFAULT_MAX_FRAME,
    ) -> None:
        if max_frame_size <= 0:
            raise ValueError("max_frame_size must be positive")
        self.max_frame_size = max_frame_size
        self.storage = storage or MemoryKVCache(
            max_entries,
            max_bytes=max_bytes,
            size_fn=lambda value: len(value) if isinstance(value, bytes) else 0,
        )
        self._server = _ThreadingTCPServer((host, port), _RequestHandler)
        self._server.cache_server = self
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> tuple[str, int]:
        """Start serving in a daemon thread and return the bound address."""
        if self._thread is not None:
            raise RuntimeError("server is already running")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="nano-lmcache-server",
            daemon=True,
        )
        self._thread.start()
        return self.address

    def close(self) -> None:
        """Stop the server and release its listening socket."""
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._server.server_close()

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        if operation == "get":
            key = self._key(request)
            value = self.storage.get(key)
            if value is not None and not isinstance(value, bytes):
                raise RemoteCacheError("server storage contains non-byte payload")
            return {
                "ok": True,
                "payload": None if value is None else _encode_payload(value),
            }
        if operation == "contains":
            return {"ok": True, "present": self._key(request) in self.storage}
        if operation == "put":
            key = self._key(request)
            payload = _decode_payload(request.get("payload"))
            parent = request.get("parent_key")
            if parent is not None and not isinstance(parent, str):
                raise RemoteCacheProtocolError("parent_key must be text or null")
            admitted = self.storage.try_put(key, payload, parent_key=parent)
            return {"ok": True, "admitted": admitted}
        if operation == "get_many":
            keys = request.get("keys")
            if not isinstance(keys, list) or not all(
                isinstance(key, str) for key in keys
            ):
                raise RemoteCacheProtocolError("keys must be a list of strings")
            payloads = []
            for key in keys:
                value = self.storage.get(key)
                if value is not None and not isinstance(value, bytes):
                    raise RemoteCacheError(
                        "server storage contains non-byte payload"
                    )
                payloads.append(
                    None if value is None else _encode_payload(value)
                )
            return {"ok": True, "payloads": payloads}
        raise RemoteCacheProtocolError(f"unknown operation {operation!r}")

    @staticmethod
    def _key(request: dict[str, Any]) -> str:
        key = request.get("key")
        if not isinstance(key, str) or not key:
            raise RemoteCacheProtocolError("key must be non-empty text")
        return key


class RemoteCacheClient:
    """`CacheStorage` client with serializer bytes at the network boundary."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        serializer: Serializer | None = None,
        timeout: float = 5.0,
        max_frame_size: int = _DEFAULT_MAX_FRAME,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_frame_size <= 0:
            raise ValueError("max_frame_size must be positive")
        self.host = host
        self.port = port
        self.serializer = serializer or JsonSerializer()
        self.timeout = timeout
        self.max_frame_size = max_frame_size

    def put(
        self, key: str, value: object, *, parent_key: str | None = None
    ) -> None:
        if value is None:
            raise ValueError("None is reserved for a cache miss")
        payload = self.serializer.dumps(value)
        response = self._request(
            {
                "op": "put",
                "key": key,
                "parent_key": parent_key,
                "payload": _encode_payload(payload),
            }
        )
        if not response.get("admitted", False):
            raise RemoteCacheError(f"remote cache rejected key {key!r}")

    def get(self, key: str) -> object | None:
        try:
            response = self._request({"op": "get", "key": key})
        except (OSError, RemoteCacheError):
            return None
        encoded = response.get("payload")
        if encoded is None:
            return None
        return self.serializer.loads(_decode_payload(encoded))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            return bool(self._request({"op": "contains", "key": key}).get("present"))
        except (OSError, RemoteCacheError):
            return False

    def get_many(self, keys: Sequence[str]) -> list[object | None]:
        """Fetch many keys in one round trip, preserving key order."""
        response = self._request({"op": "get_many", "keys": list(keys)})
        payloads = response.get("payloads")
        if not isinstance(payloads, list) or len(payloads) != len(keys):
            raise RemoteCacheProtocolError("server returned an invalid payload list")
        return [
            None if payload is None else self.serializer.loads(_decode_payload(payload))
            for payload in payloads
        ]

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        with socket.create_connection((self.host, self.port), self.timeout) as stream:
            stream.settimeout(self.timeout)
            _send_frame(stream, request, max_frame_size=self.max_frame_size)
            response = _receive_frame(stream, max_frame_size=self.max_frame_size)
        if response.get("ok") is not True:
            raise RemoteCacheError(str(response.get("error", "remote request failed")))
        return response
