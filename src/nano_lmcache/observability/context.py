"""Request-scoped trace context for storage and engine callbacks."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

_REQUEST_ID: ContextVar[str | None] = ContextVar(
    "nano_lmcache_request_id",
    default=None,
)


def current_request_id() -> str | None:
    """Return the request ID active in the current execution context."""
    return _REQUEST_ID.get()


@contextmanager
def request_context(request_id: str | None) -> Iterator[None]:
    """Temporarily associate storage and engine callbacks with a request."""
    if request_id is None:
        yield
        return
    token: Token[str | None] = _REQUEST_ID.set(request_id)
    try:
        yield
    finally:
        _REQUEST_ID.reset(token)
