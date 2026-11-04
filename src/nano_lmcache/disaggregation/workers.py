"""Prefill and decode as separate workers, and the colocated baseline.

    colocated:     prompt ──▶ [prefill + decode] ──▶ tokens

    disaggregated: prompt ──▶ [prefill] ──KVHandoff──▶ [decode] ──▶ tokens

Each worker owns its own model instance, as each would own its own GPU.
The disaggregated path must produce exactly the colocated tokens while the
decode worker computes only the generated positions, never the prompt.

A prefill worker may consult a `ChunkedPrefixCache` (earlier lessons) to
skip recomputing a shared prompt prefix. The cache and the handoff solve
different problems: the cache reuses KV *across requests*; the handoff
moves one request's KV *between roles*.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.disaggregation.handoff import KVHandoff
from nano_lmcache.disaggregation.toy_model import KVEntry, ToyLanguageModel
from nano_lmcache.disaggregation.transport import KVTransport


class HandoffNotFoundError(LookupError):
    """No pending handoff: never sent, cancelled, or claimed by another worker."""


@dataclass(frozen=True)
class PrefillResult:
    request_id: str
    prompt_tokens: int
    reused_tokens: int
    computed_tokens: int
    first_token: int
    transferred_bytes: int
    num_blocks: int


@dataclass(frozen=True)
class DecodeResult:
    request_id: str
    tokens: tuple[int, ...]
    imported_tokens: int
    computed_tokens: int


def generate_colocated(
    model: ToyLanguageModel, prompt_token_ids: Sequence[int], max_new_tokens: int
) -> list[int]:
    """Reference: prefill and decode in one place, no handoff."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    kv, token = model.prefill(prompt_token_ids)
    tokens = [token]
    while len(tokens) < max_new_tokens:
        token = model.decode_step(kv, token)
        tokens.append(token)
    return tokens


def _chunk_payload(chunk: object) -> list[KVEntry]:
    if not isinstance(chunk, list | tuple):
        raise TypeError(f"unexpected prefix-cache payload {type(chunk).__name__}")
    return [tuple(int(value) for value in entry) for entry in chunk]


class PrefillWorker:
    """Computes prompt KV and the first token, then sends a handoff."""

    def __init__(
        self,
        model: ToyLanguageModel,
        transport: KVTransport,
        *,
        block_size: int = 4,
        prefix_cache: ChunkedPrefixCache | None = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self._model = model
        self._transport = transport
        self._block_size = block_size
        self._prefix_cache = prefix_cache

    @property
    def model(self) -> ToyLanguageModel:
        return self._model

    def prefill(
        self, request_id: str, prompt_token_ids: Sequence[int]
    ) -> PrefillResult:
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")

        kv: list[KVEntry] = []
        matched_chunks = 0
        if self._prefix_cache is not None:
            hit = self._prefix_cache.lookup(prompt_token_ids, request_id=request_id)
            matched_chunks = hit.matched_chunks
            for chunk in hit.values:
                kv.extend(_chunk_payload(chunk))
        reused = len(kv)

        before = self._model.computed_tokens
        kv.extend(self._model.extend(kv, prompt_token_ids[reused:]))
        computed = self._model.computed_tokens - before
        first_token = self._model.next_token(kv)

        if self._prefix_cache is not None:
            size = self._prefix_cache.chunk_size
            complete = len(prompt_token_ids) // size
            new_chunks = [
                [list(entry) for entry in kv[i * size : (i + 1) * size]]
                for i in range(matched_chunks, complete)
            ]
            self._prefix_cache.store(
                prompt_token_ids,
                new_chunks,
                start_chunk=matched_chunks,
                request_id=request_id,
            )

        handoff = KVHandoff.from_prefill(
            request_id=request_id,
            model_fingerprint=self._model.config.fingerprint,
            prompt_token_ids=prompt_token_ids,
            first_token=first_token,
            kv=kv,
            block_size=self._block_size,
        )
        sent = self._transport.send(handoff)
        return PrefillResult(
            request_id=request_id,
            prompt_tokens=len(prompt_token_ids),
            reused_tokens=reused,
            computed_tokens=computed,
            first_token=first_token,
            transferred_bytes=sent,
            num_blocks=len(handoff.blocks),
        )


class DecodeWorker:
    """Claims a handoff, checks it, and generates the remaining tokens."""

    def __init__(self, model: ToyLanguageModel, transport: KVTransport) -> None:
        self._model = model
        self._transport = transport

    @property
    def model(self) -> ToyLanguageModel:
        return self._model

    def decode(self, request_id: str, *, max_new_tokens: int) -> DecodeResult:
        """Continue a prefilled request.

        Raises HandoffNotFoundError when there is nothing to claim, and a
        HandoffError when the claimed KV is corrupt, incomplete, or from
        another model. In every failure case the handoff is gone: the
        request must be prefilled again rather than decoded from bad KV.
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        handoff = self._transport.receive(request_id)
        if handoff is None:
            raise HandoffNotFoundError(request_id)
        handoff.validate(expected_fingerprint=self._model.config.fingerprint)

        kv = handoff.kv()
        before = self._model.computed_tokens
        token = handoff.first_token
        tokens = [token]
        while len(tokens) < max_new_tokens:
            token = self._model.decode_step(kv, token)
            tokens.append(token)
        return DecodeResult(
            request_id=request_id,
            tokens=tuple(tokens),
            imported_tokens=handoff.num_kv_tokens,
            computed_tokens=self._model.computed_tokens - before,
        )
