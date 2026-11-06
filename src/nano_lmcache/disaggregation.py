"""Prefill/decode (PD) disaggregation in miniature.

    colocated:     prompt ─▶ [prefill + decode] ─▶ tokens
    disaggregated: prompt ─▶ [prefill] ──KV bytes──▶ [decode] ─▶ tokens

A prefill worker computes the prompt's KV and the first token, then hands
them over as bytes. A decode worker claims the bytes, checks them, and
continues without recomputing the prompt. A toy model stands in for a
real one so outputs can be compared exactly. `timeline` shows why anyone
bothers: a long prefill no longer stalls other requests' decoding.
"""

import hashlib
import json
import zlib
from dataclasses import dataclass

KV = list[list[int]]  # one small list per token position


@dataclass
class ToyModel:
    """Deterministic stand-in for a causal LM.

    Each position's KV depends on its token and all earlier ones; the next
    token reads every KV entry (like attention); `seed` plays the weights.
    `computed` counts positions computed, to show what decode skips.
    """

    seed: int = 0
    computed: int = 0

    def extend(self, kv: KV, tokens: list[int]) -> None:
        for token in tokens:
            self.computed += 1
            previous = kv[-1] if kv else None
            digest = hashlib.sha256(f"{self.seed}|{previous}|{token}".encode()).digest()
            kv.append([digest[0], digest[1]])

    def next_token(self, kv: KV) -> int:
        if not kv:
            raise ValueError("cannot predict a token from an empty context")
        return sum(entry[0] * (i + 1) for i, entry in enumerate(kv)) % 1000


def _continue(model: ToyModel, kv: KV, token: int, max_new_tokens: int) -> list[int]:
    """Decode loop: feed the last token, predict the next."""
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    tokens = [token]
    while len(tokens) < max_new_tokens:
        model.extend(kv, [token])
        token = model.next_token(kv)
        tokens.append(token)
    return tokens


def generate(model: ToyModel, prompt: list[int], max_new_tokens: int) -> list[int]:
    """Colocated reference: prefill and decode on one worker."""
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    kv: KV = []
    model.extend(kv, prompt)
    return _continue(model, kv, model.next_token(kv), max_new_tokens)


def prefill(
    model: ToyModel, request_id: str, prompt: list[int], channel: dict[str, bytes]
) -> None:
    """Compute the prompt's KV and first token; hand them over as bytes."""
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    kv: KV = []
    model.extend(kv, prompt)
    payload = json.dumps(
        {
            "model": model.seed,
            "prompt": prompt,
            "first": model.next_token(kv),
            "kv": kv,
        },
        separators=(",", ":"),
    ).encode()
    channel[request_id] = zlib.crc32(payload).to_bytes(4, "big") + payload


def decode(
    model: ToyModel, request_id: str, max_new_tokens: int, channel: dict[str, bytes]
) -> list[int]:
    """Claim a handoff, check it, and generate without recomputing the prompt.

    `pop` makes the claim exclusive: a second decoder gets KeyError. A
    rejected handoff is gone too; the request must be prefilled again.
    """
    data = channel.pop(request_id)
    payload = data[4:]
    if int.from_bytes(data[:4], "big") != zlib.crc32(payload):
        raise ValueError("corrupt handoff: checksum mismatch")
    handoff = json.loads(payload)
    if handoff["model"] != model.seed:
        raise ValueError("handoff KV comes from a different model")
    if not handoff["prompt"] or len(handoff["kv"]) != len(handoff["prompt"]):
        raise ValueError("incomplete handoff: KV missing for some prompt tokens")
    return _continue(model, handoff["kv"], handoff["first"], max_new_tokens)


@dataclass(frozen=True)
class Request:
    id: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id must not be empty")
        if self.arrival_ms < 0:
            raise ValueError("arrival_ms must not be negative")
        if self.prompt_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("prompt_tokens and output_tokens must be positive")


def timeline(
    requests: list[Request],
    *,
    disaggregated: bool,
    prefill_ms: float = 0.5,  # per prompt token
    step_ms: float = 20.0,  # per decode step, all running requests together
    transfer_ms: float = 0.05,  # per prompt token of KV
) -> dict[str, list[float]]:
    """Simulated time of each emitted token, per request.

    Colocated: one GPU; a waiting prefill runs before the next decode step.
    Disaggregated: a prefill GPU, then a KV transfer, then the decode GPU.
    """
    if len({request.id for request in requests}) != len(requests):
        raise ValueError("request IDs must be unique")
    waiting = sorted(requests, key=lambda r: r.arrival_ms)
    times: dict[str, list[float]] = {r.id: [] for r in requests}
    ready: list[tuple[float, Request]] = []  # first token is out, KV on decode GPU
    if disaggregated:
        free = 0.0  # when the prefill GPU is next idle
        for r in waiting:
            free = max(free, r.arrival_ms) + r.prompt_tokens * prefill_ms
            ready.append((free + r.prompt_tokens * transfer_ms, r))
        waiting = []

    running: list[Request] = []
    ready_index = 0
    now = 0.0
    while waiting or ready_index < len(ready) or running:
        while ready_index < len(ready) and ready[ready_index][0] <= now:
            at, r = ready[ready_index]
            ready_index += 1
            times[r.id].append(at)
            running.append(r)
        running = [r for r in running if len(times[r.id]) < r.output_tokens]
        if waiting and waiting[0].arrival_ms <= now:
            r = waiting.pop(0)
            now += r.prompt_tokens * prefill_ms
            ready.append((now, r))
        elif running:
            now += step_ms
            for r in running:
                times[r.id].append(now)
        elif waiting or ready_index < len(ready):
            next_ready = (
                ready[ready_index][0] if ready_index < len(ready) else float("inf")
            )
            next_arrival = waiting[0].arrival_ms if waiting else float("inf")
            now = min(next_ready, next_arrival)
    return times
