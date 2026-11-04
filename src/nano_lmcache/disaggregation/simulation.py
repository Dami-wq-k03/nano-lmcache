"""Timeline model: why separate prefill and decode at all?

Prefill is one large, compute-bound pass over the prompt; decode is many
small steps, one token each. On one GPU they compete: while a long prompt
prefills, every running request's next token waits. That shows up as a
spike in inter-token latency (ITL). Moving prefill to its own GPU removes
the stall, at the price of transferring KV before decode can start.

This module replays requests through both arrangements using a simple
cost model, in simulated milliseconds. It deliberately ignores batching
effects, memory limits, and queueing across many GPUs; the point is to
make the trade-off visible, not to predict real numbers.

Scheduling rules:

- Colocated: one GPU. Waiting prefills run first, in arrival order (as in
  nano-vLLM's scheduler); otherwise one decode step advances every running
  request by one token. The first token is emitted when prefill ends.
- Disaggregated: a prefill GPU runs prefills in arrival order. Each
  request's KV then transfers, and its first token is emitted when the
  decode GPU has it. The decode GPU runs one step at a time over every
  request whose KV has arrived; newcomers join at the next step boundary.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    prefill_ms_per_token: float = 0.5
    decode_ms_per_step: float = 20.0
    transfer_ms_per_token: float = 0.05

    def prefill_ms(self, prompt_tokens: int) -> float:
        return prompt_tokens * self.prefill_ms_per_token

    def transfer_ms(self, prompt_tokens: int) -> float:
        return prompt_tokens * self.transfer_ms_per_token


@dataclass(frozen=True)
class SimRequest:
    request_id: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.prompt_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("prompt_tokens and output_tokens must be positive")
        if self.arrival_ms < 0:
            raise ValueError("arrival_ms must not be negative")


@dataclass(frozen=True)
class RequestTiming:
    request_id: str
    arrival_ms: float
    token_times_ms: tuple[float, ...]

    @property
    def ttft_ms(self) -> float:
        """Time to first token, from arrival."""
        return self.token_times_ms[0] - self.arrival_ms

    @property
    def max_itl_ms(self) -> float:
        """Longest wait between two consecutive tokens; 0 for one token."""
        times = self.token_times_ms
        return max((b - a for a, b in zip(times, times[1:], strict=False)), default=0.0)

    @property
    def finish_ms(self) -> float:
        return self.token_times_ms[-1]


def _check_unique(requests: Sequence[SimRequest]) -> list[SimRequest]:
    ids = [request.request_id for request in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("request IDs must be unique")
    return sorted(requests, key=lambda request: request.arrival_ms)


def _results(
    requests: Sequence[SimRequest], times: dict[str, list[float]]
) -> dict[str, RequestTiming]:
    return {
        request.request_id: RequestTiming(
            request.request_id, request.arrival_ms, tuple(times[request.request_id])
        )
        for request in requests
    }


def simulate_colocated(
    requests: Sequence[SimRequest], cost: CostModel | None = None
) -> dict[str, RequestTiming]:
    """One GPU doing both prefill and decode, prefill first."""
    cost = cost if cost is not None else CostModel()
    queue = _check_unique(requests)
    times: dict[str, list[float]] = {request.request_id: [] for request in queue}
    running: list[SimRequest] = []
    now = 0.0
    while queue or running:
        if queue and queue[0].arrival_ms <= now:
            request = queue.pop(0)
            now += cost.prefill_ms(request.prompt_tokens)
            times[request.request_id].append(now)
            if request.output_tokens > 1:
                running.append(request)
        elif running:
            now += cost.decode_ms_per_step
            for request in running:
                times[request.request_id].append(now)
            running = [r for r in running if len(times[r.request_id]) < r.output_tokens]
        else:
            now = queue[0].arrival_ms
    return _results(requests, times)


def simulate_disaggregated(
    requests: Sequence[SimRequest], cost: CostModel | None = None
) -> dict[str, RequestTiming]:
    """A prefill GPU feeding a decode GPU through a KV transfer."""
    cost = cost if cost is not None else CostModel()
    ordered = _check_unique(requests)
    times: dict[str, list[float]] = {request.request_id: [] for request in ordered}

    # Prefill GPU: FIFO, independent of decode.
    kv_ready: list[tuple[float, SimRequest]] = []
    prefill_free = 0.0
    for request in ordered:
        start = max(prefill_free, request.arrival_ms)
        prefill_free = start + cost.prefill_ms(request.prompt_tokens)
        kv_ready.append(
            (prefill_free + cost.transfer_ms(request.prompt_tokens), request)
        )
    kv_ready.sort(key=lambda item: item[0])

    # Decode GPU: the first token is emitted when the KV lands.
    running: list[SimRequest] = []
    now = 0.0
    while kv_ready or running:
        while kv_ready and kv_ready[0][0] <= now:
            ready_at, request = kv_ready.pop(0)
            times[request.request_id].append(ready_at)
            if request.output_tokens > 1:
                running.append(request)
        if running:
            now += cost.decode_ms_per_step
            for request in running:
                times[request.request_id].append(now)
            running = [r for r in running if len(times[r.request_id]) < r.output_tokens]
        elif kv_ready:
            now = kv_ready[0][0]
    return _results(requests, times)
