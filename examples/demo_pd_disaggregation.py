"""Split prefill and decode into two workers and hand KV between them.

    uv run python examples/demo_pd_disaggregation.py
"""

from itertools import pairwise

from nano_lmcache.disaggregation import (
    Request,
    ToyModel,
    decode,
    generate,
    prefill,
    timeline,
)

prompt = list(range(100, 127))
channel: dict[str, bytes] = {}  # stands in for the network between workers

print("=== 1. Same tokens, prompt computed once ===")
prefill(ToyModel(), "req-1", prompt, channel)
print(f"handoff: {len(channel['req-1'])} bytes for {len(prompt)} prompt tokens")
decoder = ToyModel()
tokens = decode(decoder, "req-1", 8, channel)
print(f"colocated:     {generate(ToyModel(), prompt, 8)}")
print(f"disaggregated: {tokens}")
print(f"decode worker computed {decoder.computed} positions (prompt: 0)")

print("\n=== 2. When the decode worker must refuse ===")
prefill(ToyModel(), "x", prompt, channel)
channel["x"] = channel["x"][:-1] + b"?"  # damage one byte in transit
try:
    decode(ToyModel(), "x", 4, channel)
except ValueError as error:
    print(f"corrupt bytes -> {error}")

prefill(ToyModel(seed=1), "x", prompt, channel)  # different weights
try:
    decode(ToyModel(seed=2), "x", 4, channel)
except ValueError as error:
    print(f"other model   -> {error}")

prefill(ToyModel(), "y", prompt, channel)
decode(ToyModel(), "y", 4, channel)
try:
    decode(ToyModel(), "y", 4, channel)
except KeyError:
    print("second claim  -> KeyError (already claimed)")

print("\n=== 3. Timeline: one GPU vs. prefill GPU + decode GPU ===")
requests = [Request("chat", 0, 200, 20), Request("long-doc", 150, 4000, 5)]
print(f"{'':<14} {'request':<9} {'TTFT':>7} {'max gap':>8}")
for disaggregated in (False, True):
    times = timeline(requests, disaggregated=disaggregated)
    for r in requests:
        t = times[r.id]
        gap = max((b - a for a, b in pairwise(t)), default=0.0)
        label = "disaggregated" if disaggregated else "colocated"
        print(f"{label:<14} {r.id:<9} {t[0] - r.arrival_ms:>5.0f}ms {gap:>6.0f}ms")
