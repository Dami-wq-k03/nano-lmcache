English | [简体中文](zh-CN/pd-disaggregation.md)

# Prefill/decode disaggregation

PD disaggregation runs prefill and decode on different workers. The
prefill worker computes the prompt's KV and the first token and hands them
to a decode worker, which generates the rest without recomputing the
prompt. The whole teaching version is one module,
[`disaggregation.py`](../src/nano_lmcache/disaggregation.py), about 160
lines, using a toy model so outputs can be compared exactly.

```bash
uv run python examples/demo_pd_disaggregation.py
uv run pytest tests/unit/test_disaggregation.py
```

## Why split the phases

Prefill is one big pass over the prompt, and decode is many small steps.
On one GPU, a long prompt that arrives mid-generation stalls every running
request until its prefill ends. The demo's timeline uses 0.5 ms per prompt
token, 20 ms per decode step, and 0.05 ms per transferred token:

| Arrangement | Request | TTFT | Max gap between tokens |
| --- | --- | ---: | ---: |
| colocated | chat (200-token prompt) | 100 ms | **2020 ms** |
| colocated | long-doc (4000 tokens, arrives at 150 ms) | 2010 ms | 20 ms |
| disaggregated | chat | 110 ms | **20 ms** |
| disaggregated | long-doc | **2200 ms** | 20 ms |

The stall disappears, but every TTFT now includes the KV transfer. These
numbers are simulated, not measured.

## What the handoff must guarantee

1. **Same answer.** `decode` after `prefill` returns exactly what
   `generate` returns on one worker.
2. **No prompt recomputation.** The first token travels in the handoff, so
   *m* new tokens cost the decode model *m − 1* positions.
3. **Checks before use.** `decode` rejects the handoff if it fails any of
   three checks:
   - its CRC32 does not match (bytes damaged in transit);
   - it comes from a different model (KV from other weights has the right
     shape but gives wrong tokens);
   - KV is missing for some prompt tokens.
4. **One owner.** `decode` *pops* the handoff from the channel, so a second
   decoder gets `KeyError`. A rejected handoff is gone too: the request
   must be prefilled again.

## What this version leaves out

Real tensors and engines (roadmap PD-3), streaming KV while prefill still
runs (PD-4), cross-process transports such as files, sockets, or RDMA,
routing between many workers, and timeouts for handoffs nobody claims.
Prefill can also reuse a prefix cache. The handoff stays the same because
the decode worker doesn't care where prefill got its KV.

Return to the [guided learning path](tutorial/README.md#8-split-prefill-and-decode).
