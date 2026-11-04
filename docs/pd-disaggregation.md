English | [简体中文](zh-CN/pd-disaggregation.md)

# Prefill/decode disaggregation (teaching version)

Prefill/decode (PD) disaggregation runs the two phases of generation on
different workers: a *prefill worker* turns the prompt into KV and the
first token, then hands that KV to a *decode worker* that generates the
rest. The point is not to split two functions. A system that disaggregates
must define who owns the KV, what format it travels in, how it moves, how
long it lives, and what happens when any of that goes wrong.

This version runs on CPU with a deterministic toy model, so every claim
below is checked exactly by a test. It does not move real tensors or
drive nano-vLLM; see [What this version leaves out](#what-this-version-leaves-out).

```bash
uv run python examples/demo_pd_disaggregation.py
uv run pytest tests/unit/test_pd_handoff.py tests/unit/test_pd_workers.py \
  tests/unit/test_pd_simulation.py
```

## Why split the phases at all

Prefill processes the whole prompt in one large pass; decode produces one
token per small step. When both share one GPU, a long prompt that arrives
mid-generation stalls every running request until its prefill ends. The
demo's timeline section shows this using its cost model (0.5 ms per prompt
token, 20 ms per decode step, 0.05 ms per transferred token):

| Arrangement | Request | TTFT | Max inter-token latency | Finish |
| --- | --- | ---: | ---: | ---: |
| colocated | chat (200-token prompt) | 100 ms | **2020 ms** | 2480 ms |
| colocated | long-doc (4000-token prompt, arrives at 150 ms) | 2010 ms | 20 ms | 2240 ms |
| disaggregated | chat | 110 ms | **20 ms** | 490 ms |
| disaggregated | long-doc | **2200 ms** | 20 ms | 2430 ms |

Disaggregation removes the chat request's 2-second stall. The cost: every
request's TTFT now includes the KV transfer, so the long request's first
token arrives later. These are simulated numbers from
[`simulation.py`](../src/nano_lmcache/disaggregation/simulation.py),
which ignores batching, memory limits, and multi-GPU queueing. Use it to
see the trade-off, not to predict real speedups.

Experiment: in the demo, raise `transfer_ms_per_token` until disaggregation
also hurts the chat request's TTFT, or move `long-doc`'s arrival past the
chat request's finish time and watch the stall disappear from the
colocated row.

## The pieces

```text
Client
  │ prompt
  ▼
PrefillWorker ──KVHandoff (bytes)──▶ KVTransport ──claim──▶ DecodeWorker
  │ optional ChunkedPrefixCache                              │
  ▼                                                          ▼
first token                                         remaining tokens
```

| Module | Role |
| --- | --- |
| [`toy_model.py`](../src/nano_lmcache/disaggregation/toy_model.py) | KV of each token is chained to all earlier tokens; the next token reads every KV entry; a different `seed` (weights) gives different KV. The model counts computed positions. |
| [`handoff.py`](../src/nano_lmcache/disaggregation/handoff.py) | `KVHandoff`: request ID, model fingerprint, prompt tokens, first token, KV in blocks. Versioned wire format with length and CRC32. |
| [`transport.py`](../src/nano_lmcache/disaggregation/transport.py) | `InMemoryTransport` (threads) and `DirectoryTransport` (processes sharing a directory). Both move bytes and enforce single-claim ownership. |
| [`workers.py`](../src/nano_lmcache/disaggregation/workers.py) | `PrefillWorker`, `DecodeWorker`, and the `generate_colocated` reference. |
| [`simulation.py`](../src/nano_lmcache/disaggregation/simulation.py) | Colocated vs. disaggregated timelines: TTFT, max inter-token latency, finish time. |

## Rules the handoff must keep

**1. Same answer as colocated.** The disaggregated tokens must equal
`generate_colocated` for the same prompt. Tests check this for both
transports and for 1, 2, and 9 new tokens.

**2. The decode worker never recomputes the prompt.** After a handoff of
an *n*-token prompt, generating *m* tokens costs the decode model exactly
*m − 1* positions: the first token was produced by prefill and travels in
the handoff. `test_decode_worker_never_recomputes_the_prompt` checks the
counters.

**3. Ship every prompt position, including a partial last block.** The
prefix cache skips an incomplete trailing chunk because it can recompute
it later. A decode worker cannot: missing KV would force it to run prefill
itself. An 11-token prompt with block size 4 therefore ships blocks of
4, 4, and 3. `validate` rejects a handoff whose KV count differs from its
prompt length, or whose non-final blocks are short.

**4. Check identity before using foreign KV.** KV from other weights or
shapes is silently wrong: it has the right structure but produces
different tokens. The handoff carries `model_fingerprint`; the decode
worker rejects a mismatch with `ConfigMismatchError` before computing
anything. A real system's fingerprint must cover everything that changes
KV: weights, dtype, layer and head counts, tensor-parallel layout, and
position encoding.

**5. Detect damaged transfers.** `from_bytes` checks the format version,
payload length (truncation), and CRC32 (bit flips), raising
`CorruptHandoffError`. Checking integrity is separate from checking
meaning: an intact handoff can still come from the wrong model.

**6. Exactly one owner at a time.** After `send`, the transport owns the
handoff. `receive` *claims* it: the in-memory transport pops it under a
lock, and the directory transport renames the file, which succeeds for
only one process. A second decode worker gets `HandoffNotFoundError`.
Sending a second handoff for a pending request ID raises
`DuplicateHandoffError` rather than overwriting the first.

**7. Failures end the handoff.** Whether the KV is corrupt, incomplete, or
from another model, the claimed handoff is removed and the decode worker
computes nothing. Recovery means prefilling again, never decoding from bad
KV. A cancelled request calls `discard`, so its KV does not linger.

## The prefix cache on the prefill side

The prefix cache and the handoff solve different problems. The prefix
cache reuses KV **across requests**, and the handoff moves **one
request's** KV between roles. They compose: give `PrefillWorker` a
`ChunkedPrefixCache` and a second request that shares a document prefix
computes only its new tokens:

```text
q-1: reused  0, computed 27 of 27 prompt tokens; tokens match colocated: True
q-2: reused 24, computed  3 of 27 prompt tokens; tokens match colocated: True
```

The handoff still carries all 27 positions: what the decode worker needs
does not depend on where prefill got its KV. Use the model fingerprint as
the cache namespace, as the demo does, so the two checks agree on what
"the same model" means.

## Directory transport details

`DirectoryTransport` writes each handoff to a temporary file, then
hard-links it to its final name. Linking fails if the name already exists
(duplicate detection), and readers never see a partial file. File names
are SHA-256 hashes of request IDs, so no ID can address a path outside
the directory. `receive` renames the file to a per-process claimed name
before reading it, then deletes it whether or not the bytes are valid.
`test_concurrent_receivers_claim_once` races eight receivers on one handoff.

The directory needs hard-link and atomic-rename support (any local file
system). Network file systems may weaken these guarantees.

## What this version leaves out

- **Real tensors and engines.** A nano-vLLM version would export a
  request's KV blocks after prefill (the block format in
  [`kv_blocks.py`](../src/nano_lmcache/integrations/nanovllm/kv_blocks.py)
  already exists), and import them into a second engine's block table
  before decode. See the roadmap's PD tasks.
- **Streaming and overlap.** Real systems transfer layer by layer or block
  by block while prefill is still running, and over RDMA/NCCL rather than
  files. This version sends one message after prefill ends.
- **Scheduling and routing.** Which decode worker gets a request, how many
  prefill vs. decode GPUs to run, and backpressure when decode memory is
  full are all outside this version.
- **Timeouts and garbage collection.** A handoff that is never claimed
  stays pending until `discard`. A real system needs leases or a TTL.

Return to the [guided learning path](tutorial/README.md#8-split-prefill-and-decode)
or the [documentation index](README.md).
