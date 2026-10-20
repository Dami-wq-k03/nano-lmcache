# Prefix-safe eviction performance (M2)

## Design decision

The original `PrefixLRUIndex._find_victim` scans all entries in recency
order until it finds an unprotected leaf. This is O(n) per eviction in
the worst case, especially when most cached entries are internal nodes
of deep chains.

| Approach | Candidate maintenance | Newly exposed parent | Policy |
| --- | --- | --- | --- |
| Original linear scan | O(1) access; O(n) victim search | Already in global recency order | Exact current LRU |
| Recency-ordered leaf list, append on exposure | O(1) | Becomes most recently used | Changes LRU order |
| Heap with lazy deletion (chosen) | O(log n) amortized | Re-enters with its last actual access tick | Exact current LRU |

An `OrderedDict` of leaves cannot insert a newly exposed parent into the
middle by its original access time in O(1). We retain the existing policy:
if `a` was accessed before `x`, and `a` becomes a leaf only after its child
is evicted, `a` must still be evicted before `x`. This also lets
`tests/unit/test_bounded_memory.py` pass without modification.

Bounded indexes retain the existing global recency order for callers and
recovery, and maintain a heap of `(last_access_tick, key)` candidates:

- Insertions and leaf hits push a candidate; internal-node hits only
  update the recorded tick.
- A record is stale if its key was removed, has children, or has a newer
  tick. Victim selection pops stale records lazily.
- Removing a last child pushes its parent with the parent's last actual
  access tick, rather than refreshing it.
- A protected parent is temporarily skipped and then returned to the
  heap, including when admission fails.
- Once the heap exceeds both 64 records and twice the stored entry
  count, it is rebuilt from current leaves. This bounds retained records
  by O(n), including repeated hits with no evictions. Rebuilding costs
  O(n) but only follows enough added records to amortize that cost.
- Unbounded indexes do not maintain ticks or candidates.

Heap operations are O(log n) amortized over the access/admission sequence;
a single operation can still clean up many stale records or rebuild.
The extra bookkeeping costs more for short chains. The measured gains
for deep chains justify the heap here; this is not a universal speedup.

## Measurement setup

Measured on 2026-10-20, macOS 27.0.1, Apple arm64, Python 3.11.15.
Before is commit `2aa6d0a` (the original scan); after is the heap
implementation on `codex/faster-prefix-eviction`. Both used the same
machine, sequential runs, `CHUNK_SIZE=16`, and the best of five repeats,
as defined by the existing I3 benchmark. All values below are milliseconds.
These are local measurements, not portable latency guarantees.

The standard commands were run before and after:

```bash
uv run python benchmarks/bench_prefix_cache.py
uv run python benchmarks/bench_prefix_cache.py --chunks 100 1000 --capacity 64
```

### Default command: store and lookup

| Chunks | Capacity | Store before | Store after | Lookup before | Lookup after | Matched (both) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | unbounded | 0.022 | 0.025 | 0.019 | 0.022 | 10 |
| 10 | 64 | 0.021 | 0.026 | 0.019 | 0.022 | 10 |
| 100 | unbounded | 0.205 | 0.236 | 0.180 | 0.207 | 100 |
| 100 | 64 | 0.204 | 0.236 | 0.118 | 0.135 | 64 |
| 1000 | unbounded | 2.047 | 2.075 | 1.805 | 1.819 | 1000 |
| 1000 | 64 | 1.962 | 1.948 | 0.186 | 0.188 | 64 |

### Explicit `--chunks 100 1000 --capacity 64`: store and lookup

| Chunks | Capacity | Store before | Store after | Lookup before | Lookup after | Matched (both) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | unbounded | 0.203 | 0.206 | 0.178 | 0.182 | 100 |
| 100 | 64 | 0.208 | 0.215 | 0.119 | 0.122 | 64 |
| 1000 | unbounded | 2.045 | 2.058 | 1.795 | 1.826 | 1000 |
| 1000 | 64 | 1.942 | 1.954 | 0.185 | 0.189 | 64 |

### Standard command churn (2000 prompts, 8 chunks each)

| Command | Capacity | Before | After | Entries kept (both) |
| --- | ---: | ---: | ---: | ---: |
| Default | unbounded | 37.238 | 36.089 | 16000 |
| Default | 64 | 41.550 | 45.320 | 64 |
| Explicit chunks/capacity | unbounded | 35.949 | 36.352 | 16000 |
| Explicit chunks/capacity | 64 | 41.158 | 46.235 | 64 |

### Churn across chain depths

These use the same I3 `run_churn` function, including real prefix-cache
key construction and storage, not just the index in isolation:

| Prompts | Chunks per prompt | Capacity | Before | After | Before / after | Entries kept (both) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 1 | 2048 | 95.470 | 106.209 | 0.90x | 2048 |
| 2000 | 8 | 64 | 42.095 | 46.760 | 0.90x | 64 |
| 128 | 256 | 2048 | 165.605 | 101.134 | 1.64x | 2048 |
| 64 | 1024 | 8192 | 788.278 | 207.856 | 3.79x | 8192 |

The 1024-chunk workload improves by about 74%; short-chain churn takes
roughly 9-12% longer because candidate maintenance replaces a cheap short
scan. Small store/lookup timings also vary between runs; they do not
establish an improvement outside the deep-chain workloads.

For reproducibility, this was the before/after depth sweep:

```bash
uv run python - <<'PY'
from benchmarks.bench_prefix_cache import format_churn, run_churn

for depth, capacity, prompts in [
    (1, 2048, 32768),
    (8, 64, 2000),
    (256, 2048, 128),
    (1024, 8192, 64),
]:
    print(format_churn(run_churn(prompts, depth, capacity, 5)))
PY
```

The benchmark now also exposes `--churn-prompts` and `--churn-chunks`, so
the deep cases can be rerun without an inline script:

```bash
uv run python benchmarks/bench_prefix_cache.py --capacity 2048 \
  --churn-prompts 128 --churn-chunks 256
uv run python benchmarks/bench_prefix_cache.py --capacity 8192 \
  --churn-prompts 64 --churn-chunks 1024
```

## Correctness checks

`tests/unit/test_leaf_eviction.py` covers original access times on newly
exposed leaves, internal-node hits, protected candidates, failed
admissions, stale records, key reuse, recovery-style recency replay, and
bounded candidate memory after repeated reads. Randomized sequences
compare every admission, eviction, parent relation, recency order, and
counter against an independent implementation of the original scan
policy (12 scenarios, 600 operations each).

A deep-chain test counts entry iteration rather than asserting noisy
timing thresholds. The existing bounded-memory, disk-recovery, and
backend-contract tests remain unchanged.
