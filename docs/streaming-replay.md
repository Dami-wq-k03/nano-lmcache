# Streaming workload replay and bounded reports

Replay a workload without retaining every request or event. The command-line
viewer keeps exact full-workload totals separately from the detail embedded
in HTML. Everything here uses the core package: no model, GPU, or runtime
dependencies are needed.

## Choose input and ordering

`--workload-jsonl` accepts one serialized `WorkloadRequest` per nonblank line.
For example, save these records as `/tmp/workload.jsonl`:

```jsonl
{"request_id":"a","session_id":"s","arrival_time_ms":0,"model_id":"m","prompt_tokens":6,"token_ids":[1,2,3,4,5,6]}
{"request_id":"b","session_id":"s","arrival_time_ms":1,"model_id":"m","prompt_tokens":6,"token_ids":[1,2,3,4,7,8],"observed_cached_tokens":2}
```

Run:

```bash
uv run python -m nano_lmcache.visualize \
  --workload-jsonl /tmp/workload.jsonl --stream --chunk-size 2 \
  --max-entries 4 --max-events 1000 --max-requests 100 --sample-every 2 \
  --summary-json /tmp/workload-summary.json --output /tmp/workload.html
```

The second request reuses two chunks (four tokens). Across both requests,
the simulation reuses 2/6 complete chunks; the dataset separately reports
two observed cached tokens. These numbers describe different measurements.
Use one model per token replay runner; mixing model labels raises an error
rather than reusing incompatible KV. Runner-owned token caches include
that label in their namespace, isolating persistent backends across models.
If supplying a prebuilt `ChunkedPrefixCache`, the caller owns its namespace
and must ensure it belongs to the workload's model.

For source-specific opaque block IDs, replace `token_ids` with
`prefix_blocks`, add `--opaque-blocks`, and set `--block-tokens` (default
16). These identities are not token IDs; replay models reuse, not model
execution. Opaque keys include the model label and preceding prefix.

Raw FreeInference session JSONL is also supported:

```bash
uv run python -m nano_lmcache.visualize \
  --freeinference-jsonl /path/to/sessions.jsonl --stream \
  --max-entries 4096 --max-events 10000 --max-requests 200 \
  --sample-every 10 --summary-json /tmp/freeinference-summary.json \
  --output /tmp/freeinference.html
```

`--stream` requires nondecreasing `(arrival_time_ms, request_id)` order.
Ties use lexicographic request IDs. Input is checked as it is consumed;
out-of-order requests cause a clear error, not a hidden sort. FreeInference
adapts and sorts **one session record** at a time, including nested sessions;
different records can still interleave. For large interleaved sources,
normalize to one-request-per-line workload JSONL and sort externally before
replay. Without `--stream`, replay sorts the entire selected workload in
memory, preserving the small-input behavior.

`--limit N` stops after N selected requests, without reading a later JSONL
record. It does not bound the size of an individual record. No input file is
changed. On disk/tiered replay, writes made before an input error remain in
the selected cache directory; use a fresh directory when retrying a cold run.

## Budgets and summary semantics

| Option | CLI default | Meaning |
| --- | --- | --- |
| `--max-entries` | 10,000 | Maximum simulated cache entries, per configured tier |
| `--max-events` | 10,000 | Maximum event objects retained for HTML |
| `--max-requests` | 200 | Maximum request details retained for HTML |
| `--sample-every` | 1 | Select original event sequences 0, k, 2k, … until the event cap |
| `--limit` | No limit | Stop processing input after N requests |
| `--summary-json` | No file | Export aggregates, configuration, and retention metadata |

Use zero events and/or zero requests for statistics without that detail.
`sample_every` is deterministic **prefix sampling**, not a random or
representative sample of the entire run. Once the event cap is reached,
later events are counted but not retained. Sequence numbers keep their
original values, so gaps are visible. Request detail keeps the first N
requests independently of event selection.

The online summary counts every processed request, complete chunk, simulated
reused token, dataset-observed cached token, and enabled trace event. It also
reports successful storage reads by backend. `observed_requests` states how
many requests supplied observed cache data: missing is distinct from zero.
Event-type counts include evictions and promotions even when no event
details are retained. They describe observed operations, not estimated GPU
time. The JSON summary contains no request/event arrays.

HTML explains the retained fraction. The request panel pages 25 retained
requests and displays at most 200 chunk states per request. The timeline
renders a window of 100 retained events; use Seek event to jump through
the report. The prefix tree shows up to 500 keys found in retained events.
Omitted ancestors are possible. It shows historical relationships, not
current cache residency or an exact reconstruction of an eviction sequence.

With `--stream`, retained memory is bounded by the cache budget, retained
events/requests, and the largest input record/request, plus counters for
distinct event types and storage-tier names. Entries and events are counts,
not byte limits: a single huge JSONL line, long request, metadata value, or
custom cache payload can still be large. Python integers grow with totals.
This is not an operating-system RSS limit. Disk space and temporary HTML/JSON
serialization also depend on retained data sizes.

The Python API keeps its full-detail defaults for compatibility. To use the
same budgets programmatically, configure all three retention sources:

```python
from nano_lmcache.datasets import iter_workload_jsonl
from nano_lmcache.observability import TraceRecorder
from nano_lmcache.replay import WorkloadReplayRunner
from nano_lmcache.storage.memory import MemoryKVCache

report = WorkloadReplayRunner(
    chunk_size=2,
    storage=MemoryKVCache(max_entries=4096),
    recorder=TraceRecorder(max_events=10000, sample_every=10),
).run(
    iter_workload_jsonl("/tmp/workload.jsonl"),
    sort_by_arrival=False,
    max_requests=200,
)
print(report.hit_rate, report.to_dict()["summary"])
```

Each runner call clears its attached recorder for a new report and retains
the simulated cache for warm replay. Reports snapshot their own counters.
Runner/recorder ownership is sequential; do not share a recorder with a
concurrently running engine or another replay. Disabled recorders do not
count events; request-level reuse totals still work.
`TraceRecorder.to_json()` / `write_json()` export retained events only;
use the replay report or summary export for full-workload statistics.
An event file recorded with sampling cannot reconstruct missing engine hits.

## Compare cache capacity

Use a fresh in-memory cache for each run, with the same workload, order,
chunk size, and detail budgets:

```bash
uv run python -m nano_lmcache.visualize --scenario lru_eviction \
  --max-entries 2 --summary-json /tmp/capacity-2.json --output /tmp/capacity-2.html
uv run python -m nano_lmcache.visualize --scenario lru_eviction \
  --max-entries 3 --summary-json /tmp/capacity-3.json --output /tmp/capacity-3.html
```

With capacity 2, the revisit misses (0/4 hits, two evictions). Capacity 3
retains the prefix until the revisit (1/4 hits, no eviction). Read
`summary.hit_rate`, `summary.event_counts.eviction` (absent means zero),
`summary.storage_hits`, and the recorded configuration in the JSON files.
The same flags apply to opaque-block replay and the memory/disk/tiered
backends. Disk and tiered cold comparisons need separate empty directories;
reusing a directory intentionally measures a warm persistent cache instead.
The eviction policy remains prefix-safe leaf LRU; alternative policies are
the separate M11 task.

## Validate memory growth

```bash
uv run python benchmarks/bench_streaming_replay.py --counts 1000 10000 100000
uv run python benchmarks/bench_streaming_replay.py --counts 1000 10000 --batch-baseline
uv run pytest tests/unit/test_streaming_replay.py
```

The benchmark generates requests incrementally, shares two blocks, and
introduces a new third block on every request. It bounds the simulated cache
to 64 entries, retains 20 requests and 100 events, and samples every tenth
event. The baseline sorts the input and retains all request/event detail.
It measures peak Python allocations with `tracemalloc`, not RSS, file I/O,
or tensor memory. Exact reuse and observed-token totals are checked in both
modes. The unit test checks lazy consumption and compares memory at two
workload lengths to catch accidental accumulation.

Measured on 2026-11-05, macOS 27.0.1 arm64, CPython 3.11.12, using the
commands above (KiB = 1,024 bytes):

| Requests | Batch, all detail: peak KiB | Streaming, bounded detail: peak KiB | Streaming seconds |
| --- | --- | --- | --- |
| 1,000 | 2,820.3 | 129.3 | 0.125 |
| 10,000 | 27,398.4 | 113.5 | 1.190 |
| 100,000 | Not run | 113.5 | 12.158 |

The streaming runs retain 20 requests and 100 events at every size. Cache
evictions are 938, 9,938, and 99,938 respectively. This illustrates bounded
retention for this generated workload, not a universal memory/latency bound
for arbitrary input records or tensor payloads.

## Recorded engine JSON

The existing `--trace-json` importer still parses a whole JSON array or
envelope and groups engine events in memory. It applies HTML detail caps
**after** exact engine statistics are calculated, and supports
`--summary-json`; `--stream` rejects this input format. This preserves
committed L1/L2 hit accounting rather than inferring engine reuse from a
sample. Large workload experiments should use the streaming JSONL paths
above. Streaming raw engine-event ingestion is a separate follow-up.
