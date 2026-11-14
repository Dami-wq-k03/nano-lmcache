English | [简体中文](zh-CN/README.md)

# Documentation

Start with the [guided learning path](tutorial/README.md) if you are new to
KV cache reuse. It connects each concept to a runnable experiment, an
expected observation, source code, and tests. The first five steps need no
model or GPU.

## Learn and understand

| Document | What it explains |
| --- | --- |
| [Guided learning path](tutorial/README.md) | Chunking → prefix matching → eviction → storage tiers → traces → real tensors → engine restoration → PD disaggregation |
| [Glossary and FAQ](glossary.md) | KV, prefill, decode, TTFT, chunks, blocks, and common misconceptions |
| [Architecture](architecture.md) | Current modules, invariants, capacity accounting, serialization, and concurrency boundaries |
| [nano-vLLM integration](nanovllm-integration.md) | Engine block layout, L1/L2 lookup, restore/save hooks, and correctness rules |
| [vLLM connector guide](vllm-kv-connector.md) | Pinned version, no-op connector, real request logs, output/APC comparisons and block/chunk mapping |
| [PD disaggregation](pd-disaggregation.md) | Prefill/decode workers, KV handoff format, ownership, failure handling, and the TTFT vs. stall trade-off |

## Run and inspect

| Task | Entry point | Requirements |
| --- | --- | --- |
| Run the smallest cache example | [Project quick start](../README.md#start-here) | Python 3.11+, uv |
| Replay synthetic workloads or FreeInference blocks | [Replay tutorial](tutorial/README.md#5-follow-a-lookup-in-the-viewer) | Core package; JSONL input for FreeInference |
| Stream large workloads and compare bounded cache runs | [Streaming replay](streaming-replay.md) | Core package; JSONL sorted by arrival time and request ID |
| Inspect persisted entries without changing them | [Disk tutorial](tutorial/README.md#4-persist-and-promote-chunks) | Core package, existing cache directory |
| Reuse Hugging Face model KV | [Real-model tutorial](tutorial/README.md#6-reuse-real-model-tensors) | `hf` extra, model weights; CPU supported |
| Record engine events and render offline HTML | [Engine traces](engine-traces.md) | Linux/NVIDIA GPU/local model to record; core package to render |
| Hand KV from a prefill worker to a decode worker | [PD tutorial](tutorial/README.md#8-split-prefill-and-decode) | Core package |

## Experiments and trade-offs

These reports record particular models, hardware, and workloads. Their
results are measurements under those conditions, not universal speedup claims.

| Report | Question |
| --- | --- |
| [Eviction performance](eviction-performance.md) | How does a leaf heap preserve LRU order while reducing victim-selection cost? |
| [INT8 KV compression](quantized-kv.md) | How much space does quantization save, and what errors and latency does it add? |
| [nano-vLLM TTFT](nanovllm-benchmark.md) | When do memory/disk loads cost less than recomputing the prefix? |
| [GPU/data-plane experiments](gpu-experiment-report.md) | What do pinned/asynchronous copies, page operations, INT8, restart recovery, and write-behind measure on the development GPU? |
| [Serving validation](serving-validation-report.md) | Real vLLM external hits, async export, cold NVMe and two-GPU model handoff |

The [GPU validation record](gpu-validation.md) documents a complete test run,
real-model examples, two-process disk restoration, and engine trace rendering.

## Contribute and maintain

- [ROADMAP](../ROADMAP.md) owns task status, prerequisites, and acceptance criteria.
- [CONTRIBUTING](../CONTRIBUTING.md) owns setup, dependency groups, checks, and the contribution workflow.
- Architecture and topic guides describe implemented behavior. Completed
  task specifications in the roadmap provide historical context.
- Keep the English and Chinese README, index, tutorial, and architecture
  aligned when changing the concepts they explain. Specialist reports are
  currently in English; the [Chinese index](zh-CN/README.md) labels them.

Return to the [project README](../README.md).
