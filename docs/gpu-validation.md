# GPU validation record

Validated on 2026-11-03 against runtime source base
[`bffb23c`](https://github.com/Dami-wq-k03/nano-lmcache/commit/bffb23cbba74e6f7aaae6a95c8c82242f1e42aa8),
including the thread-safe storage change. The documentation working tree
does not change runtime code or tests. This record checks the runnable
[learning path](tutorial/README.md); it is not a performance benchmark.

## Environment

| Component | Value |
| --- | --- |
| GPU | One NVIDIA GeForce RTX 4090, 24 GiB; selected before starting Python |
| Driver | 580.178.04 |
| Python | 3.11.16, Linux |
| PyTorch | 2.6.0+cu124, CUDA 12.4 |
| Transformers | 5.20.2 |
| FlashAttention | 2.7.4.post1, prebuilt CUDA 12 / torch 2.6 wheel |
| nano-vLLM | `bb823b3e06983d71485a8e1f23715ebd87d98ef8` |
| Model | Local Qwen3-0.6B weights; HF demo uses FP32, engine uses BF16 |
| Runtime settings | `OMP_NUM_THREADS=1`; model downloads disabled during execution |

A separate workspace and fresh cache directories kept the run independent
of existing server checkouts and caches. Dependencies were installed with:

```bash
uv sync --locked --extra hf --group nanovllm
```

After installation, execution used `uv run --offline --no-sync --locked
--extra hf --group nanovllm` to avoid repeating network dependency checks.
The installed environment came from the lock file. The full test invocation
was equivalent to:

```bash
NANO_LMCACHE_TEST_MODEL=/path/to/Qwen3-0.6B \
  uv run --offline --no-sync --locked --extra hf --group nanovllm pytest -q
```

## Results

| Check | Observation |
| --- | --- |
| Complete test suite, with CUDA and local model enabled | **575 passed, no skipped tests**, 94.97 seconds |
| Ruff and mypy in the GPU environment | Both passed |
| HF model demo on CPU | Prompt B reused 256 tokens and computed 11; maximum logit difference `5.67e-5` |
| HF model demo on CUDA | Prompt B reused 256 tokens and computed 11; maximum logit difference `6.82e-5` |
| CUDA with pinned CPU offload | Same reuse and logit difference as direct CUDA; 56.0 MiB of cached KV in CPU memory |
| CPU-offload + disk, cold process | Prompt A reused 0 tokens; Prompt B reused 256; no disk reads were needed for reuse |
| CPU-offload + disk, second process | Prompt A restored 256 tokens and computed 12; 16 chunks served from disk and 16 from memory across both prompts |
| nano-vLLM, cold process | First request had no L1/L2 hits; five complete blocks were persisted |
| nano-vLLM, second process | First request restored five L2 blocks with empty L1; later requests hit five GPU L1 blocks |
| Generated answers | Both questions produced identical answers in the cold and restarted processes |
| Engine Trace JSON → HTML | Passed; the imported events preserved cold/L2/L1 attribution and produced standalone HTML |

The full suite includes CUDA tensor/offload checks and seven real-model
nano-vLLM integration tests. Those tests check restored KV, generation
against a no-cache baseline, prefix sharing, chunked prefill, bounded L2,
disk reopening, and trace rendering.

Reproduction commands for the demos are in tutorial
[step 6](tutorial/README.md#6-reuse-real-model-tensors) and
[step 7](tutorial/README.md#7-restore-a-real-engines-cache). The engine demo
was invoked twice as separate Python processes with the same disk directory.

## Interpretation

These results verify the documented examples and correctness boundaries
for this model and environment. Floating-point logit differences are small
but nonzero. A cache hit is not a speedup guarantee, and demo TTFT includes
tracing and synchronous writes. Use the controlled
[TTFT experiment](nanovllm-benchmark.md) for performance comparisons.

Return to the [documentation index](README.md).
