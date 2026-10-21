# Contributing to nano-lmcache

Thanks for your interest! nano-lmcache is an educational, pure-Python
reimplementation of the core ideas behind LMCache (KV cache reuse via
chunked prefix keys). Clarity and testability matter more here than
performance.

## Project stage

The project grows one concept at a time. Open tasks, their difficulty,
and hardware requirements are listed in [ROADMAP.md](ROADMAP.md); the
design is described in `docs/architecture.md`.

- **Nano** (done): chunking, prefix-aware keys, in-memory storage,
  prefix lookup, hit/miss metrics.
- **Mini** (done, follow-ups open): storage interface, prefix-safe
  eviction, disk and tiered backends.
- **GPU** (done, follow-ups open): real KV tensors, GPU to CPU offload.
- **nano-vLLM integration** (open for contributors): nano-lmcache as the
  external KV cache of nano-vLLM; see
  [docs/nanovllm-integration.md](docs/nanovllm-integration.md).

The core package must stay free of runtime dependencies. PyTorch,
nano-vLLM, and similar libraries belong in optional dependency groups and separate
subpackages (see task I1 in the roadmap), so people learning the cache
concepts never have to install them. If you are unsure whether an idea
fits, open an issue first.

## Setup

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Dami-wq-k03/nano-lmcache.git
cd nano-lmcache
uv sync
```

## Checks

CI runs exactly these commands. Please make sure they pass locally
before opening a pull request:

```bash
uv run pytest
uv run ruff check .
uv run mypy src/nano_lmcache
uv sync --locked --group tensor-test
uv run --locked --group tensor-test pytest -m "torch and not gpu"
```

- Ruff uses a line length of 88 with the `E`, `F`, `I`, `UP`, `B` rules.
- mypy runs in `strict` mode; public code needs complete type hints.

## Optional dependencies and GPU tests

The core package has no runtime dependencies, and a plain `uv sync`
installs only the dev tools. Heavy libraries are optional extras:

```bash
uv sync --extra torch   # PyTorch only
uv sync --extra hf      # PyTorch + transformers
uv sync --group tensor-test  # CPU PyTorch + transformers for tests
```

On Linux, `torch` is installed from PyTorch's CUDA 12.4 wheel index (see
`[tool.uv.sources]` in `pyproject.toml`). PyPI's default Linux wheel targets
CUDA 13 and needs a very recent NVIDIA driver, while CUDA 12 wheels run on
drivers >= 525. `torch >= 2.6` is required by current transformers.
The `tensor-test` dependency group uses PyTorch's CPU wheel index so the same
tests can run efficiently without a GPU.
Check that your GPU is visible with:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

nano-vLLM integration tests need the `nanovllm` dependency group (Linux,
Python 3.11 or 3.12; flash-attn comes as a prebuilt wheel), a CUDA GPU,
and a local model directory:

```bash
uv sync --extra hf --group nanovllm
CUDA_VISIBLE_DEVICES=0 NANO_LMCACHE_TEST_MODEL=/path/to/Qwen3-0.6B \
    uv run pytest -m nanovllm
```

nano-vLLM always runs on the first visible GPU, so choose one with
`CUDA_VISIBLE_DEVICES`. Without the model variable, the real-engine tests
skip and the scheduler-level tests still run (they need no GPU).

Code that needs PyTorch lives in `src/nano_lmcache/tensor/`, and inference
engine adapters live in `src/nano_lmcache/integrations/` (for example
`integrations/nanovllm/`). Never import torch or an engine from
`nano_lmcache` itself, `core`, or `storage`; a test enforces this.

Tests that need heavy dependencies carry a marker and are skipped, not
failed, when the environment cannot run them:

| Marker | Needs |
| --- | --- |
| `torch` | PyTorch installed |
| `gpu` | PyTorch and a CUDA GPU |
| `vllm` | vLLM and a CUDA GPU |
| `nanovllm` | The `nanovllm` dependency group (add `gpu` for real-engine tests) |

```python
import pytest


@pytest.mark.gpu
def test_copy_to_gpu():
    import torch  # import inside, so collection works without torch
    ...
```

CI installs the `tensor-test` group in a separate job and runs tests marked
`torch` on CPU. GPU tests only run on your machine. Run just them with
`uv run pytest -m gpu`, and say in your PR which GPU you tested on.

## Development workflow

This project is developed test-first:

1. Write the tests for the new behavior and confirm they fail.
2. Implement the behavior until they pass.
3. Run the three checks above.

Every new feature or bug fix needs tests in `tests/unit/`. If it changes
user-visible behavior, update the relevant example in `examples/` and
`docs/architecture.md`.

Write all code, comments, docs, and commit messages in English, so the
project stays readable for everyone.

Code should explain itself through clear names and docstrings. Add
comments for the reasoning behind non-obvious decisions, for example why
eviction must respect prefix chains.

## Benchmarks

Performance work (for example roadmap task M2) must come with numbers.
Run the benchmark before and after your change, on the same machine, and
paste both tables in the PR:

```bash
uv run python benchmarks/bench_prefix_cache.py
uv run python benchmarks/bench_prefix_cache.py --chunks 100 1000 --capacity 64
```

GPU work has its own benchmark, which needs a CUDA GPU:

```bash
uv run python benchmarks/bench_offload.py --device cuda:0
```

Benchmarks are not part of CI, because timings vary between machines.

## Branches, commits, and pull requests

- One pull request per feature. Keep changes small and focused.
- Branch from an up-to-date `main`, using a prefix that matches the
  change: `feat/...`, `fix/...`, `docs/...`, `chore/...`.
- Write commit messages in the
  [Conventional Commits](https://www.conventionalcommits.org/) style,
  for example `feat: add bounded memory cache`.
- In the pull request, describe what changed and why, and how you tested
  it. CI must pass before merging.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE). Only submit code you wrote yourself or that is
under a license compatible with MIT, and keep the original copyright
notice if you adapt someone else's code.
