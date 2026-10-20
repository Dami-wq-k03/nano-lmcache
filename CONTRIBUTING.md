# Contributing to nano-lmcache

Thanks for your interest! nano-lmcache is an educational, pure-Python
reimplementation of the core ideas behind LMCache (KV cache reuse via
chunked prefix keys). Clarity and testability matter more here than
performance.

## Project stage

The project grows one concept at a time. See `docs/architecture.md` for
the roadmap:

- **Nano** (done): chunking, prefix-aware keys, in-memory storage,
  prefix lookup, hit/miss metrics.
- **Mini** (in progress): storage interface and bounded, prefix-safe
  eviction are in place; more cache behavior and backends come next.
- **Integration** (later): real KV tensors and inference engines such as
  vLLM.

Please do not add PyTorch, vLLM, or other heavy runtime dependencies
ahead of the stage that needs them. If you are unsure whether an idea
fits, open an issue first.

## Setup

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Kangwenqiao/nano-lmcache.git
cd nano-lmcache
uv sync
```

## Checks

CI runs exactly these three commands. Please make sure they pass locally
before opening a pull request:

```bash
uv run pytest
uv run ruff check .
uv run mypy src/nano_lmcache
```

- Ruff uses a line length of 88 with the `E`, `F`, `I`, `UP`, `B` rules.
- mypy runs in `strict` mode; public code needs complete type hints.

## Development workflow

This project is developed test-first:

1. Write the tests for the new behavior and confirm they fail.
2. Implement the behavior until they pass.
3. Run the three checks above.

Every new feature or bug fix needs tests in `tests/unit/`. If it changes
user-visible behavior, update the relevant example in `examples/` and
`docs/architecture.md`.

Code should explain itself through clear names and docstrings. Add
comments for the reasoning behind non-obvious decisions, for example why
eviction must respect prefix chains.

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
