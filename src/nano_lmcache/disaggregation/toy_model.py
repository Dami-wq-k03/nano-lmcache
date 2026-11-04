"""A deterministic stand-in for a causal language model.

Real KV tensors need PyTorch and model weights. To study the handoff
itself we only need a model with the same *shape of dependencies*:

- The KV of token i depends on token i and on every earlier token
  (here through a hash chain), so KV cannot be reused across different
  prefixes.
- The next token depends on *all* KV entries in the context (a crude
  stand-in for attention), so missing or wrong KV changes the output.
- Two models with different `seed` values ("weights") produce different
  KV for the same tokens, so KV from one model is useless to another.

The model counts how many token positions it computed. Tests use that
counter to prove a decode worker never recomputes the prompt.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

KVEntry = tuple[int, ...]
"""KV for one token position: a few integers instead of K/V tensors."""

_KV_VALUE_RANGE = 10_000


@dataclass(frozen=True)
class ToyModelConfig:
    """Everything that decides whether two models' KV are interchangeable."""

    name: str = "toy-lm"
    seed: int = 0
    vocab_size: int = 1000
    kv_dim: int = 4

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.kv_dim < 2:
            raise ValueError("kv_dim must be at least 2")

    @property
    def fingerprint(self) -> str:
        """Identity a decode worker compares before accepting foreign KV."""
        return (
            f"{self.name}:seed={self.seed}:vocab={self.vocab_size}:kv_dim={self.kv_dim}"
        )


class ToyLanguageModel:
    """Computes toy KV and next tokens; counts the positions it computed."""

    def __init__(self, config: ToyModelConfig | None = None) -> None:
        self._config = config if config is not None else ToyModelConfig()
        self._computed_tokens = 0

    @property
    def config(self) -> ToyModelConfig:
        return self._config

    @property
    def computed_tokens(self) -> int:
        """Token positions whose KV this model instance computed."""
        return self._computed_tokens

    def token_kv(self, previous: KVEntry | None, token: int) -> KVEntry:
        """KV for `token`, chained to the KV of the position before it."""
        self._computed_tokens += 1
        digest = hashlib.sha256(
            f"{self._config.fingerprint}|{previous}|{token}".encode()
        ).digest()
        return tuple(
            int.from_bytes(digest[4 * i : 4 * i + 4], "big") % _KV_VALUE_RANGE
            for i in range(self._config.kv_dim)
        )

    def next_token(self, kv: Sequence[KVEntry]) -> int:
        """The token predicted after the context whose KV is `kv`.

        Reads every position, like attention, so dropping or altering any
        entry can change the result.
        """
        if not kv:
            raise ValueError("cannot predict a token from an empty context")
        attended = sum(entry[0] * (position + 1) for position, entry in enumerate(kv))
        return (attended + kv[-1][1]) % self._config.vocab_size

    def extend(self, kv: Sequence[KVEntry], token_ids: Sequence[int]) -> list[KVEntry]:
        """Compute KV for `token_ids` placed after the context `kv`."""
        computed: list[KVEntry] = []
        previous = kv[-1] if kv else None
        for token in token_ids:
            previous = self.token_kv(previous, token)
            computed.append(previous)
        return computed

    def prefill(self, token_ids: Sequence[int]) -> tuple[list[KVEntry], int]:
        """Compute the prompt's KV and the first generated token."""
        if not token_ids:
            raise ValueError("prompt must contain at least one token")
        kv = self.extend([], token_ids)
        return kv, self.next_token(kv)

    def decode_step(self, kv: list[KVEntry], last_token: int) -> int:
        """Append `last_token`'s KV to `kv` in place and predict the next token."""
        kv.extend(self.extend(kv, [last_token]))
        return self.next_token(kv)
