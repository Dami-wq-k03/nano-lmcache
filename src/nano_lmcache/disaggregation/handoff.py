"""The message a prefill worker sends to a decode worker.

A handoff answers four questions a decode worker must settle before it
may skip the prompt's computation:

1. **Whose KV is this?** `model_fingerprint` must equal the decode
   model's, or the KV was produced by different weights or shapes.
2. **For which tokens?** `prompt_token_ids` travel with the KV, so the
   decode worker knows exactly which context it is continuing.
3. **Is all of it here?** Every prompt position must have KV, including
   a trailing partial block. (Contrast the prefix cache, which skips
   incomplete chunks: it may recompute them later, but a decode worker
   that is missing KV would have to run prefill itself.)
4. **Did it arrive intact?** The wire format carries a CRC32 and the
   payload length, so truncation and bit flips are detected.

Wire format, all UTF-8 JSON:

    {"version": 1, "payload_bytes": N, "payload_crc32": C}\\n<payload>

Check 4 happens in `from_bytes`; checks 1–3 in `validate`.
"""

import json
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

from nano_lmcache.disaggregation.toy_model import KVEntry

HANDOFF_VERSION = 1


class HandoffError(ValueError):
    """A handoff that a decode worker must not use."""


class CorruptHandoffError(HandoffError):
    """The bytes are truncated, altered, or not a handoff at all."""


class ConfigMismatchError(HandoffError):
    """The KV was produced by a different model configuration."""


class IncompleteHandoffError(HandoffError):
    """Some prompt positions have no KV, or the blocks are malformed."""


@dataclass(frozen=True)
class KVHandoff:
    """Prompt KV in transfer blocks, plus the token prefill already produced."""

    request_id: str
    model_fingerprint: str
    prompt_token_ids: tuple[int, ...]
    first_token: int
    block_size: int
    blocks: tuple[tuple[KVEntry, ...], ...]

    @classmethod
    def from_prefill(
        cls,
        *,
        request_id: str,
        model_fingerprint: str,
        prompt_token_ids: Sequence[int],
        first_token: int,
        kv: Sequence[KVEntry],
        block_size: int,
    ) -> "KVHandoff":
        """Group per-token KV into blocks; the last block may be partial."""
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        blocks = tuple(
            tuple(kv[start : start + block_size])
            for start in range(0, len(kv), block_size)
        )
        return cls(
            request_id=request_id,
            model_fingerprint=model_fingerprint,
            prompt_token_ids=tuple(prompt_token_ids),
            first_token=first_token,
            block_size=block_size,
            blocks=blocks,
        )

    @property
    def num_kv_tokens(self) -> int:
        return sum(len(block) for block in self.blocks)

    def kv(self) -> list[KVEntry]:
        """Per-token KV in prompt order, ready for decoding."""
        return [entry for block in self.blocks for entry in block]

    def validate(self, *, expected_fingerprint: str) -> None:
        """Raise unless this handoff is safe for that model to continue."""
        if self.model_fingerprint != expected_fingerprint:
            raise ConfigMismatchError(
                f"handoff {self.request_id!r} was produced by "
                f"{self.model_fingerprint!r}, decode model is "
                f"{expected_fingerprint!r}"
            )
        if not self.prompt_token_ids:
            raise IncompleteHandoffError("handoff has an empty prompt")
        for index, block in enumerate(self.blocks):
            is_last = index == len(self.blocks) - 1
            if (
                not block
                or len(block) > self.block_size
                or (not is_last and len(block) != self.block_size)
            ):
                raise IncompleteHandoffError(
                    f"block {index} has {len(block)} entries; only the last "
                    f"block may hold fewer than {self.block_size}"
                )
        if self.num_kv_tokens != len(self.prompt_token_ids):
            raise IncompleteHandoffError(
                f"handoff {self.request_id!r} has KV for {self.num_kv_tokens} "
                f"of {len(self.prompt_token_ids)} prompt tokens"
            )
        widths = {len(entry) for block in self.blocks for entry in block}
        if len(widths) != 1:
            raise IncompleteHandoffError(
                f"KV entries have inconsistent widths {sorted(widths)}"
            )

    def to_bytes(self) -> bytes:
        payload = json.dumps(
            {
                "request_id": self.request_id,
                "model_fingerprint": self.model_fingerprint,
                "prompt_token_ids": self.prompt_token_ids,
                "first_token": self.first_token,
                "block_size": self.block_size,
                "blocks": self.blocks,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        header = {
            "version": HANDOFF_VERSION,
            "payload_bytes": len(payload),
            "payload_crc32": zlib.crc32(payload) & 0xFFFFFFFF,
        }
        return (
            json.dumps(header, separators=(",", ":")).encode("utf-8") + b"\n" + payload
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "KVHandoff":
        """Decode bytes from `to_bytes`; raise CorruptHandoffError otherwise."""
        header_line, newline, payload = data.partition(b"\n")
        if not newline:
            raise CorruptHandoffError("missing header line")
        try:
            header = json.loads(header_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CorruptHandoffError("unreadable header") from error
        if not isinstance(header, dict):
            raise CorruptHandoffError("header is not an object")
        if header.get("version") != HANDOFF_VERSION:
            raise CorruptHandoffError(
                f"unsupported handoff version {header.get('version')!r}"
            )
        if header.get("payload_bytes") != len(payload):
            raise CorruptHandoffError(
                f"expected {header.get('payload_bytes')!r} payload bytes, "
                f"got {len(payload)} (truncated transfer?)"
            )
        if header.get("payload_crc32") != zlib.crc32(payload) & 0xFFFFFFFF:
            raise CorruptHandoffError("payload checksum mismatch")
        try:
            fields = json.loads(payload)
            return cls(
                request_id=str(fields["request_id"]),
                model_fingerprint=str(fields["model_fingerprint"]),
                prompt_token_ids=tuple(int(t) for t in fields["prompt_token_ids"]),
                first_token=int(fields["first_token"]),
                block_size=int(fields["block_size"]),
                blocks=tuple(
                    tuple(tuple(int(v) for v in entry) for entry in block)
                    for block in fields["blocks"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CorruptHandoffError("malformed payload") from error
