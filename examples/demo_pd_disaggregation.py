"""Split prefill and decode into two workers and hand KV between them.

    uv run python examples/demo_pd_disaggregation.py

Runs on CPU with a deterministic toy model (no weights, no GPU):

1. Correctness: the two-worker path produces the colocated tokens, and the
   decode worker never recomputes the prompt.
2. Prefix cache on the prefill side: a second request sharing a document
   prefix computes only its new tokens.
3. Failures: corrupt bytes, a different model, cancellation, a second
   decode worker trying to claim the same request.
4. Timeline: why a long prompt hurts other requests when prefill and
   decode share one GPU, and what disaggregation costs instead.
"""

import shutil
import tempfile
from pathlib import Path

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.disaggregation import (
    CostModel,
    DecodeWorker,
    DirectoryTransport,
    HandoffError,
    HandoffNotFoundError,
    PrefillWorker,
    SimRequest,
    ToyLanguageModel,
    ToyModelConfig,
    generate_colocated,
    simulate_colocated,
    simulate_disaggregated,
)

CONFIG = ToyModelConfig(name="toy-lm", seed=7)
DOCUMENT = list(range(100, 124))  # 24 shared "document" tokens
MAX_NEW_TOKENS = 8


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def correctness(transport: DirectoryTransport) -> None:
    section("1. Same tokens, prompt computed once")
    prompt = DOCUMENT + [7, 8, 9]  # 27 tokens: 6 full blocks + 1 partial block

    reference = generate_colocated(ToyLanguageModel(CONFIG), prompt, MAX_NEW_TOKENS)

    prefill = PrefillWorker(ToyLanguageModel(CONFIG), transport, block_size=4)
    decode = DecodeWorker(ToyLanguageModel(CONFIG), transport)
    sent = prefill.prefill("req-1", prompt)
    print(
        f"prefill worker: computed {sent.computed_tokens}/{sent.prompt_tokens} "
        f"prompt tokens, sent {sent.num_blocks} blocks ({sent.transferred_bytes} B), "
        f"first token {sent.first_token}"
    )
    print(f"pending handoff files: {transport.pending_count()}")
    result = decode.decode("req-1", max_new_tokens=MAX_NEW_TOKENS)
    print(
        f"decode worker:  imported KV for {result.imported_tokens} tokens, "
        f"computed {result.computed_tokens} new positions"
    )
    print(f"colocated tokens:     {reference}")
    print(f"disaggregated tokens: {list(result.tokens)}")
    print(f"identical: {reference == list(result.tokens)}")


def prefix_reuse(transport: DirectoryTransport) -> None:
    section("2. Prefill worker reuses a shared prefix")
    cache = ChunkedPrefixCache(chunk_size=8, namespace=CONFIG.fingerprint)
    prefill = PrefillWorker(ToyLanguageModel(CONFIG), transport, prefix_cache=cache)
    decode = DecodeWorker(ToyLanguageModel(CONFIG), transport)
    for request_id, question in (("q-1", [1, 2, 3]), ("q-2", [4, 5, 6])):
        prompt = DOCUMENT + question
        sent = prefill.prefill(request_id, prompt)
        result = decode.decode(request_id, max_new_tokens=MAX_NEW_TOKENS)
        expected = generate_colocated(ToyLanguageModel(CONFIG), prompt, MAX_NEW_TOKENS)
        print(
            f"{request_id}: reused {sent.reused_tokens:>2}, computed "
            f"{sent.computed_tokens:>2} of {sent.prompt_tokens} prompt tokens; "
            f"tokens match colocated: {list(result.tokens) == expected}"
        )


def failures(transport: DirectoryTransport) -> None:
    section("3. When the decode worker must refuse")
    prompt = DOCUMENT + [42]
    prefill = PrefillWorker(ToyLanguageModel(CONFIG), transport)
    decode = DecodeWorker(ToyLanguageModel(CONFIG), transport)

    prefill.prefill("corrupt", prompt)
    path = transport.path_for("corrupt")
    data = bytearray(path.read_bytes())
    data[-5] ^= 0x01  # flip one bit in the payload
    path.write_bytes(bytes(data))
    try:
        decode.decode("corrupt", max_new_tokens=4)
    except HandoffError as error:
        print(f"corrupt bytes      -> {type(error).__name__}: {error}")

    prefill.prefill("other-model", prompt)
    stranger = DecodeWorker(ToyLanguageModel(ToyModelConfig(seed=8)), transport)
    try:
        stranger.decode("other-model", max_new_tokens=4)
    except HandoffError as error:
        print(f"different weights  -> {type(error).__name__}")

    prefill.prefill("cancelled", prompt)
    print(f"cancel before decode -> discarded: {transport.discard('cancelled')}")
    try:
        decode.decode("cancelled", max_new_tokens=4)
    except HandoffNotFoundError:
        print("decode after cancel  -> HandoffNotFoundError")

    prefill.prefill("claimed", prompt)
    decode.decode("claimed", max_new_tokens=4)
    try:
        DecodeWorker(ToyLanguageModel(CONFIG), transport).decode(
            "claimed", max_new_tokens=4
        )
    except HandoffNotFoundError:
        print("second decode worker -> HandoffNotFoundError (already claimed)")
    print(f"pending handoff files left: {transport.pending_count()}")


def timeline() -> None:
    section("4. Timeline: one GPU vs. a prefill GPU + a decode GPU")
    cost = CostModel(
        prefill_ms_per_token=0.5, decode_ms_per_step=20, transfer_ms_per_token=0.05
    )
    requests = [
        SimRequest("chat", arrival_ms=0, prompt_tokens=200, output_tokens=20),
        SimRequest("long-doc", arrival_ms=150, prompt_tokens=4000, output_tokens=5),
    ]
    print(
        f"cost: prefill {cost.prefill_ms_per_token} ms/token, decode "
        f"{cost.decode_ms_per_step} ms/step, "
        f"transfer {cost.transfer_ms_per_token} ms/token"
    )
    print(
        f"{'arrangement':<14} {'request':<9} {'TTFT':>8} {'max ITL':>9} {'finish':>8}"
    )
    for name, simulate in (
        ("colocated", simulate_colocated),
        ("disaggregated", simulate_disaggregated),
    ):
        for timing in simulate(requests, cost).values():
            print(
                f"{name:<14} {timing.request_id:<9} {timing.ttft_ms:>6.0f}ms "
                f"{timing.max_itl_ms:>7.0f}ms {timing.finish_ms:>6.0f}ms"
            )
    print(
        "colocated: chat stalls while long-doc prefills (max ITL).\n"
        "disaggregated: chat keeps its pace; long-doc pays the KV transfer in TTFT."
    )


def main() -> None:
    directory = Path(tempfile.mkdtemp(prefix="nano-lmcache-pd-"))
    try:
        transport = DirectoryTransport(directory)
        print(f"Handoff directory: {directory}")
        correctness(transport)
        prefix_reuse(transport)
        failures(transport)
        timeline()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    main()
