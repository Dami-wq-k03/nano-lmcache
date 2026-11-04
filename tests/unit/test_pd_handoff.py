import threading

import pytest

from nano_lmcache.disaggregation import (
    ConfigMismatchError,
    CorruptHandoffError,
    DirectoryTransport,
    DuplicateHandoffError,
    IncompleteHandoffError,
    InMemoryTransport,
    KVHandoff,
    KVTransport,
    ToyLanguageModel,
    ToyModelConfig,
)

CONFIG = ToyModelConfig(seed=1)


def make_handoff(request_id="r", prompt=(1, 2, 3, 4, 5), block_size=2):
    kv, first = ToyLanguageModel(CONFIG).prefill(list(prompt))
    return KVHandoff.from_prefill(
        request_id=request_id,
        model_fingerprint=CONFIG.fingerprint,
        prompt_token_ids=prompt,
        first_token=first,
        kv=kv,
        block_size=block_size,
    )


@pytest.fixture(params=["memory", "directory"])
def transport(request, tmp_path):
    if request.param == "memory":
        return InMemoryTransport()
    return DirectoryTransport(tmp_path / "handoffs")


def test_partial_last_block_is_transferred():
    handoff = make_handoff(prompt=(1, 2, 3, 4, 5), block_size=2)

    assert [len(block) for block in handoff.blocks] == [2, 2, 1]
    assert handoff.num_kv_tokens == 5
    handoff.validate(expected_fingerprint=CONFIG.fingerprint)


def test_bytes_round_trip():
    handoff = make_handoff()

    assert KVHandoff.from_bytes(handoff.to_bytes()) == handoff


@pytest.mark.parametrize(
    "damage",
    [
        lambda data: data[:-3],  # truncated transfer
        lambda data: data[:-2] + bytes([data[-2] ^ 1]) + data[-1:],  # bit flip
        lambda data: b"not a handoff",
        lambda data: data.replace(b'"version":1', b'"version":2', 1),
    ],
)
def test_damaged_bytes_are_rejected(damage):
    with pytest.raises(CorruptHandoffError):
        KVHandoff.from_bytes(damage(make_handoff().to_bytes()))


def test_other_model_is_rejected():
    handoff = make_handoff()

    with pytest.raises(ConfigMismatchError):
        handoff.validate(expected_fingerprint=ToyModelConfig(seed=2).fingerprint)


def test_missing_block_is_rejected():
    handoff = make_handoff(prompt=(1, 2, 3, 4, 5), block_size=2)
    truncated = KVHandoff(
        request_id=handoff.request_id,
        model_fingerprint=handoff.model_fingerprint,
        prompt_token_ids=handoff.prompt_token_ids,
        first_token=handoff.first_token,
        block_size=handoff.block_size,
        blocks=handoff.blocks[:-1],
    )

    with pytest.raises(IncompleteHandoffError, match="4 of 5"):
        truncated.validate(expected_fingerprint=CONFIG.fingerprint)


def test_short_middle_block_is_rejected():
    handoff = make_handoff(prompt=(1, 2, 3, 4, 5), block_size=2)
    blocks = (
        handoff.blocks[0][:1],
        handoff.blocks[1],
        handoff.blocks[2] + handoff.blocks[0][1:],
    )
    malformed = KVHandoff(
        request_id=handoff.request_id,
        model_fingerprint=handoff.model_fingerprint,
        prompt_token_ids=handoff.prompt_token_ids,
        first_token=handoff.first_token,
        block_size=handoff.block_size,
        blocks=blocks,
    )

    with pytest.raises(IncompleteHandoffError, match="block 0"):
        malformed.validate(expected_fingerprint=CONFIG.fingerprint)


def test_transports_satisfy_protocol(tmp_path):
    assert isinstance(InMemoryTransport(), KVTransport)
    assert isinstance(DirectoryTransport(tmp_path), KVTransport)


def test_receive_claims_exactly_once(transport):
    handoff = make_handoff()
    sent = transport.send(handoff)

    assert sent == len(handoff.to_bytes())
    assert transport.receive("r") == handoff
    assert transport.receive("r") is None


def test_duplicate_pending_send_is_rejected(transport):
    transport.send(make_handoff())

    with pytest.raises(DuplicateHandoffError):
        transport.send(make_handoff())
    # The original is untouched and still claimable.
    assert transport.receive("r") == make_handoff()


def test_discard_cancels_pending_handoff(transport):
    transport.send(make_handoff())

    assert transport.discard("r") is True
    assert transport.discard("r") is False
    assert transport.receive("r") is None


def test_receive_missing_returns_none(transport):
    assert transport.receive("never-sent") is None


def test_directory_corrupt_file_is_removed_after_failed_receive(tmp_path):
    transport = DirectoryTransport(tmp_path)
    transport.send(make_handoff())
    path = transport.path_for("r")
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(CorruptHandoffError):
        transport.receive("r")
    assert list(tmp_path.iterdir()) == []


def test_directory_paths_stay_inside_directory(tmp_path):
    transport = DirectoryTransport(tmp_path)

    assert transport.path_for("../../etc/passwd").parent == tmp_path


def test_directory_shared_between_transport_objects(tmp_path):
    # Two objects on one directory stand in for two processes.
    DirectoryTransport(tmp_path).send(make_handoff())

    assert DirectoryTransport(tmp_path).receive("r") == make_handoff()


def test_concurrent_receivers_claim_once(tmp_path):
    DirectoryTransport(tmp_path).send(make_handoff())
    results: list[object] = []
    barrier = threading.Barrier(8)

    def claim() -> None:
        receiver = DirectoryTransport(tmp_path)
        barrier.wait()
        results.append(receiver.receive("r"))

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(result is not None for result in results) == 1
    assert list(tmp_path.iterdir()) == []
