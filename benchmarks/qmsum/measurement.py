"""Benchmark-only timing hooks; the inference engine itself stays unchanged."""

import threading
import time

import torch

from nano_lmcache.integrations.nanovllm import engine
from nano_lmcache.storage.async_write import AsyncWriteBehindStorage
from nano_lmcache.tensor.offload import CPUOffloadKVCache


class LockedOffload:
    """Serialize backend metadata access with the write-behind worker."""

    def __init__(self):
        self.backend = CPUOffloadKVCache(device="cuda")
        self.lock = threading.Lock()

    def put(self, key, value, *, parent_key=None):
        with self.lock:
            self.backend.put(key, value, parent_key=parent_key)

    def get(self, key):
        with self.lock:
            return self.backend.get(key)

    def __contains__(self, key):
        with self.lock:
            return key in self.backend


def host_snapshot(value):
    # Generic AsyncWriteBehindStorage.prepare is synchronous. This owned,
    # completed snapshot is safe for the CPU worker; it does NOT hide D2H.
    held = torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
    held.copy_(value, non_blocking=False)
    return held


class TimedWriter(AsyncWriteBehindStorage):
    def __init__(self, backend, capacity):
        super().__init__(backend, max_pending=capacity, prepare=host_snapshot)
        self.queue_wait_ms = 0.0

    def flush(self):
        start = time.perf_counter()
        super().flush()
        self.queue_wait_ms += (time.perf_counter() - start) * 1000


def new_storage(mode, capacity):
    backend = LockedOffload()
    return TimedWriter(backend, capacity) if mode.startswith("async") else backend


class Probe:
    """Per-request counters plus CUDA event spans (not extra per-hook syncs)."""

    def __init__(self, llm, external):
        self.llm = llm
        self.external = external
        self.metrics = {}
        self.current = None
        self.events = []
        self.original_import = engine.import_blocks
        self.original_postprocess = llm.scheduler.postprocess

    def span(self, name, action):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        host_start = time.perf_counter()
        result = action()
        host_ms = (time.perf_counter() - host_start) * 1000
        end.record()
        metric = self.metrics[self.current]
        metric[name + "_ms"] += host_ms
        self.events.append((metric, name + "_cuda_ms", start, end))
        return result

    def install(self, manager):
        self.manager = manager
        if self.external is not None:
            original_fetch = self.external.fetch

            def fetch(*args):
                return self.span("fetch", lambda: original_fetch(*args))

            self.external.fetch = fetch
            allocate, hash_blocks = manager.allocate, manager.hash_blocks

            def measured_allocate(seq, cached):
                self.current = seq.seq_id
                before = manager.stats.l2_hit_blocks
                l1_before = manager.stats.l1_hit_blocks
                allocate(seq, cached)
                self.metrics[seq.seq_id].update(
                    l2_hit_blocks=manager.stats.l2_hit_blocks - before,
                    l1_hit_blocks=manager.stats.l1_hit_blocks - l1_before,
                )

            def measured_hash(seq):
                start = seq.num_cached_tokens // 256
                end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // 256
                if start == end:
                    return hash_blocks(seq)
                self.current = seq.seq_id
                before = manager.stats.stored_blocks
                self.span("store", lambda: hash_blocks(seq))
                self.metrics[seq.seq_id]["stored_blocks"] += (
                    manager.stats.stored_blocks - before
                )

            manager.allocate = measured_allocate
            manager.hash_blocks = measured_hash

            # import_blocks processes all scheduled requests together. Do not
            # attribute that entire batch span separately to every request.
            def measured_import(*args):
                self.current = "batch"
                return self.span("import", lambda: self.original_import(*args))

            engine.import_blocks = measured_import

        def postprocess(seqs, tokens, is_prefill):
            for seq in seqs:
                if is_prefill:
                    self.metrics[seq.seq_id]["prefill_tokens"] += (
                        seq.num_scheduled_tokens
                    )
            return self.original_postprocess(seqs, tokens, is_prefill)

        self.llm.scheduler.postprocess = postprocess

    @staticmethod
    def empty():
        return {
            "l1_hit_blocks": 0,
            "l2_hit_blocks": 0,
            "prefill_tokens": 0,
            "stored_blocks": 0,
            "fetch_ms": 0.0,
            "store_ms": 0.0,
            "fetch_cuda_ms": 0.0,
            "store_cuda_ms": 0.0,
            "import_ms": 0.0,
            "import_cuda_ms": 0.0,
        }

    def finish(self):
        # The caller synchronizes at each step boundary, so all spans are ready.
        for metric, name, start, end in self.events:
            metric[name] += start.elapsed_time(end)
        engine.import_blocks = self.original_import
        self.llm.scheduler.postprocess = self.original_postprocess
        if self.external is not None:
            # Break callback cycles so old multi-GB host caches do not wait for GC.
            del self.manager.allocate, self.manager.hash_blocks, self.external.fetch


def replay_wave(llm, requests, params, external, storage=None):
    """All arrivals in a wave share a timestamp; step completion is observable TTFT."""
    from nano_lmcache.integrations.nanovllm.engine import attach, detach

    manager = detach(llm) if external is None else attach(llm, external)
    probe = Probe(llm, external)
    probe.metrics["batch"] = probe.empty()
    probe.install(manager)
    queue_start = getattr(storage, "queue_wait_ms", 0)
    seqs, first, end = [], {}, {}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        for request in requests:
            llm.add_request(request["input_ids"], params)
            seq = llm.scheduler.waiting[-1]
            seqs.append(seq)
            probe.metrics[seq.seq_id] = probe.empty()
        while not llm.is_finished():
            llm.step()
            # Covers stream-enqueued sync-store copies. Does not flush the
            # CPU write-behind queue, including in the continuous condition.
            torch.cuda.current_stream().synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            for seq in seqs:
                if seq.num_completion_tokens:
                    first.setdefault(seq.seq_id, elapsed)
                if seq.is_finished:
                    end.setdefault(seq.seq_id, elapsed)
        wave_ms = (time.perf_counter() - start) * 1000
    finally:
        probe.finish()
    records = []
    for request, seq in zip(requests, seqs, strict=True):
        metrics = probe.metrics[seq.seq_id]
        if metrics["l1_hit_blocks"]:
            raise RuntimeError("L1 contaminated the external-cache experiment")
        if metrics["prefill_tokens"] != (
            request["prompt_tokens"] - metrics["l2_hit_blocks"] * 256
        ):
            raise RuntimeError("prefill accounting mismatch (possibly preemption)")
        record = {
            k: v
            for k, v in request.items()
            if k not in ("input_ids", "reference_answer", "question")
        }
        record.update(
            metrics,
            ttft_ms=first[seq.seq_id],
            e2e_ms=end[seq.seq_id],
            output_token_ids=seq.completion_token_ids,
            output_text=llm.tokenizer.decode(seq.completion_token_ids),
        )
        records.append(record)
    return records, {
        "wave_ms": wave_ms,
        "arrival_monotonic": start,
        "finished_monotonic": start + wave_ms / 1000,
        "batch_import_ms": probe.metrics["batch"]["import_ms"],
        "batch_import_cuda_ms": probe.metrics["batch"]["import_cuda_ms"],
        "queue_wait_ms": getattr(storage, "queue_wait_ms", 0) - queue_start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "arrivals": len(requests),
    }
