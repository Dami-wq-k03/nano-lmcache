# mypy: disable-error-code="unused-ignore"
"""Scheduler prefix lookup and worker chunk saving for vLLM 0.33.0.

Import this module in the separate vLLM environment. The scheduler tracks
each request's prompt, block table and already-published prefix; workers save
only newly completed chunks from the paged KV cache. Loading is not
implemented. Save planning supports one full-attention KV cache group.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.kv_cache_interface import FullAttentionSpec

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.tensor.kv_chunks import LayerKV, split_kv

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorTransferResults,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(f"vllm.{__name__}")


def _cache_namespace(config: VllmConfig) -> str:
    model = config.model_config
    identity = {
        "model": model.model,
        "revision": getattr(model.hf_config, "_commit_hash", None) or model.revision,
        "hf_config": model.hf_config.to_dict(),
        "dtype": str(model.dtype),
        "quantization": model.quantization,
        "kv_dtype": config.cache_config.cache_dtype,
        "chunk_size": config.cache_config.block_size,
        "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
        "pipeline_parallel_size": config.parallel_config.pipeline_parallel_size,
        "layout": "layers,kv,heads,tokens,head_dim/v1",
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return "vllm-kv/v1/" + hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class SaveSpec:
    """How much of a request's prefix is already published."""

    skip_leading_tokens: int = 0
    can_save: bool = True


@dataclass
class RequestTracker:
    """Scheduler-side state kept for one request across scheduling steps.

    The tracker owns the full prompt, the latest block table, and how far
    vLLM has computed and this connector has published. It never leaves the
    scheduler process.
    """

    req_id: str
    prompt_token_ids: tuple[int, ...]
    block_ids: list[int]
    num_computed_tokens: int = 0
    num_saved_tokens: int = 0

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @classmethod
    def from_new_request(cls, request: NewRequestData) -> RequestTracker:
        return cls(
            req_id=request.req_id,
            prompt_token_ids=tuple(request.prompt_token_ids or ()),
            block_ids=list(request.block_ids[0]),
            num_computed_tokens=request.num_computed_tokens,
        )

    def update(
        self,
        *,
        num_computed_tokens: int,
        new_block_ids: tuple[list[int], ...] | None,
        resumed: bool = False,
    ) -> None:
        """Apply the scheduler's latest block table and compute progress."""
        self.num_computed_tokens = num_computed_tokens
        if resumed:
            if new_block_ids is None:
                raise ValueError(f"Resumed request {self.req_id} has no blocks")
            self.block_ids = list(new_block_ids[0])
        elif new_block_ids is not None:
            self.block_ids.extend(new_block_ids[0])


@dataclass(frozen=True)
class ReqMeta:
    """A complete prompt prefix that will be valid after this step's forward."""

    req_id: str
    token_ids: tuple[int, ...]
    slot_mapping: tuple[int, ...]
    save_spec: SaveSpec = field(default_factory=SaveSpec)

    @classmethod
    def from_request_tracker(
        cls,
        tracker: RequestTracker,
        *,
        num_scheduled_tokens: int,
        block_size: int,
        chunk_size: int,
    ) -> ReqMeta | None:
        """Plan the newly completed chunks for one scheduled request."""
        if num_scheduled_tokens <= 0:
            return None
        if tracker.num_computed_tokens >= tracker.prompt_len:
            return None

        compute_end = min(
            tracker.prompt_len,
            tracker.num_computed_tokens + num_scheduled_tokens,
        )
        save_end = compute_end // chunk_size * chunk_size
        skip = tracker.num_saved_tokens // chunk_size * chunk_size
        if save_end == 0 or save_end <= skip:
            return None

        if save_end > len(tracker.block_ids) * block_size:
            raise ValueError(
                f"Not enough KV blocks for {save_end} tokens: "
                f"{len(tracker.block_ids)} blocks of {block_size}"
            )

        slot_mapping = tuple(
            tracker.block_ids[position // block_size] * block_size
            + position % block_size
            for position in range(save_end)
        )
        tracker.num_saved_tokens = save_end

        return cls(
            req_id=tracker.req_id,
            token_ids=tracker.prompt_token_ids[:save_end],
            slot_mapping=slot_mapping,
            save_spec=SaveSpec(skip_leading_tokens=skip),
        )


@dataclass
class NanoLMCacheConnectorMetadata(KVConnectorMetadata):  # type: ignore[misc]
    step_id: int = 0
    request_ids: tuple[str, ...] = ()
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(self, req_meta: ReqMeta) -> None:
        self.requests.append(req_meta)


def _log_call(connector: NanoLMCacheConnector, method: str, **details: Any) -> None:
    """Log entry order per instance without inspecting tensor contents."""
    connector._call_sequence += 1
    metadata = connector._trace_metadata
    logger.info(
        "nano_lmcache pid=%s role=%s call=%s step=%s request_ids=%s method=%s args=%s",
        os.getpid(),
        connector.role.name,
        connector._call_sequence,
        metadata.step_id if metadata is not None else None,
        metadata.request_ids if metadata is not None else (),
        method,
        details,
    )


class NanoLMCacheConnector(KVConnectorBase_V1):  # type: ignore[misc]
    """Scheduler save planning and worker chunk saving with lifecycle tracing."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self._kv_role = vllm_config.kv_transfer_config.kv_role
        self._block_size = vllm_config.cache_config.block_size
        self._chunk_size = self._block_size
        self._namespace = _cache_namespace(vllm_config)
        self._validate_scope(vllm_config, kv_cache_config)

        self._call_sequence = 0
        self._step_sequence = 0
        self._trace_metadata: NanoLMCacheConnectorMetadata | None = None

        # Scheduler-side state.
        self._request_trackers: dict[str, RequestTracker] = {}
        self._lookup_cache: ChunkedPrefixCache | None = None

        # Worker-side state.
        self._layer_names: list[str] = []
        self._pending_save_layers: dict[str, dict[str, LayerKV]] = {}
        self._save_cache: ChunkedPrefixCache | None = None

        # Role-specific resources, mirroring LMCache's lookup client/engine
        # split. Neither cache touches CUDA in the constructor.
        if role == KVConnectorRole.SCHEDULER:
            self._lookup_cache = ChunkedPrefixCache(
                chunk_size=self._chunk_size,
                namespace=self._namespace,
            )
        else:
            self._save_cache = ChunkedPrefixCache(
                chunk_size=self._chunk_size,
                namespace=self._namespace,
            )

        _log_call(self, "__init__", num_blocks=kv_cache_config.num_blocks)

    @staticmethod
    def _validate_scope(
        vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
    ) -> None:
        """Reject configurations whose KV layout V3 cannot save correctly."""
        groups = kv_cache_config.kv_cache_groups
        if len(groups) > 1:
            raise NotImplementedError(
                "nano-lmcache V3 supports one KV cache group, "
                f"got {len(groups)}"
            )
        if groups and not isinstance(groups[0].kv_cache_spec, FullAttentionSpec):
            raise NotImplementedError(
                "nano-lmcache V3 supports full attention only"
            )
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise NotImplementedError(
                "nano-lmcache V3 does not support speculative decoding"
            )

    # Worker-side methods.
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self._layer_names and set(kv_caches) != set(self._layer_names):
            raise ValueError("Registered KV cache layer set must remain unchanged")

        self._layer_names = sorted(
            kv_caches,
            key=lambda name: [
                (1, int(part)) if part.isdigit() else (0, part)
                for part in re.split(r"(\d+)", name)
            ],
        )
        _log_call(
            self,
            "register_kv_caches",
            layers={
                name: (tuple(tensor.shape), str(tensor.dtype), str(tensor.device))
                for name, tensor in kv_caches.items()
            },
        )

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        self._pending_save_layers.clear()
        self._trace_metadata = (
            connector_metadata
            if isinstance(connector_metadata, NanoLMCacheConnectorMetadata)
            else None
        )
        _log_call(
            self,
            "bind_connector_metadata",
            metadata_type=type(connector_metadata).__name__,
        )
        super().bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        _log_call(self, "clear_connector_metadata")
        super().clear_connector_metadata()
        self._trace_metadata = None
        self._pending_save_layers.clear()

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        _log_call(
            self,
            "start_load_kv",
            context_type=type(forward_context).__name__,
            attn_metadata_type=type(forward_context.attn_metadata).__name__,
            kwargs_keys=sorted(kwargs),
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        _log_call(self, "wait_for_layer_load", layer_name=layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Save this layer's newly completed prompt chunks from the paged cache."""
        _log_call(
            self,
            "save_kv_layer",
            layer_name=layer_name,
            shape=tuple(kv_layer.shape),
            dtype=str(kv_layer.dtype),
            device=str(kv_layer.device),
            attn_metadata_type=type(attn_metadata).__name__,
            kwargs_keys=sorted(kwargs),
        )

        if self._kv_role == "kv_consumer":
            return

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, NanoLMCacheConnectorMetadata)

        if not connector_metadata.requests:
            return

        expected_layers = set(self._layer_names)
        if not expected_layers:
            raise RuntimeError("KV cache layers must be registered before saving")
        if layer_name not in expected_layers:
            raise ValueError(f"Cannot save unregistered KV layer: {layer_name}")

        save_cache = self._save_cache
        assert save_cache is not None

        head_dim = kv_layer.shape[-1] // 2
        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if not save_spec.can_save:
                continue

            start = save_spec.skip_leading_tokens
            end = len(request.token_ids) // self._chunk_size * self._chunk_size
            if start >= end:
                continue
            if start % self._chunk_size:
                raise ValueError(
                    f"Unaligned skip_leading_tokens {start} "
                    f"for chunk size {self._chunk_size}"
                )

            slot_ids = request.slot_mapping[start:end]
            with gpu_sync_allowed():
                slots = torch.tensor(slot_ids, device=kv_layer.device)
                block_idxs = slots // self._block_size
                offsets = slots % self._block_size
                selected = kv_layer[block_idxs, :, offsets]
                kv_cache = selected.detach().cpu()

            keys = kv_cache[..., :head_dim].permute(1, 0, 2)
            values = kv_cache[..., head_dim:].permute(1, 0, 2)
            layers = self._pending_save_layers.setdefault(request.req_id, {})
            layers[layer_name] = (keys, values)

            if layers.keys() == expected_layers:
                ordered_layers = [layers[name] for name in self._layer_names]
                chunks = split_kv(ordered_layers, self._chunk_size)
                save_cache.store(
                    request.token_ids[:end],
                    chunks,
                    start_chunk=start // self._chunk_size,
                    request_id=request.req_id,
                )

    def wait_for_save(self) -> None:
        _log_call(self, "wait_for_save")
        if self._kv_role == "kv_consumer":
            return
        for request_id, layers in self._pending_save_layers.items():
            missing = set(self._layer_names) - layers.keys()
            if missing:
                raise RuntimeError(
                    f"Incomplete KV save for {request_id}: "
                    f"missing layers {sorted(missing)}"
                )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        _log_call(self, "get_finished", finished_req_ids=sorted(finished_req_ids))
        result: tuple[set[str] | None, set[str] | None] = super().get_finished(
            finished_req_ids
        )
        return result

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        _log_call(
            self, "get_transfer_results", finished_req_ids=sorted(finished_req_ids)
        )
        # The base calls get_finished() and builds the empty transfer snapshot.
        result: KVConnectorTransferResults = super().get_transfer_results(
            finished_req_ids
        )
        return result

    def get_block_ids_with_load_errors(self) -> set[int]:
        _log_call(self, "get_block_ids_with_load_errors")
        result: set[int] = super().get_block_ids_with_load_errors()
        return result

    def get_kv_connector_kv_cache_events(self) -> None:
        _log_call(self, "get_kv_connector_kv_cache_events")
        return None

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        # Preemption handling may run before metadata is bound on the worker.
        _log_call(
            self,
            "handle_preemptions",
            incoming_step=getattr(kv_connector_metadata, "step_id", None),
            incoming_request_ids=getattr(kv_connector_metadata, "request_ids", ()),
        )
        super().handle_preemptions(kv_connector_metadata)

    def finish_forward(self) -> None:
        _log_call(self, "finish_forward")
        super().finish_forward()

    def shutdown(self) -> None:
        _log_call(self, "shutdown")
        # vLLM 0.33.0's base shutdown() has no type annotations.
        super().shutdown()  # type: ignore[no-untyped-call]

    # Scheduler-side methods.
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Report additional complete chunks beyond the local prefix."""
        token_ids = request.prompt_token_ids
        request_id = request.request_id
        chunk_size = self._chunk_size
        lookup_cache = self._lookup_cache
        assert lookup_cache is not None

        external_prefix_tokens = 0
        matched_end = 0
        additional_matched_tokens = 0
        reason = "no_prompt_token_ids"

        if token_ids:
            result = lookup_cache.lookup(token_ids, request_id=request_id)
            external_prefix_tokens = result.matched_tokens
            max_reusable_tokens = (len(token_ids) - 1) // chunk_size * chunk_size
            matched_end = min(external_prefix_tokens, max_reusable_tokens)
            additional_matched_tokens = max(0, matched_end - num_computed_tokens)
            reason = "lookup"

        _log_call(
            self,
            "get_num_new_matched_tokens",
            request_id=request.request_id,
            prompt_tokens=len(token_ids) if token_ids is not None else 0,
            num_computed_tokens=num_computed_tokens,
            external_prefix_tokens=external_prefix_tokens,
            matched_end=matched_end,
            additional_matched_tokens=additional_matched_tokens,
            chunk_size=chunk_size,
            block_size=self._block_size,
            reason=reason,
            result=(additional_matched_tokens, False),
        )
        return additional_matched_tokens, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        _log_call(
            self,
            "update_state_after_alloc",
            request_id=request.request_id,
            block_ids=blocks.get_block_ids(),
            num_external_tokens=num_external_tokens,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> NanoLMCacheConnectorMetadata:
        for req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(req_id, None)

        for request in scheduler_output.scheduled_new_reqs:
            self._request_trackers[request.req_id] = (
                RequestTracker.from_new_request(request)
            )

        cached = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            self._request_trackers[req_id].update(
                num_computed_tokens=cached.num_computed_tokens[index],
                new_block_ids=cached.new_block_ids[index],
                resumed=req_id in cached.resumed_req_ids,
            )

        requests: list[ReqMeta] = []
        block_state = scheduler_output.kv_connector_block_state
        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            tracker = self._request_trackers[req_id]
            if block_state is not None:
                block_ids = block_state.get_block_ids(req_id)
                if block_ids is not None:
                    tracker.block_ids = list(block_ids[0])

            req_meta = ReqMeta.from_request_tracker(
                tracker,
                num_scheduled_tokens=num_tokens,
                block_size=self._block_size,
                chunk_size=self._chunk_size,
            )
            if req_meta is not None:
                requests.append(req_meta)

        self._step_sequence += 1
        metadata = NanoLMCacheConnectorMetadata(
            step_id=self._step_sequence,
            request_ids=tuple(
                sorted(scheduler_output.num_scheduled_tokens)
            ),
            requests=requests,
        )
        self._trace_metadata = metadata
        _log_call(
            self,
            "build_connector_meta",
            num_scheduled_tokens=dict(scheduler_output.num_scheduled_tokens),
        )
        self._trace_metadata = None
        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        _log_call(
            self,
            "update_connector_output",
            finished_sending=sorted(connector_output.finished_sending or ()),
            finished_recving=sorted(connector_output.finished_recving or ()),
            invalid_block_ids=sorted(connector_output.invalid_block_ids),
        )
        super().update_connector_output(connector_output)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        _log_call(
            self, "request_finished", request_id=request.request_id, block_ids=block_ids
        )
        self._request_trackers.pop(request.request_id, None)
        result: tuple[bool, dict[str, Any] | None] = super().request_finished(
            request, block_ids
        )
        return result