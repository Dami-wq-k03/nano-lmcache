# mypy: disable-error-code="unused-ignore"
"""Scheduler prefix lookup with lifecycle tracing for vLLM 0.33.0.

Import this module in the separate vLLM environment. Workers save complete
CPU chunks from scheduler-generated prompt save plans. Loading is not
implemented; positive lookup results remain for controlled scheduler
validation. Save planning supports one full-attention KV cache group.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

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
    from vllm.v1.core.sched.output import SchedulerOutput
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
class ReqMeta:
    """A complete prompt prefix that will be valid after this step's forward."""

    request_id: str
    token_ids: tuple[int, ...]
    slot_mapping: tuple[int, ...]


@dataclass
class _SaveRequestState:
    prompt_token_ids: tuple[int, ...]
    block_ids: list[int]


@dataclass
class NanoLMCacheConnectorMetadata(KVConnectorMetadata):  # type: ignore[misc]
    step_id: int = 0
    request_ids: tuple[str, ...] = ()
    requests: tuple[ReqMeta, ...] = ()


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
    """Scheduler lookup and worker chunk saving with lifecycle tracing."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._call_sequence = 0
        self._step_sequence = 0
        self._trace_metadata: NanoLMCacheConnectorMetadata | None = None
        _log_call(self, "__init__", num_blocks=kv_cache_config.num_blocks)

        self._block_size = vllm_config.cache_config.block_size
        self._pending_save_layers: dict[str, dict[str, LayerKV]] = {}
        self._save_requests: dict[str, _SaveRequestState] = {}
        self._layer_names: list[str] = []
        self._nano_lmcache_engine = ChunkedPrefixCache(
            chunk_size=self._block_size, namespace=_cache_namespace(vllm_config)
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
        """Save complete all-layer chunks from a computed prompt prefix."""
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

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, NanoLMCacheConnectorMetadata)

        if not connector_metadata.requests:
            return

        expected_layers = set(self._layer_names)
        if not expected_layers:
            raise RuntimeError("KV cache layers must be registered before saving")
        if layer_name not in expected_layers:
            raise ValueError(f"Cannot save unregistered KV layer: {layer_name}")

        chunk_size = self._nano_lmcache_engine.chunk_size
        head_dim = kv_layer.shape[-1] // 2
        for request in connector_metadata.requests:
            num_tokens = len(request.token_ids) // chunk_size * chunk_size
            if num_tokens == 0:
                continue
            slot_ids = request.slot_mapping[:num_tokens]

            with gpu_sync_allowed():
                slots = torch.tensor(slot_ids, device=kv_layer.device)
                block_idxs = slots // self._block_size
                offsets = slots % self._block_size
                selected = kv_layer[block_idxs, :, offsets]
                kv_cache = selected.detach().cpu()

            keys = kv_cache[..., :head_dim].permute(1, 0, 2)
            values = kv_cache[..., head_dim:].permute(1, 0, 2)
            layers = self._pending_save_layers.setdefault(request.request_id, {})
            layers[layer_name] = (keys, values)

            if layers.keys() == expected_layers:
                ordered_layers = [layers[name] for name in self._layer_names]
                chunks = split_kv(ordered_layers, chunk_size)
                self._nano_lmcache_engine.store(
                    request.token_ids[:num_tokens],
                    chunks,
                    request_id=request.request_id,
                )

    def wait_for_save(self) -> None:
        _log_call(self, "wait_for_save")
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
        chunk_size = self._nano_lmcache_engine.chunk_size

        external_prefix_tokens = 0
        matched_end = 0
        additional_matched_tokens = 0
        reason = "no_prompt_token_ids"

        if token_ids:
            result = self._nano_lmcache_engine.lookup(token_ids, request_id=request_id)
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
        for request_id in scheduler_output.finished_req_ids:
            self._save_requests.pop(request_id, None)

        computed_tokens: dict[str, int] = {}
        for request in scheduler_output.scheduled_new_reqs:
            self._save_requests[request.req_id] = _SaveRequestState(
                prompt_token_ids=tuple(request.prompt_token_ids or ()),
                block_ids=list(request.block_ids[0]),
            )
            computed_tokens[request.req_id] = request.num_computed_tokens

        cached = scheduler_output.scheduled_cached_reqs
        for index, request_id in enumerate(cached.req_ids):
            state = self._save_requests[request_id]
            new_blocks = cached.new_block_ids[index]
            if request_id in cached.resumed_req_ids:
                assert new_blocks is not None
                state.block_ids = list(new_blocks[0])
            elif new_blocks is not None:
                state.block_ids.extend(new_blocks[0])
            computed_tokens[request_id] = cached.num_computed_tokens[index]

        requests: list[ReqMeta] = []
        chunk_size = self._nano_lmcache_engine.chunk_size
        scheduled = scheduler_output.num_scheduled_tokens
        for request_id, scheduled_tokens in scheduled.items():
            state = self._save_requests[request_id]
            computed = computed_tokens[request_id]
            if scheduled_tokens <= 0 or computed >= len(state.prompt_token_ids):
                continue
            computed_end = min(len(state.prompt_token_ids), computed + scheduled_tokens)
            save_end = computed_end // chunk_size * chunk_size
            if save_end == 0:
                continue

            block_state = scheduler_output.kv_connector_block_state
            if block_state is not None:
                block_ids = block_state.get_block_ids(request_id)
                if block_ids is not None:
                    state.block_ids = list(block_ids[0])

            slots = tuple(
                state.block_ids[position // self._block_size] * self._block_size
                + position % self._block_size
                for position in range(save_end)
            )
            requests.append(
                ReqMeta(request_id, state.prompt_token_ids[:save_end], slots)
            )

        self._step_sequence += 1
        metadata = NanoLMCacheConnectorMetadata(
            step_id=self._step_sequence,
            request_ids=tuple(sorted(scheduler_output.num_scheduled_tokens)),
            requests=tuple(requests),
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
            failed_recving=sorted(connector_output.failed_recving),
            invalid_block_ids=sorted(connector_output.invalid_block_ids),
        )
        super().update_connector_output(connector_output)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        _log_call(
            self, "request_finished", request_id=request.request_id, block_ids=block_ids
        )
        self._save_requests.pop(request.request_id, None)
        result: tuple[bool, dict[str, Any] | None] = super().request_finished(
            request, block_ids
        )
        return result
