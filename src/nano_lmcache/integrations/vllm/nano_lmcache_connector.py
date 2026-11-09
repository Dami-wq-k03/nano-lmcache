"""Scheduler prefix lookup with lifecycle tracing for vLLM 0.33.0.

Import this module in the separate vLLM environment. Worker hooks still
perform no KV loads or saves: positive lookup results are for controlled
scheduler validation until the worker load path is implemented.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.tensor.hf import namespace_for

if TYPE_CHECKING:
    import torch
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

# vLLM configures handlers only for its own logger hierarchy. Use a child so
# external-connector INFO traces inherit its level, formatter and handlers.
logger = init_logger(f"vllm.{__name__}")


@dataclass
class NanoLMCacheConnectorMetadata(KVConnectorMetadata):  # type: ignore[misc]
    """Serializable trace context; contains no KV load or save plans."""

    step_id: int = 0
    request_ids: tuple[str, ...] = ()


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
    """Scheduler prefix lookup and no-op worker hooks with lifecycle tracing."""

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
        self._nano_lmcache_engine = ChunkedPrefixCache(
            chunk_size=self._block_size,
            namespace="vllm"
        )

    # Worker-side methods.
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        _log_call(
            self,
            "register_kv_caches",
            layers={
                name: (tuple(tensor.shape), str(tensor.dtype), str(tensor.device))
                for name, tensor in kv_caches.items()
            },
        )
        super().register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
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
        # Log against the outgoing step, then discard both base and trace state.
        _log_call(self, "clear_connector_metadata")
        super().clear_connector_metadata()
        self._trace_metadata = None

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

    def wait_for_save(self) -> None:
        _log_call(self, "wait_for_save")

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
        super().shutdown()

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
            result = self._nano_lmcache_engine.lookup(
                token_ids, request_id=request_id
            )
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
        self._step_sequence += 1
        metadata = NanoLMCacheConnectorMetadata(
            step_id=self._step_sequence,
            request_ids=tuple(sorted(scheduler_output.num_scheduled_tokens)),
        )
        self._trace_metadata = metadata
        _log_call(
            self,
            "build_connector_meta",
            num_scheduled_tokens=dict(scheduler_output.num_scheduled_tokens),
        )
        # Other scheduler callbacks need not belong to the step just built.
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
        result: tuple[bool, dict[str, Any] | None] = super().request_finished(
            request, block_ids
        )
        return result
