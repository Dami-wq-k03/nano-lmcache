# Engine traces

The simplified nano-vLLM integration in `integrations/nanovllm` does not
record trace events. `ExternalBlockManager` restores and stores KV blocks
and reports counters through `EngineCacheStats`; it no longer emits
`request_started`, `l1_hit`, `l2_lookup`, `gpu_import`, `gpu_export`, or
`prefill_step` events. `examples/demo_nanovllm.py` has no `--trace-output`
or `--memory-max-entries` options.

The offline viewer is unchanged. It still renders trace JSON produced by
other sources, such as synthetic `ChunkedPrefixCache` experiments,
bounded workload replay, and the storage/tier tracing wrappers. See
[streaming replay](streaming-replay.md) for the replay path, and
[the integration design](nanovllm-integration.md) for the current
nano-vLLM behavior.
