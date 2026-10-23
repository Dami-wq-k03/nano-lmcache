# INT8 KV compression experiment

`QuantizedTensorSerializer` is an optional, lossy disk serializer. This
experiment measures its storage and serialization costs, then checks how
quantized prefix KV affects a tiny model's logits and greedy choices.
See [Architecture](architecture.md#optional-int8-kv-compression) for the
scaling rule, usage example, and when to consider enabling it.

## Reproduce on CPU

```bash
uv sync --locked --group tensor-test
uv run --locked --group tensor-test python benchmarks/bench_quantized_kv.py
```

The default runs without a GPU or model download: it constructs a tiny,
randomly initialized Qwen3 model with seed 0. The JSON output includes
sizes, median encode/decode times, tensor errors, and model comparisons.
For an already downloaded model, use `--model /path/to/model`; the script
uses `local_files_only=True`. Use `--device cuda` only with a compatible
GPU environment. Evaluate actual target workloads separately before
adopting lossy storage.

## Measurement conditions

Measured on 2026-10-23 using Apple M4, Darwin 27.0.0 arm64, Python 3.11.12,
PyTorch 2.14.1, and Transformers 5.20.2. The device was CPU, with one
PyTorch thread, five warmup iterations, and 30 timed iterations per
operation. Times below are medians and are specific to this machine.
The serializer and benchmark source were commit `23d7633`.

[Raw results](benchmarks/quantized-kv-apple-m4-cpu.json) record the full
precision numbers and environment. These are CPU measurements; the
benchmark's synthetic chunk and tiny model are separate experiments.

## Serialized size and latency

The synthetic chunk has shape `[28, 2, 8, 16, 128]`, generated with seed 0.
Raw bytes count tensor elements only; serialized sizes include the
`torch.save` container. Compression ratio is raw bytes divided by INT8
serialized bytes. Encode/decode times measure in-memory serialization and
deserialization, excluding `DiskKVCache` file I/O and model execution.

| Dtype | Raw bytes | Lossless bytes | INT8 bytes | Raw / INT8 | INT8 encode ms | INT8 decode ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 1,835,008 | 1,836,585 | 1,148,773 | 1.597x | 0.728 | 0.290 |
| BF16 | 1,835,008 | 1,836,585 | 1,148,773 | 1.597x | 0.740 | 0.360 |

The scale tensor is FP32, so the result is about 1.60x rather than 2x
smaller than the original FP16/BF16 payload. Changing the token count
changes the scale overhead per value. A smaller payload alone does not
show that end-to-end inference is faster.

The synthetic chunk's reconstruction errors are:

| Dtype | Maximum absolute error | Relative L2 error |
| --- | ---: | ---: |
| FP16 | 0.018066406 | 0.004701464 |
| BF16 | 0.017578125 | 0.004957295 |

Relative L2 means `norm(restored - original) / norm(original)`, computed
in FP32.

## G1 comparison: full recompute versus disk-backed reuse

The tiny Qwen3 has three layers, hidden size 32, two KV heads, and head
dimension 8. It warms a 12-token shared prefix, then reuses that prefix
for a 15-token prompt with a different three-token tail. Chunk size is
four. The disk cache is reopened before reuse to exercise persisted
payloads. The full 15-token forward pass is baseline A; B uses lossless
KV and C uses INT8 KV. The logit comparison covers the three computed
suffix positions.

| Dtype | Reuse format | Max K error | Max V error | Max logit error vs A | Logit relative L2 vs A | Greedy agreement vs A |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | Lossless reuse | 0 | 0 | 0 | 0 | 3/3 |
| FP16 | INT8 reuse | 0.009765625 | 0.0014038086 | 0.00048828125 | 0.0013924002 | 3/3 |
| BF16 | Lossless reuse | 0 | 0 | 0 | 0 | 3/3 |
| BF16 | INT8 reuse | 0.01171875 | 0.001953125 | 0.001953125 | 0.0050682644 | 3/3 |

K/V errors compare restored cached chunks against the original model KV.
Logit errors compare suffix logits against the matching positions in the
full recompute. Both INT8 cases also match the next token predicted at
the final position.

These three greedy comparisons use the fixed prompt's suffix positions;
they are not a three-token autoregressive generation test. The model has
random weights, and the prompt is short. In other environments even
lossless reuse can produce slightly different low-precision logits
because full-prefill and suffix execution use different computation
shapes. The exact lossless numbers above describe this run.

## How to judge the trade-off

Quantization changes K/V and can change logits, selected tokens, and
later generated text. It remains disabled by default because no single
error threshold establishes acceptable behavior for every model or task.
Use lossless `TensorSerializer` when exact payload preservation is needed.

Before enabling INT8, define acceptable logits error, token agreement,
and task-quality thresholds for the actual model and workload. Compare
full recompute, lossless reuse, and INT8 reuse on representative prompts
and generation lengths; measure total inference latency as well as
serialization cost. A local 3/3 greedy match is evidence for that sample,
not a guarantee about other inputs or long generations.

Exercise: change the token dimension in `bench_tensor` in
`benchmarks/bench_quantized_kv.py` to 4, 16, and 32 and rerun the experiment.
Predict the storage ratio using `2 / (1 + 4 / tokens)` before counting
serialized bytes. Explain why container overhead and scale storage keep
the measured ratio below the idealized ratio.
