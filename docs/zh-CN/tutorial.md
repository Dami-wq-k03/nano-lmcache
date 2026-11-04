[English](../tutorial/README.md) | 简体中文

# 从分块键到引擎恢复

按顺序完成以下步骤。每步都包含小实验、预期现象、源码与测试入口。
在仓库根目录执行 `uv sync --locked` 后开始。第 1–5 步只需核心包；
第 6 步需要模型权重和可选依赖；第 7 步需要 Linux 与 NVIDIA GPU；第 8 步只需核心包。

| 步骤 | 要回答的问题 |
| --- | --- |
| [1. 分块键](#1-生成分块键) | 为什么键必须依赖完整前缀？ |
| [2. 前缀匹配](#2-在首次未命中时停止) | 首次未命中和不完整分块会怎样处理？ |
| [3. 淘汰](#3-分叉前缀与叶节点淘汰) | 有界缓存如何保证共享前缀仍可访问？ |
| [4. 分层存储](#4-持久化与提升分块) | 重启后什么会保留，命中来自哪一层？ |
| [5. Trace 查看器](#5-在查看器中跟踪查询) | 哪些分块命中、未命中或没有检查？ |
| [6. 真实张量](#6-复用真实模型张量) | 复用是否保持模型输出，是否节省时间？ |
| [7. 引擎恢复](#7-恢复真实引擎的缓存) | 外部 L2 块如何回到引擎的 GPU L1？ |
| [8. PD 分离](#8-拆分-prefill-与-decode) | Prefill Worker 要交出什么，Decode Worker 才能接着生成？ |

## 1. 生成分块键

```bash
uv run python examples/demo_chunking.py
uv run pytest tests/unit/test_chunking.py
```

示例以大小为 2 的分块比较两个请求。前两块相同，因此汇总显示共享两块、可复用四个 Token。
这个入门示例比较的是独立分块的哈希；真正的缓存还会把上一块的键作为 `prefix_key` 传入。

实验：通过 `uv run python` 打开 Python，比较不同前缀后的同一后缀：

```python
from nano_lmcache.core.chunking import make_chunk_key

a = make_chunk_key([10, 20], namespace="lesson")
b = make_chunk_key([99, 20], namespace="lesson")
assert make_chunk_key([30, 40], namespace="lesson", prefix_key=a) != make_chunk_key(
    [30, 40], namespace="lesson", prefix_key=b
)
```

前面的 Token 不同，同一后缀就不能共享 KV。命名空间还用于区分不兼容的模型和缓存配置；
通用缓存需要调用方正确选择它。

阅读[键生成过程](architecture.md#如何生成分块键)与
[`core/chunking.py`](../../src/nano_lmcache/core/chunking.py)。
`test_chunk_key_depends_on_prefix` 覆盖这个实验。

## 2. 在首次未命中时停止

```bash
uv run python examples/demo_prefix_cache.py
uv run pytest tests/unit/test_prefix_cache.py
```

示例中的无界缓存从冷启动开始，随后复用共享前缀。查询从根开始，首次缺失即停止。
后续完整分块虽然没有被探测，仍然在复用指标中计为未命中。

实验：在 `uv run python` 中运行：

```python
from nano_lmcache.core.prefix_cache import ChunkedPrefixCache

cache = ChunkedPrefixCache(chunk_size=2, namespace="lesson")
cache.store([10, 20, 30, 40, 50], ["kv-0", "kv-1"])
assert cache.lookup([10, 20, 30, 40, 50]).matched_tokens == 4
assert cache.lookup([10, 20, 99, 40, 50]).matched_tokens == 2
assert cache.lookup([99, 20, 30, 40, 50]).matched_tokens == 0
```

末尾 Token 50 构成不完整分块，不存储也不复用。修改第一块后，即使后续块的 Token 相同，
也不能复用它们。

阅读[查询流程](architecture.md#lookuptoken_ids)与
[`core/prefix_cache.py`](../../src/nano_lmcache/core/prefix_cache.py)。
测试 `test_lookup_stops_when_first_chunk_misses` 和
`test_incomplete_trailing_chunk_is_not_reused` 覆盖这两个边界。

## 3. 分叉前缀与叶节点淘汰

前缀缓存示例的有界部分说明普通逐条目 LRU 的问题：删除父节点会使全部已存后代不可访问。
因此只能淘汰叶节点，并在叶节点中优先淘汰最近最少使用的一个。

```bash
uv run python -m nano_lmcache.visualize --scenario prefix_branching --max-entries 3 \
  --output /tmp/nano-branching.html
uv run python -m nano_lmcache.visualize --scenario lru_eviction --max-entries 2 \
  --output /tmp/nano-eviction.html
uv run pytest tests/unit/test_bounded_memory.py tests/unit/test_leaf_eviction.py tests/unit/test_byte_capacity.py
```

打开 HTML。分叉场景保留公共祖先、替换不同分支的叶节点；淘汰场景重新访问一个已被两条目容量缓存淘汰的根。

实验：把分叉命令改成 `--max-entries 2`。共享的前两块可以容纳，第三块均无法接纳，
但被拒绝的写入不能破坏已有前缀。再比较示例中的字节预算：一个较大载荷可能迫使缓存淘汰多个叶节点。

阅读[容量与淘汰](architecture.md#容量与淘汰)、
[`storage/eviction.py`](../../src/nano_lmcache/storage/eviction.py)和
[性能报告（英文）](../eviction-performance.md)。随机叶节点测试将堆实现与原始 LRU 策略逐步对照。

## 4. 持久化与提升分块

第一次使用新目录，第二次复用同一目录：

```bash
uv run python examples/demo_disk_cache.py --dir /tmp/nano-learning-kv
uv run python examples/demo_disk_cache.py --dir /tmp/nano-learning-kv
uv run python -m nano_lmcache.inspect /tmp/nano-learning-kv
uv run python -m nano_lmcache.inspect /tmp/nano-learning-kv --json
uv run python examples/demo_tiered_storage.py
uv run pytest tests/unit/test_disk_cache.py tests/unit/test_tiered_storage.py tests/unit/test_thread_safe_storage.py
```

全新目录第一次显示 `Reused 0/3 chunks`，第二个进程显示 `Reused 3/3 chunks`。
检查器只读持久化元数据，不更新最近使用时间，也不修复文件。
分层示例展示内存命中、磁盘命中、提升，以及内存为空的模拟重启。

实验：换一个新目录，两次磁盘命令都加上 `--max-bytes 400`。
容量包含头部和序列化载荷，因此可能只容纳前缀。若给已有目录设置低于现有内容大小的预算，
打开时会报错，而不是自动淘汰有效文件。

阅读[磁盘存储](architecture.md#磁盘存储)、[分层存储](architecture.md#分层存储)与
[线程安全](architecture.md#线程安全)。共享的内存、磁盘和分层实例会串行执行后端操作；
整个多分块查询不是快照，不同对象或进程也不能并发写同一磁盘目录。

## 5. 在查看器中跟踪查询

```bash
uv run python -m nano_lmcache.visualize --scenario partial_prefix_hit \
  --output /tmp/nano-partial.html
uv run python -m nano_lmcache.visualize --scenario incomplete_chunk \
  --output /tmp/nano-incomplete.html
uv run pytest tests/unit/test_replay.py tests/unit/test_trace.py tests/unit/test_engine_trace.py
```

直接用浏览器打开。部分前缀场景的第二个请求命中两块、第三块未命中；冷请求在首次未命中后停止探测，
后续完整块标记为 `NOT_CHECKED`。不完整分块场景的末尾 Token 标记为 `INCOMPLETE`。
用 Next/Previous 将时间线与前缀树、事件详情对照。

实验：改成 `--scenario full_prefix_hit` 比较第二个请求。
再添加 `--storage tiered --storage-dir /tmp/nano-learning-trace` 查看具名存储层事件；冷启动请使用新目录。

已有 FreeInference JSONL 文件时，还可以运行：

```bash
uv run python -m nano_lmcache.visualize --freeinference-jsonl /path/to/sample.jsonl --limit 100 \
  --output /tmp/nano-freeinference.html
```

其中 Block ID 是不透明标识，不是模型 Token ID。数据集中观测到的缓存 Token 与模拟命中分别统计。
当前回放和 HTML 在内存中累积；`--limit` 限制选中的请求数，不是进程总内存上限。
后续工作见[路线图（英文）](../../ROADMAP.md)的 VIS-11。

阅读 [`replay/runner.py`](../../src/nano_lmcache/replay/runner.py)与
[`FreeInferenceAdapter`](../../src/nano_lmcache/datasets/adapters/freeinference.py)。
回放测试覆盖首次未命中即停止、前缀分叉和不透明块处理。

## 6. 复用真实模型张量

除非 `--model` 指向本地目录，否则这一步需要下载模型权重。安装可选 `hf` 依赖后可在 CPU 上运行：

```bash
uv sync --extra hf
uv run python examples/demo_kv_reuse.py --model Qwen/Qwen3-0.6B
uv run --extra hf pytest tests/tensor/test_hf_reuse.py tests/tensor/test_kv_chunks.py
```

Prompt B 复用 Prompt A 处理文档时保存的前缀。输出包括复用/计算的 Token 数、缓存耗时、
从头计算耗时和最大 Logit 差异。无损存储保持已有 KV，但不同浮点执行路径仍可能产生数值差异。
查询、重建和存储都有成本，因此命中不一定更快。

NVIDIA GPU 实验：比较直接保留在 GPU 与 CPU 卸载，再用同一目录运行两次磁盘版本：

```bash
uv run --extra hf python examples/demo_kv_reuse.py --device cuda
uv run --extra hf python examples/demo_kv_reuse.py --device cuda --offload
uv run --extra hf python examples/demo_kv_reuse.py --device cuda --offload \
  --disk /tmp/nano-learning-tensors
```

阅读[真实张量](architecture.md#真实-kv-张量)、[CPU 卸载](architecture.md#gpu-到-cpu-的卸载)与
[`tensor/hf.py`](../../src/nano_lmcache/tensor/hf.py)。有损压缩另见
[INT8 实验（英文）](../quantized-kv.md)。更换序列化器或不兼容的模型配置时，应使用不同缓存目录。

## 7. 恢复真实引擎的缓存

要求：Linux、NVIDIA GPU、Python 3.11 或 3.12，以及本地 Qwen3-0.6B 模型。
第一条引擎命令使用新磁盘目录：

```bash
uv sync --extra hf --group nanovllm
uv run python examples/demo_nanovllm.py --model /path/to/Qwen3-0.6B \
  --dir /tmp/nano-learning-engine --trace-output /tmp/nano-cold.json
uv run python examples/demo_nanovllm.py --model /path/to/Qwen3-0.6B \
  --dir /tmp/nano-learning-engine --trace-output /tmp/nano-restart.json
uv run python -m nano_lmcache.visualize --trace-json /tmp/nano-restart.json \
  --output /tmp/nano-engine.html
```

第二个进程以空 GPU L1 和空内存 L2 启动，从磁盘恢复共享前缀并导入引擎 KV 块。
同一进程的后续请求可复用 GPU L1。示例的快速 L2 是 `MemoryKVCache`，
张量保留在原有设备上，并不是显式的页锁定 CPU 卸载层。

实验：对比冷启动和重启 Trace，区分 `l2_lookup` 可用性探测与真正的 `l2_hit`/`l2_restore`。
GPU 复制事件表示入队顺序，不是 CUDA 完成时间。即使最后一个 Prompt 块完整，引擎仍需计算它以获取 Logits；
仅用通用缓存的不完整分块规则不能解释这一行为。

在同一 GPU 环境执行真实模型检查：

```bash
NANO_LMCACHE_TEST_MODEL=/path/to/Qwen3-0.6B uv run \
  --group nanovllm pytest tests/integration/test_nanovllm_engine.py
```

测试覆盖 KV 恢复、生成 Token 一致、进程重启后的磁盘复用和 Trace 转 HTML。
没有可选引擎依赖时，CPU 上仍可运行
[`test_engine_trace.py`](../../tests/unit/test_engine_trace.py) 验证导入与渲染。

阅读[集成规则（英文）](../nanovllm-integration.md#rules-the-integration-must-keep)和
[引擎 Trace 指南（英文）](../engine-traces.md)。性能比较应使用受控
[TTFT 基准（英文）](../nanovllm-benchmark.md)：示例 TTFT 包含 Trace 开销和同步写入，并非单独的缓存加速测量。

[GPU 验证记录（英文）](../gpu-validation.md)记录了上述示例的实际运行结果，包含跨进程恢复和 Trace 渲染。

## 8. 拆分 Prefill 与 Decode

```bash
uv run python examples/demo_pd_disaggregation.py
uv run pytest tests/unit/test_pd_handoff.py tests/unit/test_pd_workers.py \
  tests/unit/test_pd_simulation.py
```

本步只需核心包。Prefill Worker 计算 Prompt 的 KV 和第一个 Token，通过共享目录发送 `KVHandoff`；
Decode Worker 认领后生成其余 Token。示例会显示合并与分离两种方式输出完全一致，
且 Decode Worker 只计算新增位置；接着演示四种拒绝：字节损坏、来自不同权重的 KV、请求已取消、
第二个 Worker 认领已被认领的请求。最后的时间线对比单 GPU 与“Prefill GPU + Decode GPU”：
合并部署下长 Prompt 让正在运行的 chat 请求卡顿两秒；分离后卡顿消失，但每个 TTFT 都包含 KV 传输。

实验：把 27 个 Token 的 Prompt 被拆成 7 块这一点，与第 2 步前缀缓存的规则对比。
交接会发送不完整的最后一块，因为 Decode Worker 无法自行重算它。然后像示例第二部分那样给
`PrefillWorker` 一个 `ChunkedPrefixCache`：前缀复用减少了 Prefill 计算量，交接内容却不变。

阅读 [PD 分离指南](pd-disaggregation.md)，了解交接必须遵守的七条规则和本教学版没有覆盖的内容，
再阅读 [`disaggregation/`](../../src/nano_lmcache/disaggregation/)。

返回[文档目录](README.md)，或从[路线图（英文）](../../ROADMAP.md)选择贡献任务。
