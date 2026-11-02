<p align="center">
  <img src="logo/nano-lmcache-logo.svg" alt="nano-lmcache 标志" width="420">
</p>

[English](README.md) | 简体中文

本文同步自英文版 [README.md](README.md) commit `479aef1`（2026-11-02，含 VIS-10 更新），内容以英文版为准。架构说明见 [docs/zh-CN/architecture.md](docs/zh-CN/architecture.md)。

# nano-lmcache

一个循序渐进的 KV 缓存复用教学项目，从纯 Python 原型逐步发展到真实 GPU 张量和 nano-vLLM 外部缓存集成。

nano-lmcache 是一个教学项目，而不是 [LMCache](https://github.com/LMCache/LMCache) 的重新实现或替代品。它将核心概念逐一重新实现，每个概念都配有测试和可运行示例，帮助学习者日后阅读 LMCache 和 vLLM 等生产级系统，并理解它们的设计原因。

## 学习目标

- 理解 KV 缓存复用和前缀匹配。
- 实现缓存键生成和存储抽象。
- 先探索内存缓存，再逐步引入更复杂的后端。
- 逐步实现与真实大语言模型推理引擎的集成。

## 开发环境

要求：Python 3.11 或更高版本，以及 uv。

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src/nano_lmcache
```

## 试运行

```bash
uv run python examples/demo_chunking.py       # 单独演示分块键
uv run python examples/demo_prefix_cache.py   # 完整缓存流程和指标
uv run python examples/demo_disk_cache.py     # 连续运行两次，验证重启后的复用
uv run python examples/demo_tiered_storage.py # 内存层位于磁盘层之前
uv run python -m nano_lmcache.inspect /path/to/cache  # 只读检查磁盘缓存
uv run python -m nano_lmcache.inspect /path/to/cache --json
```

使用真实模型（需要执行 `uv sync --extra hf`）：

```bash
uv run python examples/demo_kv_reuse.py --model Qwen/Qwen3-0.6B
uv run python examples/demo_kv_reuse.py --device cuda --offload
uv run python examples/demo_kv_reuse.py --device cuda --offload --disk /tmp/kv
```

作为 nano-vLLM 的外部 KV 缓存（需要 Linux、NVIDIA GPU 和 `uv sync --extra hf --group nanovllm`）：

```bash
uv run python examples/demo_nanovllm.py --model /path/to/Qwen3-0.6B  # 连续运行两次
```

记录真实引擎运行过程，并生成离线 HTML：

```bash
uv run python examples/demo_nanovllm.py --model /path/to/Qwen3-0.6B \
  --trace-output engine.json
uv run python -m nano_lmcache.visualize \
  --trace-json engine.json --output engine-trace.html
```

页面展示 L1/L2 复用、GPU 导入/导出、各存储层事件和实测单 Token TTFT。两次运行使用同一 `--dir`，即可观察新进程从磁盘恢复 KV。JSON 转 HTML 无需 PyTorch 或 GPU；采集真实引擎 Trace 才需要 GPU 环境。事件只保存元数据，不保存 Prompt、Token ID 或 KV Tensor。GPU 拷贝事件表示入队，TTFT 包含 Trace 开销和同步磁盘写入，并非独立的缓存加速基准。完整实验和事件语义见 [真实引擎 Trace 指南](docs/engine-traces.md)。

前缀缓存示例会使用同一个缓存处理四个请求，并输出每个分块的命中或未命中情况，以及累计复用指标。

也可以直接回放可控工作负载并生成离线 Trace 可视化页面：

```bash
uv run python -m nano_lmcache.visualize \
  --scenario partial_prefix_hit \
  --output trace.html
```

生成的 HTML 不依赖 CDN 或常驻服务，包含 Chunk 状态、Prefix Key、Prefix Tree、事件详情和逐步播放控制。FreeInference JSONL 会使用独立的 opaque block replay，不会把数据集中的 Block ID 当作模型 Token ID：

```bash
uv run python -m nano_lmcache.visualize \
  --freeinference-jsonl sample.jsonl \
  --limit 100 \
  --output freeinference.html
```

## 路线图

1. **Nano（已完成）**：Token 分块、感知前缀的分块键、内存后端、连续前缀查询，以及命中/未命中指标。
2. **Mini（已完成）**：存储接口、有界且保持前缀安全的淘汰机制、磁盘后端和内存/磁盘分层后端。
3. **GPU（已完成）**：通过真实模型验证 KV 张量复用、GPU 到 CPU 的卸载，以及 GPU → CPU → Disk 存储路径。
4. **nano-vLLM 集成（V1 已完成，后续优化开放）**：作为 GPU L1 之后的外部 L2 缓存。[集成设计](docs/nanovllm-integration.md)和[性能实验](docs/nanovllm-benchmark.md)分别解释正确性约束和缓存何时值得使用。
5. **Visualization Lab（进行中）**：Trace、Synthetic/FreeInference Workload Replay、离线 HTML 播放、Storage 事件，以及真实 nano-vLLM 的 L1/L2 复用和 GPU 传输 Trace。

初学者可以从[术语表和 FAQ](docs/glossary.md)开始。

完整任务列表请参阅 [ROADMAP.md](ROADMAP.md)。欢迎贡献者认领开放任务。

## 项目范围

本项目优先考虑代码的清晰性和可测试性，而不是生产环境中的性能。分布式缓存、GPU 专项优化和推理引擎集成会在核心概念建立之后逐步引入。

## 贡献与许可证

欢迎贡献。请从 [ROADMAP.md](ROADMAP.md) 中选择任务，并阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献流程。nano-lmcache 使用 MIT 许可证发布，详情请参阅 [LICENSE](LICENSE)。
