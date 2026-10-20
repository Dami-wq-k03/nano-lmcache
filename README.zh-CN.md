<!--
Chinese translation of README.md
English source commit: 2aa6d0ae92b07eb9c8b0053e6a9a9c51df77ba40
-->

<p align="center">
  <img src="logo/nano-lmcache-logo.svg" alt="nano-lmcache 标志" width="420">
</p>

# nano-lmcache

一个循序渐进的 KV 缓存复用教学项目，从纯 Python 原型逐步发展到真实 GPU 张量和 vLLM 集成。

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
```

使用真实模型（需要执行 `uv sync --extra hf`）：

```bash
uv run python examples/demo_kv_reuse.py --model Qwen/Qwen3-0.6B
uv run python examples/demo_kv_reuse.py --device cuda --offload
uv run python examples/demo_kv_reuse.py --device cuda --offload --disk /tmp/kv
```

前缀缓存示例会使用同一个缓存处理四个请求，并输出每个分块的命中或未命中情况，以及累计复用指标。

## 路线图

1. **Nano（已完成）**：Token 分块、感知前缀的分块键、内存后端、连续前缀查询，以及命中/未命中指标。
2. **Mini（进行中）**：存储接口和有界且保持前缀安全的淘汰机制已经就绪；磁盘和分层后端即将实现。
3. **GPU 和 vLLM**：使用真实模型验证真实 KV 张量、GPU 到 CPU 的卸载，以及 vLLM KV 连接器。

完整任务列表请参阅 [ROADMAP.md](ROADMAP.md)。欢迎贡献者认领开放任务。

## 项目范围

本项目优先考虑代码的清晰性和可测试性，而不是生产环境中的性能。分布式缓存、GPU 专项优化和推理引擎集成会在核心概念建立之后逐步引入。

## 贡献与许可证

欢迎贡献。请从 [ROADMAP.md](ROADMAP.md) 中选择任务，并阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献流程。nano-lmcache 使用 MIT 许可证发布，详情请参阅 [LICENSE](LICENSE)。