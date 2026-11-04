<p align="center">
  <img src="logo/nano-lmcache-logo.svg" alt="nano-lmcache 标志" width="420">
</p>

[English](README.md) | 简体中文

# nano-lmcache

循序渐进地学习 KV 缓存复用：从 Token 分块、前缀匹配和分层存储，到真实模型张量，以及
[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的外部缓存。

这是一个教学项目。每个概念都配有测试和可运行示例，帮助你理解
[LMCache](https://github.com/LMCache/LMCache) 和 vLLM 等系统。
项目优先考虑清晰性和可测试性，不以生产性能或兼容 LMCache API 为目标。

## 从这里开始

要求：Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。以下命令均在仓库根目录执行：

```bash
git clone https://github.com/Dami-wq-k03/nano-lmcache.git
cd nano-lmcache
uv sync --locked
uv run python examples/demo_prefix_cache.py
```

示例展示冷缓存未命中、共享前缀命中，以及条目数和字节容量限制下的淘汰。
数据载荷使用短字符串，不需要下载模型、安装 PyTorch 或使用 GPU。

也可以生成离线页面，逐步查看每次查询：

```bash
uv run python -m nano_lmcache.visualize --scenario partial_prefix_hit \
  --output /tmp/nano-lmcache-trace.html
```

在浏览器中打开 `/tmp/nano-lmcache-trace.html`，即可查看前缀树和事件时间线，无需服务端或 CDN。

## 接下来读什么

| 目标 | 指南 | 环境要求 |
| --- | --- | --- |
| 从分块键一路学到引擎重启恢复 | [循序渐进教程](docs/zh-CN/tutorial.md) | 前五步只需核心包 |
| 查找概念、命令和实验 | [文档目录](docs/zh-CN/README.md) | 按主题说明 |
| 理解术语和设计 | [术语与 FAQ（英文）](docs/glossary.md)、[架构](docs/zh-CN/architecture.md) | 核心包即可 |
| 在 CPU 或 GPU 上复用真实模型 KV | [真实模型教程](docs/zh-CN/tutorial.md#6-复用真实模型张量) | 可选 `hf` 扩展；CUDA 卸载需要 GPU |
| 在重启后恢复 nano-vLLM KV | [引擎教程](docs/zh-CN/tutorial.md#7-恢复真实引擎的缓存)、[集成设计（英文）](docs/nanovllm-integration.md) | Linux、NVIDIA GPU、本地模型和 `nanovllm` 依赖组 |
| 采集和查看 L1/L2 引擎事件 | [真实引擎 Trace（英文）](docs/engine-traces.md) | 采集需要 GPU；查看只需核心包 |
| 比较缓存加载与重新计算 | [TTFT 实验（英文）](docs/nanovllm-benchmark.md) | 参阅报告中的环境与限制 |

核心包没有运行时依赖。PyTorch 和推理引擎依赖均为可选项，安装与测试命令统一见
[贡献指南（英文）](CONTRIBUTING.md)。本页与英文 README 同步维护。

## 项目状态与贡献

已实现前缀缓存、有界内存与磁盘存储、内存/磁盘/分层后端的线程安全访问、真实张量、
CPU 卸载和 nano-vLLM V1 集成。离线查看器支持合成工作负载、FreeInference 不透明块回放，
以及已采集的引擎 Trace。

任务状态、前置条件和后续工作统一维护在 [ROADMAP.md（英文）](ROADMAP.md)。
贡献前请选择开放任务，并阅读 [CONTRIBUTING.md（英文）](CONTRIBUTING.md)。

项目使用 [MIT 许可证](LICENSE)。
