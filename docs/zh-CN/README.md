[English](../README.md) | 简体中文

# 文档目录

初次接触 KV 缓存复用，建议从[循序渐进教程](tutorial.md)开始。
每一步都有可运行实验、预期现象、源码和测试入口。前五步不需要模型或 GPU。

## 学习与理解

| 文档 | 内容 |
| --- | --- |
| [循序渐进教程](tutorial.md) | 分块 → 前缀匹配 → 淘汰 → 分层存储 → Trace → 真实张量 → 引擎恢复 |
| [术语表与 FAQ（英文）](../glossary.md) | KV、Prefill、Decode、TTFT、Chunk、Block 和常见误解 |
| [架构](architecture.md) | 当前模块、不变量、容量统计、序列化和并发边界 |
| [nano-vLLM 集成（英文）](../nanovllm-integration.md) | 引擎块布局、L1/L2 查询、恢复/保存钩子及正确性规则 |

## 运行与观察

| 任务 | 入口 | 要求 |
| --- | --- | --- |
| 运行最小缓存示例 | [项目快速开始](../../README.zh-CN.md#从这里开始) | Python 3.11+、uv |
| 回放合成工作负载或 FreeInference 块 | [回放教程](tutorial.md#5-在查看器中跟踪查询) | 核心包；FreeInference 需要 JSONL 文件 |
| 只读检查持久化条目 | [磁盘教程](tutorial.md#4-持久化与提升分块) | 核心包和已有缓存目录 |
| 复用 Hugging Face 模型 KV | [真实模型教程](tutorial.md#6-复用真实模型张量) | `hf` 扩展和模型权重，支持 CPU |
| 采集引擎事件并生成离线 HTML | [引擎 Trace（英文）](../engine-traces.md) | 采集需要 Linux/NVIDIA GPU/本地模型；渲染只需核心包 |

## 实验与取舍

以下报告记录特定模型、硬件和工作负载下的结果，不能据此假定缓存总能加速。

| 报告 | 回答的问题 |
| --- | --- |
| [淘汰性能（英文）](../eviction-performance.md) | 叶节点堆如何保持 LRU 顺序并降低候选选择成本？ |
| [INT8 KV 压缩（英文）](../quantized-kv.md) | 节省多少空间，会增加哪些误差和延迟？ |
| [nano-vLLM TTFT（英文）](../nanovllm-benchmark.md) | 何时从内存或磁盘加载比重新计算前缀便宜？ |

[GPU 验证记录（英文）](../gpu-validation.md)记录了完整测试、真实模型示例、跨进程磁盘恢复和引擎 Trace 渲染的实测结果。

## 贡献与维护

- [ROADMAP（英文）](../../ROADMAP.md) 统一维护任务状态、前置条件和验收标准。
- [CONTRIBUTING（英文）](../../CONTRIBUTING.md) 统一维护环境配置、依赖组、检查命令和贡献流程。
- 架构与专题指南解释当前实现；路线图中已完成任务的原始规格用于追溯设计背景。
- 修改相关概念时，同步维护中英文 README、目录、教程和架构说明。专题实验报告目前以英文提供。

返回[项目 README](../../README.zh-CN.md)。
