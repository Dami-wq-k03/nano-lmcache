[English](../pd-disaggregation.md) | 简体中文

# Prefill/Decode 分离

PD 分离把 Prefill 和 Decode 放到不同 Worker 上：Prefill Worker 计算 Prompt 的 KV
和第一个 Token，交给 Decode Worker，后者不重算 Prompt 直接继续生成。
整个教学版只有一个模块 [`disaggregation.py`](../../src/nano_lmcache/disaggregation.py)，
约 140 行，用玩具模型代替真实模型，以便精确比较输出。

```bash
uv run python examples/demo_pd_disaggregation.py
uv run pytest tests/unit/test_disaggregation.py
```

## 为什么要拆分

Prefill 是对 Prompt 的一次大计算，Decode 是很多次小步。共用一张 GPU 时，生成中途到达的长 Prompt
会让所有正在运行的请求一直等到它的 Prefill 结束。示例时间线按每个 Prompt Token 0.5 ms、
每个 Decode 步 20 ms、每个传输 Token 0.05 ms 计算：

| 部署方式 | 请求 | TTFT | Token 最大间隔 |
| --- | --- | ---: | ---: |
| 合并 | chat（200 Token） | 100 ms | **2020 ms** |
| 合并 | long-doc（4000 Token，150 ms 到达） | 2010 ms | 20 ms |
| 分离 | chat | 110 ms | **20 ms** |
| 分离 | long-doc | **2200 ms** | 20 ms |

卡顿消失了，但每个 TTFT 都包含 KV 传输时间。这些是模拟数值，不是实测。

## 交接必须保证什么

1. **结果相同。** `prefill` 之后的 `decode`，与单 Worker 的 `generate` 输出完全一致。
2. **不重算 Prompt。** 第一个 Token 随交接一起发送，生成 *m* 个 Token 只让 Decode 模型计算 *m − 1* 个位置。
3. **先检查再使用。** `decode` 遇到以下任一情况就拒绝：CRC32 不匹配（传输中损坏）；
   来自不同模型（其他权重的 KV 形状正确，但会生成错误的 Token）；缺少部分 Prompt Token 的 KV。
4. **只有一个所有者。** `decode` 从通道中 *pop* 出交接，第二个 Decoder 会得到 `KeyError`。
   被拒绝的交接同样被移除，请求必须重新 Prefill。

## 本版本没有覆盖的内容

真实张量和引擎（路线图 PD-3）、Prefill 进行中流式传输 KV（PD-4）、文件/Socket/RDMA 等跨进程传输、
多 Worker 路由，以及无人认领交接的超时。Prefill 也可以复用前缀缓存，交接保持不变，
因为 Decode Worker 不关心 Prefill 的 KV 从哪里来。

返回[循序渐进教程](tutorial.md#8-拆分-prefill-与-decode)。
