[English](../pd-disaggregation.md) | 简体中文

# Prefill/Decode 分离（教学版）

PD 分离把生成的两个阶段放到不同 Worker 上：**Prefill Worker** 把 Prompt
计算成 KV 和第一个 Token，再把 KV 交给 **Decode Worker** 继续生成。
关键不是把两个函数拆开，而是要定义清楚：KV 归谁所有、用什么格式传输、
怎么传、能存活多久，以及任何一步出错时如何处理。

本版本在 CPU 上运行，使用确定性的玩具模型，因此下面每条结论都有测试精确验证。
它不传输真实张量，也不驱动 nano-vLLM，见[本版本没有覆盖的内容](#本版本没有覆盖的内容)。

```bash
uv run python examples/demo_pd_disaggregation.py
uv run pytest tests/unit/test_pd_handoff.py tests/unit/test_pd_workers.py \
  tests/unit/test_pd_simulation.py
```

## 为什么要拆分

Prefill 一次性处理整个 Prompt，是一次大计算；Decode 每步只生成一个 Token，是很多次小计算。
两者共用一张 GPU 时，生成过程中到达的长 Prompt 会让所有正在 Decode 的请求一直等到它的 Prefill 结束。
示例的时间线部分用一个成本模型演示这一点（每个 Prompt Token 0.5 ms，
每个 Decode 步 20 ms，每个传输 Token 0.05 ms）：

| 部署方式 | 请求 | TTFT | 最大 Token 间隔 | 完成时间 |
| --- | --- | ---: | ---: | ---: |
| 合并 | chat（200 Token Prompt） | 100 ms | **2020 ms** | 2480 ms |
| 合并 | long-doc（4000 Token Prompt，150 ms 到达） | 2010 ms | 20 ms | 2240 ms |
| 分离 | chat | 110 ms | **20 ms** | 490 ms |
| 分离 | long-doc | **2200 ms** | 20 ms | 2430 ms |

分离消除了 chat 请求 2 秒的卡顿；代价是每个请求的 TTFT 都包含 KV 传输时间，
长请求的第一个 Token 来得更晚。这些是
[`simulation.py`](../../src/nano_lmcache/disaggregation/simulation.py)
的模拟数值，忽略了批处理、显存限制和多 GPU 排队，只用于理解取舍，不能预测真实加速比。

实验：在示例中调大 `transfer_ms_per_token`，直到分离也损害 chat 的 TTFT；
或者把 `long-doc` 的到达时间推迟到 chat 完成之后，观察合并部署的卡顿消失。

## 组成部分

```text
Client
  │ prompt
  ▼
PrefillWorker ──KVHandoff（字节）──▶ KVTransport ──认领──▶ DecodeWorker
  │ 可选 ChunkedPrefixCache                                  │
  ▼                                                          ▼
第一个 Token                                             其余 Token
```

| 模块 | 作用 |
| --- | --- |
| [`toy_model.py`](../../src/nano_lmcache/disaggregation/toy_model.py) | 每个 Token 的 KV 链接到所有之前的 Token；下一个 Token 读取全部 KV；不同 `seed`（权重）产生不同 KV。模型会统计计算过的位置数。 |
| [`handoff.py`](../../src/nano_lmcache/disaggregation/handoff.py) | `KVHandoff`：请求 ID、模型指纹、Prompt Token、第一个 Token、分块 KV；带版本、长度和 CRC32 的传输格式。 |
| [`transport.py`](../../src/nano_lmcache/disaggregation/transport.py) | `InMemoryTransport`（线程间）和 `DirectoryTransport`（共享目录的进程间），都传输字节，并保证只能被认领一次。 |
| [`workers.py`](../../src/nano_lmcache/disaggregation/workers.py) | `PrefillWorker`、`DecodeWorker` 和参考实现 `generate_colocated`。 |
| [`simulation.py`](../../src/nano_lmcache/disaggregation/simulation.py) | 合并与分离两种时间线：TTFT、最大 Token 间隔、完成时间。 |

## 交接必须遵守的规则

**1. 结果与合并部署一致。** 同一 Prompt 下，分离路径的输出必须等于
`generate_colocated`。测试覆盖两种传输方式以及生成 1、2、9 个 Token 的情况。

**2. Decode Worker 不重新计算 Prompt。** 交接 *n* 个 Token 的 Prompt 后，生成 *m* 个
Token 只让 Decode 模型计算 *m − 1* 个位置：第一个 Token 由 Prefill 产生并随交接一起发送。
`test_decode_worker_never_recomputes_the_prompt` 检查计数器。

**3. 发送所有 Prompt 位置，包括不完整的最后一块。** 前缀缓存会跳过不完整的尾部分块，
因为以后可以重新计算；Decode Worker 不行，缺少 KV 就只能自己再做 Prefill。
所以 11 个 Token、块大小 4 的 Prompt 会发送 4、4、3 三块。若 KV 数量与 Prompt 长度不一致，
或者非最后一块不满，`validate` 都会拒绝。

**4. 使用外来 KV 前先核对身份。** 来自其他权重或形状的 KV 是“静默错误”：结构正确，
生成的 Token 却不同。交接携带 `model_fingerprint`，不匹配时 Decode Worker 在计算前就抛出
`ConfigMismatchError`。真实系统的指纹必须覆盖所有会改变 KV 的因素：权重、dtype、
层数和头数、张量并行布局、位置编码。

**5. 检测传输损坏。** `from_bytes` 检查格式版本、负载长度（截断）和 CRC32（比特翻转），
失败时抛出 `CorruptHandoffError`。完整性与语义是两回事：完好的交接仍可能来自错误的模型。

**6. 同一时刻只有一个所有者。** `send` 之后交接归传输层所有；`receive` 会**认领**它：
内存传输在锁内弹出，目录传输通过重命名文件实现，只有一个进程能成功。
第二个 Decode Worker 会得到 `HandoffNotFoundError`。对仍在等待的请求 ID 再次发送会抛出
`DuplicateHandoffError`，而不是覆盖。

**7. 失败即结束交接。** 无论 KV 损坏、不完整还是来自其他模型，被认领的交接都会被删除，
Decode Worker 不做任何计算。恢复方式是重新 Prefill，绝不能用错误的 KV 继续 Decode。
请求取消时调用 `discard`，避免 KV 残留。

## Prefill 侧的前缀缓存

前缀缓存和交接解决的是不同问题：前缀缓存在**请求之间**复用 KV，交接在**角色之间**移动
**一个请求**的 KV。两者可以组合：给 `PrefillWorker` 一个 `ChunkedPrefixCache`，
共享文档前缀的第二个请求只计算新增 Token：

```text
q-1: reused  0, computed 27 of 27 prompt tokens; tokens match colocated: True
q-2: reused 24, computed  3 of 27 prompt tokens; tokens match colocated: True
```

交接仍然包含全部 27 个位置，因为 Decode Worker 需要什么与 Prefill 的 KV 从哪里来无关。
像示例一样用模型指纹作为缓存命名空间，让两处检查对“同一个模型”的定义保持一致。

## 目录传输细节

`DirectoryTransport` 先把交接写入临时文件，再用硬链接放到最终文件名。目标已存在时链接失败
（检测重复发送），读取方也永远看不到写了一半的文件。文件名是请求 ID 的 SHA-256，
任何 ID 都无法指向目录外的路径。`receive` 先把文件重命名为本进程专属的认领名再读取，
无论字节是否有效都会删除它。`test_concurrent_receivers_claim_once` 让 8 个接收者争抢同一个交接。

目录所在的文件系统需要支持硬链接和原子重命名（本地文件系统均可），网络文件系统可能削弱这些保证。

## 本版本没有覆盖的内容

- **真实张量和引擎。** nano-vLLM 版本需要在 Prefill 后导出请求的 KV 块
  （[`kv_blocks.py`](../../src/nano_lmcache/integrations/nanovllm/kv_blocks.py) 中的块格式已经存在），
  并在 Decode 前导入第二个引擎的块表。见路线图中的 PD 任务。
- **流式传输与重叠。** 真实系统在 Prefill 仍在运行时按层或按块传输，并使用 RDMA/NCCL 而不是文件；
  本版本在 Prefill 结束后一次性发送。
- **调度与路由。** 请求分配给哪个 Decode Worker、Prefill 与 Decode GPU 的配比、Decode 显存满时的背压，
  都不在本版本范围内。
- **超时与垃圾回收。** 从未被认领的交接会一直等待，直到 `discard`；真实系统需要租约或 TTL。

返回[循序渐进教程](tutorial.md#8-拆分-prefill-与-decode)或[文档目录](README.md)。
