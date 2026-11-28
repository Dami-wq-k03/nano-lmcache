# QMSum-based Prefix KV Cache Reuse Benchmark

[English](../qmsum-benchmark.md)

> 本页记录历史 Pilot，不是当前主分支的全量重测。当前 PR 将 benchmark 的 Storage
> 显式传入计时函数，以兼容集成精简后移除的 `external.storage` 属性；原始测量值不变。

本实验使用真实会议的完整转录和不同问题，检验 **外部 L2 KV 恢复**是否减少
重复 Prefill，而不是仅证明 Key 存在。运行方法见
[`benchmarks/bench_nanovllm/README.md`](../../benchmarks/bench_nanovllm/README.md)。

## 固定实验设计

- 数据：官方 [Yale-LILY/QMSum](https://github.com/Yale-LILY/QMSum)，
  审计于 2026-11-22；上游 main 为
  `83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e`，另保存三个 JSONL 的 SHA-256。
- 基础代码：`a8652d9` 加本 benchmark 工作区；实际运行保存源码 SHA-256。
- nano-vLLM 固定为 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`；
  PyTorch 2.6.0+cu124、Transformers 4.57.1、FlashAttention 2.7.4.post1。
- TP=1，BF16，eager；完整会议，不截断；最大上下文 32,768，预留 64 个输出 Token。
- 模型 Chat Template，`enable_thinking=False`；对最终 Token IDs 求所有问题的 LCP，
  以 256 对齐，并遵守 nano-vLLM 最后一个 Prompt Block 不复用的限制。
- Pilot：优先选测试集最长的 20 场会议，每场 4 问；不是按长度均匀抽样。
- Off / Sync / Async-Flush（C1）/ Async-Continuous（C2）使用相同输入、顺序和参数。
  各模式在**同一 GPU、同一模型**内比较；每轮顺序轮转。
- 输出 1 / 32 Token，`ignore_eos=True`，temperature=1e-6，固定种子；
  请求并发 1 / 4 / 8；每个配置 3 轮，模式和并发各 2 次 Warmup，不计入结果。
- 每一波请求前重建 L1；同一 cohort 内保留 L2，cohort/轮次之间新建空 L2。
  并发波次由不同会议构成，后续 Query 在上一波返回后进入。

## 数据审计：实际结果

| Split | 会议数 | 原始 Query 条目 | 去重后有效 Query | 32K 合格会议 |
| --- | ---: | ---: | ---: | ---: |
| train | 162 | 1,257 | 1,256 | 161 |
| val | 35 | 272 | 271 | 35 |
| test | 35 | 281 | 280 | 34 |
| 合计 | **232** | **1,810** | **1,807** | **230** |

这里的 1,810 是下载文件逐条计数，不把它强行改成方案引用的 1,808。
三个 split 各有一个会议内重复问题；去掉这 3 个重复条目后是 1,807 个不同问题，
没有空问题。合格会议共提供 **1,794** 个有效请求，合格测试集提供 **273** 个。

两个会议因完整 Prompt 加预留输出超过 32K 而排除：
`train-113`（最大 Prompt 35,741 Token）、`test-016`（33,831 Token）。
没有截断原文来保留它们。

| 共同 Prefix Bucket | 全部合格会议 | 合格测试集 | Pilot |
| --- | ---: | ---: | ---: |
| 1K–4K | 13 | 1 | **0** |
| 4K–8K | 46 | 6 | **0** |
| 8K–16K | 106 | 16 | **9** |
| >=16K | 65 | 11 | **11** |
| 合计 | 230 | 34 | **20** |

Pilot 固定 **80 个请求**：

- 平均共同 Prefix：**16,988.75 Token**；最大 **26,958**。
- 平均完整可缓存 Prefix：**16,857.6 Token**；范围 **11,520–26,880**。
- 工作负载 SHA-256：
  `a7c325109d8d6836c83696744d9bef61768ffef51cfc89c0e4dc9e2aac92718d`。
- 这批 Pilot **不能**回答 1K/4K 下的收益或 crossover；这些 Bucket 没有样本。
  后续全量测试集实验才能覆盖较短会议，也不能由大 Prefix 的结果外推。

## 模型与资源

服务器报告 GPU 0 为 RTX 4090 / 48 GiB，GPU 1 为 RTX 4090 / 24 GiB。
按资源使用要求，两张卡同时运行独立的 TP=1 实验：

| GPU | 模型 | 用途 | GPU Memory Utilization | Rendezvous Port |
| --- | --- | --- | ---: | ---: |
| 0 | Qwen3-14B，已有 FP8 权重离线重建为 BF16 | 主实验 | 0.74 | 23410 |
| 1 | 已下载的 Qwen3-0.6B，BF16 | 独立小模型对照 | 0.35 | 23411 |

当前固定 nano-vLLM 的 loader 不支持 AWQ/FP8；Qwen3.6 对应的架构也不受支持。
因此没有把 32B-AWQ、14B-FP8 或 Qwen3.6 直接当作可运行模型。

14B 使用已有 `Qwen/Qwen3-14B-FP8` 快照
`9a283b4a5efbc09ce247e0ae5b02b744739e525a`，按 128×128 Block Scale 在 FP32
中反量化再转 BF16，保存独立权重和原始 Shard 哈希，**不修改原始模型缓存**。
这是“重建量化权重后的 BF16 推理”，不是原生 FP8 推理，也不是原始未量化 BF16
权重。0.6B 快照为 `c1899de289a04d12100db370d81485cdf75e47ca`。

显存预算同时为 GPU KV、恢复 Payload 和 Stack/Scatter 临时 Tensor 预留空间；
不能把 GPU KV 池吃满后假定恢复不再需要显存。并发值表示**同时到达请求数**；
显存容量可能使实际执行并发更小，报告不将二者混同。

两个进程共享 CPU、主存和 PCIe 资源，因此结果标为 **co-run 条件**，不声称是
独占整机的性能。不能把 GPU 0 的 Off 和 GPU 1 的 On 拼成 Speedup。
两模型的 GPU/显存预算不同，也没有交换到同一张卡上做控制实验，因此跨模型
Speedup 差异不能单独归因于模型参数规模。

## 测量与验收

1. TTFT：从本地请求入队到第一个可观察 Token 的 step/postprocess 结束；包含
   当步 Store。这不是 HTTP 流式服务 TTFT。32 Token 的 E2E 单独记录。
2. L2：只在 Payload 确实取回、导入 GPU 并完成对应 Stream 工作后记录；检查
   `prefill_tokens = prompt_tokens - l2_hit_blocks * 256`，断言没有 L1 命中。
3. C1：请求外显式 flush；C2：不插入显式 flush，实际 pending-key 等待另记。
   重置 L1、收集测量等带来的实际 inter-wave gap 也保存，不声称零间隔。
4. Generic Async 的 prepare 在前台完成 D2H，后台再存入 Pinned CPU Backend；
   不能宣称 GPU 导出/D2H 已完全隐藏。没有为放大收益添加人工 Backend 延迟。
5. Store Hook、Fetch、整批 Import 的 CPU/CUDA 时间分开；整批 Import 不重复归给
   每个请求。Throughput 使用包含 C1 flush 和最终异步 drain 的 cohort wall time。
6. 分别给出冷请求、后续请求的样本数和 P50/P95；同配置逐请求比较生成 Token IDs。
   不一致、没有基准配对或运行未覆盖所有预定请求，分析程序都返回非零状态。
7. 不把“至少一个区间提升 10%”预先写成结论；BF16 数值变化导致的 Token 不一致
   也必须保留，不能按语义相似把它算作严格正确性通过。

## 当前验证状态

- 本地完整 CPU 回归：**769 passed、5 skipped**；Ruff、`mypy src` 通过。
  新增 QMSum 测试单独运行：**16 passed**（含真正字节比较对正负零的检查）。
- 服务器固定环境 CPU 回归：**801 passed、1 failed、2 skipped**。
  失败的是原有 `test_nested_dictionary_order_does_not_change_namespace`：
  它直接访问 Transformers 4.57.1 不存在的 `rope_parameters` 字段。
  已确认该测试和核心源码没有被本实验修改；这里如实保留这个兼容性失败，
  不把服务器完整套件写成全部通过。
- 两种模型的单请求长 Prefix 冒烟测试：每种模型 **6/6** 个 Off/On 配对的生成
  Token 一致，四种 Cache 模式都完成真实 GPU 推理。
- 两卡同时执行的 8 并发到达压力测试：每种模型 **32/32** 条结果齐全。
  0.6B 的 **16/16** 个 Off/On 配对一致；14B 为 **15/16**。
  14B 的 `test-030 / specific-0` 在 Off 下输出首 Token `The`，C2 下输出 `Based`。
  该请求确实恢复 77 个 Block，Prefill 从 19,927 降为 215 Token，但这**不等于**
  严格生成正确性通过。后续诊断发现，原生 L1 与外部 L2 的该样本 logits 完全一致。
- 0.6B 已完成 **5,760/5,760** 条测量；4,320 个 Off/On 配对中，**801** 个生成
  序列不一致（其中 54 个首 Token 不一致）。三个外部 Cache 模式之间的 1,440 组
  对应输出则全部一致，不能将此等同于原先要求的 Off/On 一致。
- 14B 的并发 1 已完整完成 **1,920/1,920** 条测量；1,440 个 Off/On 配对中，
  **162** 个生成序列不一致（其中 18 个首 Token 不一致）。进入并发 4 后，
  GPU Restore 的 5.35 GiB `torch.stack` 分配因显存碎片失败。
  保留完整顺序组；不把中途产生的 144 条并发 4 观测混入最终完整矩阵。
- 相同显存预算 0.74 下，设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  后，原触发 OOM 的 cohort 已复测通过；并发 4/8 的 **3,840/3,840** 条已完整补跑。
  14B 最终有效矩阵为 **5,760/5,760**，4,320 个 Off/On 配对中 **531** 个生成序列
  不一致（其中 54 个首 Token 不一致），无缺失配对。
  两种模型合计 **11,520 条有效测量**，Warmup 和诊断不计入其中；两张 GPU 均已释放。
  原始代码快照、失败日志和恢复条件均保留，不隐藏 OOM，也不混入失败阶段的部分组。
  补跑时 GPU 1 从原生 L1 诊断转为空闲，利用率日志记录两张卡；因此跨并发
  Throughput 曲线不能被当作严格的“其他整机负载恒定”扩展性实验。

## 观察性性能结果：严格 Token-ID 验收尚未通过

以下数值来自已完成的配置，**不是最终验收成功的声明**。TTFT 表仅统计同一会议
后续 Query 的热请求；每模式有 9 场/81 条或 11 场/99 条观测，包含三轮重复，
不是 81/99 场独立会议。P50/P95 均为毫秒，C2 为 Async-Continuous。

| 模型 | 共同 Prefix | 会议/观测数（每模式） | Off P50 / P95 | C2 P50 / P95 | P50 比值 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.6B | 8K–16K | 9 / 81 | 232.6 / 264.6 | 113.3 / 178.7 | 2.05× |
| 0.6B | >=16K | 11 / 99 | 466.2 / 777.2 | 161.2 / 227.2 | 2.89× |
| 14B 重建 BF16 | 8K–16K | 9 / 81 | 2,840.5 / 3,140.4 | 245.8 / 295.5 | 11.56× |
| 14B 重建 BF16 | >=16K | 11 / 99 | 4,740.5 / 6,953.5 | 372.0 / 447.0 | 12.74× |

32 Token 输出、包括冷请求和最终 drain 的请求吞吐（三轮中位数，req/s）：

| 模型 | 请求并发 | Off | Sync | C1 | C2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.6B | 1 | 0.914 | 1.053 | 1.000 | 1.005 |
| 0.6B | 4 | 1.412 | 1.832 | 1.659 | 1.679 |
| 0.6B | 8 | 1.427 | 1.835 | 1.684 | 1.679 |
| 14B 重建 BF16 | 1 | 0.198 | 0.423 | 0.413 | 0.413 |
| 14B 重建 BF16 | 4 | 0.210 | 0.484 | 0.465 | 0.466 |
| 14B 重建 BF16 | 8 | 0.210 | 0.480 | 0.467 | 0.463 |

这里不能把热请求的 12× TTFT 比值写成整体吞吐 12×。也没有观察到 Generic Async
优于同步 Store：它的前台 D2H 和额外 Host Copy 成本仍在。14B 三种请求并发下
GPU KV 池都只有 **38,144 Token 槽位**，容纳不下 8 份完整长会议，不能把请求到达
并发等同于执行并发。图中附有正确性未通过的警示；14B 合并图另标注混合运行条件。

以 C2 / 并发 1 / 输出 1 Token 的三轮为例，每个模型都是 240 条请求，其中 180 条
热请求，成功恢复 **11,853 个 Block**，实际节省 **3,034,368 个 Prefill Token**。
按全部可复用 Prompt Block 计算，包含冷请求的命中率为 **74.93%**，热请求为
**99.90%**。这不是按 Key 存在性推算的命中率。

有效测量中的 PyTorch 峰值 allocated 显存：0.6B **18.77 GiB**、14B **44.06 GiB**。
另外保存 1 秒采样的 GPU 利用率、显存、功耗与温度 CSV；进程显存和 allocator 的
allocated/reserved 不是同一个指标，不能混称。

## 正确性诊断：不要把原生 Prefix 数值路径当作 L2 数据损坏

1. **0.6B 的 `test-013 / specific-2`**：Full Prefill 输出 `They`，原生 L1 输出
   `The`；Sync/C2 与原生 L1 的 logits 完全一致（最大绝对差 0）。额外 Roundtrip
   检查确认：两种 L2 模式各 **45 个 Block** 的导出 GPU Payload、存储的 CPU 副本、
   导入后的 GPU 块均逐位一致。
2. **0.6B 完整 20 场会议 / 80 个 32-Token 请求**，并发 1：原生 L1 与
   Sync/C1/C2 各 **80/80** 个输出序列一致，与 Full Prefill 有 **26/80** 个不同。
   这是额外正确性诊断，不混入性能统计。
3. **14B 的 `test-030 / specific-0`**：Sync/C2 同样与原生 L1 的 logits 完全一致，
   两种 L2 模式各 **77 个 Block** 也通过导出 GPU Payload、CPU 副本、导入 GPU 块
   的逐字节检查（按 dtype/shape 和 uint8 视图比较）。单独 Full Prefill 时，
   `Based` 和 `The` 的
   BF16 logits 同为 **41.75**；原生 L1 为 **42.0 / 41.5**。
4. 固定引擎的采样器即使 temperature=1e-6，仍通过指数随机噪声抽样，并非真正
   `argmax`。完整/分块/缓存路径消耗随机数的次数也不同。因此既有 Prefix 路径
   数值差异，也有同分近贪心抽样因素；不能笼统地把全部差异归为一个原因。

这些证据支持“已检查样本的 L2 拷贝无损且与原生 L1 等价”，**没有把原定的
Off/On 严格 Token 一致条件改判为通过**。后续建议使用真正确定性的 argmax 单独
重做正确性审计，并区分 KV 传输一致性、Prefix 数值误差、生成质量三个问题。
当前这一轮不在中途更换采样器。

对保存的 logits 再做 CPU `argmax` 审计：0.6B 的四条路径都选择 `The`（ID 785）；
14B 的 Full Prefill 选择 `The`（785），原生 L1 和两种 L2 则选择 `Based`（28715）。
因此，真正 argmax 可以排除近贪心抽样噪声，**仍不能保证 BF16 完整/Prefix 路径
生成逐 Token 相同**。这一步基于已保存 logits，不冒充另一轮 GPU 性能实验。

可复用诊断脚本为 `benchmarks/bench_nanovllm/diagnose_qmsum.py`；它保存四条路径的首 Token
logits、Top-5 和逐 Block 检查结果。为检查 CPU 存储副本会显式 drain 写队列，
诊断运行不用于声称 C2 等待时间，也不把其计时计入性能结果。

## 验收与复现交付

| 项目 | 本轮结论 |
| --- | --- |
| Off/On 生成 Token IDs 完全一致 | **未通过**：0.6B 801/4,320，14B 531/4,320 个配对序列不同 |
| 实际 L2 恢复与 Prefill 节省 | 完成逐请求计数校验；另有上述逐位 KV 和原生 L1 诊断证据 |
| 至少一个区间热 TTFT 改善超过 10% | 观察性数据满足，但不因此覆盖正确性失败、宣布整体验收通过 |
| 可复现条件 | 两模型三轮配置齐全，记录输入/源码/模型版本、环境及 OOM 恢复条件；未验证跨机器稳定性 |

测量版源码 SHA-256 为
`80f5ea49149512041d74f6d8cc42a765a7e903bbcd9f21a0a36b744009138b90`，
完整快照保存在实验根目录的 `source-v1.tar.gz`。后续公开代码新增环境元数据、
诊断工具和图中正确性警示；没有回写本轮已测量的输入、时间或输出。
报告另外保存分析脚本的 `analysis_code_sha256`。

输出目录中的主要交付物：

- `pilot-0.6b/`：完整小模型结果、Summary、CSV、浅色/深色三类图。
- `pilot-14b-combined/`：完整顺序组与补跑并发组的汇总，来源文件哈希和环境嵌入
  `environment.json`；不是伪装成一次无中断、条件恒定的运行。
- `pilot-14b/`、`pilot-14b-c1/`、`pilot-14b-c4c8-retry/`：原始失败记录、有效顺序
  子集和完整补跑原始数据，均保留。
- `recovery.json`、运行日志、GPU CSV、`diagnostic-*/`：OOM 恢复说明和正确性证据。

本轮完成的是 **20 场会议 Pilot**。全量 34 场合格测试会议 / 273 个有效 Query、
5 轮、64 Token 输出、Disk 后端，以及同会议同时到达的 Stampede 还没有运行。
先明确严格正确性与数值路径一致性的验收边界，再决定是否扩大正式实验。

性能观测、逐请求原始文件和图表保存在实验输出目录，不将原始会议、模型权重或大型
结果放入 Git。最终数值以对应运行的 `summary.json`、CSV 和完整性检查为准。
