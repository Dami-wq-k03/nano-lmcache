[English](../streaming-replay.md) | 简体中文

# 流式工作负载回放与有界报告

无需模型、GPU 或额外运行时依赖，即可逐条处理工作负载。完整统计与 HTML 中保留的
请求、事件明细分别维护，采样不会改变模拟命中率。

## 输入与顺序

`--workload-jsonl` 接受每个非空行一条 `WorkloadRequest`。例如保存为 `/tmp/workload.jsonl`：

```jsonl
{"request_id":"a","session_id":"s","arrival_time_ms":0,"model_id":"m","prompt_tokens":6,"token_ids":[1,2,3,4,5,6]}
{"request_id":"b","session_id":"s","arrival_time_ms":1,"model_id":"m","prompt_tokens":6,"token_ids":[1,2,3,4,7,8],"observed_cached_tokens":2}
```

```bash
uv run python -m nano_lmcache.visualize \
  --workload-jsonl /tmp/workload.jsonl --stream --chunk-size 2 \
  --max-entries 4 --max-events 1000 --max-requests 100 --sample-every 2 \
  --summary-json /tmp/workload-summary.json --output /tmp/workload.html
```

第二个请求复用两块（4 个 Token），两次请求合计命中 2/6 块；数据集单独报告观测到的
缓存 Token 数为 2。这两个指标不是同一测量。Token 回放每个 runner 只接受一个模型，
遇到不同 `model_id` 会报错，避免复用不兼容的 KV。
Runner 自建 Token 缓存会将模型标识加入命名空间，隔离共享持久化后端中的不同模型。
若传入已经构建的 `ChunkedPrefixCache`，调用方需保证其命名空间与模型匹配。

若输入提供不透明 Block ID，将 `token_ids` 换成 `prefix_blocks`，添加 `--opaque-blocks`，
并指定 `--block-tokens`（默认 16）。Block ID 不是 Token ID；这里模拟缓存复用，
不执行模型。不透明块键同时包含模型标识和前缀上下文。

FreeInference 原始会话 JSONL 可以直接回放：

```bash
uv run python -m nano_lmcache.visualize \
  --freeinference-jsonl /path/to/sessions.jsonl --stream \
  --max-entries 4096 --max-events 10000 --max-requests 200 \
  --sample-every 10 --summary-json /tmp/freeinference-summary.json \
  --output /tmp/freeinference.html
```

`--stream` 要求输入按 `(arrival_time_ms, request_id)` 非递减排列；时间相同时按请求 ID
字典序排列。发现逆序会报错，不会悄悄收集所有请求再排序。FreeInference 适配器会展开并
排序**单条会话记录**（含嵌套会话），但不同记录的时间可能交错。大型交错数据应先标准化成
每行一请求的 JSONL，在外部完成排序。省略 `--stream` 时，仍会在内存中排序全部选中请求。

`--limit N` 处理 N 条请求后停止，不再读取下一条 JSONL 记录；单条记录的大小不受它限制。
输入文件不变。磁盘或分层回放遇到输入错误时，此前已写入的缓存仍会保留；重试冷启动实验请换新目录。

## 预算与统计

| 参数 | CLI 默认值 | 含义 |
| --- | --- | --- |
| `--max-entries` | 10,000 | 每个配置存储层的模拟缓存条目上限 |
| `--max-events` | 10,000 | HTML 保留的事件数上限 |
| `--max-requests` | 200 | HTML 保留的请求明细数上限 |
| `--sample-every` | 1 | 从序号 0 开始，每隔 k 个事件选一个，直到达到事件上限 |
| `--limit` | 不限制 | 处理 N 条请求后停止 |
| `--summary-json` | 不输出文件 | 导出完整统计、配置和明细保留范围 |

事件或请求预算为 0 时，不保留对应明细，但仍统计已处理的完整工作负载。采样是确定性的
**前段采样**，不是全程随机抽样；达到上限后，后续事件仅计数。保留事件使用原始序号，
不会把缺口重新编号。请求明细独立保留前 N 条。

汇总逐条累计请求数、完整块数、模拟复用 Token、数据集观测缓存 Token、事件类型计数，
以及各存储层成功读取次数。`observed_requests` 表示多少请求提供了观测值，缺失值不等于零。
即使没有保留事件明细，淘汰与提升仍会计数。这些是操作统计，不是 GPU 时间估计。
JSON 汇总不包含请求或事件数组。

Viewer 会显示实际保留比例。请求面板每页 25 条，每个请求最多显示前 200 个块状态；
时间线每次渲染 100 个保留事件，可用 Seek event 定位；前缀树最多显示保留事件中的 500 个键。
祖先可能被采样省略，因此树表示历史关系，不表示当前缓存驻留状态，也不能完整还原淘汰过程。

在流式模式下，保留内存由缓存预算、事件/请求预算、最大单条输入记录/请求，以及不同事件类型、
存储层名称的计数器决定。预算按数量计算，不是字节上限：超长 JSONL 行、请求、元数据或自定义
缓存载荷仍可能很大，统计整数也会随总量增长。这不是进程 RSS 硬限制；磁盘空间和 HTML/JSON
序列化临时分配也取决于保留的数据大小。

Python API 为兼容保留完整明细的默认行为；调用方需要同时设置 `MemoryKVCache(max_entries=...)`、
`TraceRecorder(max_events=..., sample_every=...)`，并向 `run` 传入
`sort_by_arrival=False, max_requests=...`。完整代码见[英文版](../streaming-replay.md)。
每次 `run` 都会清空所绑定 recorder，开始独立报告；模拟缓存则保留，可用于热缓存实验。
不要将此 recorder 与其他并发回放或正在运行的引擎共享。禁用 recorder 时不统计事件，
请求级复用总量仍然可用。
`TraceRecorder.to_json()` / `write_json()` 仅导出保留事件；完整统计请使用回放报告或汇总导出。
已经在采集时采样的引擎事件文件无法还原被省略的命中。

## 比较容量与验证内存

```bash
uv run python -m nano_lmcache.visualize --scenario lru_eviction \
  --max-entries 2 --summary-json /tmp/capacity-2.json --output /tmp/capacity-2.html
uv run python -m nano_lmcache.visualize --scenario lru_eviction \
  --max-entries 3 --summary-json /tmp/capacity-3.json --output /tmp/capacity-3.html
uv run python benchmarks/bench_streaming_replay.py --counts 1000 10000 100000
uv run python benchmarks/bench_streaming_replay.py --counts 1000 10000 --batch-baseline
uv run pytest tests/unit/test_streaming_replay.py
```

容量为 2 时，重访未命中（0/4 命中、淘汰两次）；容量为 3 时，重访命中（1/4 命中、无淘汰）。
比较 JSON 中的 `summary.hit_rate`、`summary.event_counts.eviction`（缺失表示零）、
`summary.storage_hits` 与配置。磁盘/分层冷启动比较需要分别使用空目录；沿用目录会测到持久化热缓存。
当前淘汰策略仍是前缀安全的叶节点 LRU，替代策略属于 M11。

内存基准会生成共享前两块、第三块始终不同的请求；缓存限 64 条，保留 20 个请求、100 个事件，
每 10 个事件采样一次。基线模式会排序全部输入并保留全部明细。它使用 `tracemalloc` 测量
Python 峰值分配，不是 RSS、文件 I/O 或张量显存，并验证完整命中量与观测 Token 总数。

2026-11-05 在 macOS 27.0.1 arm64、CPython 3.11.12 上的测量（KiB = 1,024 字节）：

| 请求数 | 批量排序、完整明细峰值 KiB | 流式、有界明细峰值 KiB | 流式耗时（秒） |
| --- | --- | --- | --- |
| 1,000 | 2,820.3 | 129.3 | 0.125 |
| 10,000 | 27,398.4 | 113.5 | 1.190 |
| 100,000 | 未运行 | 113.5 | 12.158 |

三次流式实验都保留 20 条请求与 100 个事件，分别淘汰 938、9,938、99,938 次。
这些结果说明该生成工作负载的保留内存不会随请求数线性增长，不是任意输入或张量载荷的通用内存/延迟上限。

## 已记录的引擎 JSON

现有 `--trace-json` 仍会整批解析事件数组/封装并在内存中按请求分组，然后在计算准确引擎统计后
裁剪 HTML 明细，也支持 `--summary-json`；此格式不能与 `--stream` 一起使用。
这样不会从采样事件误推引擎命中。大型工作负载实验请使用上述流式 JSONL 路径；
原始引擎事件的流式导入属于后续工作。
