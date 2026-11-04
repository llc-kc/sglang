# 在线请求回放

`replay_online_request.py` 将 JSONL 文件逐条发送到 SGLang KV capacity estimator 的 `/v1/chat/completions`。默认输入就是 `ksyun10084_glm53_20261030_1600-1700_sorted.evalscope.32600.jsonl` 的完整绝对路径。

在当前环境直接使用容器中的 Python（已安装 aiohttp）：

```bash
docker exec -it \
  -w '/mnt/c/Users/liluchang/Documents/ChatGPT/KV cache estimator' \
  llc_sglang5 python replay_online_request.py --concurrency 4
```

默认回放全部请求，默认并发为 4。先试跑前 100 条：

```bash
docker exec -it \
  -w '/mnt/c/Users/liluchang/Documents/ChatGPT/KV cache estimator' \
  llc_sglang5 python replay_online_request.py \
  --concurrency 4 --num-requests 100 \
  --summary-output replay_100_summary.json
```

要从空缓存开始一轮独立分析，可加 `--reset`。这会清空服务的缓存历史、累计结果和 Prometheus 计数。不要在其他客户端还在发送请求时执行。默认不重置，因此试跑后的完整分析应显式加 `--reset`。

```bash
docker exec -it \
  -w '/mnt/c/Users/liluchang/Documents/ChatGPT/KV cache estimator' \
  llc_sglang5 python replay_online_request.py \
  --concurrency 4 --reset \
  --summary-output replay_all_summary.json \
  --results-jsonl replay_all_results.jsonl
```

也可以在其他安装了 `aiohttp` 的 Python 3.10+ 环境运行（`python3 -m pip install aiohttp`），使用 `--base-url http://127.0.0.1:30000` 指定服务地址。参数也接受以 `/v1` 结尾的地址。

| 参数 | 说明 |
| --- | --- |
| `--input PATH` | JSONL 路径；默认使用指定的 14.1 GB 数据文件 |
| `--concurrency N` | 固定异步 worker 数，默认 4，必须大于 0 |
| `--num-requests N` | 回放前 N 条非空记录；不传则回放全部。解析失败、HTTP 失败也计入 N |
| `--model NAME` | 覆盖模型名；默认从服务 status 获取并替换原始数据的 model |
| `--timeout SECONDS` | 单次 HTTP 请求超时，默认 600 秒 |
| `--log-interval N` | 每完成 N 条输出进度，默认 100 |
| `--summary-output PATH` | 汇总 JSON，默认 `replay_online_summary.json`，会覆盖已有同名文件 |
| `--results-jsonl PATH` | 可选逐条结果，边回放边写盘，不包含原始提示词；会覆盖已有同名文件 |
| `--reset` | 回放前清空服务状态，默认关闭 |
| `--finalize` | 回放结束后调用 finalize，保存完整聚合结果；默认关闭 |
| `--api-key KEY` | 可选 Bearer token，也可设置环境变量 `SGLANG_API_KEY` |

脚本逐行读取，不预先计数，不创建全量任务或结果列表；内存保留的请求数量上限为并发数，实际内存取决于同时处理的单条请求大小。文件读取在线程中串行执行，HTTP 使用 aiohttp 异步并发。messages、tools、chat_template_kwargs 和其他请求字段原样保留，仅替换 model、设置 stream=false 并去掉 stream_options。回放尽可能快地发送，不按原始时间间隔节流。

终端输出各容量的累计页面命中率和前缀命中率；汇总文件包含本次成功/失败数、prompt token 数、服务回放前后状态。服务状态及 Prometheus 指标包含服务的全部历史；只有使用 `--reset` 且无其他客户端流量时，它们才对应本轮独立回放。`--results-jsonl` 每行含原始行号、成功状态、usage 和 kv_capacity_estimation；失败时含错误信息，输出顺序是完成顺序。

单条 JSON/HTTP/超时错误打印行号后继续，存在失败时退出码为 1。失败不自动重试，因为超时请求可能已经进入模拟器，重试可能重复计入缓存历史。Ctrl+C 退出码为 130，中断时不会 finalize；已到达服务的请求可能仍在完成，最终状态可通过 `/kv-capacity-estimator/status` 查看。

并发情况下，服务按 tokenizer 完成顺序更新 LRU 历史，因此结果可能随调度变化。严格按文件顺序分析时使用 `--concurrency 1`。`--finalize` 会使服务拒绝后续请求，直到调用 reset；默认仅查询 status，便于继续使用服务。

验证：

```bash
docker exec -w '/mnt/c/Users/liluchang/Documents/ChatGPT/KV cache estimator' \
  llc_sglang5 python -m unittest -v test_replay_online_request.py
```
