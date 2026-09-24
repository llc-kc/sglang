# KV capacity estimator mode

This mode keeps SGLang's OpenAI request validation, chat-template rendering, tool-call handling, and tokenizer, then sends the prompt token IDs directly to `kv-capacity-estimator`. It does not start scheduler or detokenizer processes and does not load model weights.

## Prerequisites

- Linux with Python 3.10 or newer
- An SGLang environment that supports the tokenizer for `MODEL_PATH`
- The simulator wheel installed in the same environment

```bash
python -m pip install /path/to/kv_capacity_estimator-0.2.0-py3-none-any.whl
```

## Start the server

```bash
sglang serve MODEL_PATH \
  --model-type llm \
  --tool-call-parser glm47 \
  --kv-capacity-estimator \
  --tokenizer-worker-num 4 \
  --enable-metrics \
  --kv-capacity-estimator-config '{
    "kv_bytes_per_token": 61505,
    "capacities": ["200GiB", "400GiB", "800GiB", "1TiB", "2TiB"],
    "target_hit_rate_ratio": 1.0,
    "slide_window_size": 3000,
    "output_path": "/tmp/kv-cache-result.json"
  }'
```

`--model-type llm` bypasses automatic diffusion-backend selection. The model path still needs model configuration, tokenizer, and chat-template files, but weight files are not read.

`--kv-capacity-estimator-config` accepts one JSON object. Only `kv_bytes_per_token` is required because it depends on the deployed model. All omitted simulation fields use defaults from the installed
`kv-capacity-estimator` wheel rather than defaults copied into SGLang:

| JSON field | Wheel default |
| --- | --- |
| `page_size` | `64` |
| `capacities` | `200GiB` through `32TiB` |
| `include_partial_page` | `false` |
| `target_hit_rate_ratio` | `0.99` |

`capacities` accepts a comma-separated string, an array of size strings, or `[label, bytes-or-size]` pairs. The three `*_bytes` search fields accept integer bytes or strings such as `2TiB`. SGLang also accepts these adapter-specific fields:

| JSON field | Default | Description |
| --- | --- | --- |
| `slide_window_size` | `3000` | Number of latest successful requests retained for online required-capacity percentiles. |
| `output_path` | unset | Writes the aggregate result at finalization. |

`warm_up_ratio` is intentionally not accepted by the SGLang online adapter.
Warm-up prefix exclusion is an offline replay concept: status, Prometheus
metrics, finalization, and the output file all measure every admitted request.

## Submit chat requests

Use the normal OpenAI endpoint:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MODEL_PATH",
    "messages": [{"role": "user", "content": "Explain prefix caching."}]
  }'
```

The response remains an OpenAI-compatible chat completion with an empty assistant message and zero completion tokens. The `kv_capacity_estimation`
extension contains the prompt token count, page hits for each configured capacity, prefix hits, and this request's required capacity. Streaming requests receive the same extension on their single completion chunk.

`--tokenizer-worker-num` controls a process pool. Each worker initializes its own SGLang tokenizer, `TemplateManager`, and `OpenAIServingChat` instance. The main HTTP process receives completed token-ID sequences and commits them to one central `ReplaySimulator`, so all requests still share one LRU history. Results enter that history in tokenizer completion order rather than strict HTTP arrival order; this avoids a reorder barrier on the tokenize hot path.

Multimodal requests are rejected because their final token expansion belongs to the model execution pipeline. Text, tools, reasoning options, custom chat templates, and pre-tokenized `input_ids` use SGLang's chat request path.

## Read or finalize results

Check progress without changing simulator state:

```bash
curl http://127.0.0.1:30000/kv-capacity-estimator/status
```

Finalize and return aggregate capacity statistics:

```bash
curl -X POST http://127.0.0.1:30000/kv-capacity-estimator/finalize
```

Finalization is idempotent and stops the server from accepting more chat requests. The server also finalizes during graceful shutdown. When `output_path` is set in the JSON config, either path writes the aggregate result to that file.

## Metrics

The status response always includes a `metrics` object with cumulative request, prompt-token, page-access, unlimited-cache hit, per-capacity hit-rate, and required-capacity statistics. These online counters and the finalized simulator result cover the same whole trace.

With `--enable-metrics`, Prometheus metrics are available on the standard endpoint (without API-key authentication, matching normal SGLang behavior):

```bash
curl http://127.0.0.1:30000/metrics
```

The server exports SGLang-compatible `sglang:prompt_tokens_total`, `sglang:prompt_tokens_histogram`, and `sglang:num_requests_total` series. The `sglang:kv_capacity_estimator_*` series additionally cover:

- successful/error and pending requests;
- tokenizer worker count, worker time, executor queue/IPC time, simulator time, and end-to-end request time;
- page accesses, unlimited-cache reusable page/prefix hits;
- page/prefix hits and cumulative hit rates for every configured capacity;
- required-capacity distribution and unreachable-target count;
- nearest-rank p90, p95, p99, and p99.9 required capacities over the latest `slide_window_size` successful requests.

The sliding-window series is `sglang:kv_capacity_estimator_required_capacity_slide_window_bytes` with `quantile="0.9"`, `quantile="0.95"`, `quantile="0.99"`, and `quantile="0.999"` labels. The status response exposes the same values under `metrics.required_capacity.slide_window` as `p90_bytes`, `p95_bytes`, `p99_bytes`, and `p999_bytes`. Requests whose target capacity is unreachable occupy a window slot but are excluded from percentile calculation; the status payload reports reachable and unreachable sample counts.

`--extra-metric-labels` is supported. Capacity is represented by both the human-readable `capacity` label and numeric `capacity_bytes` label.
