# KV capacity estimator mode

This mode keeps SGLang's OpenAI request validation, chat-template rendering, tool-call handling, and tokenizer, then sends the prompt token IDs directly to `kv-capacity-estimator`. It does not start scheduler or detokenizer processes and does not load model weights.

## Prerequisites

- Linux with Python 3.10 or newer; no accelerator hardware is required
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
    "page_size": 256,
    "target_hit_rate_ratio": 0.99
  }'
```

Simulator mode resolves only tokenizer and HTTP settings, using CPU tokenization without accelerator detection or inference-backend initialization. No `--device cpu` flag is needed.

`--model-type llm` bypasses automatic diffusion-backend selection. The model path still needs model configuration, tokenizer, and chat-template files, but weight files are not read.

`--kv-capacity-estimator-config` accepts one JSON object. Only `kv_bytes_per_token` is required because it depends on the deployed model. All omitted simulation fields use defaults from the installed
`kv-capacity-estimator` wheel rather than defaults copied into SGLang:

| JSON field | Wheel default |
| --- | --- |
| `page_size` | `64` |
| `capacities` | `200GiB` through `32TiB` |
| `include_partial_page` | `false` |
| `target_hit_rate_ratio` | `0.99` |

`capacities` accepts a comma-separated string, an array of size strings, or `[label, bytes-or-size]` pairs. SGLang also accepts this adapter-specific field:

| JSON field | Default | Description |
| --- | --- | --- |
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

## Reset for a new workload

```bash
curl -X POST http://127.0.0.1:30000/kv-capacity-estimator/reset
```

Reset works during an active run or after finalization. It stops admission,
waits for admitted requests to finish, clears the LRU history, request numbering,
aggregate results, and Prometheus counters/histograms, then accepts new chat
requests. The response is the fresh status (`requests: 0`, `finalized: false`).
Tokenizer workers and simulation configuration are reused. Chat requests during
reset receive HTTP 409; retry after reset completes. Reset and finalize operations
are serialized. If `--api-key` is set, these endpoints require its Bearer token.

Reset does not finalize the previous run or modify its output file. Call finalize
first to save that run; the next finalization overwrites `output_path` with the
new run's result. Prometheus retains previously scraped samples and treats the
new zero counters as a reset.

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
- cumulative required-capacity histogram buckets for time-range quantiles.

`slide_window_size` and all `required_capacity_slide_window_*` series have
been removed. Remove the old JSON setting before starting the server. The
status endpoint now reports whole-trace counters and the mean capacity only;
select time ranges and request percentiles in Prometheus/Grafana.

`--extra-metric-labels` is supported. Capacity is represented by both the
human-readable `capacity` label and numeric `capacity_bytes` label.

## Time-range monitoring with Prometheus and Grafana

Scrape `/metrics` every 2 seconds (see [prometheus.yml](prometheus.yml)).
Import [grafana-dashboard.json](grafana-dashboard.json), select a Prometheus
source and model, then select `window` (10s, 20s, 30s, 1m, 5m, 10m, 30m, or 1h; default 30s) and request coverage `q`.
The dashboard shows token hit rates, a capacity/hit-rate table, and capacity
thresholds. The queries below also work directly in Prometheus.

### Capacity versus token hit rate

For each configured capacity, divide the increase of prefix-hit tokens by the
increase of all prompt tokens over the last five minutes:

```promql
sum by (model_name, capacity, capacity_bytes) (
  increase(sglang:kv_capacity_estimator_capacity_prefix_hit_tokens_total[5m])
)
/ on (model_name) group_left
sum by (model_name) (
  increase(sglang:prompt_tokens_total{engine_type="kv_capacity_estimator"}[5m])
)
```

Use `[10m]` for ten minutes, `[$window]` for the dashboard selector, or
`[$__range]` to cover the Grafana time picker range. Use an instant query for a
capacity/hit-rate snapshot; a range query tracks each capacity over time. In
Grafana, convert `capacity_bytes` to a numeric field and sort numerically when
plotting capacity on the X axis. Never average the cumulative hit-rate Gauges
over time: those Gauges remain available for whole-process compatibility.

These are prefix token hits, the reusable contiguous prefix before the first
miss. With `include_partial_page=false`, unmodeled trailing tokens still count
in the prompt-token denominator and cannot hit. With it enabled, a hit on the
partial final page counts only its actual token length. Page-hit curves remain
available using `capacity_page_hits_total / page_accesses_total` with the same
`increase()` and label matching; they measure simulated pages rather than all
prompt tokens.

### Capacity threshold for a fraction of requests

The p99 capacity over the last five minutes is:

```promql
histogram_quantile(
  0.99,
  sum by (model_name, le) (
    increase(sglang:kv_capacity_estimator_required_capacity_bytes_bucket[5m])
  )
)
```

Replace `0.99` with `0.9`, `0.95`, or `0.999`, and `[5m]` with `[10m]` as needed.
`target_hit_rate_ratio=0.99` means that **each request** aims to preserve 99% of
its unlimited-cache reusable prefix. The query's `0.99` means that **99% of
requests** need no more than the estimated capacity. These two ratios describe
different quantities. Cold requests with no reusable prefix need zero bytes
for that target and are included in the zero bucket, along with empty requests.

The histogram includes zero, quarter-octave bounds from 1 MiB to 1 PiB, and all
configured capacities as explicit bounds, plus `+Inf`. Quantiles use bucket
interpolation and are estimates, unlike the removed exact nearest-rank
request-window percentiles. A value in the overflow bucket is not a reliable
finite threshold; add larger configured capacities if needed.

For exact observed coverage at a bucket boundary, for example 1 GiB, use:

```promql
sum by (model_name) (
  increase(sglang:kv_capacity_estimator_required_capacity_bytes_bucket{le="1.073741824e+09"}[5m])
)
/ on (model_name)
sum by (model_name) (
  increase(sglang:kv_capacity_estimator_requests_total{status="success"}[5m])
)
```

Match the `le` label as stored by your Prometheus server (inspect bucket series
if its float formatting differs). Compare this coverage with the desired request
fraction and select the smallest bucket boundary meeting it to get a conservative observed threshold. Unreachable
requests increment `required_capacity_unreachable_total` and are excluded from
the histogram; therefore histogram quantiles are conditional on reachable
requests. The current simulator's relative-prefix target always has a finite
answer, but exporters retain this counter for compatibility. If it becomes
nonzero, check the reachable fraction before interpreting any quantile as
coverage of all requests.

A query interval only selects **observations**. It does not reset the LRU cache
history at its start. These thresholds describe recent traffic under the
accumulated trace, not a forecast of an unseen workload. Prometheus `increase()`
handles counter resets and extrapolates between scrapes; no-traffic ratios and
quantiles return NaN, and at least two scrapes are needed. Apply `increase()`
before summing across replicas. Only aggregate replicas with the same model,
page size, target ratio, and histogram bucket configuration; add deployment/job
filters or grouping when comparing separate workloads.

See the [Prometheus query function documentation](https://prometheus.io/docs/prometheus/latest/querying/functions/)
for counter increases and histogram interpolation.
