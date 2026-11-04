# SGLang KV capacity estimator 监控

此目录用于你已经启动的 `sglang serve ... --kv-capacity-estimator --enable-metrics`。
已核对本地 SGLang 实现及运行中的 `/metrics`，复用仓库内匹配的五面板仪表盘。
无需重启 SGLang，也不需要额外 exporter。

## 当前环境与连接方式

检查时发现本地已有 `prom/prometheus:latest`、`grafana/grafana:latest`，Docker 服务运行在 Ubuntu，且没有 `docker compose` 子命令。因此使用 `docker run` 脚本，不需要另装 Compose。

两个容器均使用 Linux Docker 的 `--network host`：

- Prometheus → `http://127.0.0.1:30000/metrics`（SGLang）。
- Grafana → `http://127.0.0.1:9090`（Prometheus）。
- 浏览器 → `http://localhost:3000`（Grafana）。

这里 host 网络使容器共享 Docker 宿主机网络，因此配置中的 `127.0.0.1` 才能访问宿主机服务。这份配置假设 SGLang 与 Docker 服务在同一个 Linux/WSL 网络环境；不适用于直接切换到普通 bridge 网络或其他远程 Docker 服务。如果 SGLang 运行在另一个容器，它需要通过 `-p 30000:30000` 发布端口，或使用同一个 host 网络。

默认 Prometheus 和 Grafana 只监听 `127.0.0.1`。在同一台 Windows 的 WSL 环境中，可先尝试从 Windows 浏览器打开 localhost；若未开启 WSL localhost 转发，可在对应 Linux 环境访问或配置端口转发。

## 启动

在 WSL/Linux 终端执行：

```bash
cd /root/luchangli/sglang_llc_kc/kv_capacity_estimator/monitoring
bash start.sh
```

脚本使用本地镜像，`--pull=never` 不会下载镜像。首次登录账号/密码默认 `admin` / `admin`，按 Grafana 页面提示修改。也可以在首次创建 Grafana 数据卷前指定密码：

```bash
GRAFANA_ADMIN_PASSWORD='替换为你的密码' bash start.sh
```

如本地镜像不是默认标签，可在启动时指定实际名称：

```bash
PROMETHEUS_IMAGE='prom/prometheus:你的标签' \
GRAFANA_IMAGE='grafana/grafana:你的标签' \
bash start.sh
```

账号密码初始化仅在新的 Grafana 数据卷上生效；已有数据卷的账号不会因环境变量而重置。

打开以下地址：

- Prometheus Targets：<http://localhost:9090/targets>，`kv-capacity-estimator` 应显示 **UP**。
- Grafana 仪表盘：<http://localhost:3000/d/kv-capacity-estimator>。
- 也可在 Grafana 的 Dashboards → SGLang 下找到 `KV capacity estimator`。

Prometheus 数据源与仪表盘均由 provisioning 文件自动加载，无需手动添加/导入。

## 验证采集

```bash
curl -fsS http://127.0.0.1:30000/metrics | head -n 20
curl -fsS http://127.0.0.1:9090/-/ready
curl -fsS http://127.0.0.1:3000/api/health
docker ps --filter name=kv-estimator-
```

在 Prometheus 查询页输入：

```promql
up{job="kv-capacity-estimator"}
```

值为 `1` 表示抓取成功。再查询：

```promql
sglang:kv_capacity_estimator_requests_total{status="success"}
```

应能看到服务启动以来成功请求的累计数量。

## 如何看仪表盘

选择 `Prometheus` 数据源、`job` 与 `model`；观察窗口 `window` 支持 `10s、20s、30s、1m、5m、10m、30m、1h`，默认 `30s`（`1m` 即 1 分钟），请求覆盖率 `q` 默认 `0.99`。

五个面板分别展示：

1. 各容量下前缀 token 命中率随时间变化。
2. 最近窗口内的所需容量分位数（只统计目标可达的请求，直方图插值估算）。
3. 各容量下前缀 token 命中率快照表。
4. 各直方图容量边界能够覆盖的请求比例（包括零容量冷请求，以全部成功请求为分母）。
5. 目标不可达的请求比例。

你的 `target_hit_rate_ratio=0.99` 表示每个请求要保留其无限缓存下可复用前缀的 99%；页面上的 `q=0.99` 表示对目标可达请求取所需容量的 99 分位。两者含义不同。

**请在 Prometheus 启动后继续发送测试流量。** 采集间隔为 2 秒，Grafana 数据源采样间隔同步为 2 秒，仪表盘每 5 秒刷新。`increase()` 至少需要两个采样点；10 秒窗口正常约含 5 个采样点。建议先等待约 10–15 秒并持续发送请求。Prometheus 无法补回启动前的历史采样；之前的请求会体现在初次抓取的累计计数中，但不会自动成为后续窗口内的请求增量。

没有新请求时命中率可能为 No data/NaN；这是窗口内分母为零。只有无复用前缀的冷请求时，所需容量可能为 0。测试命中行为应包含重复或共享前缀的请求。超过窗口没有新流量时，窗口图表也会变空。

短窗口按真实经过的时间统计，而不是仿真请求中的逻辑时间。缩短抓取间隔会增加采集与存储开销。旧的 15 秒采样历史无法补出 10 秒粒度的数据，请观察新配置生效后的时间段。查看短时变化时，可将右上角时间范围切换到最近 5 分钟。

## 日志、停止、重新创建

```bash
docker logs --tail 100 kv-estimator-prometheus
docker logs --tail 100 kv-estimator-grafana
```

停止并删除本套监控容器，保留历史数据，不操作 SGLang：

```bash
bash stop.sh
```

重新启动：

```bash
bash start.sh
```

启动脚本遇到同名容器会退出，不会自动覆盖它。修改 Prometheus 配置后可执行 `docker restart kv-estimator-prometheus`；修改 Grafana provisioning 后可执行 `docker restart kv-estimator-grafana`。修改启动环境变量/镜像需先 `bash stop.sh` 再启动。

Prometheus 历史保留 15 天。命名数据卷分别为 `kv-estimator-prometheus-data` 和 `kv-estimator-grafana-data`，停止脚本不会删除它们。

## 远程访问

从其他机器访问，优先通过 SSH 隧道转发本地端口：

```bash
ssh -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 用户名@运行Docker的主机
```

然后在本机浏览器打开 localhost。如果明确需要监听全部网卡，可在重新创建容器时设置 `LISTEN_ADDRESS=0.0.0.0 bash start.sh`；需自行限制网络访问，Prometheus 默认没有登录认证。

## 文件与校验

- `start.sh` / `stop.sh`：启动、停止脚本。
- `prometheus.yml`：2 秒抓取一次 SGLang `/metrics`。
- `grafana/provisioning/datasources/prometheus.yml`：自动创建 Prometheus 数据源。
- `grafana/provisioning/dashboards/default.yml`：自动加载仪表盘。
- `grafana/dashboards/kv-capacity-estimator.json`：匹配本地实现的仪表盘。

生成时已通过 `promtool check config`、Bash 语法检查、YAML/JSON 解析，并核对所有面板查询使用的指标均由当前运行中的 SGLang 导出。尚未启动长期运行的监控容器，也未验证浏览器显示。

参考官方文档：[Prometheus Docker 安装](https://prometheus.io/docs/prometheus/latest/installation/)；[Grafana provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)。
