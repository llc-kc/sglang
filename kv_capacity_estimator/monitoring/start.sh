#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
monitoring_dir="$PWD"
prometheus_image="${PROMETHEUS_IMAGE:-prom/prometheus:latest}"
grafana_image="${GRAFANA_IMAGE:-grafana/grafana:latest}"
listen_address="${LISTEN_ADDRESS:-127.0.0.1}"
# Require local images; never pull during startup.
docker image inspect "$prometheus_image" >/dev/null
docker image inspect "$grafana_image" >/dev/null
for container in kv-estimator-prometheus kv-estimator-grafana; do
  if docker container inspect "$container" >/dev/null 2>&1; then
    echo "Container $container already exists. Run bash stop.sh before recreating it." >&2
    exit 1
  fi
done
# Named volumes persist when stop.sh removes the containers.
docker volume create kv-estimator-prometheus-data >/dev/null
docker volume create kv-estimator-grafana-data >/dev/null
docker run -d --pull=never --name kv-estimator-prometheus \
  --restart unless-stopped --network host \
  -v "$monitoring_dir/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  -v kv-estimator-prometheus-data:/prometheus \
  "$prometheus_image" \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --storage.tsdb.retention.time=15d \
  --web.listen-address="$listen_address:9090"
docker run -d --pull=never --name kv-estimator-grafana \
  --restart unless-stopped --network host \
  -e "GF_SERVER_HTTP_ADDR=$listen_address" \
  -e GF_SERVER_HTTP_PORT=3000 \
  -e GF_USERS_ALLOW_SIGN_UP=false \
  -e GF_SECURITY_ADMIN_USER=admin \
  -e "GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-admin}" \
  -v kv-estimator-grafana-data:/var/lib/grafana \
  -v "$monitoring_dir/grafana/provisioning:/etc/grafana/provisioning:ro" \
  -v "$monitoring_dir/grafana/dashboards:/var/lib/grafana/dashboards:ro" \
  "$grafana_image"
echo 'Prometheus: http://localhost:9090/targets'
echo 'Grafana: http://localhost:3000/d/kv-capacity-estimator'
