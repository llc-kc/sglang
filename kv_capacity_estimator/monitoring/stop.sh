#!/usr/bin/env bash
set -euo pipefail
# Remove only these monitoring containers; keep all stored data.
for container in kv-estimator-grafana kv-estimator-prometheus; do
  if docker container inspect "$container" >/dev/null 2>&1; then
    docker stop "$container"
    docker rm "$container"
  fi
done
