#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Uso: $0 DATASET.jsonl NOMBRE [argumentos adicionales de train.py]" >&2
  exit 2
fi

dataset=$1
adapter_name=$2
shift 2

if [[ ! -x trainer/.venv/bin/python ]]; then
  echo "Falta trainer/.venv. Créalo e instala trainer/requirements.txt." >&2
  exit 1
fi

trainer/.venv/bin/python trainer/validate_dataset.py "$dataset"
vllm_running=false
if docker compose ps --status running --services 2>/dev/null | grep -qx vllm; then
  vllm_running=true
  docker compose stop vllm
fi

restart_vllm() {
  if [[ "$vllm_running" == true ]]; then docker compose up -d vllm; fi
}
trap restart_vllm EXIT

trainer/.venv/bin/python trainer/train.py --dataset "$dataset" --name "$adapter_name" "$@"
