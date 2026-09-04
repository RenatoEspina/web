#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Uso: $0 DATASET.jsonl NOMBRE [argumentos adicionales de train.py]" >&2
  exit 2
fi

dataset=$1
adapter_name=$2
shift 2

if [[ ! -f "$dataset" ]]; then
  echo "No existe el dataset '$dataset'." >&2
  exit 1
fi

trainer_python=${FINE_TUNE_PYTHON:-}
if [[ -z "$trainer_python" ]]; then
  if [[ -x trainer/.venv/bin/python ]]; then
    trainer_python=trainer/.venv/bin/python
  elif [[ -x .venv/bin/python ]]; then
    # Compatibilidad con un venv raíz que el usuario ya haya preparado.
    trainer_python=.venv/bin/python
  fi
fi

if [[ -z "$trainer_python" || ! -x "$trainer_python" ]]; then
  echo "Falta el entorno Python de fine-tuning. Ejecuta ./comandos.fish fine-tune-setup." >&2
  exit 1
fi

"$trainer_python" trainer/validate_dataset.py "$dataset"

# El CLI de Fish usa este nombre de proyecto explícitamente. Mantenerlo aquí
# evita que `docker compose ps` consulte otro proyecto y deje vLLM ocupando la
# GPU durante el entrenamiento. Se puede sobrescribir para instalaciones
# distintas.
compose_project=${COMPOSE_PROJECT_NAME:-llm-bridge}
compose() {
  docker compose -p "$compose_project" "$@"
}

vllm_running=false
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  if compose ps --status running --services 2>/dev/null | grep -qx vllm; then
    vllm_running=true
    compose stop vllm
  fi
else
  echo "Aviso: Docker no está disponible; no se comprobará si vLLM ocupa la GPU." >&2
fi

restart_vllm() {
  if [[ "$vllm_running" == true ]]; then
    echo "Reiniciando vLLM..."
    compose up -d vllm || echo "Aviso: el entrenamiento terminó, pero vLLM no pudo reiniciarse automáticamente." >&2
  fi
}
trap restart_vllm EXIT

"$trainer_python" trainer/train.py --dataset "$dataset" --name "$adapter_name" "$@"
