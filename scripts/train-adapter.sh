#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Uso: $0 DATASET.jsonl NOMBRE [argumentos adicionales de train.py]" >&2
  exit 2
fi

dataset=$1
adapter_name=$2
shift 2

# Los datasets Terraria JSONL son artefactos generados a partir del corpus fuente
# curado. Si aún no existen (por ejemplo tras un clone limpio), materialízalos
# antes de validar el path solicitado.
if [[ ! -f "$dataset" && "$dataset" == *"terraria-training.jsonl" ]]; then
  python_builder=$(command -v python3 || command -v python || true)
  if [[ -z "$python_builder" ]]; then
    echo "No existe '$dataset' y no se encontró Python para reconstruir el corpus Terraria." >&2
    exit 1
  fi
  "$python_builder" trainer/corpora/terraria/build.py
fi

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

# Un venv existente no implica que siga siendo compatible con el repositorio.
# Verifica los pins y la ruta de fingerprint de Hugging Face antes de reservar
# VRAM. Si está desactualizado, sincroniza requirements.txt una sola vez.
if ! "$trainer_python" trainer/check_dependencies.py; then
  echo "== Sincronizando dependencias del fine-tuning =="
  "$trainer_python" -m pip install -r trainer/requirements.txt
  "$trainer_python" trainer/check_dependencies.py
fi

"$trainer_python" trainer/validate_dataset.py "$dataset"

extra_args=("$@")
has_validation=false
for arg in "${extra_args[@]}"; do
  if [[ "$arg" == "--validation-dataset" || "$arg" == --validation-dataset=* ]]; then
    has_validation=true
    break
  fi
done

if [[ "$has_validation" == false ]]; then
  validation_dataset=""
  if [[ "$dataset" == *"terraria-training.jsonl" ]]; then
    validation_dataset="trainer/corpora/terraria/validation.jsonl"
  elif [[ "$dataset" == *-training.jsonl ]]; then
    sibling_validation="${dataset%-training.jsonl}-validation.jsonl"
    if [[ -f "$sibling_validation" ]]; then
      validation_dataset="$sibling_validation"
    fi
  fi

  if [[ -n "$validation_dataset" && -f "$validation_dataset" ]]; then
    echo "== Validación separada detectada: $validation_dataset =="
    "$trainer_python" trainer/validate_dataset.py "$validation_dataset"
    extra_args+=("--validation-dataset" "$validation_dataset")
  fi
fi

# El CLI de Fish usa este nombre de proyecto explícitamente. Mantenerlo aquí
# evita que `docker compose ps` consulte otro proyecto y deje vLLM ocupando la
# GPU durante el entrenamiento. Se puede sobrescribir para instalaciones
# distintas.
compose_project=${COMPOSE_PROJECT_NAME:-llm-bridge}
compose() {
  docker compose -p "$compose_project" "$@"
}

vllm_running=false
vllm_model=""
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  if compose ps --status running --services 2>/dev/null | grep -qx vllm; then
    vllm_running=true
    # Guarda el modelo base realmente servido antes de liberar la GPU. Esto es
    # más fiable que asumir el default de docker-compose al restaurar el servicio.
    vllm_model=$("$trainer_python" - <<'PY' 2>/dev/null || true
import json
from urllib.request import urlopen

try:
    with urlopen("http://127.0.0.1:8000/v1/models", timeout=3) as response:
        data = json.load(response).get("data", [])
    base = next(
        (
            item.get("id")
            for item in data
            if isinstance(item, dict) and item.get("id") and not item.get("parent")
        ),
        "",
    )
    print(base or "")
except Exception:
    print("")
PY
)
    if [[ -n "$vllm_model" ]]; then
      echo "Modelo vLLM detectado antes del entrenamiento: $vllm_model"
    else
      echo "Aviso: no se pudo detectar el modelo base activo; el reinicio usará la configuración normal de Compose." >&2
    fi
    compose stop vllm
  fi
else
  echo "Aviso: Docker no está disponible; no se comprobará si vLLM ocupa la GPU." >&2
fi

restart_vllm() {
  if [[ "$vllm_running" == true ]]; then
    echo "Reiniciando vLLM..."
    if [[ -n "$vllm_model" ]]; then
      VLLM_MODEL="$vllm_model" compose up -d vllm || \
        echo "Aviso: el entrenamiento terminó, pero vLLM no pudo reiniciarse automáticamente." >&2
    else
      compose up -d vllm || \
        echo "Aviso: el entrenamiento terminó, pero vLLM no pudo reiniciarse automáticamente." >&2
    fi
  fi
}
trap restart_vllm EXIT

"$trainer_python" trainer/train.py --dataset "$dataset" --name "$adapter_name" "${extra_args[@]}"
