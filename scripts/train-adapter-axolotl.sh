#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Uso: $0 DATASET.jsonl NOMBRE [argumentos adicionales de train_axolotl.py]" >&2
  exit 2
fi

dataset=$1
adapter_name=$2
shift 2

# --- lock GPU/vLLM (bloque idéntico en scripts/train-adapter.sh y scripts/train-adapter-axolotl.sh) ---
# Exclusión mutua con las GUIs (trainer/runtime_lock.py) y con otros
# entrenamientos por CLI. Una GUI que ya tomó el lock lo pasa heredado en
# LLM_BRIDGE_GPU_LOCK_FD; solo se acepta si ese descriptor es el mismo archivo.
gpu_lock_file="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/.runtime/gpu.lock"
acquire_gpu_lock() {
  if ! command -v flock >/dev/null 2>&1; then
    echo "Error: falta 'flock' (util-linux); no se puede garantizar el uso exclusivo de la GPU." >&2
    exit 1
  fi
  local inherited=${LLM_BRIDGE_GPU_LOCK_FD:-}
  if [[ "$inherited" =~ ^[0-9]+$ && -e "/dev/fd/$inherited" && "/dev/fd/$inherited" -ef "$gpu_lock_file" ]] \
    && flock -n "$inherited"; then
    return 0
  fi
  mkdir -p -- "$(dirname -- "$gpu_lock_file")"
  exec {gpu_lock_fd}>>"$gpu_lock_file"
  if ! flock -n "$gpu_lock_fd"; then
    echo "Error: otra operación de GPU/vLLM está en curso (GUI o CLI; lock $gpu_lock_file). Espera a que termine antes de entrenar." >&2
    exit 1
  fi
}
acquire_gpu_lock
# --- fin lock GPU/vLLM ---

if [[ ! -f "$dataset" && "$dataset" == *"terraria-training.jsonl" ]]; then
  python_builder=$(command -v python3 || command -v python || true)
  [[ -n "$python_builder" ]] || { echo "No se encontró Python para reconstruir Terraria." >&2; exit 1; }
  "$python_builder" trainer/corpora/terraria/build.py
fi

if [[ "$dataset" == *"unidades-training.jsonl" && ( ! -f "$dataset" || ! -f "trainer/corpora/unidades/validation.jsonl" ) ]]; then
  python_builder=$(command -v python3 || command -v python || true)
  [[ -n "$python_builder" ]] || { echo "No se encontró Python para reconstruir unidades." >&2; exit 1; }
  "$python_builder" trainer/corpora/unidades/build.py
fi

if [[ "$dataset" == *"dolly-training.jsonl" && ( ! -f "$dataset" || ! -f "trainer/corpora/dolly/validation.jsonl" ) ]]; then
  python_builder=$(command -v python3 || command -v python || true)
  [[ -n "$python_builder" ]] || { echo "No se encontró Python para reconstruir Dolly." >&2; exit 1; }
  "$python_builder" trainer/corpora/dolly/build.py
fi

[[ -f "$dataset" ]] || { echo "No existe el dataset '$dataset'." >&2; exit 1; }

trainer_python=${FINE_TUNE_AXOLOTL_PYTHON:-trainer/.venv-axolotl/bin/python}
if [[ ! -x "$trainer_python" ]]; then
  echo "Falta el entorno Axolotl. Inicia ./fine-tune-gui-axolotl y usa 'Preparar / actualizar entorno'." >&2
  exit 1
fi

"$trainer_python" trainer/check_axolotl_environment.py
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
  elif [[ "$dataset" == *"unidades-training.jsonl" ]]; then
    validation_dataset="trainer/corpora/unidades/validation.jsonl"
  elif [[ "$dataset" == *"dolly-training.jsonl" ]]; then
    validation_dataset="trainer/corpora/dolly/validation.jsonl"
  elif [[ "$dataset" == *-training.jsonl ]]; then
    sibling_validation="${dataset%-training.jsonl}-validation.jsonl"
    [[ -f "$sibling_validation" ]] && validation_dataset="$sibling_validation"
  fi

  if [[ -n "$validation_dataset" && -f "$validation_dataset" ]]; then
    echo "== Validación separada detectada: $validation_dataset =="
    "$trainer_python" trainer/validate_dataset.py "$validation_dataset"
    extra_args+=("--validation-dataset" "$validation_dataset")
  fi
fi

compose_project=${COMPOSE_PROJECT_NAME:-llm-bridge}
compose() {
  docker compose -p "$compose_project" "$@"
}

vllm_running=false
vllm_model=""
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  if compose ps --status running --services 2>/dev/null | grep -qx vllm; then
    vllm_running=true
    vllm_model=$("$trainer_python" - <<'PY' 2>/dev/null || true
import json
from urllib.request import urlopen
try:
    with urlopen("http://127.0.0.1:8000/v1/models", timeout=3) as response:
        data = json.load(response).get("data", [])
    print(next((item.get("id") for item in data if isinstance(item, dict) and item.get("id") and not item.get("parent")), ""))
except Exception:
    print("")
PY
)
    [[ -n "$vllm_model" ]] && echo "Modelo vLLM detectado antes del entrenamiento: $vllm_model"
    # Detener vLLM descarta los LoRA cargados: la reconciliación de la allowlist
    # queda pendiente hasta completarse (trainer/runtime_adapters.py borra esta
    # marca). Si una cancelación la interrumpe, la GUI la termina al volver vLLM.
    mkdir -p -- "$(dirname -- "$gpu_lock_file")"
    : > "$(dirname -- "$gpu_lock_file")/adapters-reconcile-pending"
    compose stop vllm
  fi
else
  echo "Aviso: Docker no está disponible; no se comprobará si vLLM ocupa la GPU." >&2
fi

restart_vllm() {
  if [[ "$vllm_running" == true ]]; then
    echo "Reiniciando vLLM..."
    local restarted=true
    if [[ -n "$vllm_model" ]]; then
      VLLM_MODEL="$vllm_model" compose up -d vllm || { restarted=false; echo "Aviso: vLLM no pudo reiniciarse automáticamente." >&2; }
    else
      compose up -d vllm || { restarted=false; echo "Aviso: vLLM no pudo reiniciarse automáticamente." >&2; }
    fi
    if [[ "$restarted" == true ]]; then
      # El reinicio descarta los LoRA cargados dinámicamente: se recargan con
      # validación de base o se retiran de LLM_ADAPTER_MODELS. La exportación
      # Qwen3.5 usa este mismo intérprete Axolotl.
      FINE_TUNE_PYTHON="$trainer_python" "$trainer_python" trainer/runtime_adapters.py --wait-vllm 600 || \
        echo "Aviso: no se pudo reconciliar LLM_ADAPTER_MODELS; revisa la allowlist en la GUI." >&2
    fi
  fi
}
trap restart_vllm EXIT

"$trainer_python" trainer/train_axolotl.py --dataset "$dataset" --name "$adapter_name" "${extra_args[@]}"
