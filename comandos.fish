#!/usr/bin/env fish

# Arranque local de vLLM/Ollama para LLM Bridge Chat.
# Uso:
#   ./comandos.fish vllm [modelo]       # vLLM + embeddings de Ollama
#   ./comandos.fish ollama [modelo]
#   ./comandos.fish embeddings [modelo]
#   ./comandos.fish fine-tune-setup
#   ./comandos.fish fine-tune-check
#   ./comandos.fish fine-tune-validate DATASET.jsonl
#   ./comandos.fish fine-tune-train DATASET.jsonl NOMBRE [opciones]
#   ./comandos.fish fine-tune-config NOMBRE
#   ./comandos.fish fine-tune-list
#   ./comandos.fish fine-tune-evaluate DATASET.jsonl MODELO SALIDA.json
#   ./comandos.fish status
#   ./comandos.fish stop
#   ./comandos.fish down

set -g project_dir (dirname (status --current-filename))
cd "$project_dir"; or exit 1

set -g compose_project "llm-bridge"
set -g default_ollama_model "llama3.2:1b-instruct-fp16"
set -g default_embedding_model "qwen3-embedding:4b"

function compose
    docker compose -p $compose_project $argv
end

function fail
    echo "Error: $argv[1]" >&2
    exit 1
end

function require_command
    command -sq $argv[1]; or fail "No se encontró el comando '$argv[1]'."
end

function docker_ready
    docker info >/dev/null 2>&1; or fail "Docker no está disponible. Ejecuta: sudo systemctl start docker"
end

function require_docker
    require_command docker
    docker_ready
end

function trainer_python
    if set -q FINE_TUNE_PYTHON; and test -n "$FINE_TUNE_PYTHON"; and test -x "$FINE_TUNE_PYTHON"
        echo "$FINE_TUNE_PYTHON"
    else if test -x "$project_dir/trainer/.venv/bin/python"
        echo "$project_dir/trainer/.venv/bin/python"
    else if test -x "$project_dir/.venv/bin/python"
        # Permite reutilizar el venv raíz que ya tenga el usuario, pero el
        # entorno documentado y preferido sigue siendo trainer/.venv.
        echo "$project_dir/.venv/bin/python"
    else
        echo "$project_dir/trainer/.venv/bin/python"
    end
end

function require_trainer
    set -l python_path (trainer_python)
    test -x "$python_path"; or fail \
        "No existe el entorno de fine-tuning. Ejecuta: ./comandos.fish fine-tune-setup"
end

function fine_tune_help
    echo "Uso del CLI de fine-tuning:"
    echo
    echo "  ./comandos.fish fine-tune-setup"
    echo "      Crea trainer/.venv e instala las dependencias fijadas."
    echo
    echo "  ./comandos.fish fine-tune-check"
    echo "      Comprueba PyTorch, CUDA, GPU y una operación NF4 de bitsandbytes."
    echo
    echo "  ./comandos.fish fine-tune-validate DATASET.jsonl"
    echo "      Valida el formato conversacional sin iniciar Docker."
    echo
    echo "  ./comandos.fish fine-tune-train DATASET.jsonl NOMBRE [opciones]"
    echo "      Detiene vLLM, entrena QLoRA y lo vuelve a iniciar si estaba activo."
    echo
    echo "  ./comandos.fish fine-tune-config NOMBRE"
    echo "      Comprueba el adaptador y muestra cómo registrarlo en vLLM y la web."
    echo
    echo "  ./comandos.fish fine-tune-list"
    echo "      Lista los adaptadores con manifest.json."
    echo
    echo "  ./comandos.fish fine-tune-evaluate DATASET.jsonl MODELO SALIDA.json"
    echo "      Evalúa un modelo ya servido por vLLM."
end

function fine_tune_setup
    set -l python_command python

    if command -sq python3.13
        set python_command python3.13
    end

    require_command $python_command

    set -l existing_python (trainer_python)
    if test -x "$existing_python"
        echo "Ya existe un entorno Python en $existing_python; se actualizarán sus dependencias."
    else
        echo "Creando trainer/.venv con $python_command..."
        $python_command -m venv trainer/.venv; or fail \
            "No fue posible crear el entorno virtual con $python_command."
    end

    set -l python_path (trainer_python)
    "$python_path" -m pip install --upgrade pip; or fail \
        "No fue posible actualizar pip."
    "$python_path" -m pip install -r trainer/requirements.txt; or fail \
        "No fue posible instalar las dependencias de fine-tuning."

    echo
    fine_tune_check
end

function fine_tune_check
    require_trainer
    set -l python_path (trainer_python)
    "$python_path" -c 'import torch; assert torch.cuda.is_available(), "CUDA no está disponible"; import bitsandbytes; from bitsandbytes.functional import quantize_4bit; x=torch.ones((2,2), device="cuda"); quantize_4bit(x, quant_type="nf4"); print(f"PyTorch: {torch.__version__}"); print(f"CUDA de PyTorch: {torch.version.cuda}"); print(f"GPU: {torch.cuda.get_device_name(0)}"); print(f"bitsandbytes: {bitsandbytes.__version__}"); print("bitsandbytes NF4: OK")'
    if test $status -ne 0
        fail "La comprobación CUDA/bitsandbytes falló. No inicies el entrenamiento."
    end
end

function fine_tune_validate
    if test (count $argv) -ne 1
        fail "Uso: ./comandos.fish fine-tune-validate DATASET.jsonl"
    end
    require_trainer
    test -f "$argv[1]"; or fail "No existe el dataset '$argv[1]'."
    set -l python_path (trainer_python)
    "$python_path" trainer/validate_dataset.py "$argv[1]"
end

function fine_tune_train
    if test (count $argv) -lt 2
        fail "Uso: ./comandos.fish fine-tune-train DATASET.jsonl NOMBRE [opciones]"
    end
    require_trainer
    require_docker
    test -f "$argv[1]"; or fail "No existe el dataset '$argv[1]'."
    fine_tune_validate "$argv[1]" >/dev/null; or exit 1

    set -l dataset "$argv[1]"
    set -l adapter_name "$argv[2]"
    set -l extra_args $argv[3..-1]
    set -lx FINE_TUNE_PYTHON (trainer_python)
    set -lx COMPOSE_PROJECT_NAME $compose_project
    echo "Dataset validado. Iniciando entrenamiento de '$adapter_name'..."
    bash scripts/train-adapter.sh "$dataset" "$adapter_name" $extra_args; or fail \
        "El entrenamiento terminó con error. Revisa el log anterior."

    if test -f "adapters/$adapter_name/manifest.json"
        echo
        echo "Adaptador creado: adapters/$adapter_name"
        echo "Manifiesto: adapters/$adapter_name/manifest.json"
    end
end

function fine_tune_config
    if test (count $argv) -ne 1
        fail "Uso: ./comandos.fish fine-tune-config NOMBRE"
    end

    set -l adapter_name "$argv[1]"
    if not string match -rq '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$' -- "$adapter_name"
        fail "Nombre de adaptador inválido: usa letras, números, punto, guion o guion bajo (máximo 64)."
    end

    set -l adapter_dir "adapters/$adapter_name"
    test -d "$adapter_dir"; or fail "No existe el adaptador '$adapter_dir'."
    test -f "$adapter_dir/adapter_config.json"; or fail "Falta $adapter_dir/adapter_config.json."
    test -f "$adapter_dir/adapter_model.safetensors"; or fail "Falta $adapter_dir/adapter_model.safetensors."

    echo "Adaptador válido: $adapter_dir"
    echo
    echo "1. En docker-compose.yml, dentro de command de vllm, agrega:"
    echo "   - --lora-modules"
    echo "   - $adapter_name=/adapters/$adapter_name"
    echo
    echo "2. En .env.local, permite el nombre que enviará el gateway:"
    echo "   LLM_ADAPTER_MODELS=$adapter_name"
    echo
    echo "3. Recrea vLLM desde la raíz del proyecto:"
    echo "   ./comandos.fish stop"
    echo "   ./comandos.fish vllm"
    echo
    echo "4. Comprueba los modelos publicados:"
    echo "   curl -fsS http://127.0.0.1:8000/v1/models"
end

function fine_tune_list
    if not test -d adapters
        echo "No existe adapters/. Todavía no hay adaptadores entrenados."
        return 0
    end

    set -l python_command python
    if test -x (trainer_python)
        set python_command (trainer_python)
    end

    set -l found false
    for manifest in adapters/*/manifest.json
        if not test -f "$manifest"
            continue
        end
        set found true
        set -l adapter_dir (path dirname (path dirname "$manifest"))
        set -l adapter_name (path basename "$adapter_dir")
        set -l summary ("$python_command" -c 'import json,sys; m=json.load(open(sys.argv[1], encoding="utf-8")); p=m.get("parameters",{}); print("base={} | ejemplos={} | rank={} | creado={}".format(m.get("baseModel","?"),m.get("examples","?"),p.get("rank","?"),m.get("createdAt","?")))' "$manifest" 2>/dev/null)
        if test (count $summary) -eq 0
            set summary "manifest.json inválido"
        end
        echo "$adapter_name · $summary"
    end

    if test "$found" = false
        echo "No se encontraron manifest.json dentro de adapters/."
    end
end

function fine_tune_evaluate
    if test (count $argv) -ne 3
        fail "Uso: ./comandos.fish fine-tune-evaluate DATASET.jsonl MODELO SALIDA.json"
    end
    require_trainer
    test -f "$argv[1]"; or fail "No existe el dataset '$argv[1]'."
    set -l python_path (trainer_python)
    "$python_path" trainer/evaluate.py \
        --dataset "$argv[1]" \
        --model "$argv[2]" \
        --output "$argv[3]"
end

function service_running
    set -l service $argv[1]
    set -l container_id (compose ps -q $service 2>/dev/null)

    if test (count $container_id) -eq 0
        return 1
    end

    test (docker inspect --format '{{.State.Running}}' $container_id 2>/dev/null) = true
end

function wait_for_url
    set -l service $argv[1]
    set -l url $argv[2]
    set -l max_seconds $argv[3]
    set -l started (date +%s)

    while not curl -fsS "$url" >/dev/null 2>&1
        if not service_running $service
            echo
            compose logs --no-color --tail=100 $service
            fail "$service terminó antes de responder en $url"
        end

        set -l elapsed (math (date +%s) - $started)

        if test $elapsed -ge $max_seconds
            echo
            compose logs --no-color --tail=100 $service
            fail "Timeout esperando $service ($max_seconds segundos)."
        end

        sleep 1
    end
end

function stop_service
    compose stop $argv[1] >/dev/null 2>&1
end

function start_embeddings
    require_docker
    set -l model $argv[1]

    if test -z "$model"
        set model $default_embedding_model
    end

    require_command curl

    echo "Iniciando Ollama para embeddings..."

    compose up -d ollama; or begin
        compose logs --no-color --tail=100 ollama
        fail "No fue posible iniciar Ollama para embeddings."
    end

    wait_for_url ollama http://127.0.0.1:11434/api/tags 120

    echo "Descargando/verificando el modelo de embeddings: $model"

    compose exec -T ollama ollama pull "$model"; or fail \
        "No fue posible descargar el modelo de embeddings '$model'."

    set -l embedding_payload "{\"model\":\"$model\",\"input\":\"prueba de embeddings\"}"

    curl -fsS http://127.0.0.1:11434/api/embed \
        -H "Content-Type: application/json" \
        -d "$embedding_payload" \
        >/dev/null; or fail \
        "Ollama está activo, pero no pudo generar embeddings con '$model'."

    echo "Embeddings disponibles en http://127.0.0.1:11434 usando $model"
end

function start_vllm
    require_docker
    set -l model "Qwen/Qwen3.5-0.8B"

    if set -q VLLM_MODEL; and test -n "$VLLM_MODEL"
        set model "$VLLM_MODEL"
    end

    if test (count $argv) -ge 1; and test -n "$argv[1]"
        set model "$argv[1]"
    end

    set -gx VLLM_MODEL $model

    if not set -q HF_TOKEN; or test -z "$HF_TOKEN"
        read -P 'HF_TOKEN de Hugging Face: ' -s HF_TOKEN
        echo
    end

    set -gx HF_TOKEN $HF_TOKEN

    if not set -q VLLM_IMAGE
        set -gx VLLM_IMAGE "vllm/vllm-openai:v0.24.0"
    end

    require_command curl

    # Ollama genera los embeddings y vLLM genera las respuestas.
    start_embeddings

    if not docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1
        echo "Descargando $VLLM_IMAGE..."
        compose pull vllm; or fail \
            "No fue posible descargar la imagen de vLLM."
    end

    echo "Modelo vLLM: $model"
    echo "Iniciando vLLM. La primera ejecución puede descargar el modelo..."

    compose up -d vllm; or begin
        compose logs --no-color --tail=100 vllm
        fail "No fue posible iniciar vLLM."
    end

    wait_for_url vllm http://127.0.0.1:8000/health 900

    echo "vLLM está disponible en http://127.0.0.1:8000"
    compose ps vllm ollama
end

function start_ollama
    require_docker
    set -l model $argv[1]

    if test -z "$model"
        set model $default_ollama_model
    end

    stop_service vllm
    require_command curl

    echo "Iniciando Ollama..."

    compose up -d ollama; or begin
        compose logs --no-color --tail=100 ollama
        fail "No fue posible iniciar Ollama."
    end

    wait_for_url ollama http://127.0.0.1:11434/api/tags 120

    echo "Descargando/verificando el modelo Ollama: $model"

    compose exec -T ollama ollama pull "$model"; or fail \
        "No fue posible descargar el modelo '$model'."

    echo "Ollama está disponible en http://127.0.0.1:11434"
    echo "Asegúrate de usar LLM_MODEL=$model en .env.local."

    compose ps ollama
end

function show_status
    require_docker
    compose ps

    echo
    echo "vLLM:       http://127.0.0.1:8000/health"
    echo "Ollama:     http://127.0.0.1:11434/api/tags"
    echo "Embeddings: http://127.0.0.1:11434/api/embed"
end

if test (count $argv) -eq 0
    echo "Uso: ./comandos.fish {vllm|ollama|embeddings|fine-tune-*|status|stop|down} [argumentos]"
    echo
    fine_tune_help
    exit 1
end

switch $argv[1]
    case vllm
        start_vllm $argv[2]

    case ollama
        start_ollama $argv[2]

    case embeddings
        start_embeddings $argv[2]

    case fine-tune-help help
        fine_tune_help

    case fine-tune-setup
        fine_tune_setup

    case fine-tune-check
        fine_tune_check

    case fine-tune-validate
        fine_tune_validate $argv[2..-1]

    case fine-tune-train
        fine_tune_train $argv[2..-1]

    case fine-tune-config
        fine_tune_config $argv[2..-1]

    case fine-tune-list
        fine_tune_list

    case fine-tune-evaluate
        fine_tune_evaluate $argv[2..-1]

    case status
        show_status

    case stop
        require_docker
        compose stop vllm ollama

    case down
        require_docker
        compose down --remove-orphans

    case '*'
        echo "Acción desconocida: $argv[1]" >&2
        echo "Acciones: vllm, ollama, embeddings, fine-tune-setup, fine-tune-check, fine-tune-validate, fine-tune-train, fine-tune-config, fine-tune-list, fine-tune-evaluate, status, stop, down"
        exit 1
end
