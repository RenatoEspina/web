#!/usr/bin/env fish

# Arranque local de vLLM/Ollama para LLM Bridge Chat.
# Uso:
#   ./comandos.fish vllm [modelo]
#   ./comandos.fish ollama [modelo]
#   ./comandos.fish status
#   ./comandos.fish stop
#   ./comandos.fish down

set -l project_dir (dirname (status --current-filename))
cd "$project_dir"; or exit 1

set -g compose_project "llm-bridge"
set -g default_ollama_model "llama3.2:1b-instruct-fp16"

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

function start_vllm
    set -l model "Qwen/Qwen3.5-0.8B"
    if set -q VLLM_MODEL; and test -n "$VLLM_MODEL"
        set model "$VLLM_MODEL"
    end
    if test (count $argv) -ge 1; and test -n "$argv[1]"
        set model "$argv[1]"
    end
    set -gx VLLM_MODEL $model

    stop_service ollama

    if not set -q HF_TOKEN; or test -z "$HF_TOKEN"
        read -P 'HF_TOKEN de Hugging Face: ' -s HF_TOKEN
        echo
    end
    set -gx HF_TOKEN $HF_TOKEN

    if not set -q VLLM_IMAGE
        set -gx VLLM_IMAGE "vllm/vllm-openai:v0.24.0"
    end

    require_command curl

    if not docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1
        echo "Descargando $VLLM_IMAGE..."
        compose pull vllm; or fail "No fue posible descargar la imagen de vLLM."
    end

    echo "Modelo vLLM: $model"
    echo "Iniciando vLLM. La primera ejecución puede descargar el modelo..."
    compose up -d vllm; or begin
        compose logs --no-color --tail=100 vllm
        fail "No fue posible iniciar vLLM."
    end

    wait_for_url vllm http://127.0.0.1:8000/health 900
    echo "vLLM está disponible en http://127.0.0.1:8000"
    compose ps vllm
end

function start_ollama
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
    compose exec -T ollama ollama pull "$model"; or fail "No fue posible descargar el modelo '$model'."

    echo "Ollama está disponible en http://127.0.0.1:11434"
    echo "Asegúrate de usar LLM_MODEL=$model en .env.local."
    compose ps ollama
end

function show_status
    compose ps
    echo
    echo "vLLM:   http://127.0.0.1:8000/health"
    echo "Ollama: http://127.0.0.1:11434/api/tags"
end

require_command docker
docker_ready

if test (count $argv) -eq 0
    echo "Uso: ./comandos.fish {vllm|ollama|status|stop|down} [modelo]"
    exit 1
end

switch $argv[1]
    case vllm
        start_vllm $argv[2]
    case ollama
        start_ollama $argv[2]
    case status
        show_status
    case stop
        compose stop vllm ollama
    case down
        compose down --remove-orphans
    case '*'
        echo "Acción desconocida: $argv[1]" >&2
        echo "Uso: ./comandos.fish {vllm|ollama|status|stop|down} [modelo]"
        exit 1
end
