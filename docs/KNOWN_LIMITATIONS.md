# Limitaciones y consideraciones conocidas

Este documento registra decisiones que se mantienen deliberadamente fuera de la prioridad inmediata. No representan necesariamente errores del flujo principal actual.

## Alcance operativo actual

El perfil de referencia del proyecto es:

- **Generación:** vLLM.
- **Embeddings:** Ollama ejecutando `qwen3-embedding:4b` en CPU.
- **Despliegue:** gateway local expuesto opcionalmente mediante Cloudflare Tunnel.
- **Hardware:** la configuración de Docker está ajustada al equipo usado durante la investigación.

Mientras este perfil siga siendo el objetivo principal, la portabilidad hacia otros engines o hardware no debe condicionar las correcciones funcionales del sistema.

## Cambio del modelo de embeddings

Los documentos mantienen sus vectores en memoria junto con metadatos del proveedor, modelo y dimensión usados al indexarlos. El flujo actual presupone que `qwen3-embedding:4b` continúa siendo el modelo de embeddings durante la vida del índice.

Cambiar `EMBEDDING_MODEL`, sus prefijos o el proveedor después de haber indexado documentos puede dejar vectores antiguos en memoria. Antes de considerar soportado ese cambio debe implementarse una política explícita de reindexado o invalidación del índice.

**Práctica actual:** si se cambia la configuración de embeddings, reiniciar el gateway y volver a cargar los PDF.

## Engines generativos alternativos

El gateway conserva un adaptador para Ollama y una abstracción OpenAI-compatible, pero las pruebas, el ciclo LoRA, CAG con prefix caching y la operación normal se priorizan para vLLM.

No se considera prioridad inmediata garantizar equivalencia funcional completa entre vLLM y otros engines. En particular, health checks, gestión dinámica de LoRA y reutilización del prefijo documental pueden tener semánticas distintas.

## Configuración específica del equipo

`docker-compose.yml` contiene parámetros deliberadamente ajustados al entorno de investigación, incluyendo límites de VRAM, CPU y runtime NVIDIA. La generalización a otros equipos se pospone hasta que la arquitectura funcional esté estabilizada.

## Persistencia documental

Los PDF, chunks, embeddings y cachés RAG/CAG viven actualmente en memoria. Reiniciar el gateway elimina el índice. Este comportamiento es aceptado para el MVP local, pero no es apropiado para múltiples réplicas, serverless o una biblioteca documental persistente.

## Presupuesto por caracteres

RAG y CAG limitan actualmente el contexto mediante cantidad de caracteres, no mediante tokens del tokenizer real. Es una aproximación conservadora para el modelo de referencia, pero debe reemplazarse por presupuesto de tokens antes de realizar pruebas de contexto extensas o comparar modelos con tokenizadores muy distintos.
