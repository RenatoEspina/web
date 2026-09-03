# Evolución hacia un sistema multiagente en estrella

## Diseño recomendado

El LLM central no debería conectarse directamente a procesos arbitrarios. Conviene introducir un orquestador determinista que mantenga estado, presupuestos, permisos, timeouts y trazas:

```text
cliente → API/orquestador → router central → especialistas
                         ↘ estado y trazas ↙
```

El router recibe descripciones estructuradas de capacidades (`id`, tareas, costo, latencia, contexto, herramientas y disponibilidad), genera un plan y el orquestador valida ese plan antes de ejecutar. Los especialistas solo reciben el mínimo contexto necesario y devuelven una respuesta estructurada con resultado, confianza, evidencias y errores. El central sintetiza, pero no debe ocultar fallos ni inventar respuestas faltantes.

## Escalamiento por etapas

1. **Una máquina:** procesos separados, cola en memoria, máximo de concurrencia por modelo, timeouts y trazas con `requestId`. Usar el modelo base más varios LoRA sobre el mismo vLLM reduce duplicación de pesos.
2. **Varias máquinas:** registro de capacidades y estado de salud, cola durable, almacenamiento de artefactos, workers idempotentes y un gateway OpenAI-compatible por nodo. Separar el plano de control del plano de inferencia.
3. **Producción:** réplicas, balanceo, autoscaling, observabilidad, límites por usuario y aislamiento de herramientas. Ray Serve LLM ya ofrece despliegues multi-modelo/multi-nodo, autoscaling, balanceo, routing y Multi-LoRA sobre vLLM: [documentación oficial](https://docs.ray.io/en/latest/serve/llm/index.html).
4. **Modelos grandes:** tensor/pipeline parallelism solo cuando un modelo no cabe en una GPU o nodo. Ray documenta ambos mecanismos entre nodos: [paralelismo multi-nodo](https://docs.ray.io/en/latest/serve/llm/user-guides/cross-node-parallelism.html).

## Qué necesita el proyecto antes de agregar agentes

- Un contrato de tarea versionado y respuestas JSON Schema.
- Un registro de especialistas y herramientas con permisos explícitos.
- Una cola con cancelación, reintentos acotados e idempotencia.
- Presupuestos por solicitud: tokens, tiempo, dinero y cantidad máxima de delegaciones.
- Trazas por cada decisión del router, sin guardar secretos ni cadenas internas de razonamiento.
- Un benchmark de routing: tarea, especialista correcto, calidad final, latencia y costo.
- Defensa contra prompt injection entre agentes; los resultados de una punta son datos no confiables, no instrucciones del sistema.

## Cuello de botella más probable

Al principio no será la red sino la GPU. Con una RTX 3060 de 6 GB, varias puntas generativas simultáneas competirán por VRAM. El MVP debería ejecutar especialistas en secuencia o representar especialidades mediante LoRA precargados sobre un solo modelo base. Al disponer de más GPU, se separan puntas por worker y se permite paralelismo solo entre subtareas independientes.

Para decidir cuándo escalar hay que medir tasa de errores, cola, tokens/s, tiempo al primer token y tiempo por token. Ray expone estas métricas de servicio y de vLLM mediante Prometheus/Grafana: [observabilidad oficial](https://docs.ray.io/en/latest/serve/llm/user-guides/observability.html).
