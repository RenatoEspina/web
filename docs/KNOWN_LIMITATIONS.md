# Limitaciones aceptadas y errores conocidos

Este documento separa dos categorías que antes estaban mezcladas:

- **Limitaciones aceptadas:** decisiones deliberadas del alcance actual. No se consideran errores prioritarios mientras el perfil de investigación siga siendo el mismo.
- **Errores conocidos:** comportamientos inconsistentes o engañosos que deben corregirse, aunque no bloqueen el flujo principal actual.

Cuando una limitación aceptada pase a formar parte del alcance soportado, debe retirarse de la primera sección y tratarse como requisito funcional.

## Limitaciones aceptadas

### Cambio del modelo de embeddings con un índice existente

Los documentos mantienen sus vectores en memoria junto con metadatos del proveedor,
modelo, dimensión y prefijos usados al indexarlos. Si el perfil actual no coincide,
RAG no compara esos vectores: degrada a recuperación léxica hasta reindexar.

Cambiar `EMBEDDING_MODEL`, sus prefijos o el proveedor después de haber indexado
documentos deja vectores antiguos en memoria, pero ya no los mezcla con consultas
del perfil nuevo.

**Práctica aceptada actualmente:** reiniciar el gateway y volver a cargar los PDF después de cambiar la configuración de embeddings. Una política automática de reindexado o invalidación queda fuera de la prioridad inmediata.

### Engines generativos alternativos

El gateway conserva un adaptador para Ollama y una abstracción OpenAI-compatible, pero las pruebas, el ciclo LoRA, CAG con prefix caching y la operación normal se priorizan para vLLM.

No se considera prioridad inmediata garantizar equivalencia funcional completa entre vLLM y otros engines.

### Configuración específica del equipo de investigación

`docker-compose.yml` contiene parámetros deliberadamente ajustados al entorno de investigación, incluyendo límites de VRAM, CPU y runtime NVIDIA.

La generalización y optimización para hardware distinto se pospone hasta que la arquitectura funcional esté estabilizada.

### Persistencia documental

Los PDF, chunks, embeddings y cachés RAG/CAG viven actualmente en memoria. Reiniciar el gateway elimina el índice.

Este comportamiento se acepta para el MVP local, aunque no sería apropiado para múltiples réplicas, serverless o una biblioteca documental persistente.

### Presupuesto de contexto basado en caracteres

RAG y CAG limitan actualmente el contexto mediante cantidad de caracteres y no mediante tokens del tokenizer real.

Se acepta como aproximación para el perfil actual. Debe reemplazarse por presupuesto de tokens antes de realizar pruebas de contexto extensas o comparar modelos con tokenizadores muy distintos.

### Historial conversacional después de cambiar documentos

Las respuestas anteriores permanecen en el historial aunque hayan sido obtenidas con ayuda de un PDF que posteriormente se desmarque o aunque el modo cambie a `none`.

Por ahora se considera un comportamiento conversacional aceptado: el modelo puede recordar información que ya apareció explícitamente en la conversación. La selección documental controla el **nuevo contexto recuperado**, no borra el contenido textual de turnos anteriores.

Si en el futuro se necesita aislamiento estricto por corpus, deberá versionarse el contexto de la conversación o reiniciarse el historial al cambiar de selección documental.

## Errores conocidos pendientes

No hay errores funcionales conocidos en este momento. Las comprobaciones de entorno,
la carga de variables `HOST`/`PORT` y los mensajes de estado de embeddings y selección
documental se mantienen en el código y sus regresiones correspondientes.

## Correcciones recientes relacionadas

Los siguientes problemas ya no se consideran errores conocidos porque cuentan con corrección y regresión:

- una selección explícita `documentIds: []` ya no utiliza todos los documentos por accidente;
- RAG solo publica fuentes cuyos chunks realmente entraron al contexto;
- RAG sin candidatos ya no afirma en el `system` prompt que se recuperaron fragmentos relevantes;
- la carga y descarga dinámica de LoRA intenta revertir el estado de vLLM si falla la actualización de `LLM_ADAPTER_MODELS`, y reporta explícitamente si también falla el rollback;
- la API evita procesar un nuevo PDF cuando el workspace ya alcanzó el límite de documentos.
- la GUI distingue un entorno virtual presente de uno verificado mediante CUDA/NF4;
- Vite carga `HOST` y `PORT` desde `.env.local` antes de construir su configuración;
- la UI dice **embeddings habilitados** (configuración), no **activos** (salud no comprobada),
  y advierte cuando RAG/CAG no tiene documentos seleccionados;
- los vectores indexados se comparan solo con el mismo proveedor, modelo y prefijos de embeddings;
- la GUI rechaza adaptadores cuyo modelo base no coincide con el modelo servido por vLLM.
