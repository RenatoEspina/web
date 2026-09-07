# Limitaciones aceptadas y errores conocidos

Este documento separa dos categorías que antes estaban mezcladas:

- **Limitaciones aceptadas:** decisiones deliberadas del alcance actual. No se consideran errores prioritarios mientras el perfil de investigación siga siendo el mismo.
- **Errores conocidos:** comportamientos inconsistentes o engañosos que deben corregirse, aunque no bloqueen el flujo principal actual.

Cuando una limitación aceptada pase a formar parte del alcance soportado, debe retirarse de la primera sección y tratarse como requisito funcional.

## Limitaciones aceptadas

### Cambio del modelo de embeddings con un índice existente

Los documentos mantienen sus vectores en memoria junto con metadatos del proveedor, modelo y dimensión usados al indexarlos. El flujo actual presupone que el modelo de embeddings no cambia durante la vida del índice.

Cambiar `EMBEDDING_MODEL`, sus prefijos o el proveedor después de haber indexado documentos puede dejar vectores antiguos en memoria.

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

### La GUI puede marcar el entorno de fine-tuning como listo sin haberlo verificado

`trainer/gui_server.py` considera `environmentReady=true` cuando encuentra un ejecutable Python en el entorno virtual. Eso no garantiza que PyTorch, CUDA, `bitsandbytes` y NF4 estén funcionando.

**Impacto:** un entorno creado parcialmente o con dependencias rotas puede mostrarse como **Entorno: listo** hasta ejecutar la comprobación real.

**Corrección prevista:** separar al menos `venvPresent` de `environmentVerified` y basar el estado verde en una comprobación real o en el resultado persistido de la última comprobación válida.

### `HOST` y `PORT` pueden parecer configurables desde `.env.local` sin afectar a Vite

`.env.example` presenta `HOST` y `PORT` como configuración normal, mientras `vite.config.ts` consulta `process.env.HOST` y `process.env.PORT` durante la evaluación de la configuración.

**Impacto:** si esos valores existen únicamente en `.env.local`, pueden no modificar el servidor de desarrollo como espera el usuario.

**Corrección prevista:** cargar explícitamente los archivos de entorno en `vite.config.ts` mediante el mecanismo de configuración de Vite o retirar esas variables de la configuración documentada si no se desea soportarlas allí.

### La interfaz dice “embeddings activos” sin comprobar la salud del servicio

La interfaz deriva ese texto de `embedding.enabled`, que significa que el uso de embeddings está habilitado por configuración. `/api/health` comprueba actualmente el proveedor generativo, no la disponibilidad real del servicio de embeddings.

**Impacto:** Ollama/embeddings puede estar caído y la UI seguir mostrando **embeddings activos**. La indexación degrada correctamente a recuperación léxica, pero el estado visual resulta engañoso.

**Corrección prevista:** usar **habilitados** para el estado de configuración y reservar **activos/disponibles** para un health check real.

### La UI puede mostrar RAG aunque no haya documentos seleccionados

El selector puede permanecer en `RAG` con `selectedDocumentIds=[]`. El backend interpreta correctamente una selección vacía como ausencia de contexto documental y devuelve `mode=none`.

**Impacto:** la semántica efectiva del backend es correcta, pero la interfaz puede sugerir que la siguiente pregunta usará RAG cuando no utilizará ningún PDF.

**Corrección prevista:** mostrar un estado explícito de “selecciona al menos un documento” o reflejar visualmente el modo efectivo `none` mientras la selección esté vacía.

## Correcciones recientes relacionadas

Los siguientes problemas ya no se consideran errores conocidos porque cuentan con corrección y regresión:

- una selección explícita `documentIds: []` ya no utiliza todos los documentos por accidente;
- RAG solo publica fuentes cuyos chunks realmente entraron al contexto;
- RAG sin candidatos ya no afirma en el `system` prompt que se recuperaron fragmentos relevantes;
- la carga y descarga dinámica de LoRA intenta revertir el estado de vLLM si falla la actualización de `LLM_ADAPTER_MODELS`, y reporta explícitamente si también falla el rollback;
- la API evita procesar un nuevo PDF cuando el workspace ya alcanzó el límite de documentos.
