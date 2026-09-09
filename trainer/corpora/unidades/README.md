# Unidades ES: fine-tuning con respuestas comprobables

Dataset sintético original en español para convertir una cantidad entre unidades y devolver JSON. Incluye longitud, masa, volumen, tiempo, superficie y temperatura. El resultado se puede comprobar sin usar otro LLM como juez. Ejemplo: «Convierte 2,5 kilómetros a metros» debe producir `{"estado":"ok","valor":"2500","unidad":"m"}`.

**Estado de la entrega:** corpus generado y comprobado; entrenamiento e inferencia con modelos pendientes. Los objetivos de este documento son propuestas para el experimento, no resultados medidos ni garantías de calidad máxima.

## Datos y particiones

| Partición | Casos | Uso |
|---|---:|---|
| Entrenamiento | 2.400 | Actualizar pesos del adaptador |
| Validación | 400 | Comparar configuraciones y seleccionar candidatos |
| Prueba final | 600 | Comparación final base/adaptador, después de congelar decisiones |
| Desafío final | 300 | Redacciones, magnitudes y notación científica adicionales |

Cada partición contiene 70% de conversiones válidas, 10% de unidades incompatibles, 10% de solicitudes incompletas y 10% de unidades fuera del catálogo. Las seis dimensiones tienen el mismo peso entre los casos válidos, salvo el redondeo de cuotas en validación. Hay números negativos, cero, coma decimal, exponentes de superficie/volumen y conversiones de temperatura bajo cero. El desafío incluye cantidades con más dígitos y notación científica.

El generador usa semilla `20260909`. Los grupos semánticos se asignan a una sola partición mediante SHA-256 antes de aceptar ejemplos. Para conversiones válidas, el grupo es la dimensión y la cantidad exacta en la unidad base: cambiar de alias, unidad equivalente o destino no permite cruzar particiones. También se normalizan las cantidades conocidas en errores cuando es posible. Las plantillas exteriores son distintas por partición y las conversiones válidas tienen además estructuras interiores distintas.

No hay preguntas idénticas ni grupos compartidos entre particiones. Sí puede haber varias variantes de un grupo **dentro** de una partición; el manifiesto informa cuántas. Las unidades y las reglas se comparten deliberadamente: se mide transferencia a nuevas cantidades y redacciones, no descubrimiento de unidades nuevas.

La verdad de referencia numérica se calcula con fracciones exactas y se contrasta con una implementación decimal separada. Las dos implementaciones comparten la tabla de factores lineales; las pruebas incluyen resultados conocidos escritos independientemente para detectar errores de esa tabla. Las fórmulas de temperatura corresponden a las [conversiones exactas de NIST](https://www.nist.gov/pml/owm/si-units-temperature).

## Contrato de respuesta

Tres claves exactas: `estado`, `valor`, `unidad`. No se permiten explicaciones, Markdown o claves adicionales.

| Estado | Cuándo | valor / unidad |
|---|---|---|
| `ok` | Conversión dentro de una dimensión soportada | Cadena decimal / símbolo de destino |
| `falta_dato` | Falta cantidad, origen o destino | `null` / `null` |
| `no_soportada` | Alguna unidad está fuera del catálogo | `null` / `null` |
| `incompatible` | Las dimensiones son diferentes | `null` / `null` |

Las comprobaciones se aplican en ese orden de errores: falta de datos, unidad no soportada, incompatibilidad. No se infiere densidad para convertir masa a volumen. Las temperaturas representan lecturas absolutas, no incrementos.

Catálogo canónico: `mm, cm, m, km; mg, g, kg, t; mL, L, m3; s, min, h; mm2, cm2, m2, km2; °C, °F, K`. Se admiten los nombres españoles y superíndices definidos en el generador. Fahrenheit, litros, minutos y horas están incluidos aunque el corpus no se limita estrictamente a unidades SI.

El valor usa punto decimal y un máximo de seis decimales con empate al par; no lleva ceros decimales finales, exponente o separadores de miles. El cero se escribe `"0"`. La salida se compara exactamente **después de ese redondeo prescrito**. No se está prometiendo precisión matemática infinita. En la entrada, punto y coma significan decimal; las entradas ambiguas con separadores de miles quedan fuera del alcance.

## Archivos y uso en el proyecto

Desde la raíz del repositorio:

```bash
python3 trainer/corpora/unidades/build.py
python3 trainer/corpora/unidades/build.py --check
python3 trainer/validate_dataset.py trainer/examples/unidades-training.jsonl
python3 trainer/validate_dataset.py trainer/corpora/unidades/validation.jsonl
```

El generador materializa:

- `trainer/examples/unidades-training.jsonl`: SFT con mensajes system/user/assistant; seleccionable en la GUI.
- `trainer/corpora/unidades/validation.jsonl`: SFT de validación, fuera de las carpetas seleccionables para entrenar.
- `validation-eval.jsonl`: los mismos 400 casos, sin respuesta assistant, para medir generación durante el desarrollo.
- `evaluation.jsonl` y `challenge.jsonl`: referencias reservadas para la evaluación final.
- `train-audit.jsonl`: metadatos para auditar el entrenamiento; no es un archivo para entrenar.
- `manifest.json`: tamaños, categorías, grupos, controles y hashes de los archivos.

Abrir `./fine-tune-gui` genera automáticamente el corpus. Seleccionar `unidades-training.jsonl` conecta su validación separada en el lanzador de entrenamiento. Los JSONL se reconstruyen a partir del generador versionado, como el corpus Terraria del proyecto. No se mezclan ambos temas.

El formato SFT `messages` también permite preparar estos datos para servicios compatibles con ese formato, pero el flujo principal de esta entrega es el entrenador QLoRA del repositorio. El [formato de SFT de OpenAI](https://developers.openai.com/api/docs/guides/supervised-fine-tuning) usa igualmente JSONL y ejemplos conversacionales; sus hiperparámetros administrados no son intercambiables con los de QLoRA.

## Configuración inicial para tu entrenador

Se conserva el modelo por defecto del proyecto, `Qwen/Qwen3.5-0.8B`. Antes de un experimento formal, fija además su revisión inmutable con `--model-revision` y registra el modelo realmente servido en inferencia.

| Parámetro | Inicio propuesto |
|---|---|
| Método | QLoRA, NF4 de 4 bits, módulos lineales |
| Épocas | 2 |
| Learning rate | `1e-4` |
| LoRA rank / alpha | `16 / 32` |
| LoRA dropout | `0.05` |
| Batch por dispositivo / acumulación | `1 / 8` (efectivo 8 en una GPU) |
| Longitud máxima | 1024 tokens; verificar con la auditoría del trainer |
| Semilla inicial | 42 |
| Pérdida | Solo tokens de respuesta assistant |

`1e-4` es un punto de partida habitual para adaptadores según [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer); aquí se propone como candidato, no como óptimo demostrado. El trainer verifica si se truncarían respuestas. Aumenta la longitud si lo necesita y evita `--allow-truncation` en el experimento.

Con el entorno de entrenamiento del proyecto ya preparado, este comando inicia el entrenamiento cuando tú lo ejecutes:

```bash
bash scripts/train-adapter.sh trainer/examples/unidades-training.jsonl unidades-es-lr1e4-s42 \
  --model Qwen/Qwen3.5-0.8B \
  --epochs 2 --learning-rate 1e-4 \
  --rank 16 --alpha 32 --dropout 0.05 \
  --batch-size 1 --gradient-accumulation 8 --max-length 1024 --seed 42
```

No se ha ejecutado este comando para generar la entrega. El tamaño en tokens, consumo de GPU y duración dependen del tokenizer y del equipo; no se han estimado como si fueran mediciones.

## Métricas para elegir calidad

Las métricas de generación tienen escala 0–1 en el reporte; la tabla expresa porcentajes. Las metas son umbrales iniciales propuestos para este benchmark.

| Métrica | Objetivo propuesto | Interpretación |
|---|---|---|
| `exact_accuracy` | ≥98% en prueba final | Estado, valor canónico y unidad correctos; los errores también deben responder correctamente |
| `numeric_unit_accuracy` | ≥98% | Cálculo y unidad correctos entre **todas** las consultas válidas; no excluye fallos de formato/inferencia |
| `schema_validity` | ≥99,5% | Objeto JSON con las claves y tipos requeridos |
| `invalid_request_error_rate` | ≤1% | Errores al gestionar consultas incompletas, incompatibles o no soportadas, incluyendo respuestas ilegibles |
| `status_macro_f1` | ≥0,98 | Equilibrio entre los cuatro estados |
| Exactitud del desafío | ≥90% | Robustez ante las variaciones adicionales reservadas |
| Mejora base → adaptador | Delta >0 y límite inferior del IC95% >0 | Evidencia de mejora en las mismas preguntas |

Para seleccionar candidatos usa primero la exactitud de **validación**, exige buen manejo de errores y revisa la exactitud por cada categoría. Un promedio puede esconder un fallo sistemático en temperatura o superficie. En empates, favorece menor pérdida de validación y el modelo más temprano/sencillo. Prueba final y desafío no participan en esa selección.

Durante entrenamiento registra `loss`, `eval_loss`, `mean_token_accuracy`, `grad_norm` y learning rate. El proyecto exporta pérdida/perplejidad de validación antes y después; perplejidad es `exp(loss)`. Las definiciones token a token están en [TRL](https://huggingface.co/docs/trl/sft_trainer). **No hay un valor universal de loss que garantice conversiones correctas**: una cifra equivocada puede coexistir con alta exactitud de tokens por el texto JSON repetido. Tampoco conviene minimizar entropía o norma de gradiente como objetivos de calidad.

El trainer actual evalúa por época y exporta el checkpoint con menor `eval_loss`. El nuevo evaluador mide la generación de ese candidato; no cambia la selección interna a exactitud. El trainer conserva como máximo dos checkpoints, por lo que esta entrega no promete comparar automáticamente todas las épocas por exactitud generada.

## Experimento para comprobar la mejora

1. Evalúa el modelo base con `validation-eval.jsonl` y el mismo system prompt, plantilla de chat y ajustes de inferencia que usarás con el adaptador. Desactiva RAG/CAG y otras fuentes externas en ambos. Comprueba que el servidor realmente carga el adaptador; una comparación contra el mismo modelo base no prueba nada.
2. Entrena tres candidatos con learning rate `5e-5`, `1e-4` y `2e-4`, manteniendo dos épocas y lo demás fijo. Compara los adaptadores exportados sobre validación. Si quedan errores de ajuste, prueba una o tres épocas para la tasa seleccionada, observando loss y exactitud.
3. Si la validación sigue mejorando, conserva ese candidato. Si la pérdida de entrenamiento baja mientras empeoran validación o exactitud, reduce épocas/tasa y revisa ejemplos. Elige por generación, no solo por pérdida.
4. Repite la configuración elegida con semillas 42, 43 y 44. Informa media y dispersión; no selecciones la semilla mirando la prueba final.
5. Congela la configuración y ejecuta base/adaptador sobre los 600 casos finales y los 300 del desafío. Reporta ambos conjuntos separados, denominadores, intervalos y resultados por categoría. Si reutilizas estas pruebas para modificar datos o parámetros, pasan a ser desarrollo y necesitas un conjunto final nuevo.

El baseline puede ser ya muy alto: en ese caso no hay margen para demostrar una mejora. No se debe prometer un incremento porcentual antes de medir. Tampoco se debe reemplazar el conjunto de prueba hasta conseguir un resultado favorable.

## Evaluación automática

El evaluador existente del proyecto usa subcadenas; aquí se usa un evaluador propio. `10` nunca pasa por aparecer dentro de `100`. El orden de las claves JSON no importa. Una salida `"10.0"` frente a `"10"` cuenta como cálculo correcto, pero incumple el formato canónico y falla la exactitud completa.

Con ambos modelos servidos por el mismo endpoint compatible y usando sus identificadores reales:

```bash
python3 trainer/corpora/unidades/evaluate.py \
  --dataset trainer/corpora/unidades/validation-eval.jsonl \
  --model Qwen/Qwen3.5-0.8B --compare-model unidades-es-lr1e4-s42 \
  --base-url http://127.0.0.1:8000 \
  --output trainer/corpora/unidades/results/validation-comparison.json
```

Para la comparación final cambia `--dataset` por `evaluation.jsonl` y después por `challenge.jsonl`, con archivos de salida diferentes. El identificador del adaptador es el que muestre tu servidor, no necesariamente su carpeta. Se envían solo system/user, nunca las referencias ni los metadatos. La temperatura es 0; aun así, reproducibilidad bit a bit depende del servidor. Si se requiere autenticación, configura la variable de entorno indicada por `--api-key-env`.

También puedes evaluar respuestas generadas por otro sistema:

```bash
python3 trainer/corpora/unidades/evaluate.py \
  --dataset trainer/corpora/unidades/evaluation.jsonl \
  --predictions base.jsonl --compare-predictions ajustado.jsonl \
  --output comparacion-final.json
```

Cada línea de predicciones contiene `{"id":"id-del-caso","output":"respuesta textual"}`. El evaluador alinea por ID y rechaza duplicados, casos faltantes o adicionales. Conserva las respuestas y los errores de inferencia. Los casos que fallan no se eliminan del denominador. Guarda hashes del dataset y la configuración de inferencia en el reporte.

El reporte incluye un intervalo Wilson95 nominal por caso y un intervalo de mejora mediante bootstrap pareado por grupo semántico. Wilson presupone independencia y es aproximado cuando hay variantes de un grupo; el bootstrap agrupa esas variantes. Con 600 casos, una diferencia pequeña puede quedar dentro de la incertidumbre. La tasa auxiliar `invalid_request_numeric_response_rate` solo reconoce el campo numérico de JSON legible: usa `invalid_request_error_rate` como control completo del manejo de errores.

## Alcance de la conclusión

Este corpus sirve para un experimento acotado de seguimiento de instrucciones, cálculo y salida estructurada. Es sintético y usa plantillas; no demuestra calidad de conversación abierta ni comprensión de cualquier forma del español. Los errores se plantean explícitamente y son más sencillos que solicitudes reales ambiguas. No incluye preguntas multiturno, conversiones a la misma unidad, explicaciones extensas, unidades arbitrarias, varios valores a la vez ni todos los formatos regionales. Tampoco combina varios errores en una consulta: la prioridad entre errores forma parte del contrato, pero no queda evaluada en este corpus. Una buena nota aquí no demuestra que el modelo no haya empeorado en otros temas: para un uso general se necesita además una evaluación independiente de regresión.

Para convertir unidades en producción, una calculadora determinista sigue siendo la referencia. El interés de este dataset es que permite detectar y cuantificar con claridad si el fine-tuning mejora la tarea especificada.
