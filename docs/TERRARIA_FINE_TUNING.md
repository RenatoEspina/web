# Fine-tuning de Terraria desde la GUI local

## Qué cambió en el experimento

El corpus de Terraria ahora está en **inglés** y separa tres conjuntos por
artículo fuente:

- **128 conversaciones de entrenamiento** (32 artículos);
- **24 conversaciones de validación** (6 artículos distintos);
- **16 conversaciones de evaluación final** (4 artículos adicionales).

La GUI reconstruye automáticamente los JSONL locales desde
`trainer/corpora/terraria/data/` al abrirse. Selecciona solamente
**Ejemplo · terraria-training.jsonl**; el validation set se conecta
automáticamente durante el entrenamiento y no aparece como un dataset SFT
seleccionable.

## Por qué ya no usamos solo train_loss

Las métricas que aparecen durante SFT (`loss`, `mean_token_accuracy`, `entropy`,
`grad_norm`) se calculan sobre los ejemplos que el optimizador está viendo. Sirven
para detectar inestabilidad o si el modelo está aprendiendo, pero una mejora en
ellas no demuestra que el checkpoint generalice mejor.

Cuando existe validation set, el trainer ahora:

1. calcula una validación base antes de actualizar el LoRA;
2. calcula `eval_loss` al final de cada época;
3. guarda checkpoints por época;
4. selecciona el checkpoint con **menor `eval_loss`**;
5. recarga ese checkpoint antes de guardar el adapter final;
6. registra `eval_loss`, perplexity, checkpoint seleccionado y mejora relativa en
   `manifest.json -> qualitySelection`.

La perplexity es una transformación de la misma pérdida (`exp(loss)`), así que
sirve para lectura/comparación pero no es una métrica independiente que haya que
maximizar. Para esta fase experimental usamos **mínimo validation/eval loss** como
criterio objetivo de selección.

La evaluación final de 16 preguntas permanece completamente fuera de ese proceso.
No la uses repetidamente para escoger epochs, learning rate o rank.

## Entrenar un adapter nuevo

1. Actualiza la rama/revisión del proyecto y abre `./fine-tune-gui`.
2. En **Entorno**, usa **Comprobar GPU** y prepara/actualiza el entorno si hace
   falta.
3. En **Dataset**, selecciona **terraria-training.jsonl**. Debe mostrar 128
   conversaciones.
4. Usa un nombre nuevo, por ejemplo `terraria-en-v2`, con modelo base
   `Qwen/Qwen3.5-0.8B`.
5. Como punto de partida conserva, para aislar el efecto del dataset/validación:
   rank **16**, alpha **32**, dropout **0.05**, **2 épocas**, learning rate
   **0.0001**, batch size **1**, gradient accumulation **8**, max length **1024**,
   seed **42**.
6. Pulsa **Preparar y entrenar**. El script detectará automáticamente:

   ```text
   trainer/corpora/terraria/validation.jsonl
   ```

   y el log debe mostrar `Validación antes del entrenamiento` y evaluaciones por
   época.
7. Al finalizar revisa:

   ```bash
   jq '.qualitySelection' adapters/terraria-en-v2/manifest.json
   ```

   Un resultado típico tendrá esta forma:

   ```json
   {
     "metric": "eval_loss",
     "direction": "minimize",
     "bestCheckpoint": "checkpoint-...",
     "bestMetric": 1.8,
     "baseline": {
       "baseline_loss": 2.7,
       "baseline_perplexity": 14.9
     },
     "selected": {
       "validation_loss": 1.8,
       "validation_perplexity": 6.0
     },
     "relativeLossImprovement": 0.33
   }
   ```

   Los números anteriores son solo un ejemplo de estructura; usa los valores
   reales de tu entrenamiento.
8. Inicia vLLM con el mismo modelo base, carga el adapter y usa **Comprobar uso**.
9. Reinicia la web principal si estaba abierta para que relea la allowlist de
   adapters.

Con 128 ejemplos, batch size 1 y gradient accumulation 8 hay aproximadamente
**16 actualizaciones del optimizador por época**, unas 32 en dos épocas. Si la
segunda época baja train loss pero empeora validation loss, el trainer debería
conservar el checkpoint de la primera.

## Evaluación final Base vs LoRA

Después de cargar el adapter, genera el corpus final si todavía no existe:

```bash
python trainer/corpora/terraria/build.py
```

Luego ejecuta exactamente el mismo conjunto con ambos modelos:

```bash
python trainer/evaluate.py \
  --dataset trainer/corpora/terraria/evaluation.jsonl \
  --model Qwen/Qwen3.5-0.8B \
  --output outputs/terraria-base.json

python trainer/evaluate.py \
  --dataset trainer/corpora/terraria/evaluation.jsonl \
  --model terraria-en-v2 \
  --output outputs/terraria-lora.json
```

Compara las respuestas completas además de `passRate`. `contains` sigue siendo un
smoke test léxico: una respuesta puede contener la palabra esperada y ser
contradictoria, o fallar por usar una paráfrasis correcta.

También compara en el chat una conversación nueva, sin RAG/CAG, con el mismo
prompt, temperatura y límite de tokens para Base y LoRA. La telemetría de la web
permite revisar latencia, tokens generados, tok/s y `finish_reason`.

## Qué queremos aprender con esta versión

Este experimento cambia deliberadamente dos cosas que podían limitar la calidad:

- **selección del checkpoint**: ahora existe una señal de generalización separada
  del entrenamiento;
- **datos**: el corpus pasa de 96 ejemplos SFT en español a 128 en inglés, agrega
  temas nuevos y añade un validation set de 24 ejemplos.

Por ahora dejamos rank, alpha, dropout, learning rate, epochs y `all-linear`
iguales. Si la calidad mejora, podremos atribuir buena parte del cambio al enfoque
de datos/selección. Si `eval_loss` no mejora o la evaluación final continúa mala,
el siguiente experimento debería cambiar una sola variable a la vez (por ejemplo
learning rate o módulos LoRA) para evitar confundir causas.

## Corpus y licencia

Consulta `trainer/corpora/terraria/README.md` para el detalle de las 42 fuentes,
particiones, alcance y condiciones **CC BY-NC-SA 4.0**. El corpus es una colección
curada para experimentar; para cobertura amplia, actualizada y con fuentes, RAG
sigue siendo la herramienta adecuada.
