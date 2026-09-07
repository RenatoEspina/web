# Fine-tuning de Terraria desde la GUI local

## Antes de empezar

Actualiza el proyecto a la revisión que contiene esta guía y vuelve a abrir
**LLM Bridge Fine-tuning** desde el menú de aplicaciones. Si nunca lo has abierto,
ejecuta `./fine-tune-gui` una vez desde la raíz del proyecto.

No necesitas volver a entrenar `qwen-es-v1` para corregir su compatibilidad:
si estaba cargado, pulsa **Descargar de vLLM**, vuelve a **Cargar en vLLM** y
pulsa **Comprobar uso**. Si vLLM se reinició, basta con cargarlo y comprobarlo.
La exportación de Qwen3.5 se prepara automáticamente conservando los originales.

## Entrenar un adaptador nuevo

1. **Entorno:** pulsa **Comprobar GPU**. Si el entorno falta o está desactualizado,
   usa **Preparar / actualizar entorno**. Debe terminar con CUDA y NF4 correctos.
2. **Dataset:** selecciona **Ejemplo · terraria-training.jsonl**. Ya está incluido;
   no hace falta descargar ni subir archivos. Contiene 96 conversaciones válidas.
3. **Entrenamiento:** utiliza un nombre nuevo, por ejemplo `terraria-es-v1`, y
   el modelo base `Qwen/Qwen3.5-0.8B`. No uses `qwen-es-v1` como modelo base ni
   reutilices su nombre: ambos adaptadores deben conservarse separados.
4. Como **punto de partida experimental** para la RTX 3060 Laptop de 6 GB:
   rank **16**, alpha **32**, dropout **0.05**, **2 épocas**, learning rate
   **0.0001**, batch size **1**, gradient accumulation **8**, max length **1024**,
   seed **42**. No son hiperparámetros optimizados. Con 96 ejemplos, son unas
   24 actualizaciones del optimizador. Más épocas no garantizan mejor calidad.
5. Pulsa **Preparar y entrenar**. El flujo detiene vLLM durante el entrenamiento
   para liberar VRAM y lo restaura si estaba activo. No ejecutes otro entrenamiento
   o programa intensivo en GPU a la vez. El HF token solo hace falta si el acceso
   al modelo lo requiere o quieres autenticar la descarga; no acelera el cálculo.
6. Si aparece un error de memoria, revisa otros procesos de GPU y prueba
   max length **512** con un **nombre nuevo**. No borres un adaptador terminado
   para reutilizar su nombre. Revisa la terminal integrada antes de reintentar.
7. **vLLM:** inicia el **mismo modelo base** si no quedó activo.
8. **Adaptadores:** carga `terraria-es-v1` y pulsa **Comprobar uso**. El resultado
   debe indicar efecto detectado; si es inconcluso, no asumas que esté aplicado.
9. Reinicia la web principal si estaba abierta para que relea `LLM_ADAPTER_MODELS`.
   Abre una conversación nueva y elige **terraria-es-v1** en **Modelo o adaptador**.
   Cargar un LoRA no lo selecciona automáticamente. El nombre superior ahora refleja
   la selección; no es una certificación de calidad.

## Comprobar aprendizaje, no solo carga

La GUI distingue entre cargar el adaptador y detectar un cambio en la inferencia.
**Comprobar uso** compara las probabilidades del siguiente token en tres prompts
con una repetición del modelo base para controlar ruido numérico. Puede detectar
un LoRA aunque el texto final coincida. No confirma que cada tensor se utilice ni
que el adaptador sea mejor; una prueba inconclusa tampoco prueba ausencia absoluta
de efecto.

En el chat, compara el modelo base y `terraria-es-v1` usando conversaciones nuevas,
el mismo prompt y sin documentos/RAG/CAG. Por ejemplo, pregunta la diferencia entre
un Magic Mirror y una Recall Potion. No reutilices el historial de la otra prueba.

Para una evaluación reproducible (opcional, desde la raíz del repositorio):

```bash
python trainer/evaluate.py --dataset trainer/corpora/terraria/evaluation.jsonl --model Qwen/Qwen3.5-0.8B --output outputs/terraria-base.json
python trainer/evaluate.py --dataset trainer/corpora/terraria/evaluation.jsonl --model terraria-es-v1 --output outputs/terraria-lora.json
```

Ambas ejecuciones deben usar el mismo vLLM, plantilla y configuración de generación.
Compara respuestas completas, no solo `passRate`: la comprobación por palabras
puede dar falsos positivos y negativos. La evaluación tiene 16 casos de temas
reservados, no entrenados; úsala como comprobación exploratoria de generalización
y regresiones, no como prueba de que el modelo conoce toda la wiki.

El corpus aún es pequeño para aprender Terraria en profundidad. Amplíalo con hechos
verificados, respuestas variadas y una partición de desarrollo independiente.
Para consultar toda la wiki con fuentes actuales, combina el modelo con RAG.
No se ha ejecutado el entrenamiento de Terraria automáticamente.

## Fuentes y condiciones del corpus

La [ficha del dataset](../trainer/corpora/terraria/README.md) explica el alcance,
las 28 fuentes, el método de consulta, la licencia **CC BY-NC-SA 4.0** y las
limitaciones. Conserva esa atribución al compartir los datos. No asumas permiso
para uso comercial ni para distribuir pesos sin revisar las licencias aplicables.
