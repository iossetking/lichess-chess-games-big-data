# lichess-chess-games-big-data

Dos modelos sobre datos oficiales de [Lichess](https://database.lichess.org/):

1. **Visión**: localiza automáticamente el tablero con la webcam, rectifica la
   perspectiva y usa YOLO para identificar las 12 clases de piezas y emitir FEN.
2. **Predicción de jugadas**: recibe una posición (FEN) y devuelve la mejor
   jugada en notación UCI.

La unión de ambos es: foto del tablero → FEN → jugada sugerida.

## Requisitos

```bash
uv sync
git lfs pull   # descarga los checkpoints de data/models/
```

Los checkpoints se guardan a media precisión (`float16`) cuando aplica y se
versionan con Git LFS (ver `.gitattributes`); `git lfs pull` los descarga tras
clonar. Las muestras limpias de `data/processed/` **no** se versionan: las
genera el cuaderno 01 y se quedan en el equipo donde se ejecuta (o en la sesión
de Colab). Los pesos de visión de YOLO (`chess-pieces*.pt`,
`chess-piece-classifier.pt`) pueden venir en el repositorio; si faltan, se
generan con los entrenadores de `scripts/` (ver la sección de visión).

## Tres rutas de trabajo

El repositorio se puede usar sin entrenar nada, reconstruyendo todo desde los
datos oficiales de Lichess o entrenando en una GPU de Colab. Elige una:

### Ruta A · Usar los modelos ya guardados (rápida)

Los cinco modelos de jugadas y, si están publicados, los pesos de visión se
descargan con Git LFS:

```bash
uv sync
git lfs pull
ls data/models/approach1_scratch.pt data/models/approach1_weight_ftuning.pt \
   data/models/approach1_lora_ftuning.pt data/models/maia3_lora_human.pt \
   data/models/maia3_lora_stockfish.pt
```

Con eso funcionan directamente (si algún checkpoint todavía no está publicado en
el repositorio, genéralo con la ruta B o C):

- **Comparar modelos**: `uv run marimo edit notebooks/04_model_playground.py`
  (no entrena; solo carga los `.pt` que existan en `data/models/`).
- **Vista en vivo**: `uv run python scripts/live_chess_fen.py --weights data/models/chess-pieces.pt`
  (ver la sección «Visión en vivo» más abajo; si falta `chess-pieces.pt`,
  entrénalo con la ruta B).

No hace falta ejecutar los cuadernos 01–03 ni descargar los archivos de Lichess.
Las muestras de `data/processed/` solo son necesarias para reentrenar o para
explorar los datos en el cuaderno 01.

### Ruta B · Entrenar desde cero en local

Se parte de la muestra oficial de Lichess y se generan todas las salidas
(`data/processed/*.parquet` y `data/models/*.pt`). Los cuadernos se ejecutan sin
interfaz y leen sus parámetros de variables de entorno, así que el tamaño y las
épocas van en el mismo comando. Ajusta los valores a tu equipo:

1. **Muestra limpia** (streaming desde `database.lichess.org`):
   ```bash
   LICHESS_YEAR_FROM=2024 LICHESS_YEAR_TO=2025 LICHESS_MAX_GAMES=20000 \
   LICHESS_MAX_POSITIONS=500000 LICHESS_MAX_EVALS=500000 \
   uv run python notebooks/01_eda_and_cleaning.py
   ```
   Salidas en `data/processed/`: `games_clean_sample.parquet`,
   `positions_games_sample.parquet` y `evals_clean_sample.parquet`.
2. **Tres modelos de jugadas** (red desde cero + dos ajustes con Stockfish):
   ```bash
   LICHESS_MAX_HUMAN=500000 LICHESS_MAX_EVAL=500000 \
   LICHESS_EPOCHS_STAGE1=8 LICHESS_EPOCHS_STAGE2=5 LICHESS_EPOCHS_LORA=5 \
   LICHESS_LORA_RANK=8 LICHESS_N_FOLDS=5 \
   uv run python notebooks/02_model_scratch_and_finetune.py
   ```
3. **Dos adaptadores Maia-3 + LoRA** (GPU recomendada; en CPU tarda más de una
   hora):
   ```bash
   LICHESS_MAIA_MODEL=maia3-5m LICHESS_MAIA_MAX_TRAIN=500000 \
   LICHESS_MAIA_EPOCHS=5 LICHESS_MAIA_RANK=8 LICHESS_MAIA_LR=2e-4 \
   uv run python notebooks/03_maia.py
   ```
4. **Detector de piezas YOLO** para la visión (o reutiliza el que ya esté en
   `data/models/chess-pieces.pt`):
   ```bash
   uv run python scripts/train_chess_piece_yolo.py \
     --epochs 50 --device mps --imgsz 640 --batch 16
   ```

Los cuadernos 2 y 3 reutilizan los Parquet de `data/processed/` y, si faltan,
generan su propia muestra. La misma ruta sirve para una prueba local de segundos
(baja las variables) o para una corrida grande con GPU. Al terminar,
`data/models/` contiene los cinco modelos y puedes seguir la ruta A para
compararlos y probarlos en vivo.

### Ruta C · Entrenar en Google Colab con `colab-cli`

La sesión de Colab tiene su propio disco. Los Parquet que genera el cuaderno 01
se quedan en `/content/lichess-chess-games-big-data/data/processed` **dentro de
la sesión**, y los cuadernos 02 y 03 los leen desde ahí, sin copiarlos al
portátil. Los modelos entrenados se escriben en
`/content/lichess-chess-games-big-data/data/models` y al final se descargan a
`data/models/` del portátil.

1. **Instala el CLI con `uv`** (una sola vez):
   ```bash
   uv tool install google-colab-cli
   ```
2. **Crea una sesión con GPU** (T4, L4, A100…):
   ```bash
   colab new -s chess --gpu T4
   ```
3. **Clona el repositorio y prepara las dependencias en la sesión**:
   ```bash
   colab exec -s chess --timeout 0 <<'PY'
   !GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 -b develop https://github.com/iossetking/lichess-chess-games-big-data.git /content/lichess-chess-games-big-data
   !pip install -q marimo polars matplotlib zstandard datasets "git+https://github.com/CSSLab/maia3.git"
   PY
   ```
   `torch` y el resto de la base ya vienen en Colab. Los cuadernos añaden
   `src/` del clon a `sys.path`, así que no hace falta instalar el paquete. Si
   clonas en otra ruta, pásala con `--env
   LICHESS_PROJECT_DIR=/content/otra/ruta`.
4. **Ejecuta los tres cuadernos en la misma sesión**, un comando por cuaderno.
   El primero descarga y limpia los datos; los otros dos entrenan con esos
   Parquet de la sesión:
   ```bash
   # 01 · datos limpios (se quedan en la sesión)
   colab exec -s chess -f notebooks/01_eda_and_cleaning.py --timeout 0 \
     --env LICHESS_YEAR_FROM=2024 --env LICHESS_YEAR_TO=2025 \
     --env LICHESS_MAX_GAMES=20000 --env LICHESS_MAX_POSITIONS=2000000 \
     --env LICHESS_MAX_EVALS=2000000

   # 02 · tres modelos de jugadas
   colab exec -s chess -f notebooks/02_model_scratch_and_finetune.py --timeout 0 \
     --env LICHESS_MAX_HUMAN=2000000 --env LICHESS_MAX_EVAL=2000000 \
     --env LICHESS_EPOCHS_STAGE1=8 --env LICHESS_EPOCHS_STAGE2=5 \
     --env LICHESS_EPOCHS_LORA=5 --env LICHESS_N_FOLDS=5

   # 03 · dos adaptadores Maia-3 + LoRA
   colab exec -s chess -f notebooks/03_maia.py --timeout 0 \
     --env LICHESS_MAIA_MODEL=maia3-23m --env LICHESS_MAIA_MAX_TRAIN=2000000 \
     --env LICHESS_MAIA_EPOCHS=5
   ```
5. **Descarga los modelos a `data/models/`** del portátil:
   ```bash
   mkdir -p data/models
   for model in approach1_scratch approach1_weight_ftuning \
                approach1_lora_ftuning maia3_lora_human maia3_lora_stockfish; do
     colab download -s chess \
       "/content/lichess-chess-games-big-data/data/models/${model}.pt" \
       "data/models/${model}.pt"
   done
   ```
6. **Opcional · guarda también los Parquet limpios en `data/processed/`** del
   portátil (o usa `--env LICHESS_PROCESSED_DIR=...` para escribirlos en una
   carpeta persistente, por ejemplo la de Google Drive):
   ```bash
   mkdir -p data/processed
   for name in games_clean_sample positions_games_sample evals_clean_sample; do
     colab download -s chess \
       "/content/lichess-chess-games-big-data/data/processed/${name}.parquet" \
       "data/processed/${name}.parquet"
   done
   ```
7. **Cierra la sesión** para dejar de consumir cómputo:
   ```bash
   colab stop -s chess
   ```

Los comandos `colab exec` usan el archivo local del cuaderno (con tus cambios),
mientras que `src/` y el paquete Maia-3 salen del clon de la sesión. Haz
`git push` antes de lanzar una corrida para que ambos coincidan. Para ver el
estado o cerrar todas las sesiones: `colab status -s chess`, `colab stop --all`.

## Variables de entorno de los cuadernos

Sin interfaz, los cuadernos leen tamaños, épocas y rutas de estas variables.
Todas son opcionales y tienen un valor por defecto pensado para una prueba
rápida:

| Cuaderno | Variables |
|---|---|
| `01_eda_and_cleaning.py` | `LICHESS_YEAR_FROM`, `LICHESS_YEAR_TO`, `LICHESS_MONTH_FROM`, `LICHESS_MONTH_TO`, `LICHESS_MAX_GAMES`, `LICHESS_MIN_ELO`, `LICHESS_MAX_POSITIONS`, `LICHESS_MAX_EVALS`, `LICHESS_MIN_DEPTH` |
| `02_model_scratch_and_finetune.py` | `LICHESS_MAX_HUMAN`, `LICHESS_MAX_EVAL`, `LICHESS_EPOCHS_STAGE1`, `LICHESS_EPOCHS_STAGE2`, `LICHESS_EPOCHS_LORA`, `LICHESS_LORA_RANK`, `LICHESS_N_FOLDS`, `LICHESS_CV_WEIGHT`, `LICHESS_CV_LORA` |
| `03_maia.py` | `LICHESS_MAIA_MODEL`, `LICHESS_MAIA_MAX_TRAIN`, `LICHESS_MAIA_EPOCHS`, `LICHESS_MAIA_RANK`, `LICHESS_MAIA_LR`, `LICHESS_MAIA_ELO` |
| Todos | `LICHESS_PROJECT_DIR`, `LICHESS_DATA_DIR`, `LICHESS_PROCESSED_DIR`, `LICHESS_MODELS_DIR` |

`LICHESS_PROJECT_DIR` solo hace falta si ejecutas el cuaderno fuera del
repositorio (por ejemplo en Colab con otro nombre de carpeta).
`LICHESS_DATA_DIR` cambia la raíz de datos; `LICHESS_PROCESSED_DIR` y
`LICHESS_MODELS_DIR` cambian solo una de las dos carpetas. Por defecto son
`<repo>/data/processed` y `<repo>/data/models`, o `./data/...` cuando no hay
repositorio (Colab).

Ejemplo de prueba local mínima (segundos):

```bash
LICHESS_MAX_HUMAN=2000 LICHESS_MAX_EVAL=2000 LICHESS_EPOCHS_STAGE1=1 \
LICHESS_EPOCHS_STAGE2=1 LICHESS_EPOCHS_LORA=1 LICHESS_N_FOLDS=3 \
LICHESS_CV_WEIGHT=0 LICHESS_CV_LORA=0 \
uv run python notebooks/02_model_scratch_and_finetune.py
```

## Cuadernos de datos y predicción (entrega parcial)

Todo se lee directamente de `database.lichess.org`; no se necesita descargar un
mes completo. Cada cuaderno cierra el flujo HTTP en cuanto reúne la muestra
pedida. El cuaderno 01 escribe `data/processed/*.parquet` (local, fuera de Git)
y los cuadernos 02 y 03 escriben los checkpoints en `data/models/` (versionados
con Git LFS).

| Cuaderno | Contenido | Salidas |
|---|---|---|
| `notebooks/01_eda_and_cleaning.py` | Exploración y limpieza de partidas `.pgn.zst` y evaluaciones `lichess_db_eval.jsonl.zst` | `games_clean_sample.parquet`, `positions_games_sample.parquet`, `evals_clean_sample.parquet` |
| `notebooks/02_model_scratch_and_finetune.py` | **Enfoque 1**: red convolucional entrenada desde cero con jugadas humanas y dos ajustes finos con Stockfish (todos los pesos y LoRA en el tronco y la cabeza) | `approach1_scratch.pt`, `approach1_weight_ftuning.pt`, `approach1_lora_ftuning.pt` |
| `notebooks/03_maia.py` | **Enfoque 3**: Maia-3 preentrenado ajustado con LoRA; un modelo con jugadas humanas y otro con jugadas de Stockfish | `maia3_lora_human.pt`, `maia3_lora_stockfish.pt` |
| `notebooks/04_model_playground.py` | **Prueba visual**: detecta todos los `.pt`/`.pth` bajo `data/models/` (incluidas subcarpetas) y compara las jugadas de los checkpoints elegidos en un FEN o en una partida guardada; no entrena | — |

La **ruta A** se salta este bloque: los `.pt` ya vienen en Git LFS y los Parquet
solo hacen falta para reentrenar. La **ruta B** ejecuta los cuadernos en orden;
también se pueden abrir con interfaz:

```bash
uv run marimo edit notebooks/01_eda_and_cleaning.py
uv run marimo edit notebooks/02_model_scratch_and_finetune.py
uv run marimo edit notebooks/03_maia.py
uv run marimo edit notebooks/04_model_playground.py
```

Los cuadernos 2 y 3 generan una muestra pequeña por su cuenta si los Parquet
del cuaderno 1 todavía no existen. El cuaderno 4 detecta cualquier archivo
`.pt` o `.pth` que guardes bajo `data/models/`, incluidas subcarpetas, y permite
elegir cuáles comparar; las posiciones de partidas guardadas solo sirven para
inspección visual, no como una medida independiente de exactitud.

### Guardado de los modelos

Los cuadernos guardan cada checkpoint con `save_policy_checkpoint`, que escribe
el `state_dict` a media precisión (`float16`). Los modelos nuevos de 18 planos
ocupan unos **2.4 MB** (1.2 M parámetros) y las predicciones apenas cambian,
porque al cargar los tensores vuelven a `float32`.

- `data/models/` se versiona con **Git LFS**; `git lfs pull` descarga los pesos
  y un `.pt` sin descargar se detecta con un mensaje claro en vez de un error de
  deserialización.
- `data/processed/` es local y está ignorado por Git: el cuaderno 01 lo
  regenera desde Lichess. En Colab se queda en el disco de la sesión y se puede
  descargar con `colab download` (ruta C).
- La validación cruzada ya no escribe los pesos de cada partición en
  `data/models/cv/`: se conservan en memoria mientras dura el cuaderno, así que
  no deja cientos de megabytes de checkpoints temporales.
- `scripts/compact_checkpoints.py` convierte checkpoints antiguos a media
  precisión; `--dry-run` solo informa del ahorro.
- `load_policy_checkpoint` lee el número de canales de las convoluciones del
  archivo, así que los modelos entrenados con otra arquitectura siguen cargando
  en el cuaderno 4 y en `live_chess_fen.py`.

### Enfoque 3: Maia-3 + LoRA

`03_maia.py` descarga el [Maia-3](https://github.com/CSSLab/maia3) oficial
(5M o 23M parámetros) desde Hugging Face a `data/models/maia3/`, congela sus
pesos y entrena adaptadores LoRA (unos 100 mil parámetros con rango 8) sobre
el MLP de cada bloque y la cabeza de jugadas. Se entrenan dos adaptadores
independientes, ambos a partir del mismo Maia-3:

- **A · humano**: `positions_games_sample.parquet` (`fen` → `move_uci`).
- **B · Stockfish**: `evals_clean_sample.parquet` (`fen` → `best_move`).

El cuaderno compara los dos contra Maia-3 sin ajustar en un conjunto de
prueba que no se usa para entrenar ni para elegir la época. También se puede
ejecutar sin interfaz, y entonces entrena directamente:

```bash
uv run python notebooks/03_maia.py
```

La lógica está en `src/lichess_pipeline/model/maia3_lora.py`. El código de
Maia-3 es AGPL-3.0 y se instala desde su repositorio de GitHub.

### Cómo funciona el modelo de jugadas

- **Entrada**: una cadena FEN (incluye posición, turno, enroques y captura al
  paso). No se usa una matriz suelta: el FEN ya transporta todo lo necesario.
- **Codificación**: tensor `18 x 8 x 8`. El tablero se mira siempre desde el
  lado que tiene el turno: si mueven las negras se refleja el tablero y la
  jugada, así que los mismos patrones se aprenden una sola vez para ambos
  colores. Los planos son 6 piezas propias, 6 del rival, 4 derechos de enroque,
  1 casilla de captura al paso y 1 con el turno original.
- **Red**: torre convolucional residual (4 bloques `3x3`, campo receptivo de
  todo el tablero) y cabeza espacial: cada casilla se proyecta a un vector de
  origen y otro de destino, y una jugada puntúa como su producto escalar más
  un sesgo aprendido `64 x 64`. Son **~1.2 M parámetros** (2.4 MB a media
  precisión) en lugar de 34 M, y una jugada poco frecuente igual entrena sus
  casillas de origen y destino.
- **Salida**: una de `64 x 64 = 4096` clases `origen -> destino`.
- **Legalidad**: antes del `softmax` se enmascaran las jugadas ilegales con
  `python-chess`, así que el modelo solo puede equivocarse de elección, nunca
  proponer una jugada ilegal.
- **Dos ajustes finos opcionales**: después del entrenamiento desde cero se
  prueban dos formas de especializar el modelo con Stockfish. Ajustar todos los
  pesos puede deshacer parte de lo aprendido con una muestra pequeña, así que
  también se ofrece **LoRA**: la red se congela y solo se entrena una corrección
  de bajo rango sobre el tronco residual y las proyecciones de la cabeza de
  jugadas, conservando la época con la menor pérdida de validación. El cuaderno
  compara los tres modelos y guarda cada uno por separado
  (`approach1_scratch.pt`, `approach1_weight_ftuning.pt`,
  `approach1_lora_ftuning.pt`). El adaptador LoRA se fusiona antes de guardar,
  así que ese archivo se carga como un `ChessPolicyNet` normal.
- **Optimización**: `train_model` usa Adam con decaimiento coseno de la tasa de
  aprendizaje y restaura la época con la menor pérdida de validación, lo que
  afina las probabilidades. El dispositivo se elige solo: CUDA, luego MPS y si
  no, CPU.
- **Compatibilidad**: los checkpoints antiguos (`_v1`, `_v2`, 13 planos y cabeza
  plana) se detectan y cargan automáticamente, así que se pueden comparar con
  los nuevos en el cuaderno 4 y en la vista en vivo.
- **Evaluación**: se reserva un **20 % de prueba** que no se toca hasta el final.
  El 80 % restante (desarrollo) se mide con **validación cruzada de 5
  particiones**, agrupando partidas completas por `game_id` para que la
  validación no contenga posiciones casi idénticas a las de entrenamiento. Con
  los resultados de la cruzada se eligen los ajustes, se reajustan los tres
  modelos con todo el desarrollo y se evalúan una sola vez en la prueba.

### Límites de escala

Para el servidor final no basta con subir los números: el pipeline actual
materializa las muestras en memoria (listas de Python y DataFrames completos,
más un tensor y una máscara de 4096 posiciones por FEN). La ejecución a escala
completa requiere procesar por bloques hacia Parquet y un dataset de
entrenamiento iterable que codifique el FEN y su máscara bajo demanda. La
lógica de parseo y el contrato del modelo no cambian; solo la capa de E/S.

## Visión en vivo: cámara → tablero → piezas YOLO → FEN → jugada

La ruta de producción es [`scripts/live_chess_fen.py`](scripts/live_chess_fen.py).
No usa ArUco ni clics: OpenCV localiza automáticamente las 49 intersecciones
internas del patrón 8×8 y calcula la homografía. Después un checkpoint YOLO
entrenado para piezas se ejecuta sobre el tablero rectificado; cada centro de
caja se asigna a una casilla y se construye el FEN.

### 1 · Consigue los pesos de piezas

El lector necesita un checkpoint YOLO con las 12 clases coloreadas.

- **Ruta A (sin entrenar):** el repositorio versiona los pesos de visión en
  `data/models/` con Git LFS:
  ```bash
  git lfs pull
  ```
- **Ruta B (entrenar desde cero):** si el archivo no está publicado, entrena el
  detector público. El comando descarga el espejo combinado
  [dopaul/chess-pieces-merged](https://huggingface.co/datasets/dopaul/chess-pieces-merged),
  CC BY 4.0, exporta los datos en formato YOLO, ajusta `yolo11n.pt` y copia el
  mejor checkpoint a `data/models/chess-pieces.pt`:
  ```bash
  uv run python scripts/train_chess_piece_yolo.py \
    --epochs 50 --device mps --imgsz 640 --batch 16
  ```

Para uso confiable, añade al ajuste fotos de tu cámara, piezas e iluminación.

### 2 · Ejecuta la vista en vivo

`ultralytics` ya está en `pyproject.toml`, así que `uv sync` lo instala. Con los
pesos en su sitio:

```bash
uv run python scripts/live_chess_fen.py --weights data/models/chess-pieces.pt
```

Los nombres de clase aceptados incluyen `w-pawn`/`b-king`,
`white_pawn`/`black_king` y variantes equivalentes. Un checkpoint COCO normal
(`yolo*.pt`) **no sirve**, porque no conoce piezas de ajedrez. Un punto de
partida público es el modelo/dataset [Chess Piece Detect de Roboflow Universe](https://universe.roboflow.com/chess-ynmuj/chess-piece-detect).

La vista en vivo dibuja cada detección con su tipo y color: cajas azul claro
para blancas y rojo para negras, por ejemplo `peón negro 0.91`. El modelo que
se entrenó aquí es de **detección** y por eso usa cajas. Si se proporcionan
pesos **YOLO-seg** de piezas, el mismo comando superpone además la máscara de
píxeles de cada peón, torre, caballo, etc.:

Ya hay unos pesos YOLO-seg validados en `data/models/chess-pieces-seg.pt`
(descárgalos con `git lfs pull`). Úsalos así:

```bash
uv run python scripts/live_chess_fen.py \
  --weights data/models/chess-pieces-seg.pt \
  --require-segmentation \
  --confidence 0.25
```

No se debe entrenar YOLO-seg con este dataset de cajas: la segmentación real
necesita una máscara o polígono de cada pieza. El lector ya está preparado para
esos pesos; el siguiente conjunto de ajuste debe anotar los contornos de tus
piezas reales, especialmente con manos y oclusiones.

### Tablero digital mostrado en celular

Si la cámara mira una imagen de tablero en el celular, los pesos de tablero
físico no generalizan a sus sprites 2D. Para el diseño de piezas gris/blanco
incluido como plantilla se entrenó un YOLO-seg específico, disponible en
`data/models/screen-chess-pieces-seg.pt` (si el archivo falta, recupera los
pesos con `git lfs pull`):

```bash
uv run python scripts/live_chess_fen.py \
  --weights data/models/screen-chess-pieces-seg.pt \
  --require-segmentation --screen-board --imgsz 416 \
  --confidence 0.25 --camera 0 \
  --turn w --castling KQkq
```

El modelo detecta y segmenta las 32 piezas de la plantilla y genera el FEN
inicial correctamente. [train_screen_chess_yoloseg.py](scripts/train_screen_chess_yoloseg.py)
regenera las posiciones sintéticas y reentrena el modelo si cambias el estilo
de las piezas.

La cámara y YOLO funcionan en hilos distintos. La ventana `Camera` sigue
actualizándose mientras se calcula la siguiente segmentación; la ventana
`Board + YOLO pieces` cambia cuando termina cada inferencia. `CAM LIVE` muestra
los FPS de YOLO y la antigüedad real de la última predicción. El modo
`--screen-board` localiza rápidamente el tablero verde de esta plantilla y
guarda sus esquinas. Pulsa `r` para recalibrar al mover cámara o tablero;
`--redetect-every 60` lo hace periódicamente si el encuadre cambia. Un tamaño
`--imgsz 416` reduce el tiempo de inferencia con posible pérdida de detalle.

Con los modelos de jugadas en `data/models/` (ruta A ya incluida, o salida de
la ruta B o C), la vista en vivo los carga sin argumentos extra y dibuja una
flecha por modelo sobre el tablero real:

```bash
uv run python scripts/live_chess_fen.py \
  --weights data/models/chess-pieces.pt \
  --turn w --castling KQkq
```

Cada modelo tiene su color y su probabilidad se muestra en una etiqueta junto a
la casilla de destino y en la leyenda de la esquina:

| Modelo | Archivo | Color |
|---|---|---|
| Desde cero | `approach1_scratch.pt` | cian |
| Ajuste de pesos | `approach1_weight_ftuning.pt` | naranja |
| Ajuste LoRA | `approach1_lora_ftuning.pt` | verde |
| Maia-3 humano | `maia3_lora_human.pt` | azul |
| Maia-3 Stockfish | `maia3_lora_stockfish.pt` | magenta |

Las flechas se proyectan con la homografía inversa y siguen el estilo SeeMate:
trayectoria en L para los caballos, casilla de destino resaltada y etiquetas
con la jugada en notación algebraica (`Nf6`) y su porcentaje. Para comparar
otros archivos, pásalos con `--move-models` (uno o varios); `--move-model`
sigue aceptando un único checkpoint. La arquitectura se detecta desde el
archivo, así que ya no hace falta `--architecture`.

La ventana muestra el FEN, el turno, el recuento de piezas y actualiza las
sugerencias sólo cuando la posición se mantiene igual durante cuatro frames.
`t` alterna el turno y `f` rota la orientación si la cámara está del lado de
negras. `--board-weights` acepta de forma opcional un YOLO-seg que tenga una
clase `board` o `chessboard`; se usa como respaldo cuando las líneas de la
cuadrícula están demasiado ocultas.

La imagen no puede saber quién tiene el turno, si una torre o rey ya se movió,
ni la casilla de captura al paso. Para que el predictor sea correcto, conserva
esos campos desde el historial de la partida con `--turn`, `--castling` y
`--en-passant`; FEN no es sólo las piezas.

## Demo heredada de calibración con ArUco

Esta demo localiza un tablero con cuatro marcadores ArUco, corrige la
perspectiva y superpone las 64 casillas. No intenta todavía calcular la mejor
jugada: el punto de integración para un modelo de clasificación de piezas está
en `classify_square`.

Para entender el diseño, la geometría y el siguiente paso de entrenamiento,
consulta el [documento técnico](docs/vision_tecnica_tablero.md) o descarga la
[versión Word](docs/Documento_Tecnico_Vision_Tablero_Ajedrez.docx).

### Preparación física

Genera e imprime cuatro marcadores del diccionario `DICT_4X4_50`, con IDs 10,
11, 12 y 13:

```bash
uv run python scripts/generate_aruco_markers.py
```

Los PNG quedan en `data/aruco_markers/`. Colócalos por fuera del tablero y en este orden visto desde el
lado de las blancas: arriba-izquierda, arriba-derecha, abajo-derecha,
abajo-izquierda. No tapes las esquinas del tablero con los marcadores.

### Ejecutar

```bash
uv run python scripts/chessboard_camera_demo.py
```

Usa `--camera 1` si la cámara no es el dispositivo 0; pulsa `q` o `Esc` para
cerrar. La primera ventana muestra la cámara y la segunda el tablero visto
desde arriba con la cuadrícula `a8` a `h1`.

```bash
uv run python scripts/chessboard_camera_demo.py --camera 1 --size 800
```

### Foto tomada con celular sin marcadores ArUco

La demo de webcam necesita los cuatro ArUco. Para una foto plana o inclinada
tomada con celular, usa el selector manual:

```bash
uv run python scripts/chessboard_photo_demo.py --image /ruta/a/tablero.jpg
```

Haz clic en las esquinas **internas** de la cuadrícula 8×8 y en este orden:
`a8`, `h8`, `h1`, `a1`. Pulsa Enter y se creará un PNG rectificado junto a la
foto, con las 64 casillas sobrepuestas. Pulsa `r` para repetir la selección si
el orden o las esquinas no quedaron bien.

### Webcam viendo una imagen en el iPhone sin ArUco

Si estás mostrando una foto del tablero en la pantalla del iPhone frente a la
webcam, usa el modo manual en vivo:

```bash
uv run python scripts/chessboard_camera_demo.py --manual-corners
```

Apunta la webcam al iPhone y pulsa `c`. Haz clic sobre la pantalla en las
cuatro esquinas internas del tablero, en orden `a8`, `h8`, `h1`, `a1`, y pulsa
Enter. La ventana `Rectified board` debe mostrar la cuadrícula 8×8. Mantén
quietos el iPhone y la cámara; si los mueves, pulsa `c` otra vez para recalibrar.

### Siguiente iteración: reconocimiento de piezas

Después de entrenar un clasificador con las clases `empty`, `white_pawn`, ...,
`black_king`, reemplaza el cuerpo de `classify_square` en
`src/lichess_pipeline/chess_vision.py`. Cada imagen recibida por esa función
es ya un recorte normalizado de una casilla del tablero rectificado.
