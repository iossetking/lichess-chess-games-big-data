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
git lfs pull   # descarga los checkpoints versionados en data/models/
```

Los checkpoints se guardan a media precisión (`float16`) para ocupar la mitad y
se versionan con Git LFS (ver `.gitattributes`); `git lfs pull` los descarga
tras clonar. Los pesos de visión de YOLO (`chess-pieces*.pt`,
`chess-piece-classifier.pt`) pueden venir en el repositorio, y si faltan se
generan con los entrenadores de `scripts/` (ver la sección de visión).

## Cuadernos de datos y predicción (entrega parcial)

Todo se lee directamente de `database.lichess.org`; no se necesita descargar un
mes completo. Cada cuaderno cierra el flujo HTTP en cuanto reúne la muestra
pedida y guarda sus resultados en `data/processed/` (ignorado por git) y
`data/models/` (versionado con Git LFS).

| Cuaderno | Contenido | Salidas |
|---|---|---|
| `notebooks/01_eda_and_cleaning.py` | Exploración y limpieza de partidas `.pgn.zst` y evaluaciones `lichess_db_eval.jsonl.zst` | `games_clean_sample.parquet`, `positions_games_sample.parquet`, `evals_clean_sample.parquet` |
| `notebooks/02_model_scratch_and_finetune.py` | **Enfoque 1**: red convolucional entrenada desde cero con jugadas humanas y dos ajustes finos con Stockfish (todos los pesos y LoRA en la cabeza) | `approach1_scratch.pt`, `approach1_weight_ftuning.pt`, `approach1_lora_ftuning.pt` |
| `notebooks/03_maia.py` | **Enfoque 3**: Maia-3 preentrenado ajustado con LoRA; un modelo con jugadas humanas y otro con jugadas de Stockfish | `maia3_lora_human.pt`, `maia3_lora_stockfish.pt` |
| `notebooks/04_model_playground.py` | **Prueba visual**: detecta todos los `.pt`/`.pth` bajo `data/models/` (incluidas subcarpetas) y compara las jugadas de los checkpoints elegidos en un FEN o en una partida guardada; no entrena | — |

Ejecuta los cuadernos en orden:

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
el `state_dict` a media precisión (`float16`): el mismo modelo ocupa la mitad
(unos 64 MB en lugar de 129 MB) y las predicciones apenas cambian, porque al
cargar los tensores vuelven a `float32`.

- `data/models/` se versiona con **Git LFS**; `git lfs pull` descarga los pesos
  y un `.pt` sin descargar se detecta con un mensaje claro en vez de un error de
  deserialización.
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
- **Codificación**: tensor `13 x 8 x 8` (6 planos de piezas blancas, 6 de negras
  y 1 para el turno).
- **Salida**: una de `64 x 64 = 4096` clases `origen -> destino`.
- **Legalidad**: antes del `softmax` se enmascaran las jugadas ilegales con
  `python-chess`, así que el modelo solo puede equivocarse de elección, nunca
  proponer una jugada ilegal.
- **Dos ajustes finos opcionales**: después del entrenamiento desde cero se
  prueban dos formas de especializar el modelo con Stockfish. Ajustar todos los
  pesos puede deshacer parte de lo aprendido con una muestra pequeña, así que
  también se ofrece **LoRA**: la red se congela y solo se entrena una corrección
  de bajo rango sobre la cabeza de jugadas, conservando la época con la menor
  pérdida de validación. El cuaderno compara los tres modelos y guarda cada uno
  por separado (`approach1_scratch.pt`, `approach1_weight_ftuning.pt`,
  `approach1_lora_ftuning.pt`). El adaptador LoRA se fusiona antes de guardar,
  así que ese archivo se carga como un `ChessPolicyNet` normal.
- **Evaluación**: se reserva un **20 % de prueba** que no se toca hasta el final.
  El 80 % restante (desarrollo) se mide con **validación cruzada de 5
  particiones**, agrupando partidas completas por `game_id` para que la
  validación no contenga posiciones casi idénticas a las de entrenamiento. Con
  los resultados de la cruzada se eligen los ajustes, se reajustan los tres
  modelos con todo el desarrollo y se evalúan una sola vez en la prueba.

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

El lector necesita un checkpoint YOLO con las 12 clases coloreadas. El
repositorio versiona los pesos de visión en `data/models/` con Git LFS:

```bash
git lfs pull
```

Si no están disponibles, entrena el detector público. El comando descarga el
espejo combinado [dopaul/chess-pieces-merged](https://huggingface.co/datasets/dopaul/chess-pieces-merged),
CC BY 4.0, exporta los datos en formato YOLO, ajusta `yolo11n.pt` y copia el
mejor checkpoint a `data/models/chess-pieces.pt`:

```bash
uv run python scripts/train_chess_piece_yolo.py --epochs 50 --device mps
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

Al terminar los cuadernos de este repositorio, conecta directamente el modelo
de jugadas:

```bash
uv run python scripts/live_chess_fen.py \
  --weights data/models/chess-pieces.pt \
  --move-model data/models/approach1_lora_ftuning.pt \
  --architecture scratch --turn w --castling KQkq
```

La ventana muestra el FEN y actualiza la sugerencia sólo cuando la posición se
mantiene igual durante cuatro frames. `t` alterna el turno y `f` rota la
orientación si la cámara está del lado de negras. `--board-weights` acepta de
forma opcional un YOLO-seg que tenga una clase `board` o `chessboard`; se usa
como respaldo cuando las líneas de la cuadrícula están demasiado ocultas.

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
