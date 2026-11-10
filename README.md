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
```

## Cuadernos de datos y predicción (entrega parcial)

Todo se lee directamente de `database.lichess.org`; no se necesita descargar un
mes completo. Cada cuaderno cierra el flujo HTTP en cuanto reúne la muestra
pedida y guarda sus resultados en `data/processed/` y `data/models/` (ignorados
por git).

| Cuaderno | Contenido | Salidas |
|---|---|---|
| `notebooks/01_eda_and_cleaning.py` | Exploración y limpieza de partidas `.pgn.zst` y evaluaciones `lichess_db_eval.jsonl.zst` | `games_clean_sample.parquet`, `positions_games_sample.parquet`, `evals_clean_sample.parquet` |
| `notebooks/02_model_scratch_and_finetune.py` | **Enfoque 1**: red convolucional entrenada desde cero con jugadas humanas y ajustada con jugadas de Stockfish | `approach1_stage1_scratch.pt`, `approach1_stage2_finetuned.pt` |
| `notebooks/03_model_maia_pretrained.py` | **Enfoque 3**: red residual estilo Maia, comparada con el enfoque 1 en la misma validación | `approach3_maia_finetuned.pt` |

Ejecuta los cuadernos en orden:

```bash
uv run marimo edit notebooks/01_eda_and_cleaning.py
uv run marimo edit notebooks/02_model_scratch_and_finetune.py
uv run marimo edit notebooks/03_model_maia_pretrained.py
```

Los cuadernos 2 y 3 generan una muestra pequeña por su cuenta si los Parquet
del cuaderno 1 todavía no existen.

### Cómo funciona el modelo de jugadas

- **Entrada**: una cadena FEN (incluye posición, turno, enroques y captura al
  paso). No se usa una matriz suelta: el FEN ya transporta todo lo necesario.
- **Codificación**: tensor `13 x 8 x 8` (6 planos de piezas blancas, 6 de negras
  y 1 para el turno).
- **Salida**: una de `64 x 64 = 4096` clases `origen -> destino`.
- **Legalidad**: antes del `softmax` se enmascaran las jugadas ilegales con
  `python-chess`, así que el modelo solo puede equivocarse de elección, nunca
  proponer una jugada ilegal.
- **Dos etapas**: primero aprende de jugadas humanas de Elo alto; después se
  ajusta con las mejores jugadas de Stockfish. La tabla de resultados de cada
  cuaderno compara Top-1 y Top-3 antes y después del ajuste.

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

Instala la dependencia de inferencia y coloca unos pesos de YOLO con las 12
clases coloreadas en `data/models/`:

```bash
uv add ultralytics
uv run python scripts/live_chess_fen.py --weights data/models/chess-pieces.pt
```

Los nombres de clase aceptados incluyen `w-pawn`/`b-king`,
`white_pawn`/`black_king` y variantes equivalentes. Un checkpoint COCO normal
(`yolo*.pt`) **no sirve**, porque no conoce piezas de ajedrez. Un punto de
partida público es el modelo/dataset [Chess Piece Detect de Roboflow Universe](https://universe.roboflow.com/chess-ynmuj/chess-piece-detect). El proyecto incluye un entrenador que descarga el espejo combinado [dopaul/chess-pieces-merged](https://huggingface.co/datasets/dopaul/chess-pieces-merged), CC BY 4.0, y guarda los pesos listos para usar:

```bash
uv run python scripts/train_chess_piece_yolo.py --epochs 50 --device mps
```

El comando exporta los datos en formato YOLO, ajusta `yolo11n.pt` y copia el
mejor checkpoint a `data/models/chess-pieces.pt`. Para uso confiable, añade al
ajuste fotos de tu cámara, piezas e iluminación.

La vista en vivo dibuja cada detección con su tipo y color: cajas azul claro
para blancas y rojo para negras, por ejemplo `peón negro 0.91`. El modelo que
se entrenó aquí es de **detección** y por eso usa cajas. Si se proporcionan
pesos **YOLO-seg** de piezas, el mismo comando superpone además la máscara de
píxeles de cada peón, torre, caballo, etc.:

Ya hay unos pesos YOLO-seg validados en `data/models/chess-pieces-seg.pt`.
Úsalos así:

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
incluido como plantilla se entrenó un YOLO-seg específico y ya está disponible
en `data/models/screen-chess-pieces-seg.pt`:

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
  --move-model data/models/approach1_stage2_finetuned.pt \
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
