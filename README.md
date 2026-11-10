# lichess-chess-games-big-data

Dos modelos sobre datos oficiales de [Lichess](https://database.lichess.org/):

1. **Visión**: detecta un tablero físico con cuatro marcadores ArUco, rectifica la
   perspectiva y prepara las 64 casillas para clasificar las piezas.
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

## Demo de visión: tablero físico por webcam

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
