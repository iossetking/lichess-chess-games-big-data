# lichess-chess-games-big-data

## Demo de visión: tablero físico por webcam

Esta demo localiza un tablero con cuatro marcadores ArUco, corrige la
perspectiva y superpone las 64 casillas. No intenta todavía calcular la mejor
jugada: el punto de integración para un modelo de clasificación de piezas está
en `classify_square`.

Para entender el diseño, la geometría y el siguiente paso de entrenamiento,
consulta el [documento técnico](docs/vision_tecnica_tablero.md).

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
uv sync
uv run python scripts/chessboard_camera_demo.py
```

Usa `--camera 1` si la cámara no es el dispositivo 0; pulsa `q` o `Esc` para
cerrar. La primera ventana muestra la cámara y la segunda el tablero visto
desde arriba con la cuadrícula `a8` a `h1`.

```bash
uv run python scripts/chessboard_camera_demo.py --camera 1 --size 800
```

### Siguiente iteración: reconocimiento de piezas

Después de entrenar un clasificador con las clases `empty`, `white_pawn`, ...,
`black_king`, reemplaza el cuerpo de `classify_square` en
`src/lichess_pipeline/chess_vision.py`. Cada imagen recibida por esa función
es ya un recorte normalizado de una casilla del tablero rectificado.
