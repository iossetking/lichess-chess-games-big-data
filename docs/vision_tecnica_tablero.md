# Documento técnico: visión por computadora para un tablero físico de ajedrez

## 1. Propósito y alcance

Este componente convierte la imagen de una cámara en un FEN para un tablero de
ajedrez real. Localiza automáticamente el tablero, corrige la perspectiva,
identifica las piezas con YOLO y obtiene las 64 casillas en coordenadas
conocidas.

El entregable actual entrega el FEN estabilizado a la red de predicción de
jugadas del repositorio. Mantiene separado el problema geométrico (tablero) del
de reconocimiento (piezas), lo cual permite sustituir o ajustar el checkpoint
de YOLO sin cambiar la integración del predictor.

El sistema entrega una imagen cenital del tablero y un mapeo de las 64 regiones
desde `a8` hasta `h1`. Esto reduce el problema de reconocimiento de piezas de
"buscar objetos en una escena" a "clasificar el contenido de una casilla ya
conocida".

## 2. Problema que resuelve

Una cámara frontal o lateral ve el tablero como un cuadrilátero deformado por
la perspectiva. Las casillas más lejanas parecen más pequeñas y no se pueden
dividir simplemente por píxeles de la imagen original.

Además, detectar las piezas directamente sobre una imagen sin rectificar exige
que el modelo aprenda muchas escalas, inclinaciones, iluminaciones y posiciones
posibles. Primero normalizar la geometría hace que las piezas aparezcan casi
siempre con la misma escala y orientación.

```text
Cámara en color
      │
      ▼
OpenCV: 49 intersecciones internas del patrón 8 × 8
      │       (o YOLO-seg de tablero como respaldo)
      ▼
Transformación de perspectiva / homografía
      │
      ▼
Tablero cenital cuadrado, por defecto 800 × 800 px
      │
      ▼
YOLO de piezas: 12 clases + cajas delimitadoras
      │
      ▼
Centro de cada caja → casilla → FEN estable → predictor de jugadas
```

## 2.1 Qué checkpoint se necesita

No existe un checkpoint YOLO oficial y universal para piezas de ajedrez. Los
pesos genéricos de YOLO sólo conocen COCO y no reconocen peones, torres ni
casillas. El script exige un checkpoint con las 12 clases con color, por
ejemplo `w-pawn` hasta `b-king`; acepta las variantes `white_pawn` y
`black_king`. `scripts/train_chess_piece_yolo.py` descarga un espejo combinado
CC BY 4.0 con los splits YOLO y ajusta `yolo11n.pt` localmente. La
generalización depende mucho de las piezas, el tablero y la iluminación, por
lo que debe terminarse el ajuste con imágenes tomadas con la cámara final.

La orientación y el estado histórico no son visibles en una foto. El sistema
permite rotar el tablero y recibe turno, enroques y captura al paso como
contexto de la partida. Sin esos campos, el resultado contiene la colocación de
piezas, pero no necesariamente un FEN apto para predecir una jugada legal.

## 3. Decisión de diseño anterior: marcadores ArUco

### Qué son

ArUco es una familia de marcadores cuadrados fiduciarios: patrones en blanco y
negro que OpenCV puede detectar e identificar de forma rápida. Cada patrón
tiene un ID, por lo que no basta con saber que hay cuatro cuadrados; el
programa sabe cuál representa cada esquina.

La demo usa el diccionario `DICT_4X4_50` y estos IDs:

| ID | Posición física | Rol geométrico |
| --- | --- | --- |
| 10 | Arriba-izquierda | esquina superior izquierda |
| 11 | Arriba-derecha | esquina superior derecha |
| 12 | Abajo-derecha | esquina inferior derecha |
| 13 | Abajo-izquierda | esquina inferior izquierda |

La orientación se define al ver el tablero desde el lado de las piezas blancas:
la fila superior de la imagen rectificada corresponde a la octava fila (`8`) y
la inferior a la primera (`1`).

### Por qué se eligieron inicialmente

- **No necesita un conjunto de entrenamiento para localizar el tablero.** Se
  puede demostrar inmediatamente, incluso con pocas fotos disponibles.
- **Cada esquina tiene identidad.** Evita adivinar si una esquina es `a8` o
  `h1`, un error que invierte el tablero y produce una posición inválida.
- **Es robusto en tiempo real.** La detección es rápida, incluso si hay piezas
  cubriendo casillas interiores.
- **Simplifica la segmentación.** Al conocer los cuatro vértices, las 64
  casillas ya no se detectan individualmente: se calculan.

### Qué se sacrifica y por qué ya no son la ruta principal

Los marcadores deben imprimirse y mantenerse visibles junto al tablero. La ruta
actual usa `cv2.findChessboardCornersSB` para detectar automáticamente las
intersecciones 7 × 7 de la cuadrícula. Si las piezas, una mano o reflejos
ocultan demasiado la cuadrícula, `--board-weights` permite aportar un YOLO de
segmentación con una máscara de clase `board` o `chessboard`.

## 4. Geometría: de una cámara inclinada a una vista cenital

Una homografía es una matriz de 3 × 3 que relaciona puntos de un plano vistos
desde dos perspectivas. El tablero físico es aproximadamente un plano; por
eso, con cuatro puntos no colineales es posible calcular la transformación.

El programa toma el centro de los cuatro marcadores detectados como puntos de
origen:

```text
origen = [marcador 10, marcador 11, marcador 12, marcador 13]
       = [sup-izq, sup-der, inf-der, inf-izq]
```

Y los relaciona con los puntos destino de una imagen cuadrada de lado `S`:

```text
destino = [(0, 0), (S-1, 0), (S-1, S-1), (0, S-1)]
```

OpenCV calcula esa matriz con `cv2.getPerspectiveTransform` y transforma todos
los píxeles con `cv2.warpPerspective`. El resultado es un tablero que se ve
desde arriba. Por defecto, `S = 800`; por tanto, cada casilla mide
`800 / 8 = 100` píxeles.

Una homografía funciona correctamente solo si el tablero y los marcadores están
en el mismo plano. Si los marcadores quedan muy elevados, se pegan en una pared
o se usa un lente muy gran angular, la corrección deja de ser exacta.

## 5. Segmentación en 64 casillas

Después de rectificar, la segmentación no usa una red neuronal. Es una división
determinista de la imagen cuadrada:

```text
columna = 0..7  → archivos a..h
fila    = 0..7  → rangos 8..1

x1 = columna × tamaño_casilla
y1 = fila × tamaño_casilla
x2 = (columna + 1) × tamaño_casilla
y2 = (fila + 1) × tamaño_casilla
```

Por ejemplo, con una imagen de 800 px:

| Casilla | Región `(x1, y1, x2, y2)` |
| --- | --- |
| `a8` | `(0, 0, 100, 100)` |
| `h1` | `(700, 700, 800, 800)` |

Este mapeo está implementado en `square_regions`. Usar notación algebraica desde
esta etapa evita tener que convertir coordenadas de píxel a casillas después.

## 6. Componentes implementados

### `src/lichess_pipeline/chess_vision.py`

Es un módulo independiente de la webcam que contiene la lógica reutilizable y
probable de probar:

| Función | Qué hace | Por qué existe |
| --- | --- | --- |
| `board_destination(size)` | Crea los 4 puntos de destino de la vista cenital. | Mantiene consistente la homografía. |
| `square_regions(size)` | Devuelve las regiones de `a8` a `h1`. | Convierte el tablero rectificado en entradas para el clasificador. |
| `detect_board_corners(frame)` | Encuentra las 49 intersecciones internas con OpenCV. | Localiza el tablero sin marcadores. |
| `detections_to_fen(...)` | Asigna el centro de cada caja YOLO a una casilla y emite FEN. | Es el contrato entre visión y el predictor de jugadas. |
| `FenStabilizer` | Espera varios frames iguales antes de emitir una posición. | Evita FEN intermitentes por detecciones ruidosas. |

Los ayudantes de ArUco y `classify_square` continúan en el módulo sólo para las
demos de calibración heredadas.

### `scripts/live_chess_fen.py`

Es la aplicación de cámara. Su ciclo es:

1. Abrir la webcam seleccionada con `cv2.VideoCapture`.
2. Leer un frame.
3. Detectar automáticamente la cuadrícula interna 7 × 7; si falla, usar la
   máscara del YOLO-seg opcional.
4. Calcular la homografía y rectificar el tablero.
5. Ejecutar YOLO de piezas sobre la imagen rectificada.
6. Convertir sus cajas y clases a FEN, después de estabilizar varios frames.
7. Si se configuró un checkpoint del predictor, pedir la mejor jugada legal.
8. Repetir hasta pulsar `q` o `Esc`.

Hay dos ventanas: `Camera` enseña lo que está observando la cámara y
`Board + YOLO pieces` enseña las detecciones y la cuadrícula rectificada.

### `scripts/generate_aruco_markers.py`

Genera los cuatro PNG imprimibles mediante OpenCV. Por defecto mide 600 × 600
píxeles cada uno y los guarda en `data/aruco_markers/`. Puede cambiarse con
`--pixels` u `--output`.

### Dependencias

| Dependencia | Uso |
| --- | --- |
| `opencv-contrib-python` | Cámara, detección de cuadrícula, homografía y ventanas. |
| `ultralytics` | Carga e inferencia de los pesos YOLO de piezas y, opcionalmente, de segmentación del tablero. |
| `numpy` | Arreglos de coordenadas y de imágenes. |
| `pytest` | Pruebas automáticas del mapeo geométrico. |
| `uv` | Gestiona el entorno reproducible y el archivo `uv.lock`. |

## 7. Cómo ejecutar y demostrarlo

Desde la raíz del repositorio:

```bash
uv sync
uv run python scripts/live_chess_fen.py --weights data/models/chess-pieces.pt
```

Si hay más de una cámara:

```bash
uv run python scripts/live_chess_fen.py --weights data/models/chess-pieces.pt --camera 1
```

Antes de demostrar:

1. Montar la cámara de modo que se vean las cuatro esquinas y las líneas 8×8.
2. Asegurar iluminación uniforme y poco reflejo.
3. Usar pesos YOLO entrenados en piezas reales, no un modelo COCO genérico.
4. Verificar que la cuadrícula rectificada tenga `a8` arriba a la izquierda;
   pulsar `f` si la cámara está del lado de negras.
5. Conservar turno, enroques y captura al paso desde el historial de la partida.

## 8. Ajuste del detector YOLO de piezas

### Formulación recomendada

El detector recibe el tablero rectificado y devuelve una caja por pieza. Se
recomienda entrenarlo con **12 clases** que incluyan tipo y color:

```text
w-pawn, w-knight, w-bishop, w-rook, w-queen, w-king
b-pawn, b-knight, b-bishop, b-rook, b-queen, b-king
```

La salida son cajas delimitadoras y confianza. El centro de la caja determina
la casilla después de rectificar; si dos cajas caen en la misma casilla, se
conserva la de mayor confianza.

### Datos necesarios

Grabar fotos o video del **mismo tablero, piezas y cámara** previstos para la
demo. Conviene variar:

- iluminación cálida/fría, sombras y reflejos;
- cámara levemente desplazada, más alta y más baja;
- tipos de piezas, especialmente caballos, reyes y damas;
- aperturas, medio juego y finales, para no entrenar solo con peones;
- casillas vacías, que serán muy frecuentes.

Se puede extraer un frame cada pocos segundos, rectificarlo con este módulo y
etiquetar sus recortes. Es crucial partir por posiciones completas distintas:
muchos frames consecutivos del mismo tablero son casi duplicados y no deben
quedar separados entre entrenamiento y validación.

### Entrenamiento y conexión

Se puede partir de `yolo11n.pt` y ajustar detección con Ultralytics, incluyendo
variaciones de ángulo, escala, desenfoque e iluminación. El checkpoint debe
devolver clase, confianza y caja. En cada frame la integración recibe objetos
como:

```python
PieceDetection("b-rook", 0.99, (5, 5, 95, 95))
PieceDetection("w-king", 0.98, (410, 710, 490, 795))
```

`detections_to_fen` convierte esa lista en FEN. La visión no decide legalidad
ni la mejor jugada; esa responsabilidad queda separada en el predictor.

## 9. Respaldo de segmentación para tableros difíciles

Cuando `findChessboardCornersSB` no puede ver suficiente cuadrícula por piezas,
manos, reflejos o un tablero con líneas débiles, se etiqueta el contorno entero
del tablero y se entrena un modelo YOLO de segmentación. La máscara se reduce
a cuatro esquinas y la salida sigue siendo la misma homografía.

Esto es útil si el proyecto exige detectar cualquier tablero sin accesorios,
pero aumenta el riesgo técnico:

| Aspecto | Cuadrícula OpenCV | Segmentación YOLO |
| --- | --- | --- |
| Datos etiquetados | No | Sí |
| Latencia | Muy baja | Mayor, por inferencia |
| Tableros distintos | Depende del contraste | Puede generalizar, si los datos bastan |
| Fallas por piezas cubriendo líneas | Posibles | Depende del entrenamiento |
| Robustez inicial | Alta con rejilla visible | Debe medirse |

Una alternativa clásica es detectar líneas del tablero con bordes/Hough o usar
`findChessboardCorners`. No se tomó como solución primaria porque las piezas,
las texturas de madera, reflejos y las líneas poco contrastadas pueden ocultar
las intersecciones. Además, `findChessboardCorners` se diseñó principalmente
para patrones de calibración sin piezas encima.

## 10. Limitaciones y mitigaciones

| Riesgo | Efecto | Mitigación |
| --- | --- | --- |
| La cuadrícula queda oculta | No se rectifica el tablero. | Alejar la mano, mejorar la luz o usar `--board-weights`. |
| Cuadrícula de poco contraste | No se detecta con fiabilidad. | Usar un tablero con contraste o entrenar segmentación. |
| Reflejos o poca luz | Peor detección y clasificación. | Iluminación difusa y fijar exposición si la cámara lo permite. |
| Lente gran angular | Líneas curvas; la homografía no compensa toda la distorsión. | Alejar la cámara o calibrarla y aplicar `undistort` antes. |
| Mano sobre el tablero | El clasificador puede producir una posición transitoria falsa. | No aceptar una posición hasta que sea estable varios frames; opcionalmente detectar manos. |
| Cámara se mueve | Cambia la geometría entre frames. | Mantenerla fija en trípode; el sistema recalcula la homografía por frame. |
| Piezas no vistas en entrenamiento | Baja precisión de clase. | Entrenar con las piezas físicas finales y aumentar datos. |

## 11. Pruebas realizadas

Las pruebas en `tests/test_chess_vision.py` verifican que:

1. la imagen destino ocupa correctamente `0` a `size - 1`;
2. existen exactamente 64 regiones;
3. `a8` y `h1` se ubican donde corresponde;
4. no se acepta un tamaño que no pueda dividirse entre ocho;
5. las 49 intersecciones se extrapolan a cuatro esquinas;
6. las etiquetas YOLO se normalizan y las cajas se convierten a FEN;
7. el estabilizador publica sólo posiciones persistentes.

Para ejecutarlas junto con el resto del proyecto:

```bash
uv run python -m pytest -q
```

En la implementación actual pasaron 37 pruebas.

## 13. Opción adicional: calibración manual desde foto

Para una foto aislada se conserva `scripts/chessboard_photo_demo.py` como
herramienta de calibración manual. La ruta en vivo no necesita ArUco: intenta
la detección automática de la cuadrícula y puede usar segmentación YOLO como
respaldo. Los cuatro clics son útiles para inspeccionar una imagen problemática
o preparar datos de entrenamiento.

El usuario carga la foto y hace clic en los cuatro vértices **internos** de la
cuadrícula 8×8, en el orden `a8`, `h8`, `h1`, `a1`. No debe seleccionar el
borde decorativo de madera ni la sombra exterior del tablero. Con esos cuatro
puntos el programa aplica exactamente la misma homografía de la cámara y crea
un PNG rectificado con sus 64 regiones.

```bash
uv run python scripts/chessboard_photo_demo.py --image /ruta/a/tablero.jpg
```

El modo manual elimina el punto de fallo geométrico de una foto aislada. Si la
orientación elegida es correcta, la imagen resultante tiene `a8` en la esquina
superior izquierda y `h1` en la inferior derecha.

El mismo método está disponible para la webcam con
`uv run python scripts/chessboard_camera_demo.py --manual-corners`. Es útil al
mostrar una imagen de referencia en la pantalla de un celular: se pulsa `c`,
se seleccionan los cuatro puntos y la cuadrícula se mantiene mientras la cámara
y el celular no cambien de posición. Si cambian, se vuelve a pulsar `c` para
recalibrar.

## 12. Respuestas breves para una presentación

**¿Qué problema resuelve?** Convierte el tablero físico visto por cámara en una
matriz 8 × 8 en perspectiva cenital, lista para reconocer piezas.

**¿Por qué no se detectan las 64 casillas con IA?** Porque, una vez conocidas
las cuatro esquinas, se calculan exactamente por geometría. Es más rápido,
explicable y requiere cero entrenamiento adicional.

**¿Por qué ArUco?** Da cuatro esquinas identificadas de forma fiable y hace que
el MVP funcione sin depender de un gran dataset.

**¿Dónde entra el machine learning?** En la siguiente capa: cada una de las 64
imágenes normalizadas se clasifica como vacía o una de las 12 piezas posibles.

**¿Por qué separar visión y motor de ajedrez?** La visión responde “qué hay en
cada casilla”; el motor responde “qué jugada conviene”. Separar responsabilidades
permite probar, reemplazar y mejorar cada parte sin romper la otra.

**¿Puede funcionar sin marcadores?** Sí, entrenando un detector de las cuatro
esquinas, pero el mismo flujo de rectificación se conserva y primero hay que
reunir datos etiquetados.
