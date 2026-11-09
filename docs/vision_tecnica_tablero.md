# Documento técnico: visión por computadora para un tablero físico de ajedrez

## 1. Propósito y alcance

Este componente convierte la imagen de una cámara en una representación
geométrica estable de un tablero de ajedrez real. Localiza el tablero, corrige
la perspectiva y obtiene las 64 casillas en coordenadas conocidas.

El entregable actual **no calcula la mejor jugada ni identifica aún las piezas
reales**. Su objetivo es resolver la base de visión para que un clasificador de
piezas, y posteriormente un motor como Stockfish, reciban información ordenada
y consistente.

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
Detección de 4 marcadores ArUco
      │       (esquinas conocidas del tablero)
      ▼
Transformación de perspectiva / homografía
      │
      ▼
Tablero cenital cuadrado, por defecto 800 × 800 px
      │
      ▼
64 recortes de 100 × 100 px: a8, b8, ..., h1
      │
      └──► Futuro: clasificador de pieza → FEN → motor de ajedrez
```

## 3. Decisión de diseño: marcadores ArUco

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

### Por qué se eligieron

- **No necesita un conjunto de entrenamiento para localizar el tablero.** Se
  puede demostrar inmediatamente, incluso con pocas fotos disponibles.
- **Cada esquina tiene identidad.** Evita adivinar si una esquina es `a8` o
  `h1`, un error que invierte el tablero y produce una posición inválida.
- **Es robusto en tiempo real.** La detección es rápida, incluso si hay piezas
  cubriendo casillas interiores.
- **Simplifica la segmentación.** Al conocer los cuatro vértices, las 64
  casillas ya no se detectan individualmente: se calculan.

### Qué se sacrifica

Los marcadores deben imprimirse y mantenerse visibles junto al tablero. Para
una versión final sin marcadores se puede reemplazar esta etapa por un modelo
de *keypoints* que prediga las cuatro esquinas del tablero, pero esa alternativa
requiere imágenes etiquetadas y pruebas de generalización.

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
| `marker_centers_by_corner(...)` | Calcula el centro de cada marcador y los ordena por ID. | Conserva la orientación semántica del tablero. |
| `classify_square(square_image)` | Devuelve provisionalmente `("empty", 1.0)`. | Es el contrato donde se enchufará el modelo entrenado. |

La función temporal `classify_square` no afirma que no haya piezas: solo evita
fingir resultados de IA antes de entrenar un modelo. Se reemplaza en la siguiente
fase.

### `scripts/chessboard_camera_demo.py`

Es la aplicación de cámara. Su ciclo es:

1. Abrir la webcam seleccionada con `cv2.VideoCapture`.
2. Leer un frame.
3. Detectar los ArUco con `cv2.aruco.ArucoDetector`.
4. Verificar que estén los IDs 10, 11, 12 y 13.
5. Si están presentes, calcular la homografía y rectificar el tablero.
6. Dibujar la cuadrícula y etiquetas para inspección humana.
7. Recortar las 64 casillas y llamar a `classify_square`.
8. Repetir hasta pulsar `q` o `Esc`.

Hay dos ventanas: `Camera` enseña lo que está observando la cámara y
`Rectified board` enseña la salida geométrica que consumiría el clasificador.

### `scripts/generate_aruco_markers.py`

Genera los cuatro PNG imprimibles mediante OpenCV. Por defecto mide 600 × 600
píxeles cada uno y los guarda en `data/aruco_markers/`. Puede cambiarse con
`--pixels` u `--output`.

### Dependencias

| Dependencia | Uso |
| --- | --- |
| `opencv-contrib-python` | Cámara, detección ArUco, homografía y ventanas. Se usa la variante *contrib* porque incluye `cv2.aruco`. |
| `numpy` | Arreglos de coordenadas y de imágenes. |
| `pytest` | Pruebas automáticas del mapeo geométrico. |
| `uv` | Gestiona el entorno reproducible y el archivo `uv.lock`. |

## 7. Cómo ejecutar y demostrarlo

Desde la raíz del repositorio:

```bash
uv sync
uv run python scripts/generate_aruco_markers.py
uv run python scripts/chessboard_camera_demo.py
```

Si hay más de una cámara:

```bash
uv run python scripts/chessboard_camera_demo.py --camera 1
```

Antes de demostrar:

1. Imprimir los cuatro PNG sin escalar de forma desigual.
2. Pegarlos alrededor, no encima, de las cuatro esquinas del tablero.
3. Verificar los IDs y el orden de la tabla anterior.
4. Asegurar iluminación suficiente y que los cuatro marcadores aparezcan en la
   imagen.
5. Mostrar primero la ventana `Camera`; cuando detecte todo, explicar que la
   segunda ventana es la imagen normalizada que se usa para IA.

## 8. Entrenamiento futuro para reconocer piezas

### Formulación recomendada

No es necesario que el modelo encuentre un rey dentro de una escena completa.
La geometría ya proporciona una imagen por casilla. Por eso se recomienda
entrenar un **clasificador de 13 clases** sobre esos recortes:

```text
empty
white_pawn, white_knight, white_bishop, white_rook, white_queen, white_king
black_pawn, black_knight, black_bishop, black_rook, black_queen, black_king
```

Esto se conoce como clasificación de imagen, no segmentación semántica. Cada
ejemplo de entrenamiento es una casilla de tamaño fijo con una etiqueta. Es más
sencillo que usar detección de objetos porque la posición ya se conoce.

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

Se puede usar un CNN pequeño propio, EfficientNet/MobileNet o el modo de
clasificación de Ultralytics. El modelo debe devolver la clase con mayor
probabilidad y su confianza. Luego se sustituye el cuerpo de
`classify_square` por la carga del modelo y su inferencia.

En cada frame el resultado debe ser un diccionario como:

```python
{
    "a8": ("black_rook", 0.99),
    "e1": ("white_king", 0.98),
    "e4": ("empty", 0.97),
}
```

Con ese diccionario otro componente puede generar FEN y pedir una recomendación
al motor. La visión no debe decidir legalidad de jugadas ni la mejor jugada;
esa responsabilidad queda separada.

## 9. Alternativa sin marcadores y por qué no es la primera entrega

En una versión sin ArUco, se etiqueta un conjunto de imágenes con las cuatro
esquinas del tablero y se entrena un detector de *keypoints* (por ejemplo, un
modelo YOLO de pose). La salida sigue siendo exactamente la misma: cuatro
puntos ordenados para calcular una homografía.

Esto es útil si el proyecto exige detectar cualquier tablero sin accesorios,
pero aumenta el riesgo técnico:

| Aspecto | ArUco | Modelo de esquinas |
| --- | --- | --- |
| Datos etiquetados | No | Sí |
| Tiempo para un MVP | Bajo | Medio/alto |
| Tableros distintos | Requiere marcadores | Puede generalizar, si los datos bastan |
| Fallas por piezas cubriendo líneas | Baja | Depende del entrenamiento |
| Robustez inicial | Alta | Debe medirse |

Una alternativa clásica es detectar líneas del tablero con bordes/Hough o usar
`findChessboardCorners`. No se tomó como solución primaria porque las piezas,
las texturas de madera, reflejos y las líneas poco contrastadas pueden ocultar
las intersecciones. Además, `findChessboardCorners` se diseñó principalmente
para patrones de calibración sin piezas encima.

## 10. Limitaciones y mitigaciones

| Riesgo | Efecto | Mitigación |
| --- | --- | --- |
| Falta un marcador | No se rectifica el tablero. | Mostrar mensaje, recolocar cámara o marcador. |
| Marcador demasiado pequeño/borroso | No se detecta con fiabilidad. | Imprimir grande, usar buena luz y evitar desenfoque. |
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
5. los centros de ArUco conservan el orden explícito de sus IDs;
6. no se intenta rectificar si falta alguno de los cuatro marcadores.

Para ejecutarlas junto con el resto del proyecto:

```bash
uv run python -m pytest -q
```

En la implementación actual pasaron 38 pruebas.

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
