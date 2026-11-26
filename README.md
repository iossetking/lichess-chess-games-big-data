# lichess-chess-games-big-data

Pipeline de Big Data, Deep Learning y Visión por Computadora sobre datos oficiales de [Lichess](https://database.lichess.org/):

```text
cámara / pantalla / foto ──▶ rectificación (OpenCV) ──▶ piezas (YOLO-seg) ──▶ FEN ──▶ mejor jugada (UCI)
```

## Contenido

- [1. Descripción general](#1-descripción-general)
- [2. Arquitectura y estructura del repositorio](#2-arquitectura-y-estructura-del-repositorio)
- [3. Requisitos e instalación](#3-requisitos-e-instalación)
- [4. Entrenamiento y obtención de modelos (3 rutas)](#4-entrenamiento-y-obtención-de-modelos-3-rutas)
  - [Ruta A · Usar los modelos ya guardados (rápida)](#ruta-a--usar-los-modelos-ya-guardados-rápida)
  - [Ruta B · Entrenar todo en local](#ruta-b--entrenar-todo-en-local)
  - [Ruta C · Entrenar en Google Colab con `colab-cli`](#ruta-c--entrenar-en-google-colab-con-colab-cli)
  - [Variables de entorno de los cuadernos](#variables-de-entorno-de-los-cuadernos)
- [5. Ejecución local](#5-ejecución-local)
  - [5.1 Cuadernos interactivos (Marimo)](#51-cuadernos-interactivos-marimo)
  - [5.2 Visión en vivo (cámara / pantalla → tablero → FEN → jugada)](#52-visión-en-vivo-cámara--pantalla--tablero--fen--jugada)
  - [5.3 Demos heredadas (ArUco y fotos sin ArUco)](#53-demos-heredadas-aruco-y-fotos-sin-aruco)
- [6. Explicación técnica del proyecto](#6-explicación-técnica-del-proyecto)
  - [6.1 Predicción de jugadas (Enfoque 1 y Enfoque 3)](#61-predicción-de-jugadas-enfoque-1-y-enfoque-3)
  - [6.2 Guardado y versionado de checkpoints](#62-guardado-y-versionado-de-checkpoints)
  - [6.3 Pipeline de visión por computadora](#63-pipeline-de-visión-por-computadora)
  - [6.4 Límites de escala](#64-límites-de-escala)

---

## 1. Descripción general

El proyecto une dos módulos independientes que se integran en tiempo real:

1. **Predicción de jugadas**: recibe una posición en notación **FEN** y devuelve las mejores jugadas legales en notación **UCI** con su probabilidad. Se entrenan y comparan cinco modelos a partir de dos fuentes oficiales de `database.lichess.org`:
   - **Partidas estándar** (`lichess_db_standard_rated_AAAA-MM.pgn.zst`): jugadas humanas de Elo alto.
   - **Evaluaciones de Stockfish** (`lichess_db_eval.jsonl.zst`): mejores jugadas calculadas por motor.
2. **Visión por computadora**: localiza automáticamente el tablero con la webcam, corrige la perspectiva por homografía, detecta o segmenta las 12 clases de piezas con **YOLO**, construye el **FEN** y proyecta sobre el tablero real las flechas sugeridas por los modelos de jugadas.

Todo el streaming de Lichess se hace directamente desde `database.lichess.org` sin descargar meses completos: cada flujo HTTP se cierra en cuanto se reúne la muestra solicitada.

---

## 2. Arquitectura y estructura del repositorio

```text
database.lichess.org (.pgn.zst / .jsonl.zst)
        │
        ▼
[01_eda_and_cleaning.py]  (CPU)
        │  data/processed/*.parquet (local, fuera de Git)
        ├────────────────────────────────────┐
        ▼                                    ▼
[02_model_scratch_and_finetune.py]    [03_maia.py]         [scripts/train_*yolo*.py]
 (GPU: CNN desde cero + FT + LoRA)     (GPU: Maia-3 LoRA)   (GPU: detector/seg YOLO)
        │                                    │                        │
        └────────────────────┬───────────────┘                        │
                             ▼                                        ▼
                       data/models/*.pt (Git LFS, media precisión cuando aplica)
                             │                                        │
                             ├────────────────────────────────────────┤
                             ▼                                        ▼
               [04_model_playground.py]                    [scripts/live_chess_fen.py]
                (Local · inspección FEN)                    (Local · webcam en vivo)
```

### Cuadernos (`notebooks/`)

| Cuaderno | Función | Hardware ideal | Salidas |
|---|---|---|---|
| `notebooks/01_eda_and_cleaning.py` | Streaming, exploración y limpieza de partidas `.pgn.zst` y evaluaciones `lichess_db_eval.jsonl.zst` | **CPU** | `data/processed/games_clean_sample.parquet`, `positions_games_sample.parquet`, `evals_clean_sample.parquet` |
| `notebooks/02_model_scratch_and_finetune.py` | **Enfoque 1**: red convolucional residual desde cero (humanos) + ajuste de todos los pesos y ajuste LoRA (Stockfish) con validación cruzada | **GPU** | `data/models/approach1_scratch.pt`, `approach1_weight_ftuning.pt`, `approach1_lora_ftuning.pt` |
| `notebooks/03_maia.py` | **Enfoque 3**: Maia-3 preentrenado (5M o 23M) + dos adaptadores LoRA (uno humano, otro Stockfish) | **GPU** | `data/models/maia3_lora_human.pt`, `maia3_lora_stockfish.pt` |
| `notebooks/04_model_playground.py` | **Prueba visual**: detecta todos los `.pt`/`.pth` en `data/models/` (incluidas subcarpetas) y compara sus jugadas en un FEN o partida guardada; no entrena | **Local (CPU)** | — |

### Scripts principales (`scripts/`)

| Script | Función | Hardware ideal | Salida principal |
|---|---|---|---|
| `scripts/train_chess_piece_yolo.py` | Exporta `dopaul/chess-pieces-merged` (CC BY 4.0) y entrena `yolo11n` para las 12 clases de piezas físicas | **GPU** (`mps` / CUDA `0`) | `data/models/chess-pieces.pt` |
| `scripts/generate_screen_template.py` | Genera la plantilla 2D de posición inicial (`800×800`, colores verde/crema) para entrenar el modelo de pantalla | **Local (CPU)** | `data/vision/screen-template.png` |
| `scripts/train_screen_chess_yoloseg.py` | Genera 2,000 tableros sintéticos multi-estilo (Chess.com Neo + Lichess cburnett, 8 temas, resaltados y artefactos de cámara/pantalla) y entrena `yolo11s-seg` | **GPU** (`mps` / CUDA `0`) | `data/models/screen-chess-pieces-seg.pt` |
| `scripts/live_chess_fen.py` | Vista en vivo (webcam, captura directa de pantalla o foto): tablero → rectificación → YOLO-seg → FEN → flechas y consenso de los 5 modelos | **Local** | Ventana(s) OpenCV en vivo |
| `scripts/compact_checkpoints.py` | Convierte checkpoints antiguos a media precisión (`float16`); `--dry-run` solo informa el ahorro | **Local (CPU)** | Checkpoints compactados |
| `scripts/chessboard_camera_demo.py` / `chessboard_photo_demo.py` / `generate_aruco_markers.py` | Demos heredadas de calibración con ArUco o esquinas manuales | **Local** | Tablero rectificado |

---

## 3. Requisitos e instalación

```bash
uv sync
git lfs pull   # descarga los checkpoints ya versionados en data/models/
```

- **Entorno Python**: gestionado con [`uv`](https://docs.astral.sh/uv/) (`pyproject.toml` incluye `torch`, `polars`, `marimo`, `chess`, `ultralytics`, `maia3`, `datasets`, `opencv-contrib-python`).
- **Pesos (`data/models/`)**: versionados con **Git LFS** (ver `.gitattributes`). Tras clonar, `git lfs pull` descarga los checkpoints de jugadas y visión disponibles.
- **Datos procesados (`data/processed/`)**: **no** se versionan en Git; los genera el cuaderno `01` y solo hacen falta para reentrenar o explorar datos.
- **Google Colab CLI (opcional, solo para la Ruta C)**:
  ```bash
  uv tool install google-colab-cli
  ```

---

## 4. Entrenamiento y obtención de modelos (3 rutas)

Elige la ruta según lo que necesites hacer:

| Ruta | Cuándo usarla | Requiere entrenar | Requiere Colab |
|---|---|---|---|
| **Ruta A · Modelos guardados** | Quieres probar el playground (`04`) y la visión en vivo de inmediato | No | No |
| **Ruta B · Entrenar en local** | Quieres reconstruir datos y modelos en tu propia máquina (ej. Mac con Apple Silicon `mps` o PC con GPU) | Sí | No |
| **Ruta C · Entrenar en Colab** | Quieres acelerar `01`–`03` y YOLO en Google Colab y traer los `.parquet` y `.pt` a local | Sí | Sí |

### Ruta A · Usar los modelos ya guardados (rápida)

Descarga los checkpoints versionados en Git LFS y comprueba que estén presentes:

```bash
uv sync
git lfs pull
ls data/models/approach1_scratch.pt data/models/approach1_weight_ftuning.pt \
   data/models/approach1_lora_ftuning.pt data/models/maia3_lora_human.pt \
   data/models/maia3_lora_stockfish.pt
```

Si los pesos de visión (`data/models/chess-pieces.pt`, `data/models/chess-pieces-seg.pt`, `data/models/screen-chess-pieces-seg.pt`) también están publicados en LFS, quedan listos tras `git lfs pull`. Con esto puedes ir directamente a la sección [5. Ejecución local](#5-ejecución-local) sin ejecutar los cuadernos `01`–`03`.

---

### Ruta B · Entrenar todo en local

Los cuadernos se ejecutan sin interfaz y leen sus parámetros desde variables de entorno. Ajusta los tamaños a la memoria de tu equipo:

1. **Datos limpios (`01_eda_and_cleaning.py` · CPU)**:
   ```bash
   LICHESS_YEAR_FROM=2024 LICHESS_YEAR_TO=2025 LICHESS_MAX_GAMES=20000 \
   LICHESS_MAX_POSITIONS=500000 LICHESS_MAX_EVALS=500000 \
   uv run python notebooks/01_eda_and_cleaning.py
   ```
   Genera `games_clean_sample.parquet`, `positions_games_sample.parquet` y `evals_clean_sample.parquet` en `data/processed/`.

2. **Tres modelos de jugadas desde cero + ajustes (`02_model_scratch_and_finetune.py`)**:
   ```bash
   LICHESS_MAX_HUMAN=500000 LICHESS_MAX_EVAL=500000 \
   LICHESS_EPOCHS_STAGE1=8 LICHESS_EPOCHS_STAGE2=5 LICHESS_EPOCHS_LORA=5 \
   LICHESS_LORA_RANK=8 LICHESS_N_FOLDS=5 \
   uv run python notebooks/02_model_scratch_and_finetune.py
   ```

3. **Dos adaptadores Maia-3 + LoRA (`03_maia.py` · GPU recomendada)**:
   ```bash
   LICHESS_MAIA_MODEL=maia3-5m LICHESS_MAIA_MAX_TRAIN=500000 \
   LICHESS_MAIA_EPOCHS=5 LICHESS_MAIA_RANK=8 LICHESS_MAIA_LR=2e-4 \
   uv run python notebooks/03_maia.py
   ```

4. **Detector de piezas YOLO para visión**:
   ```bash
   # Tablero físico (12 clases coloreadas, descarga dopaul/chess-pieces-merged)
   uv run python scripts/train_chess_piece_yolo.py \
     --epochs 50 --device mps --imgsz 640 --batch 16

   # Opcional · tablero 2D en pantalla (multi-estilo Chess.com Neo + Lichess cburnett -> screen-chess-pieces-seg.pt)
   uv run python scripts/train_screen_chess_yoloseg.py \
     --epochs 15 --device mps --imgsz 640 --batch 16 --base-model yolo11s-seg.pt --force
   ```
   *(Cambia `--device mps` por `--device 0` en GPU NVIDIA o `--device cpu` si no tienes GPU. El script combina automáticamente `data/vision/screen-template.png` —p. ej. Chess.com Neo— con los sprites `cburnett` de `chess.svg.PIECES` sobre 8 temas de tablero. Si no tienes `screen-template.png`, genéralo antes con `scripts/generate_screen_template.py`).*

> Los cuadernos `02` y `03` reutilizan `data/processed/*.parquet` y, si faltan, generan una muestra pequeña por su cuenta. Al terminar, pasa a [5. Ejecución local](#5-ejecución-local).

---

### Ruta C · Entrenar en Google Colab con `colab-cli`

En Colab **no se puede cambiar el tipo de runtime (CPU ↔ T4 ↔ L4 ↔ A100) dentro de una sesión ya creada** sin recrear la máquina (lo que borra `/content`). Por eso esta ruta separa el trabajo en **dos sesiones cortas** y guarda los resultados en tu equipo local:

1. **Sesión 1 (CPU)**: ejecuta `01_eda_and_cleaning.py`, descarga los `.parquet` a tu carpeta local `data/processed/` y cierra la sesión (sin gastar GPU).
2. **Sesión 2 (GPU — T4 / L4 / A100)**: sube tus `.parquet` locales, entrena `02_model_scratch_and_finetune.py`, `03_maia.py` y (si lo necesitas) los pesos YOLO de visión, descarga todos los `.pt` a tu carpeta local `data/models/` y cierra la sesión.
3. **Paso 3 (Local, sin Colab)**: ejecuta `04_model_playground.py` y las demos de visión en vivo directamente en tu equipo.

> **Elección de GPU y límite de RAM en Colab:**
> - **T4** (16 GB VRAM, ~12.7 GB RAM del sistema): ideal para corridas pequeñas/medianas (hasta ~500k posiciones en el cuaderno `02` gracias a tensores `uint8` + máscaras empaquetadas en bits y particiones CV sin duplicar tensores; `maia3-5m` y YOLO).
> - **L4** (24 GB VRAM, ~53 GB RAM): recomendada para 500k–1M posiciones y `maia3-23m`.
> - **A100** (40/80 GB VRAM, ~83 GB RAM): recomendada si subes a 2M posiciones con validación cruzada de 5 particiones en el cuaderno `02` y `maia3-23m` en el `03`.

> **Importante antes de lanzar:** `colab exec -f <cuaderno>` envía tu archivo local del cuaderno, pero `src/` y `scripts/` salen del `git clone` dentro de la sesión. Haz `git push` antes de iniciar para que el código del repositorio remoto coincida con tu copia local.

#### Paso 1 · Sesión CPU: descargar y limpiar datos (`01_eda_and_cleaning.py`)

```bash
# 1. Crea una sesión solo con CPU (sin --gpu)
colab new -s lichess-data

# 2. Clona el repositorio e instala únicamente las dependencias de datos
colab exec -s lichess-data --timeout 0 <<'PY'
!GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 -b develop https://github.com/iossetking/lichess-chess-games-big-data.git /content/lichess-chess-games-big-data
!pip install -q marimo polars matplotlib zstandard chess
PY

# 3. Ejecuta el cuaderno 01 (ajusta MAX_POSITIONS / MAX_EVALS según la GPU del paso 2)
colab exec -s lichess-data -f notebooks/01_eda_and_cleaning.py --timeout 0 \
  --env LICHESS_YEAR_FROM=2024 --env LICHESS_YEAR_TO=2025 \
  --env LICHESS_MONTH_FROM=1 --env LICHESS_MONTH_TO=12 \
  --env LICHESS_MAX_GAMES=20000 --env LICHESS_MAX_POSITIONS=250000 \
  --env LICHESS_MAX_EVALS=250000 \
  --env LICHESS_MIN_ELO=1900 --env LICHESS_MIN_DEPTH=20

# 4. Guarda los Parquet limpios en tu equipo local (data/processed/)
mkdir -p data/processed
for name in games_clean_sample positions_games_sample evals_clean_sample; do
  colab download -s lichess-data \
    "/content/lichess-chess-games-big-data/data/processed/${name}.parquet" \
    "data/processed/${name}.parquet"
done

# 5. Cierra la sesión CPU
colab stop -s lichess-data
```

#### Paso 2 · Sesión GPU (T4 / L4 / A100): entrenar modelos de jugadas (`02` y `03`) y visión (`YOLO`)

```bash
# 1. Crea una sesión con GPU (usa T4 para <=250k posiciones, o L4 / A100 para 500k–2M)
colab new -s lichess-train --gpu T4

# 2. Clona el repositorio e instala dependencias de entrenamiento (jugadas + visión)
colab exec -s lichess-train --timeout 0 <<'PY'
!GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 -b develop https://github.com/iossetking/lichess-chess-games-big-data.git /content/lichess-chess-games-big-data
!mkdir -p /content/lichess-chess-games-big-data/data/processed /content/lichess-chess-games-big-data/data/models
!pip install -q marimo polars matplotlib zstandard chess datasets ultralytics "git+https://github.com/CSSLab/maia3.git"
PY

# 3. Sube los Parquet procesados localmente a la sesión GPU
for name in games_clean_sample positions_games_sample evals_clean_sample; do
  colab upload -s lichess-train \
    "data/processed/${name}.parquet" \
    "/content/lichess-chess-games-big-data/data/processed/${name}.parquet"
done

# 4. Entrena los tres modelos de jugadas (cuaderno 02)
colab exec -s lichess-train -f notebooks/02_model_scratch_and_finetune.py --timeout 0 \
  --env LICHESS_MAX_HUMAN=250000 --env LICHESS_MAX_EVAL=250000 \
  --env LICHESS_EPOCHS_STAGE1=8 --env LICHESS_EPOCHS_STAGE2=5 \
  --env LICHESS_EPOCHS_LORA=5 --env LICHESS_N_FOLDS=5

# 5. Entrena los dos adaptadores Maia-3 + LoRA (cuaderno 03)
#    En T4 usa maia3-5m; en L4/A100 puedes cambiar a maia3-23m y subir MAX_TRAIN
colab exec -s lichess-train -f notebooks/03_maia.py --timeout 0 \
  --env LICHESS_MAIA_MODEL=maia3-5m --env LICHESS_MAIA_MAX_TRAIN=250000 \
  --env LICHESS_MAIA_EPOCHS=5

# 6. Descarga los cinco modelos de jugadas a data/models/ local
mkdir -p data/models
for model in approach1_scratch approach1_weight_ftuning \
             approach1_lora_ftuning maia3_lora_human maia3_lora_stockfish; do
  colab download -s lichess-train \
    "/content/lichess-chess-games-big-data/data/models/${model}.pt" \
    "data/models/${model}.pt"
done
```

##### Entrenar también los modelos de Visión (YOLO) en la misma sesión GPU

Antes de cerrar `lichess-train` (o en una sesión `--gpu T4` aparte si solo quieres entrenar visión), puedes entrenar el detector de piezas en CUDA con `--device 0` y `--base-model yolo11n.pt` (así Ultralytics descarga el checkpoint base sin depender de Git LFS):

```bash
# 7a. Detector YOLO de piezas físicas (dopaul/chess-pieces-merged -> chess-pieces.pt)
colab exec -s lichess-train --timeout 0 <<'PY'
!python /content/lichess-chess-games-big-data/scripts/train_chess_piece_yolo.py \
  --epochs 50 --device 0 --imgsz 640 --batch 16 --base-model yolo11n.pt
PY

colab download -s lichess-train \
  /content/lichess-chess-games-big-data/data/models/chess-pieces.pt \
  data/models/chess-pieces.pt
```

Si además necesitas reentrenar el modelo YOLO-seg para tablero 2D en pantalla (`screen-chess-pieces-seg.pt`, ya incluido en Git LFS con mAP50 = 1.000 sobre Chess.com Neo y Lichess cburnett), sube tu `data/vision/screen-template.png` (o genéralo antes) y entrena con `yolo11s-seg.pt`:

```bash
# 7b. Opcional · YOLO-seg multi-estilo para pantalla (-> screen-chess-pieces-seg.pt)
colab exec -s lichess-train --timeout 0 <<'PY'
!mkdir -p /content/lichess-chess-games-big-data/data/vision
PY
colab upload -s lichess-train \
  data/vision/screen-template.png \
  /content/lichess-chess-games-big-data/data/vision/screen-template.png

colab exec -s lichess-train --timeout 0 <<'PY'
!python /content/lichess-chess-games-big-data/scripts/train_screen_chess_yoloseg.py \
  --epochs 15 --device 0 --imgsz 640 --batch 16 --base-model yolo11s-seg.pt --force
PY

colab download -s lichess-train \
  /content/lichess-chess-games-big-data/data/models/screen-chess-pieces-seg.pt \
  data/models/screen-chess-pieces-seg.pt
```

Finalmente, **cierra la sesión GPU** para dejar de consumir unidades de cómputo:

```bash
colab stop -s lichess-train
```

*(Comandos útiles de control: `colab sessions`, `colab status -s lichess-train`, `colab stop --all`.)*

#### Paso 3 · Ejecutar el cuaderno 04 y la visión en vivo en local (sin Colab)

Con los modelos ya guardados en `data/models/` y los Parquet en `data/processed/`, **no necesitas Colab para inferencia ni demos**. Continúa en la sección [5. Ejecución local](#5-ejecución-local).

---

### Variables de entorno de los cuadernos

Al ejecutarse como scripts (`python notebooks/...`), los cuadernos leen tamaños, épocas y rutas desde estas variables (todas son opcionales):

| Cuaderno | Variables |
|---|---|
| `01_eda_and_cleaning.py` | `LICHESS_YEAR_FROM`, `LICHESS_YEAR_TO`, `LICHESS_MONTH_FROM`, `LICHESS_MONTH_TO`, `LICHESS_MAX_GAMES`, `LICHESS_MIN_ELO`, `LICHESS_MAX_POSITIONS`, `LICHESS_MAX_EVALS`, `LICHESS_MIN_DEPTH` |
| `02_model_scratch_and_finetune.py` | `LICHESS_MAX_HUMAN`, `LICHESS_MAX_EVAL`, `LICHESS_EPOCHS_STAGE1`, `LICHESS_EPOCHS_STAGE2`, `LICHESS_EPOCHS_LORA`, `LICHESS_LORA_RANK`, `LICHESS_N_FOLDS`, `LICHESS_CV_WEIGHT`, `LICHESS_CV_LORA` |
| `03_maia.py` | `LICHESS_MAIA_MODEL`, `LICHESS_MAIA_MAX_TRAIN`, `LICHESS_MAIA_EPOCHS`, `LICHESS_MAIA_RANK`, `LICHESS_MAIA_LR`, `LICHESS_MAIA_ELO` |
| Todos | `LICHESS_PROJECT_DIR`, `LICHESS_DATA_DIR`, `LICHESS_PROCESSED_DIR`, `LICHESS_MODELS_DIR` |

`LICHESS_PROJECT_DIR` solo hace falta si ejecutas un cuaderno fuera del repositorio (p. ej. otro directorio en Colab). `LICHESS_DATA_DIR` cambia la raíz de datos; `LICHESS_PROCESSED_DIR` y `LICHESS_MODELS_DIR` cambian solo una de las dos carpetas.

Ejemplo de prueba local mínima (unos segundos):

```bash
LICHESS_MAX_HUMAN=2000 LICHESS_MAX_EVAL=2000 LICHESS_EPOCHS_STAGE1=1 \
LICHESS_EPOCHS_STAGE2=1 LICHESS_EPOCHS_LORA=1 LICHESS_N_FOLDS=3 \
LICHESS_CV_WEIGHT=0 LICHESS_CV_LORA=0 \
uv run python notebooks/02_model_scratch_and_finetune.py
```

---

## 5. Ejecución local

Todo lo de esta sección corre localmente (sin Colab) una vez tengas los pesos en `data/models/`.

### 5.1 Cuadernos interactivos (Marimo)

Para comparar todos los checkpoints guardados en `data/models/` (incluidas subcarpetas) sobre cualquier FEN o partida guardada del cuaderno `01`:

```bash
uv run marimo edit notebooks/04_model_playground.py
```

*(Nota: las posiciones de partidas guardadas en `data/processed/positions_games_sample.parquet` pueden haber participado en el entrenamiento; úsalas para inspección visual, no como métrica independiente).*

También puedes abrir los cuadernos de datos y entrenamiento con interfaz interactiva:

```bash
uv run marimo edit notebooks/01_eda_and_cleaning.py
uv run marimo edit notebooks/02_model_scratch_and_finetune.py
uv run marimo edit notebooks/03_maia.py
```

---

### 5.2 Visión en vivo (cámara / pantalla → tablero → FEN → jugada)

La ruta de producción es [`scripts/live_chess_fen.py`](scripts/live_chess_fen.py). Detecta automáticamente el tablero (en verde Chess.com, café Lichess, azul, madera, etc., o con calibración manual de 4 esquinas pulsando `c`), rectifica la perspectiva a `800×800`, segmenta las 12 clases de piezas con **YOLO-seg**, valida el **FEN** y proyecta las flechas de los 5 modelos de jugadas presentes en `data/models/`.

#### Tablero digital en pantalla (comando rápido · por defecto)

Carga automáticamente `data/models/screen-chess-pieces-seg.pt` (`yolo11s-seg` entrenado en Chess.com Neo + Lichess cburnett y 8 temas de tablero) y los 5 modelos de jugadas:

```bash
# 1. Captura directa de tu propia pantalla (sin webcam; ideal para Chess.com / Lichess en el navegador)
uv run python scripts/live_chess_fen.py --source screen

# 2. Webcam apuntando a un celular o monitor
uv run python scripts/live_chess_fen.py

# 3. Analizar una sola captura de pantalla o foto (muestra tablero + máscaras + jugadas de los 5 modelos)
uv run python scripts/live_chess_fen.py --image ruta/a/captura.png
```

**Cómo usar `--source screen` en 3 pasos (sin webcam):**
1. Abre tu partida de **Chess.com** o **Lichess** en un lado de la pantalla (p. ej. mitad izquierda) y ejecuta `uv run python scripts/live_chess_fen.py --source screen`.
2. Se abrirá una captura ajustada de tu pantalla: si el tablero aparece ya rodeado en **verde**, pulsa **Enter**; si no, **arrastra un rectángulo** con el mouse desde la esquina superior izquierda hasta la inferior derecha del tablero y pulsa **Enter** *(pulsa `r` para refrescar la captura si necesitas cambiar de ventana)*.
3. El selector se cierra y queda **una sola ventana compacta** (`Chess Assistant`) que puedes acomodar al lado de tu navegador sin tapar el tablero. Si mueves el navegador, pulsa `c` para volver a seleccionar la zona del tablero. *(En macOS, otorga permiso de «Grabación de pantalla» a tu terminal en Ajustes del Sistema → Privacidad y seguridad si la captura sale vacía.)*

#### Tablero físico 3D con webcam (`chess-pieces.pt`)

```bash
uv run python scripts/live_chess_fen.py --mode physical
```

*(Si prefieres pesos de segmentación físicos: `--mode physical --weights data/models/chess-pieces-seg.pt --require-segmentation`.)*

#### Modelos dibujados sobre el tablero y controles en vivo

Cada modelo encontrado en `data/models/` proyecta su mejor jugada con flecha estilo SeeMate (trayectoria en L para caballos y etiqueta con notación algebraica y probabilidad), además de un panel lateral con el consenso entre los 5 modelos y las 3 mejores opciones de cada uno:

| Modelo | Archivo en `data/models/` | Color de flecha | Tecla |
|---|---|---|---|
| Desde cero | `approach1_scratch.pt` | cian | `1` |
| Ajuste de pesos | `approach1_weight_ftuning.pt` | naranja | `2` |
| Ajuste LoRA | `approach1_lora_ftuning.pt` | verde | `3` |
| Maia-3 humano | `maia3_lora_human.pt` | azul | `4` |
| Maia-3 Stockfish | `maia3_lora_stockfish.pt` | magenta | `5` |

- **Teclas en la ventana**:
  - `c`: calibrar las 4 esquinas del tablero con clic (`a8 → h8 → h1 → a1`) si la detección automática no engancha.
  - `r`: reiniciar y volver a detectar el tablero automáticamente.
  - `t`: alternar el turno (blancas / negras).
  - `f`: girar el tablero 180° si juegas con negras.
  - `k`: ciclar derechos de enroque (`KQkq → KQ → kq → -`; los enroques imposibles se depuran automáticamente según la posición real de reyes y torres).
  - `m`: mostrar u ocultar las máscaras de segmentación coloreadas por bando.
  - `0` / `1`–`5`: mostrar los 5 modelos juntos (`0`) o enfocar uno solo (`1`–`5`).
  - `s`: guardar captura anotada (`data/snapshots/`) con tablero, máscaras, FEN y jugadas.
  - `q` o `Esc`: salir.
- **Flags útiles**:
  - `--source screen` y opcionalmente `--screen-region x,y,w,h` para capturar directo tu monitor.
  - `--image <ruta>` y `--save-output <ruta>` para evaluar una captura puntual.
  - `--move-models` (uno o varios archivos) o `--no-move-models` para elegir qué modelos comparar.
  - `--board-weights`: acepta un YOLO-seg con clase `board`/`chessboard` como respaldo.
  - `--imgsz 416`: acelera la inferencia YOLO si tu equipo va justo de FPS.

> **Notas sobre pesos de visión:**
> - Los pesos deben entrenarse con las 12 clases coloreadas (`w-pawn`/`b-king`, `white_pawn`/`black_king` o equivalentes). Un checkpoint COCO genérico (`yolo*.pt`) **no sirve**.
> - No entrenes YOLO-seg con un dataset solo de cajas: la segmentación requiere polígonos/máscaras por pieza. Para mejorar precisión con tus piezas reales (manos, oclusiones, iluminación), añade fotos anotadas de tu propia cámara o parte de datasets como [Chess Piece Detect en Roboflow Universe](https://universe.roboflow.com/chess-ynmuj/chess-piece-detect).
> - Si te falta `chess-pieces.pt` o `screen-chess-pieces-seg.pt`, entrénalos con la **Ruta B** (local) o la **Ruta C** (Colab) en la sección [4](#4-entrenamiento-y-obtención-de-modelos-3-rutas).

---

### 5.3 Demos heredadas (ArUco y fotos sin ArUco)

Estas utilidades solo localizan y rectifican el tablero (no predicen jugadas; el punto de integración para un clasificador por casilla es `classify_square` en `src/lichess_pipeline/chess_vision.py`). Consulta la documentación detallada en [docs/vision_tecnica_tablero.md](docs/vision_tecnica_tablero.md) o [docs/Documento_Tecnico_Vision_Tablero_Ajedrez.docx](docs/Documento_Tecnico_Vision_Tablero_Ajedrez.docx).

- **Calibración con 4 marcadores ArUco** (`DICT_4X4_50`, IDs 10, 11, 12 y 13 colocados fuera del tablero en orden arriba-izquierda, arriba-derecha, abajo-derecha, abajo-izquierda vistos desde blancas):
  ```bash
  uv run python scripts/generate_aruco_markers.py            # genera PNG en data/aruco_markers/
  uv run python scripts/chessboard_camera_demo.py            # usa --camera 1 --size 800 si aplica
  ```
- **Foto tomada con celular sin ArUco** (clic en las 4 esquinas internas `a8`, `h8`, `h1`, `a1`, Enter para guardar el PNG rectificado, `r` para repetir):
  ```bash
  uv run python scripts/chessboard_photo_demo.py --image /ruta/a/tablero.jpg
  ```
- **Webcam viendo una foto del tablero en un iPhone sin ArUco** (pulsa `c`, haz clic en `a8`, `h8`, `h1`, `a1` y Enter):
  ```bash
  uv run python scripts/chessboard_camera_demo.py --manual-corners
  ```

---

## 6. Explicación técnica del proyecto

### 6.1 Predicción de jugadas (Enfoque 1 y Enfoque 3)

#### Enfoque 1 · Red convolucional residual desde cero + dos ajustes finos (`02_model_scratch_and_finetune.py`)

- **Entrada y codificación (`18 × 8 × 8`)**: recibe un FEN y lo codifica siempre desde la perspectiva del bando al que le toca mover (si mueven negras, se reflejan tablero y jugada). Los 18 planos son: 6 piezas propias, 6 del rival, 4 derechos de enroque, 1 casilla de captura al paso y 1 con el turno original.
- **Arquitectura (`ChessPolicyNet`)**: torre convolucional residual (4 bloques `3×3`, 128 canales, campo receptivo de todo el tablero) y cabeza espacial factorizada: cada casilla se proyecta a un vector de origen y otro de destino, y cada jugada puntúa como su producto escalar más un sesgo aprendido `64 × 64`. Esto reduce el modelo a **~1.2 M parámetros** (~2.4 MB en `float16`) frente a 34 M de una cabeza plana y comparte aprendizaje entre casillas.
- **Salida y legalidad**: `64 × 64 = 4096` clases `origen → destino`. Las jugadas ilegales se enmascaran con `python-chess` antes del `softmax`, garantizando salidas 100 % legales por construcción.
- **Etapas de entrenamiento**:
  1. **Desde cero (`approach1_scratch.pt`)**: jugadas humanas de Elo alto con Adam y decaimiento coseno.
  2. **Ajuste de todos los pesos (`approach1_weight_ftuning.pt`)**: reentrena la red completa con mejores jugadas de Stockfish (`lr=1e-4`).
  3. **Ajuste LoRA (`approach1_lora_ftuning.pt`)**: congela la red base y entrena correcciones de bajo rango (`B·A`) en las convoluciones residuales, las proyecciones de la cabeza y el sesgo espacial (~86k parámetros, ≈7 % de la red). Al guardar, el adaptador se fusiona en un `ChessPolicyNet` normal.
- **Evaluación**: se reserva un **20 % de prueba** intacto hasta el final y el 80 % restante se evalúa con **validación cruzada de 5 particiones**, agrupando partidas completas por `game_id` en la etapa 1 para evitar fuga de posiciones casi idénticas. El dispositivo se selecciona automáticamente (`CUDA` → `MPS` → `CPU`).

#### Enfoque 3 · Maia-3 preentrenado + LoRA (`03_maia.py`)

- Parte de [Maia-3](https://github.com/CSSLab/maia3) (5M o 23M parámetros, arquitectura Chessformer del CSSLab de la Universidad de Toronto, licencia AGPL-3.0), descargado desde Hugging Face a `data/models/maia3/`.
- Congela los pesos base y añade adaptadores LoRA (~100k parámetros con rango 8, <1 MB en disco) sobre el MLP de cada bloque del transformer y las dos proyecciones de la cabeza de jugadas (`src/lichess_pipeline/model/maia3_lora.py`).
- Entrena dos adaptadores independientes a partir del mismo Maia-3:
  - **A · humano (`maia3_lora_human.pt`)**: `positions_games_sample.parquet` (`fen` → `move_uci`, usando el Elo medio de cada partida).
  - **B · Stockfish (`maia3_lora_stockfish.pt`)**: `evals_clean_sample.parquet` (`fen` → `best_move`, con Elo fijo configurable, 2500 por defecto).
- Usa **4352 clases** (4096 `origen → destino` más 256 promociones), distinguiendo coronar dama de caballo/torre/alfil, y entrena con precisión mixta (`torch.autocast`) en CUDA.

---

### 6.2 Guardado y versionado de checkpoints

- `save_policy_checkpoint` guarda el `state_dict` en media precisión (`float16`), reduciendo el peso a la mitad (~2.4 MB por modelo) sin degradar predicciones (al cargar vuelven a `float32`).
- `load_policy_checkpoint` detecta automáticamente la arquitectura (18 planos espaciales, checkpoints antiguos `_v1`/`_v2` de 13 planos con cabeza plana, y adaptadores Maia-3) y avisa claramente si un archivo `.pt` es un puntero de Git LFS aún sin descargar.
- La validación cruzada del cuaderno `02` mantiene los pesos intermedios de cada partición en memoria (en media precisión) en lugar de escribir cientos de MB en disco.
- Usa `scripts/compact_checkpoints.py` (`--dry-run` para simular) si quieres compactar checkpoints antiguos a `float16`.

---

### 6.3 Pipeline de visión por computadora

- **Detección multi-tema y rectificación previa**: `detect_any_screen_board` combina detección por cuadrícula de color verde (muestreando las 4 esquinas de cada casilla para resistir piezas en el centro) con detección por periodicidad de bordes 8×8 válida para cualquier tema (Lichess café, Chess.com verde/azul/madera, etc.), además de calibración manual (`c`). En modo pantalla, el tablero se rectifica a `800×800` **antes** de ejecutar YOLO-seg (`screen-chess-pieces-seg.pt`, `yolo11s-seg`, 12 clases con color, mAP50 = 1.000 en test multi-estilo).
- **Captura directa de pantalla sin retroalimentación**: `--source screen` detecta la escala Retina de macOS, abre un selector interactivo de un solo arrastre y luego captura exclusivamente la región del tablero en una sola ventana compacta (`Chess Assistant`) colocable junto al navegador.
- **Hilos desacoplados y estabilidad**: la captura (webcam o región de pantalla) y la inferencia YOLO corren en hilos separados; el suavizado por pieza (`PieceLabelSmoother`) y la estabilización de FEN (`FenStabilizer`, 2 frames por defecto) evitan parpadeos, mientras `sanitize_fen_castling` y `validate_fen_position` depuran enroques imposibles y explican en español cualquier error de posición.
- **Integración y proyección de jugadas**: cada posición válida consulta los 5 modelos cargados en `data/models/`, muestra el consenso (`MEJOR JUGADA (X/5)`) con el top-3 de cada modelo en el panel lateral y dibuja las flechas tanto en el tablero rectificado como proyectadas por homografía inversa sobre la imagen de cámara.

---

### 6.4 Límites de escala

El pipeline actual materializa los datasets de entrenamiento en memoria RAM (DataFrames de Polars, tensores de tablero `uint8` y máscaras de legalidad empaquetadas en bits en el cuaderno `02`, ~1.6 KB/posición, y booleanas de 4352 clases en `03`). Para entrenar a escala de decenas de millones de posiciones en un servidor de producción, la evolución natural consiste en escribir Parquet por bloques y sustituir `TensorDataset` por un `IterableDataset` que codifique cada FEN y su máscara bajo demanda, manteniendo intactos el parseo y el contrato de los modelos.
