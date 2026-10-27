import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import time
    from collections import Counter
    from pathlib import Path

    import chess
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from huggingface_hub import HfApi

    from lichess_pipeline.chess_utils import (
        MIDDLEGAME_MIN_PIECES,
        OPENING_MIN_PIECES,
        canonical_fen,
        first_move_is_legal,
        has_clock_expr,
        has_eval_expr,
        match_key,
        match_key_expr,
        phase_expr,
        piece_count_expr,
        puzzle_position,
    )
    from lichess_pipeline.data_ingestion import (
        EVALUATIONS_REPO_ID,
        EVALUATIONS_REVISION,
        EVALUATIONS_REVISION_DATE,
        EVALUATIONS_SCHEMA,
        EXPECTED_SCHEMA,
        PUZZLES_REPO_ID,
        PUZZLES_REVISION,
        PUZZLES_REVISION_DATE,
        PUZZLES_SCHEMA,
        REPOSITORY_ID,
        REPOSITORY_REVISION,
        list_repo_parquet_files,
        month_shard_files,
        parse_month_partitions,
        scan_hf_dataset,
    )

    PALETTE = {
        "azul": "#4C78A8",
        "verde": "#72B7B2",
        "ocre": "#E4A35A",
        "malva": "#B279A2",
        "gris": "#9D9D9D",
    }
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=list(PALETTE.values())),
            "figure.dpi": 100,
            "axes.titlesize": 11,
        }
    )
    OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "eda_evaluaciones"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def save_figure(figure, name: str) -> str:
        """Save a figure as PNG in outputs/eda_evaluaciones and return its path."""
        path = OUTPUT_DIR / f"{name}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        return str(path)

    return (
        Counter,
        EVALUATIONS_REPO_ID,
        EVALUATIONS_REVISION,
        EVALUATIONS_REVISION_DATE,
        EVALUATIONS_SCHEMA,
        EXPECTED_SCHEMA,
        HfApi,
        MIDDLEGAME_MIN_PIECES,
        OPENING_MIN_PIECES,
        PALETTE,
        PUZZLES_REPO_ID,
        PUZZLES_REVISION,
        PUZZLES_REVISION_DATE,
        PUZZLES_SCHEMA,
        REPOSITORY_ID,
        REPOSITORY_REVISION,
        canonical_fen,
        chess,
        first_move_is_legal,
        has_clock_expr,
        has_eval_expr,
        list_repo_parquet_files,
        match_key,
        match_key_expr,
        mo,
        month_shard_files,
        np,
        parse_month_partitions,
        phase_expr,
        piece_count_expr,
        pl,
        plt,
        puzzle_position,
        save_figure,
        scan_hf_dataset,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Paso 1: exploración de evaluaciones, puzzles y partidas

    Este cuaderno responde: **¿qué calidad y qué tamaño útil tienen los datos con
    los que se construirá el modelo de "mejor jugada"?** Cubre tres datasets de
    Lichess leídos de forma remota con Polars (`hf://`), con revisión fijada:

    1. **Evaluaciones** (`chess-position-evaluations`): las etiquetas de Stockfish.
       Estadísticas completas sobre 1 de los 20 archivos; el solape con puzzles usa 2.
    2. **Puzzles** (`chess-puzzles`): los 3 archivos completos.
    3. **Partidas** (`standard-chess-games`): cuántas partidas traen `[%eval]` y
       `[%clk]`, una muestra de enero de cada año, y la velocidad de lectura.

    Las lecturas remotas pesadas se guardan con `mo.persistent_cache` (carpeta
    `notebooks/__marimo__/cache`, ignorada por git), así que reabrir el cuaderno
    no vuelve a descargar. Los gráficos se guardan como PNG en
    `outputs/eda_evaluaciones/`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Fuentes y versiones fijadas
    """)
    return


@app.cell
def _(
    EVALUATIONS_REPO_ID,
    EVALUATIONS_REVISION,
    EVALUATIONS_REVISION_DATE,
    PUZZLES_REPO_ID,
    PUZZLES_REVISION,
    PUZZLES_REVISION_DATE,
    REPOSITORY_ID,
    REPOSITORY_REVISION,
    mo,
    pl,
):
    EVAL_STATS_FILE = "data/data_0000.parquet"
    EVAL_OVERLAP_FILES = ["data/data_0000.parquet", "data/data_0001.parquet"]
    EVAL_TOTAL_FILES = 20
    PUZZLE_FILES = [f"data/train-{i:05d}-of-00003.parquet" for i in range(3)]
    sources = pl.DataFrame(
        {
            "Dataset": [EVALUATIONS_REPO_ID, PUZZLES_REPO_ID, REPOSITORY_ID],
            "Revisión": [EVALUATIONS_REVISION, PUZZLES_REVISION, REPOSITORY_REVISION],
            "Fecha de la revisión": [
                EVALUATIONS_REVISION_DATE,
                PUZZLES_REVISION_DATE,
                "2025-10-16",
            ],
        }
    )
    mo.ui.table(sources, pagination=False)
    return EVAL_OVERLAP_FILES, EVAL_STATS_FILE, EVAL_TOTAL_FILES, PUZZLE_FILES


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Evaluaciones de Stockfish

    Estadísticas completas sobre `data_0000.parquet` (1 de 20 archivos). Un FEN
    tiene varias filas (varias líneas "multi-PV" y, a veces, varias profundidades).
    Supuesto: el archivo 0 es representativo de los otros 19 (no se verifica el
    orden en que se repartieron las posiciones entre archivos).
    """)
    return


@app.cell
def _(
    EVALUATIONS_REPO_ID,
    EVALUATIONS_REVISION,
    EVALUATIONS_SCHEMA,
    EVAL_STATS_FILE,
    mo,
    phase_expr,
    piece_count_expr,
    pl,
    scan_hf_dataset,
):
    # Una sola lectura remota: tabla conjunta por FEN (sin el FEN) y agregados por fila.
    with mo.persistent_cache("eval_file0_stats_v1"):
        _eval_lf = scan_hf_dataset(
            EVALUATIONS_REPO_ID,
            [EVAL_STATS_FILE],
            EVALUATIONS_REVISION,
            ["fen", "line", "depth", "knodes", "cp", "mate"],
            EVALUATIONS_SCHEMA,
        )
        eval_row_count = _eval_lf.select(pl.len()).collect().item()
        eval_joint = (
            _eval_lf.select("fen", "depth")
            .group_by("fen")
            .agg(
                n_filas=pl.len(),
                n_profundidades=pl.col("depth").n_unique(),
                max_profundidad=pl.col("depth").max(),
                n_pv_max=(pl.col("depth") == pl.col("depth").max()).sum(),
            )
            .with_columns(
                piezas=piece_count_expr("fen"),
                fase=phase_expr("fen"),
                turno=pl.col("fen").str.split(" ").list.get(1),
            )
            .group_by(
                "n_filas",
                "n_profundidades",
                "max_profundidad",
                "n_pv_max",
                "piezas",
                "fase",
                "turno",
            )
            .len(name="fens")
            .collect(engine="streaming")
        )
        _cp = pl.col("cp").cast(pl.Int32)
        eval_depth_rows = (
            _eval_lf.group_by("depth").len(name="filas").sort("depth").collect()
        )
        eval_cp_hist = (
            _eval_lf.filter(pl.col("cp").is_not_null())
            .group_by(((_cp.clip(-1000, 1000) + 1000) // 25).alias("bin"))
            .len(name="filas")
            .sort("bin")
            .collect()
        )
        eval_row_flags = _eval_lf.select(
            mate_filas=pl.col("mate").is_not_null().sum(),
            cp_nulo=pl.col("cp").is_null().sum(),
            cp_y_mate_nulos=(pl.col("cp").is_null() & pl.col("mate").is_null()).sum(),
            cp_fuera_1000=(_cp.abs() > 1000).sum(),
            cp_min=pl.col("cp").min(),
            cp_max=pl.col("cp").max(),
            knodes_nulo=pl.col("knodes").is_null().sum(),
        ).collect()
    return (
        eval_cp_hist,
        eval_depth_rows,
        eval_joint,
        eval_row_count,
        eval_row_flags,
    )


@app.cell
def _(eval_joint, eval_row_count, eval_row_flags, mo, pl):
    unique_fens = eval_joint["fens"].sum()
    multi_depth_fens = eval_joint.filter(pl.col("n_profundidades") > 1)["fens"].sum()
    mate_rows = eval_row_flags["mate_filas"][0]
    white_share = eval_joint.filter(pl.col("turno") == "w")["fens"].sum() / unique_fens
    eval_summary = pl.DataFrame(
        {
            "Indicador": [
                "Filas (archivo 0)",
                "FEN únicos",
                "Filas por FEN (media)",
                "FEN con más de una profundidad",
                "Filas con `mate` (%)",
                "Filas con cp nulo (%)",
                "cp nulo y mate nulo (filas)",
                "Filas con |cp| > 1000 (%)",
                "FEN con blancas al turno (%)",
            ],
            "Valor": [
                f"{eval_row_count:,}",
                f"{unique_fens:,}",
                f"{eval_row_count / unique_fens:.2f}",
                f"{multi_depth_fens:,} ({100 * multi_depth_fens / unique_fens:.2f} %)",
                f"{100 * mate_rows / eval_row_count:.2f}",
                f"{100 * eval_row_flags['cp_nulo'][0] / eval_row_count:.2f}",
                f"{eval_row_flags['cp_y_mate_nulos'][0]:,}",
                f"{100 * eval_row_flags['cp_fuera_1000'][0] / eval_row_count:.2f}",
                f"{100 * white_share:.2f}",
            ],
        }
    )
    mo.ui.table(eval_summary, pagination=False)
    return multi_depth_fens, unique_fens, white_share


@app.cell
def _(PALETTE, eval_depth_rows, eval_joint, mo, pl, plt, save_figure):
    _per_n = eval_joint.group_by("n_filas").agg(pl.col("fens").sum()).sort("n_filas")
    _per_pv = eval_joint.group_by("n_pv_max").agg(pl.col("fens").sum()).sort("n_pv_max")
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.2))
    _axes[0].bar(
        _per_n["n_filas"].to_list(), _per_n["fens"].to_list(), color=PALETTE["azul"]
    )
    _axes[0].set(
        title="Filas por FEN",
        xlabel="Filas por FEN",
        ylabel="FEN (escala log)",
        yscale="log",
    )
    _axes[1].bar(
        _per_pv["n_pv_max"].to_list(), _per_pv["fens"].to_list(), color=PALETTE["verde"]
    )
    _axes[1].set(
        title="Líneas alternativas a la máxima profundidad",
        xlabel="Líneas (multi-PV) con la profundidad máxima del FEN",
        ylabel="FEN (escala log)",
        yscale="log",
    )
    _axes[2].bar(
        eval_depth_rows["depth"].to_list(),
        eval_depth_rows["filas"].to_list(),
        color=PALETTE["ocre"],
    )
    _axes[2].set(
        title="Distribución de la profundidad (por fila)",
        xlabel="Profundidad de búsqueda",
        ylabel="Filas (escala log)",
        yscale="log",
    )
    _fig.tight_layout()
    _path_depth = save_figure(_fig, "01_filas_por_fen_y_profundidad")
    mo.vstack(
        [
            _fig,
            mo.md(
                "Mira la cola de la derecha: hay profundidades muy altas pero poco "
                "frecuentes; la mayoría de las filas se concentra en pocas "
                f"profundidades. PNG: `{_path_depth}`"
            ),
        ]
    )
    return


@app.cell
def _(PALETTE, eval_cp_hist, eval_row_flags, mo, plt, save_figure):
    _edges = [-1000 + 25 * b for b in eval_cp_hist["bin"].to_list()]
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.bar(
        _edges,
        eval_cp_hist["filas"].to_list(),
        width=25,
        align="edge",
        color=PALETTE["azul"],
    )
    _ax.set(
        title="Distribución de cp (centipeones, recortado a ±1000, punto de vista de las blancas)",
        xlabel="cp",
        ylabel="Filas (escala log)",
        yscale="log",
    )
    _fig.tight_layout()
    _path_cp = save_figure(_fig, "02_distribucion_cp")
    mo.vstack(
        [
            _fig,
            mo.md(
                "Mira el pico alrededor de 0 y los extremos: las filas con |cp| > 1000 se "
                "acumulan en las barras de ±1000. Rango real de cp: "
                f"{eval_row_flags['cp_min'][0]} a {eval_row_flags['cp_max'][0]}. "
                f"PNG: `{_path_cp}`"
            ),
        ]
    )
    return


@app.cell
def _(
    MIDDLEGAME_MIN_PIECES,
    OPENING_MIN_PIECES,
    PALETTE,
    eval_joint,
    mo,
    pl,
    plt,
    save_figure,
):
    _by_pieces = (
        eval_joint.group_by("piezas", "fase").agg(pl.col("fens").sum()).sort("piezas")
    )
    _colors = {
        "opening": PALETTE["azul"],
        "middlegame": PALETTE["ocre"],
        "endgame": PALETTE["verde"],
    }
    _fig, _ax = plt.subplots(figsize=(10, 4))
    for _phase, _group in _by_pieces.group_by("fase"):
        _ax.bar(
            _group["piezas"].to_list(),
            _group["fens"].to_list(),
            color=_colors[_phase[0]],
            label=_phase[0],
        )
    _ax.set(
        title="FEN únicos por número de piezas y fase",
        xlabel="Piezas en el tablero (reyes y peones incluidos)",
        ylabel="FEN únicos",
    )
    _ax.legend(title="Fase")
    _fig.tight_layout()
    _path_phase = save_figure(_fig, "03_piezas_y_fase")
    phase_table = (
        eval_joint.group_by("fase")
        .agg(pl.col("fens").sum())
        .with_columns((100 * pl.col("fens") / pl.col("fens").sum()).round(2).alias("%"))
        .sort("fens", descending=True)
    )
    mo.vstack(
        [
            mo.md(
                f"Fases por conteo de piezas: apertura >= {OPENING_MIN_PIECES}, "
                f"medio juego {MIDDLEGAME_MIN_PIECES}-{OPENING_MIN_PIECES - 1}, "
                f"final <= {MIDDLEGAME_MIN_PIECES - 1}."
            ),
            mo.ui.table(phase_table, pagination=False),
            _fig,
            mo.md(
                "Mira qué fase domina: define qué tipo de posiciones aprenderá mejor el "
                f"modelo. PNG: `{_path_phase}`"
            ),
        ]
    )
    return (phase_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Legalidad de una muestra aleatoria

    Muestra de ~10 000 filas elegidas por hash del FEN (todas las filas de un FEN
    entran o salen juntas). Se comprueba con python-chess que el FEN sea válido,
    que la primera jugada de `line` sea legal, y que el campo de captura al paso
    coincida con el que escribe python-chess (Lichess escribe la casilla tras
    cualquier salto doble de peón, python-chess solo si la captura es legal).
    """)
    return


@app.cell
def _(
    EVALUATIONS_REPO_ID,
    EVALUATIONS_REVISION,
    EVALUATIONS_SCHEMA,
    EVAL_OVERLAP_FILES,
    EVAL_STATS_FILE,
    mo,
    pl,
    scan_hf_dataset,
):
    # Lectura remota pesada (lee fen y line del archivo 0): cacheada.
    with mo.persistent_cache("eval_file0_legality_sample_v1"):
        _lf = scan_hf_dataset(
            EVALUATIONS_REPO_ID,
            [EVAL_STATS_FILE],
            EVALUATIONS_REVISION,
            ["fen", "line"],
            EVALUATIONS_SCHEMA,
        )
        eval_sample = _lf.filter(pl.col("fen").hash(7) % 5700 == 0).collect(
            engine="streaming"
        )
        # Duplicación entre archivos: ¿los FEN del archivo 0 aparecen en el archivo 1?
        _sample_fens = eval_sample["fen"].unique()
        _lf1 = scan_hf_dataset(
            EVALUATIONS_REPO_ID,
            [EVAL_OVERLAP_FILES[1]],
            EVALUATIONS_REVISION,
            ["fen"],
            EVALUATIONS_SCHEMA,
        )
        cross_file_hits = (
            _lf1.filter(pl.col("fen").is_in(_sample_fens.implode()))
            .select("fen")
            .unique()
            .collect(engine="streaming")
        )
        cross_file_sample_size = len(_sample_fens)
    return cross_file_hits, cross_file_sample_size, eval_sample


@app.cell
def _(
    Counter,
    canonical_fen,
    chess,
    cross_file_hits,
    cross_file_sample_size,
    eval_sample,
    first_move_is_legal,
    mo,
    pl,
):
    def _check(fen: str) -> tuple[bool, bool]:
        try:
            return chess.Board(fen).is_valid(), canonical_fen(fen) == fen
        except ValueError:
            return False, False

    _checks = [_check(f) for f in eval_sample["fen"].to_list()]
    _invalid_status = Counter(
        chess.Board(f).status().name
        for f in eval_sample["fen"].unique().to_list()
        if not chess.Board(f).is_valid()
    )
    invalid_breakdown = ", ".join(f"{k}: {v}" for k, v in _invalid_status.items())
    legal_flags = [
        first_move_is_legal(f, line)
        for f, line in zip(
            eval_sample["fen"].to_list(), eval_sample["line"].to_list(), strict=True
        )
    ]
    sample_n = len(eval_sample)
    legality_summary = pl.DataFrame(
        {
            "Indicador": [
                "Filas en la muestra",
                "FEN únicos en la muestra",
                "FEN válido (python-chess `is_valid`)",
                "FEN únicos inválidos, por causa (primer estado)",
                "Primera jugada de `line` legal",
                "Campo de captura al paso idéntico al de python-chess",
                "FEN de la muestra que también están en el archivo 1",
            ],
            "Valor": [
                f"{sample_n:,}",
                f"{eval_sample['fen'].n_unique():,}",
                f"{100 * sum(c[0] for c in _checks) / sample_n:.3f} %",
                invalid_breakdown or "ninguno",
                f"{100 * sum(legal_flags) / sample_n:.3f} %",
                f"{100 * sum(c[1] for c in _checks) / sample_n:.3f} %",
                (
                    f"{len(cross_file_hits):,} de {cross_file_sample_size:,} "
                    f"({100 * len(cross_file_hits) / cross_file_sample_size:.2f} %)"
                ),
            ],
        }
    )
    cross_file_share = len(cross_file_hits) / cross_file_sample_size
    legal_pct = 100 * sum(legal_flags) / sample_n
    mo.ui.table(legality_summary, pagination=False)
    return cross_file_share, legal_pct, sample_n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Puzzles

    Los 3 archivos completos. El FEN del puzzle es **antes** de la jugada del
    rival `Moves[0]`; la posición que ve quien resuelve se obtiene aplicando esa
    jugada (`puzzle_position`).
    """)
    return


@app.cell
def _(
    PUZZLES_REPO_ID,
    PUZZLES_REVISION,
    PUZZLES_SCHEMA,
    PUZZLE_FILES,
    mo,
    pl,
    scan_hf_dataset,
):
    with mo.persistent_cache("puzzles_stats_v1"):
        _lf = scan_hf_dataset(
            PUZZLES_REPO_ID,
            PUZZLE_FILES,
            PUZZLES_REVISION,
            list(PUZZLES_SCHEMA),
            PUZZLES_SCHEMA,
        )
        puzzle_counts = _lf.select(
            filas=pl.len(),
            ids_unicos=pl.col("PuzzleId").n_unique(),
            fen_unicos=pl.col("FEN").n_unique(),
            nulos_opening_tags=pl.col("OpeningTags").null_count(),
            nulos_otras=pl.sum_horizontal(
                pl.all().exclude("OpeningTags").null_count()
            ).sum(),
            rating_min=pl.col("Rating").min(),
            rating_max=pl.col("Rating").max(),
            rating_media=pl.col("Rating").mean(),
            rating_mediana=pl.col("Rating").median(),
            rd_media=pl.col("RatingDeviation").mean(),
        ).collect(engine="streaming")
        puzzle_rating_hist = (
            _lf.group_by((pl.col("Rating") // 100 * 100).alias("rating"))
            .len(name="puzzles")
            .sort("rating")
            .collect()
        )
        puzzle_themes = (
            _lf.select("Themes")
            .explode("Themes")
            .group_by("Themes")
            .len(name="puzzles")
            .sort("puzzles", descending=True)
            .head(20)
            .collect(engine="streaming")
        )
        puzzle_lengths = (
            _lf.group_by((pl.col("Moves").str.count_matches(" ") + 1).alias("jugadas"))
            .len(name="puzzles")
            .sort("jugadas")
            .collect()
        )
    return puzzle_counts, puzzle_lengths, puzzle_rating_hist, puzzle_themes


@app.cell
def _(mo, pl, puzzle_counts):
    _c = puzzle_counts.row(0, named=True)
    puzzle_summary = pl.DataFrame(
        {
            "Indicador": [
                "Puzzles",
                "PuzzleId únicos",
                "FEN únicos",
                "Nulos en OpeningTags / en el resto de columnas",
                "Rating mínimo / máximo",
                "Rating media / mediana",
                "RatingDeviation media",
            ],
            "Valor": [
                f"{_c['filas']:,}",
                f"{_c['ids_unicos']:,}",
                f"{_c['fen_unicos']:,}",
                f"{_c['nulos_opening_tags']:,} / {_c['nulos_otras']:,}",
                f"{_c['rating_min']} / {_c['rating_max']}",
                f"{_c['rating_media']:.0f} / {_c['rating_mediana']:.0f}",
                f"{_c['rd_media']:.1f}",
            ],
        }
    )
    mo.ui.table(puzzle_summary, pagination=False)
    return


@app.cell
def _(PALETTE, mo, plt, puzzle_lengths, puzzle_rating_hist, puzzle_themes, save_figure):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.2))
    _axes[0].bar(
        puzzle_rating_hist["rating"].to_list(),
        puzzle_rating_hist["puzzles"].to_list(),
        width=100,
        align="edge",
        color=PALETTE["azul"],
    )
    _axes[0].set(
        title="Dificultad de los puzzles (Rating, cajas de 100)",
        xlabel="Rating del puzzle",
        ylabel="Puzzles",
    )
    _axes[1].bar(
        puzzle_lengths["jugadas"].to_list(),
        puzzle_lengths["puzzles"].to_list(),
        color=PALETTE["verde"],
    )
    _axes[1].set(
        title="Longitud de la solución (jugadas totales, incl. la del rival)",
        xlabel="Jugadas en Moves",
        ylabel="Puzzles",
    )
    _fig.tight_layout()
    _path_rating = save_figure(_fig, "04_puzzles_rating_y_longitud")

    _fig2, _ax2 = plt.subplots(figsize=(9, 6))
    _themes = puzzle_themes.sort("puzzles")
    _ax2.barh(
        _themes["Themes"].to_list(), _themes["puzzles"].to_list(), color=PALETTE["ocre"]
    )
    _ax2.set(
        title="20 temas más frecuentes de los puzzles",
        xlabel="Puzzles (un puzzle puede tener varios temas)",
        ylabel="Tema",
    )
    _fig2.tight_layout()
    _path_themes = save_figure(_fig2, "05_puzzles_temas")
    mo.vstack(
        [
            _fig,
            mo.md(
                "Mira si la dificultad cubre todo el rango y qué longitud de solución es "
                f"típica. PNG: `{_path_rating}`"
            ),
            _fig2,
            mo.md(
                "Mira qué temas dominan: la evaluación por dificultad y tema se apoyará en "
                f"ellos. PNG: `{_path_themes}`"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Solape puzzles / evaluaciones (fuga de datos)

    Se toma una muestra de 100 000 puzzles, se calcula la posición que ve quien
    resuelve y se busca entre los FEN de 2 archivos de evaluaciones. Se compara con
    el FEN de Lichess tal cual (4 campos). La posición del puzzle se normaliza con
    python-chess (captura al paso solo si es legal); en la muestra anterior solo una
    fracción pequeña de FEN trae casilla de captura al paso, así que el sesgo es
    mínimo y va hacia **subestimar** el solape.
    """)
    return


@app.cell
def _(
    EVALUATIONS_REPO_ID,
    EVALUATIONS_REVISION,
    EVALUATIONS_SCHEMA,
    EVAL_OVERLAP_FILES,
    PUZZLES_REPO_ID,
    PUZZLES_REVISION,
    PUZZLES_SCHEMA,
    PUZZLE_FILES,
    match_key,
    match_key_expr,
    mo,
    pl,
    puzzle_position,
    scan_hf_dataset,
):
    # v2: positions are matched with match_key (en passant dropped on both sides).
    with mo.persistent_cache("puzzle_overlap_v2"):
        _pz = scan_hf_dataset(
            PUZZLES_REPO_ID,
            PUZZLE_FILES,
            PUZZLES_REVISION,
            ["FEN", "Moves", "Rating"],
            PUZZLES_SCHEMA,
        ).collect()
        _sample = _pz.sample(100_000, seed=42)
        _positions: list[str | None] = []
        for _fen, _moves in zip(
            _sample["FEN"].to_list(), _sample["Moves"].to_list(), strict=True
        ):
            try:
                _positions.append(match_key(puzzle_position(_fen, _moves)))
            except ValueError:
                _positions.append(None)
        overlap_sample = _sample.select("Rating").with_columns(
            posicion=pl.Series("posicion", _positions, dtype=pl.String)
        )
        _targets = overlap_sample["posicion"].drop_nulls().unique()
        _hits = []
        overlap_by_files = []
        for _file in EVAL_OVERLAP_FILES:
            _found = (
                scan_hf_dataset(
                    EVALUATIONS_REPO_ID,
                    [_file],
                    EVALUATIONS_REVISION,
                    ["fen"],
                    EVALUATIONS_SCHEMA,
                )
                .select(match_key_expr("fen").alias("key"))
                .filter(pl.col("key").is_in(_targets.implode()))
                .unique()
                .collect(engine="streaming")["key"]
            )
            _hits.append(_found)
            _all_hits = pl.concat(_hits).unique()
            _matched = overlap_sample.filter(
                pl.col("posicion").is_in(_all_hits.implode())
            )
            overlap_by_files.append(
                {
                    "archivos": len(_hits),
                    "puzzles_con_solape": len(_matched),
                    "rating_medio_solape": _matched["Rating"].mean(),
                }
            )
        overlap_failed = overlap_sample["posicion"].null_count()
        overlap_sample_size = len(overlap_sample)
        overlap_rating_all = overlap_sample["Rating"].mean()
    return (
        overlap_by_files,
        overlap_failed,
        overlap_rating_all,
        overlap_sample_size,
    )


@app.cell
def _(
    EVAL_TOTAL_FILES,
    mo,
    overlap_by_files,
    overlap_failed,
    overlap_rating_all,
    overlap_sample_size,
    pl,
):
    _rows = []
    for _r in overlap_by_files:
        _pct = 100 * _r["puzzles_con_solape"] / overlap_sample_size
        _rows.append(
            {
                "Archivos de evaluaciones leídos": _r["archivos"],
                "Puzzles de la muestra con solape": _r["puzzles_con_solape"],
                "Solape (%)": round(_pct, 2),
                "Extrapolado a 20 archivos (%)": round(
                    min(100.0, _pct * EVAL_TOTAL_FILES / _r["archivos"]), 1
                ),
            }
        )
    overlap_table = pl.DataFrame(_rows)
    overlap_pct_last = overlap_table["Solape (%)"][-1]
    overlap_extrapolated = overlap_table["Extrapolado a 20 archivos (%)"][-1]
    mo.vstack(
        [
            mo.ui.table(overlap_table, pagination=False),
            mo.md(
                f"Muestra: {overlap_sample_size:,} puzzles; posiciones no calculables: "
                f"{overlap_failed}. Rating medio de la muestra: {overlap_rating_all:.0f}. "
                "**Advertencia sobre la extrapolación:** multiplicar por 20/k supone que "
                "las posiciones se reparten al azar y sin repetirse entre archivos; si "
                "los archivos están ordenados por algún criterio, la cifra puede estar "
                "sesgada. Es una cota orientativa; la forma segura de evitar fuga es "
                "quitar por FEN (anti-join) con todos los archivos en el Paso 2."
            ),
        ]
    )
    return overlap_extrapolated, overlap_pct_last


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Partidas: `[%eval]` y `[%clk]` por año

    Para enero de cada año 2013-2025 (y septiembre de 2025, el último mes
    disponible) se leen solo las primeras 200 000 filas de `movetext` del primer
    shard. Supuesto: las primeras filas de un shard son representativas del mes
    (no se verifica el orden interno de los shards).
    """)
    return


@app.cell
def _(HfApi, REPOSITORY_ID, REPOSITORY_REVISION, mo):
    # Una sola llamada a la API, cacheada en disco.
    with mo.persistent_cache("games_file_listing_v1"):
        games_files = [
            f
            for f in HfApi().list_repo_files(
                REPOSITORY_ID, repo_type="dataset", revision=REPOSITORY_REVISION
            )
            if f.endswith(".parquet")
        ]
    return (games_files,)


@app.cell
def _(
    EXPECTED_SCHEMA,
    REPOSITORY_ID,
    REPOSITORY_REVISION,
    games_files,
    has_clock_expr,
    has_eval_expr,
    mo,
    month_shard_files,
    parse_month_partitions,
    pl,
    scan_hf_dataset,
):
    GAMES_SAMPLE_ROWS = 200_000
    games_months = parse_month_partitions(games_files)
    GAMES_TARGET_MONTHS = [f"{y}-01" for y in range(2013, 2026)] + [games_months[-1]]
    with mo.persistent_cache("games_eval_clock_by_year_v1"):
        _rows = []
        for _ym in GAMES_TARGET_MONTHS:
            _y, _m = (int(p) for p in _ym.split("-"))
            _shards = month_shard_files(games_files, _y, _m)
            _res = (
                scan_hf_dataset(
                    REPOSITORY_ID,
                    _shards[:1],
                    REPOSITORY_REVISION,
                    ["movetext"],
                    EXPECTED_SCHEMA,
                )
                .head(GAMES_SAMPLE_ROWS)
                .select(
                    partidas=pl.len(),
                    con_eval=has_eval_expr("movetext").sum(),
                    con_clk=has_clock_expr("movetext").sum(),
                    con_ambos=(
                        has_eval_expr("movetext") & has_clock_expr("movetext")
                    ).sum(),
                    movetext_nulo=pl.col("movetext").null_count(),
                )
                .collect()
                .row(0, named=True)
            )
            _rows.append({"mes": _ym, "shards_del_mes": len(_shards), **_res})
        games_eval_clock = pl.DataFrame(_rows).with_columns(
            (100 * pl.col("con_eval") / pl.col("partidas")).round(2).alias("pct_eval"),
            (100 * pl.col("con_clk") / pl.col("partidas")).round(2).alias("pct_clk"),
            (100 * pl.col("con_ambos") / pl.col("partidas"))
            .round(2)
            .alias("pct_ambos"),
        )
    return GAMES_SAMPLE_ROWS, games_eval_clock, games_months


@app.cell
def _(PALETTE, games_eval_clock, mo, plt, save_figure):
    _x = list(range(len(games_eval_clock)))
    _fig, _ax = plt.subplots(figsize=(11, 4.2))
    _ax.plot(
        _x,
        games_eval_clock["pct_eval"].to_list(),
        marker="o",
        color=PALETTE["azul"],
        label="Con [%eval]",
    )
    _ax.plot(
        _x,
        games_eval_clock["pct_clk"].to_list(),
        marker="s",
        color=PALETTE["ocre"],
        label="Con [%clk]",
    )
    _ax.set_xticks(_x, games_eval_clock["mes"].to_list(), rotation=45)
    _ax.set(
        title="Partidas con evaluación y reloj, por mes de muestra (primeras 200 000 filas del primer shard)",
        xlabel="Mes de la muestra",
        ylabel="% de partidas",
        ylim=(0, 100),
    )
    _ax.legend()
    _fig.tight_layout()
    _path_games = save_figure(_fig, "06_partidas_eval_clk_por_anio")
    mo.vstack(
        [
            mo.ui.table(games_eval_clock, pagination=False),
            _fig,
            mo.md(
                "Mira desde qué año `[%eval]` y `[%clk]` dejan de ser raros: de ahí "
                f"depende qué meses sirven para medir pérdida de centipeones. PNG: `{_path_games}`"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Velocidad de lectura

    Pulsa el botón para leer **toda** la columna `movetext` del primer shard del
    último mes y medir MB/s (descarga de ~1 GB; no se cachea a propósito, porque
    es una medición).
    """)
    return


@app.cell
def _(mo):
    throughput_button = mo.ui.run_button(label="Medir velocidad de lectura")
    throughput_button
    return (throughput_button,)


@app.cell
def _(
    EXPECTED_SCHEMA,
    HfApi,
    REPOSITORY_ID,
    REPOSITORY_REVISION,
    games_files,
    games_months,
    mo,
    month_shard_files,
    scan_hf_dataset,
    throughput_button,
    time,
):
    mo.stop(
        not throughput_button.value,
        mo.md("Botón sin pulsar: no se midió la velocidad en esta ejecución."),
    )
    _y, _m = (int(p) for p in games_months[-1].split("-"))
    _shards = month_shard_files(games_files, _y, _m)
    _t0 = time.perf_counter()
    _movetext = scan_hf_dataset(
        REPOSITORY_ID, _shards[:1], REPOSITORY_REVISION, ["movetext"], EXPECTED_SCHEMA
    ).collect()
    measured_seconds = time.perf_counter() - _t0
    measured_uncompressed_mb = _movetext["movetext"].estimated_size("mb")
    measured_shard_mb = (
        HfApi()
        .get_paths_info(
            REPOSITORY_ID,
            _shards[:1],
            repo_type="dataset",
            revision=REPOSITORY_REVISION,
        )[0]
        .size
        / 1e6
    )
    measured_rows = len(_movetext)
    mo.md(
        f"Shard `{_shards[0]}`: {measured_rows:,} partidas, {measured_seconds:.1f} s; "
        f"{measured_uncompressed_mb / measured_seconds:.1f} MB/s sin comprimir, "
        f"{measured_shard_mb / measured_seconds:.1f} MB/s del archivo comprimido "
        f"({measured_shard_mb:.0f} MB)."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Tabla de decisiones

    Reglas explícitas (supuestos):

    - **Profundidad mínima**: la mayor entre {20, 25, 30, 35, 40} que conserva al
      menos el 80 % de los FEN únicos (se usa la profundidad máxima por FEN).
    - **Posiciones disponibles**: FEN únicos del archivo 0 con esa profundidad,
      multiplicados por 20 archivos y reducidos por la duplicación entre archivos
      medida en la muestra (cota superior si hay duplicación no medida).
    - **Meses con `[%eval]` suficiente**: meses de muestra con al menos 5 % de
      partidas con `[%eval]`.
    - **Tiempo de lectura**: velocidad medida en un shard (ver constantes abajo).
    """)
    return


@app.cell
def _(
    EVAL_TOTAL_FILES,
    cross_file_share,
    eval_joint,
    games_eval_clock,
    mo,
    overlap_extrapolated,
    overlap_pct_last,
    pl,
    unique_fens,
):
    # Medido con scripts/step1_throughput.py el 2026-10-27 (shard 0 de 2025-09,
    # 1.4 M partidas, solo `movetext`); constante porque el botón no corre en
    # ejecuciones no interactivas. Supone que la red/CPU de esa medición se repite.
    MEASURED_SHARD_SECONDS = 47.2
    MEASURED_SHARD_MB = 1002.0
    SHARDS_PER_RECENT_MONTH = 62
    TOTAL_GAMES_TB = 4.94
    CANDIDATE_DEPTHS = [20, 25, 30, 35, 40]
    _retention = pl.DataFrame(
        {
            "profundidad_min": CANDIDATE_DEPTHS,
            "fen_conservados": [
                eval_joint.filter(pl.col("max_profundidad") >= d)["fens"].sum()
                for d in CANDIDATE_DEPTHS
            ],
        }
    ).with_columns(
        (100 * pl.col("fen_conservados") / unique_fens).round(2).alias("pct_fen")
    )
    _ok = _retention.filter(pl.col("pct_fen") >= 80)
    suggested_depth = (
        int(_ok["profundidad_min"].max()) if len(_ok) else CANDIDATE_DEPTHS[0]
    )
    _kept = int(
        _retention.filter(pl.col("profundidad_min") == suggested_depth)[
            "fen_conservados"
        ][0]
    )
    positions_20_files = int(_kept * EVAL_TOTAL_FILES * (1 - cross_file_share))
    good_eval_months = games_eval_clock.filter(pl.col("pct_eval") >= 5)["mes"].to_list()
    _eval_summary = (
        f"{len(good_eval_months)} de {len(games_eval_clock)} muestras "
        f"({good_eval_months[0]} a {good_eval_months[-1]}); "
        f"entre {games_eval_clock.filter(pl.col('pct_eval') >= 5)['pct_eval'].min()} % "
        f"y {games_eval_clock['pct_eval'].max()} %"
        if good_eval_months
        else "ninguno"
    )
    _month_min = MEASURED_SHARD_SECONDS * SHARDS_PER_RECENT_MONTH / 60
    _total_h = (
        TOTAL_GAMES_TB * 1e6 / (MEASURED_SHARD_MB / MEASURED_SHARD_SECONDS) / 3600
    )
    decision_table = pl.DataFrame(
        {
            "Decisión": [
                "Profundidad mínima sugerida",
                "FEN únicos tras el filtro (archivo 0)",
                "Posiciones disponibles (20 archivos, cota superior)",
                "Puzzles a excluir (solape medido / extrapolado, cota superior)",
                "Meses de muestra con >= 5 % de [%eval]",
                "Tiempo de lectura estimado (solo movetext, secuencial)",
            ],
            "Valor": [
                str(suggested_depth),
                f"{_kept:,}",
                f"{positions_20_files:,}",
                f"{overlap_pct_last} % / {overlap_extrapolated} %",
                _eval_summary,
                f"~{_month_min:.0f} min por mes reciente; ~{_total_h:.0f} h para 4.94 TB",
            ],
        }
    )
    mo.vstack(
        [
            mo.md("Retención de FEN por profundidad mínima (archivo 0):"),
            mo.ui.table(_retention, pagination=False),
            mo.ui.table(decision_table, pagination=False),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
