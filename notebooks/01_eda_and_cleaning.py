import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

    from lichess_pipeline.data_ingestion import download_to_disk
    from lichess_pipeline.pgn_parser import (
        VALID_RESULTS,
        clean_games_df,
        evals_url,
        extract_positions_df,
        games_sample_url,
        load_clean_evals_df,
        parse_pgn_sample_df,
    )

    PROJECT_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_DIR / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 100,
            "axes.titlesize": 11,
        }
    )

    def save_frame(frame: pl.DataFrame, name: str) -> Path:
        """Write a cleaned sample to data/processed and return its path."""
        path = PROCESSED_DIR / f"{name}.parquet"
        frame.write_parquet(path)
        return path

    return (
        RAW_DIR,
        VALID_RESULTS,
        clean_games_df,
        download_to_disk,
        evals_url,
        extract_positions_df,
        games_sample_url,
        load_clean_evals_df,
        mo,
        parse_pgn_sample_df,
        pl,
        plt,
        save_frame,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Exploración y limpieza de los datos oficiales de Lichess

    Se leen los datos **directamente de `database.lichess.org`**, y deja listos dos archivos Parquet para el modelo:

    | Fuente | Archivo oficial | Uso en el modelo |
    |---|---|---|
    | Partidas estándar | `lichess_db_standard_rated_AAAA-MM.pgn.zst` | Entrenamiento: jugadas humanas de Elo alto |
    | Evaluaciones | `lichess_db_eval.jsonl.zst` | Fine tuning: mejores jugadas de Stockfish |

    Solo se descarga el prefijo necesario: el flujo HTTP se cierra en cuanto se alcanza el número de partidas o de evaluaciones pedido.

    **Output `data/processed/`:** `games_clean_sample.parquet`, `positions_games_sample.parquet` y `evals_clean_sample.parquet`.
    """)
    return


@app.cell
def _(mo):
    year = mo.ui.dropdown(options=[2024, 2025], value=2025, label="Año de las partidas")
    month = mo.ui.dropdown(
        options=list(range(1, 13)), value=1, label="Mes de las partidas"
    )
    max_games = mo.ui.number(
        value=3000, start=200, stop=50000, step=200, label="Partidas a leer del .pgn.zst"
    )
    min_elo = mo.ui.slider(
        start=1500,
        stop=2200,
        step=100,
        value=1800,
        label="Elo mínimo (se exige a ambos jugadores)",
    )
    max_positions = mo.ui.number(
        value=20000,
        start=1000,
        stop=500000,
        step=1000,
        label="Máximo de posiciones (FEN) a extraer",
    )
    keep_raw = mo.ui.checkbox(
        value=False, label="Guardar también el .zst en data/raw (máx. 100 MB)"
    )
    mo.vstack(
        [
            mo.md("## Controles de la muestra"),
            mo.hstack([year, month], justify="start"),
            max_games,
            min_elo,
            max_positions,
            keep_raw,
        ]
    )
    return keep_raw, max_games, max_positions, min_elo, month, year


@app.cell
def _(mo):
    games_button = mo.ui.run_button(label="Leer partidas oficiales (.pgn.zst)")
    mo.vstack(
        [
            mo.md("## 1. Partidas estándar"),
            mo.md(
                "Pulsa el botón para descargar el prefijo indicado del mes elegido. "
                "Se descartan las partidas que no empiezan con jugadas legales."
            ),
            games_button,
        ]
    )
    return (games_button,)


@app.cell
def _(
    games_button,
    games_sample_url,
    max_games,
    mo,
    month,
    parse_pgn_sample_df,
    year,
):
    mo.stop(
        not games_button.value,
        mo.md("Pulsa **Leer partidas oficiales** para cargar la muestra."),
    )
    games_source = games_sample_url(int(year.value), int(month.value))
    raw_games = parse_pgn_sample_df(games_source, max_games=int(max_games.value))
    mo.callout(
        mo.md(
            f"Leídas **{raw_games.height:,}** partidas de `{games_source}` "
            f"({int(max_games.value):,} bloques PGN solicitados)."
        ),
        kind="success",
    )
    return (raw_games,)


@app.cell
def _(RAW_DIR, download_to_disk, games_sample_url, keep_raw, mo, month, year):
    mo.stop(
        not keep_raw.value,
        mo.md("El archivo `.zst` no se guarda (casilla desactivada)."),
    )
    raw_target = RAW_DIR / f"lichess_{int(year.value)}-{int(month.value):02d}.pgn.zst"
    saved_path = download_to_disk(
        games_sample_url(int(year.value), int(month.value)),
        raw_target,
        max_bytes=100_000_000,
    )
    mo.md(f"Guardado un prefijo del archivo en `{saved_path}`.")
    return


@app.cell
def _(mo, pl, raw_games):
    row_count = raw_games.height
    null_counts = pl.DataFrame(
        {
            "columna": raw_games.columns,
            "nulos": [raw_games[column].null_count() for column in raw_games.columns],
        }
    ).with_columns((100 * pl.col("nulos") / row_count).round(2).alias("nulos (%)"))
    quality = pl.DataFrame(
        {
            "Indicador": [
                "Partidas",
                "Columnas",
                "game_id duplicados",
                "Plies (mínimo / mediana / máximo)",
                "Eventos distintos",
                "Aperturas distintas",
            ],
            "Valor": [
                f"{row_count:,}",
                str(raw_games.width),
                f"{raw_games['game_id'].is_duplicated().sum():,}",
                (
                    f"{raw_games['num_plies'].min()} / "
                    f"{raw_games['num_plies'].median()} / "
                    f"{raw_games['num_plies'].max()}"
                ),
                f"{raw_games['Event'].n_unique():,}",
                f"{raw_games['Opening'].n_unique():,}",
            ],
        }
    )
    mo.vstack(
        [
            mo.md("### Calidad y estructura"),
            mo.ui.table(quality, pagination=False),
            mo.md("#### Valores faltantes por columna"),
            mo.ui.table(null_counts, pagination=False),
        ]
    )
    return


@app.cell
def _(plt, raw_games):
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].hist(
        [
            raw_games["WhiteElo"].drop_nulls().to_numpy(),
            raw_games["BlackElo"].drop_nulls().to_numpy(),
        ],
        bins=40,
        label=["Blancas", "Negras"],
        alpha=0.75,
    )
    axes[0, 0].set(title="Distribución de Elo", xlabel="Elo", ylabel="Partidas")
    axes[0, 0].legend()

    events = (
        raw_games.group_by("Event")
        .len(name="partidas")
        .sort("partidas", descending=True)
        .head(6)
    )
    axes[0, 1].barh(events["Event"].to_list(), events["partidas"].to_list())
    axes[0, 1].set(title="Eventos más frecuentes", xlabel="Partidas")
    axes[0, 1].invert_yaxis()

    terminations = (
        raw_games.group_by("Termination")
        .len(name="partidas")
        .sort("partidas", descending=True)
    )
    axes[1, 0].bar(
        terminations["Termination"].to_list(), terminations["partidas"].to_list()
    )
    axes[1, 0].set(title="Tipo de terminación", ylabel="Partidas")
    axes[1, 0].tick_params(axis="x", rotation=25)

    axes[1, 1].hist(raw_games["num_plies"].to_numpy(), bins=50)
    axes[1, 1].set(title="Duración de la partida", xlabel="Plies", ylabel="Partidas")

    figure.tight_layout()
    figure
    return


@app.cell
def _(mo, raw_games):
    openings = (
        raw_games.group_by("ECO", "Opening")
        .len(name="partidas")
        .sort("partidas", descending=True)
        .head(10)
    )
    time_controls = (
        raw_games.group_by("TimeControl")
        .len(name="partidas")
        .sort("partidas", descending=True)
        .head(10)
    )
    mo.vstack(
        [
            mo.md("#### Controles de tiempo más frecuentes"),
            mo.ui.table(time_controls, pagination=False),
            mo.md("#### Aperturas más frecuentes (no se usan como entrada del modelo)"),
            mo.ui.table(openings, pagination=False),
        ]
    )
    return


@app.cell
def _(VALID_RESULTS, clean_games_df, min_elo, mo, pl, raw_games):
    elo_limit = int(min_elo.value)
    filters = [
        (
            "Resultado y movetext válidos",
            pl.col("Result").is_in(VALID_RESULTS) & pl.col("movetext").is_not_null(),
        ),
        (
            f"Elo de ambos jugadores >= {elo_limit}",
            (pl.col("WhiteElo") >= elo_limit) & (pl.col("BlackElo") >= elo_limit),
        ),
        ("Sin Bullet ni UltraBullet", ~pl.col("Event").str.contains("Bullet")),
        ("Terminación Normal", pl.col("Termination") == "Normal"),
        ("Al menos 10 plies", pl.col("num_plies") >= 10),
    ]
    ladder_rows = [{"paso": "Partidas leídas", "partidas": raw_games.height}]
    current = raw_games
    for label, condition in filters:
        current = current.filter(condition)
        ladder_rows.append({"paso": f"Tras «{label}»", "partidas": current.height})

    ladder = pl.DataFrame(ladder_rows).with_columns(
        (100 * pl.col("partidas") / raw_games.height).round(1).alias("% retenido")
    )
    games_clean = clean_games_df(raw_games, min_elo=elo_limit)
    mo.vstack(
        [
            mo.md("### Limpieza aplicada"),
            mo.ui.table(ladder, pagination=False),
            mo.callout(
                mo.md(
                    f"Quedan **{games_clean.height:,}** partidas "
                    f"({100 * games_clean.height / max(raw_games.height, 1):.1f} %)."
                ),
                kind="info",
            ),
        ]
    )
    return (games_clean,)


@app.cell
def _(extract_positions_df, games_clean, max_positions, mo, save_frame):
    positions = extract_positions_df(games_clean, max_positions=int(max_positions.value))
    games_path = save_frame(games_clean, "games_clean_sample")
    positions_path = save_frame(positions, "positions_games_sample")
    mo.vstack(
        [
            mo.md("### Posiciones de entrenamiento (etapa 1)"),
            mo.callout(
                mo.md(
                    f"**{positions.height:,}** filas `(fen, move_uci)` extraídas al "
                    f"reproducir {games_clean.height:,} partidas.\n\n"
                    f"- `{games_path}`\n- `{positions_path}`"
                ),
                kind="success",
            ),
            mo.ui.table(positions.head(5), pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    max_evals = mo.ui.number(
        value=20000,
        start=1000,
        stop=500000,
        step=1000,
        label="Evaluaciones únicas a leer",
    )
    min_depth = mo.ui.slider(
        start=20, stop=40, step=5, value=20, label="Profundidad mínima de Stockfish"
    )
    evals_button = mo.ui.run_button(label="Leer evaluaciones oficiales (.jsonl.zst)")
    mo.vstack(
        [
            mo.md("## 2. Evaluaciones de Stockfish"),
            mo.md(
                "Cada línea del archivo trae un FEN con varias evaluaciones. Se "
                "conserva la de mayor profundidad y su primera jugada, y se "
                "descartan FEN inválidos o jugadas ilegales."
            ),
            max_evals,
            min_depth,
            evals_button,
        ]
    )
    return evals_button, max_evals, min_depth


@app.cell
def _(evals_button, evals_url, load_clean_evals_df, max_evals, min_depth, mo):
    mo.stop(
        not evals_button.value,
        mo.md("Pulsa **Leer evaluaciones oficiales** para cargar la muestra."),
    )
    evals_clean = load_clean_evals_df(
        evals_url(),
        max_evals=int(max_evals.value),
        min_depth=int(min_depth.value),
    )
    mo.callout(
        mo.md(
            f"Leídas **{evals_clean.height:,}** posiciones únicas y legales con "
            f"profundidad >= {int(min_depth.value)}."
        ),
        kind="success",
    )
    return (evals_clean,)


@app.cell
def _(evals_clean, mo, pl, plt):
    figure_evals, _axes = plt.subplots(2, 2, figsize=(13, 8))

    _axes[0, 0].hist(evals_clean["depth"].to_numpy(), bins=30)
    _axes[0, 0].set(
        title="Profundidad de búsqueda", xlabel="Profundidad", ylabel="Posiciones"
    )

    centipawns = evals_clean["cp"].drop_nulls().to_numpy()
    _axes[0, 1].hist(centipawns, bins=60, range=(-1000, 1000))
    _axes[0, 1].set(
        title="Evaluación en centipeones (punto de vista de las blancas)",
        xlabel="cp",
        ylabel="Posiciones",
    )

    phases = (
        evals_clean.group_by("game_phase")
        .len(name="posiciones")
        .sort("posiciones", descending=True)
    )
    _axes[1, 0].bar(phases["game_phase"].to_list(), phases["posiciones"].to_list())
    _axes[1, 0].set(title="Posiciones por fase", ylabel="Posiciones")

    turns = (
        evals_clean.group_by("turn").len(name="posiciones").sort("posiciones", descending=True)
    )
    _axes[1, 1].bar(
        ["Blancas" if t == "w" else "Negras" for t in turns["turn"].to_list()],
        turns["posiciones"].to_list(),
    )
    _axes[1, 1].set(title="Turno al que le toca mover", ylabel="Posiciones")

    figure_evals.tight_layout()
    mo.vstack(
        [
            mo.md("### Distribuciones de la muestra de evaluaciones"),
            mo.ui.table(
                evals_clean.select(
                    pl.len().alias("posiciones"),
                    pl.col("depth").median().alias("profundidad mediana"),
                    pl.col("cp").mean().round(0).alias("cp medio"),
                    pl.col("mate").is_not_null().sum().alias("posiciones con mate"),
                    pl.col("knodes").max().alias("knodes máximo"),
                ),
                pagination=False,
            ),
            figure_evals,
        ]
    )
    return


@app.cell
def _(evals_clean, mo, save_frame):
    evals_path = save_frame(evals_clean, "evals_clean_sample")
    mo.vstack(
        [
            mo.md("### Salida de la etapa 2"),
            mo.callout(
                mo.md(
                    f"**{evals_clean.height:,}** filas `(fen, best_move)` guardadas en "
                    f"`{evals_path}`."
                ),
                kind="success",
            ),
            mo.ui.table(evals_clean.head(5), pagination=False),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Cómo seguir

    Los tres Parquet de `data/processed/` alimentan los otros dos cuadernos:

    - `positions_games_sample.parquet` -> `02_model_scratch_and_finetune.py` (etapa 1).
    - `evals_clean_sample.parquet` -> `02_...` (etapa 2) y `03_model_maia_pretrained.py`.

    Para el servidor final se procesa por bloques (streaming -> Parquet) y se
    entrena con un dataset iterable; la lógica de parseo no cambia.
    """)
    return


if __name__ == "__main__":
    app.run()
