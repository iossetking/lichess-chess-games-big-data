import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

    from lichess_pipeline.data_ingestion import (
        LichessConnectionError,
        assumed_game_months,
        available_game_months,
        download_to_disk,
        months_in_range,
    )
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

    def month_label(month: tuple[int, int]) -> str:
        """Format a ``(year, month)`` pair as ``AAAA-MM``."""
        year, number = month
        return f"{year}-{number:02d}"

    return (
        LichessConnectionError,
        RAW_DIR,
        VALID_RESULTS,
        assumed_game_months,
        available_game_months,
        clean_games_df,
        download_to_disk,
        evals_url,
        extract_positions_df,
        games_sample_url,
        load_clean_evals_df,
        mo,
        month_label,
        months_in_range,
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

    Primero se consulta el índice de `database.lichess.org/standard/` para saber qué meses tienen archivo, y después se elige un **rango de años** y un **rango de meses** (el rango de meses se aplica a cada año). Solo se descarga el prefijo necesario de cada archivo: el flujo HTTP se cierra en cuanto se alcanza el número de partidas pedido.

    **Output `data/processed/`:** `games_clean_sample.parquet`, `positions_games_sample.parquet` y `evals_clean_sample.parquet`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `games_clean_sample.parquet` — Partidas estándar

    | Columna | Descripción |
    |---|---|
    | `game_id` | Identificador de la partida, extraído del tag `Site` de Lichess. |
    | `White` | Nombre de usuario del jugador de blancas. |
    | `Black` | Nombre de usuario del jugador de negras. |
    | `Result` | Resultado oficial: `1-0`, `0-1` o `1/2-1/2`. |
    | `WhiteElo` | Elo de las blancas al inicio de la partida. |
    | `BlackElo` | Elo de las negras al inicio de la partida. |
    | `WhiteRatingDiff` | Variación de Elo de las blancas tras la partida. |
    | `BlackRatingDiff` | Variación de Elo de las negras tras la partida. |
    | `UTCDate` | Fecha de la partida en UTC (`AAAA.MM.DD`). |
    | `UTCTime` | Hora de inicio de la partida en UTC (`HH:MM:SS`). |
    | `ECO` | Código ECO de la apertura. |
    | `Opening` | Nombre de la apertura. |
    | `Event` | Evento o ritmo de la partida (p. ej. `Rated Blitz game`). |
    | `TimeControl` | Control de tiempo (`segundos` o `segundos+incremento`). |
    | `Termination` | Motivo del final: `Normal`, `Time forfeit` o `Abandoned`. |
    | `movetext` | Jugadas en notación SAN reproducidas y validadas con `python-chess`, sin comentarios ni reloj. |
    | `num_plies` | Número de medias jugadas (plies) válidas de la partida. |

    ## `evals_clean_sample.parquet` — Evaluaciones de Stockfish

    | Columna | Descripción |
    |---|---|
    | `fen` | FEN original de la posición tal como aparece en el archivo oficial. |
    | `norm_fen` | FEN normalizado (primeros cuatro campos); se usa como clave para deduplicar posiciones. |
    | `turn` | Bando al que le toca mover: `w` (blancas) o `b` (negras). |
    | `piece_count` | Número total de piezas sobre el tablero. |
    | `game_phase` | Fase derivada del número de piezas: `opening` (>= 29), `middlegame` (15–28) o `endgame` (<= 14). |
    | `depth` | Profundidad de búsqueda de Stockfish, en plies. |
    | `knodes` | Miles de nodos visitados por Stockfish durante la búsqueda. |
    | `cp` | Evaluación en centipeones desde el punto de vista de las blancas. |
    | `mate` | Jugadas hasta el mate forzado (signo según el bando); nulo si no hay mate. |
    | `best_move` | Primera jugada de la variante principal, en UCI y validada como legal. |
    """)
    return


@app.cell
def _(mo):
    months_revision, set_months_revision = mo.state(0)
    year_selection, set_year_selection = mo.state((0, 0))
    month_selection, set_month_selection = mo.state((0, 0))
    refresh_months = mo.ui.button(
        label="Actualizar disponibilidad",
        on_click=lambda _count: set_months_revision(lambda revision: revision + 1),
    )
    mo.vstack(
        [
            mo.md("## Disponibilidad de archivos oficiales"),
            mo.md(
                "Se consulta el índice de `database.lichess.org/standard/` para saber "
                "qué meses tienen `.pgn.zst` antes de elegir el rango."
            ),
            refresh_months,
        ]
    )
    return (
        month_selection,
        months_revision,
        refresh_months,
        set_month_selection,
        set_year_selection,
        year_selection,
    )


@app.cell
def _(
    LichessConnectionError,
    assumed_game_months,
    available_game_months,
    mo,
    month_label,
    months_revision,
):
    try:
        available_months = available_game_months()
        availability_note = mo.callout(
            mo.md(
                f"**{len(available_months)}** meses con archivo oficial, de "
                f"`{month_label(available_months[0])}` a "
                f"`{month_label(available_months[-1])}`."
            ),
            kind="success",
        )
    except LichessConnectionError as exc:
        available_months = assumed_game_months()
        availability_note = mo.callout(
            mo.md(
                f"{exc}\n\n"
                "Se asume el rango continuo "
                f"`{month_label(available_months[0])}` … "
                f"`{month_label(available_months[-1])}`; los meses que falten se "
                "descartan al leer."
            ),
            kind="warn",
        )
    # Leer el estado basta para que el botón «Actualizar» vuelva a consultar el índice.
    _ = months_revision
    availability_note
    return available_months, availability_note


@app.cell
def _(
    available_months,
    mo,
    month_selection,
    set_month_selection,
    set_year_selection,
    year_selection,
):
    years = sorted({year for year, _month in available_months})
    default_year, default_month = available_months[-1]
    current_years = tuple(year_selection())
    if not (years[0] <= current_years[0] <= current_years[1] <= years[-1]):
        current_years = (default_year, default_year)
    current_months = tuple(month_selection())
    if not (1 <= current_months[0] <= current_months[1] <= 12):
        current_months = (default_month, default_month)
    year_range = mo.ui.range_slider(
        start=years[0],
        stop=years[-1],
        step=1,
        value=current_years,
        label="Años de las partidas",
        show_value=False,
        debounce=True,
        on_change=set_year_selection,
    )
    month_range = mo.ui.range_slider(
        start=1,
        stop=12,
        step=1,
        value=current_months,
        label="Meses de cada año (de inicio a fin)",
        show_value=False,
        debounce=True,
        on_change=set_month_selection,
    )
    year_readout = mo.md(
        f"Años seleccionados: **{current_years[0]} – {current_years[1]}**"
    )
    month_readout = mo.md(
        f"Meses por año: **{current_months[0]:02d} – {current_months[1]:02d}**"
    )
    max_games = mo.ui.number(
        value=3000,
        start=200,
        stop=50000,
        step=200,
        label="Partidas a leer por mes del .pgn.zst",
    )
    min_elo = mo.ui.slider(
        start=1500,
        stop=2200,
        step=100,
        value=1800,
        label="Elo mínimo (se exige a ambos jugadores)",
    )
    max_positions = mo.ui.number(
        value=40000,
        start=1000,
        stop=500000,
        step=1000,
        label="Máximo de posiciones (FEN) a extraer",
    )
    keep_raw = mo.ui.checkbox(
        value=False, label="Guardar también los .zst en data/raw (máx. 100 MB en total)"
    )
    raw_button = mo.ui.run_button(label="Guardar los .zst seleccionados")
    mo.vstack(
        [
            mo.md("## Controles de la muestra"),
            year_range,
            year_readout,
            month_range,
            month_readout,
            max_games,
            min_elo,
            max_positions,
            keep_raw,
            raw_button,
        ]
    )
    return (
        keep_raw,
        max_games,
        max_positions,
        min_elo,
        month_range,
        raw_button,
        year_range,
    )


@app.cell
def _(
    available_months,
    max_games,
    mo,
    month_label,
    month_range,
    months_in_range,
    year_range,
):
    months_selected, months_skipped = months_in_range(
        year_range.value, month_range.value, available_months
    )
    if not months_selected:
        selection_summary = mo.callout(
            mo.md("La selección no incluye ningún mes con archivo. Amplía el rango."),
            kind="danger",
        )
    else:
        listed = ", ".join(month_label(month) for month in months_selected)
        if len(months_selected) > 24:
            listed = (
                f"{month_label(months_selected[0])} … "
                f"{month_label(months_selected[-1])}"
            )
        total_cap = int(max_games.value) * len(months_selected)
        _blocks = [
            mo.callout(
                mo.md(
                    f"**{len(months_selected)}** meses seleccionados: {listed}.\n\n"
                    f"Tope de lectura: **{total_cap:,}** partidas "
                    f"({int(max_games.value):,} por mes)."
                ),
                kind="info",
            )
        ]
        if total_cap > 100_000:
            _blocks.append(
                mo.callout(
                    mo.md(
                        f"Es una muestra grande: hasta **{total_cap:,}** partidas se "
                        "guardan en memoria a la vez. Baja las partidas por mes o "
                        "reduce el rango si el equipo se queda sin memoria."
                    ),
                    kind="warn",
                )
            )
        if months_skipped:
            _blocks.append(
                mo.callout(
                    mo.md(
                        "Sin archivo, se descartan: "
                        + ", ".join(month_label(month) for month in months_skipped)
                    ),
                    kind="warn",
                )
            )
        selection_summary = mo.vstack(_blocks)
    selection_summary
    return months_selected, months_skipped


@app.cell
def _(mo):
    games_button = mo.ui.run_button(label="Leer partidas oficiales (.pgn.zst)")
    mo.vstack(
        [
            mo.md("## 1. Partidas estándar"),
            mo.md(
                "Pulsa el botón para descargar el prefijo indicado de **cada mes "
                "seleccionado**. Se descartan las partidas que no empiezan con jugadas "
                "legales."
            ),
            games_button,
        ]
    )
    return (games_button,)


@app.cell
def _(
    LichessConnectionError,
    games_button,
    games_sample_url,
    max_games,
    mo,
    month_label,
    months_selected,
    parse_pgn_sample_df,
    pl,
):
    mo.stop(
        not games_button.value,
        mo.md("Pulsa **Leer partidas oficiales** para cargar la muestra."),
    )
    mo.stop(
        not months_selected,
        mo.callout(
            mo.md("No hay meses con archivo en la selección actual."), kind="warn"
        ),
    )
    frames = []
    monthly_rows = []
    skipped_months = []
    _failures = []
    with mo.status.spinner(title="Leyendo los meses seleccionados…") as _spinner:
        for _game_year, _game_month in months_selected:
            source = games_sample_url(_game_year, _game_month)
            _spinner.update(subtitle=f"Leyendo `{source.rsplit('/', 1)[-1]}`…")
            try:
                month_frame = parse_pgn_sample_df(
                    source, max_games=int(max_games.value)
                )
            except LichessConnectionError as exc:
                skipped_months.append(month_label((_game_year, _game_month)))
                _failures.append(str(exc))
                continue
            frames.append(month_frame)
            monthly_rows.append(
                {
                    "Año": _game_year,
                    "Mes": _game_month,
                    "Partidas": month_frame.height,
                }
            )
    mo.stop(
        not frames,
        mo.callout(
            mo.md(
                "No se pudo leer ningún mes seleccionado.\n\n"
                + "\n".join(f"- {failure}" for failure in _failures)
            ),
            kind="danger",
        ),
    )
    raw_games = pl.concat(frames)
    _blocks = [
        mo.callout(
            mo.md(
                f"Leídas **{raw_games.height:,}** partidas de "
                f"**{len(frames)}** meses "
                f"({int(max_games.value):,} bloques PGN solicitados por mes)."
            ),
            kind="success",
        ),
        mo.ui.table(pl.DataFrame(monthly_rows), pagination=False),
    ]
    if skipped_months:
        _blocks.append(
            mo.callout(
                mo.md(
                    "Meses que no se pudieron leer y se omiten: "
                    + ", ".join(f"`{month}`" for month in skipped_months)
                    + ".\n\n"
                    + "\n".join(f"- {failure}" for failure in _failures)
                ),
                kind="warn",
            )
        )
    mo.vstack(_blocks)
    return (raw_games,)


@app.cell
def _(
    LichessConnectionError,
    RAW_DIR,
    download_to_disk,
    games_sample_url,
    keep_raw,
    month_label,
    mo,
    months_selected,
    raw_button,
):
    mo.stop(
        not keep_raw.value,
        mo.md("Los archivos `.zst` no se guardan (casilla desactivada)."),
    )
    mo.stop(
        not raw_button.value,
        mo.md("Pulsa **Guardar los .zst seleccionados** para descargarlos."),
    )
    mo.stop(
        not months_selected,
        mo.callout(mo.md("No hay meses seleccionados que descargar."), kind="warn"),
    )
    budget = max(1_000_000, 100_000_000 // len(months_selected))
    saved_paths = []
    _failures = []
    with mo.status.spinner(title="Guardando prefijos de los .zst…") as _spinner:
        for _game_year, _game_month in months_selected:
            _label = month_label((_game_year, _game_month))
            _spinner.update(subtitle=f"Descargando `{_label}`…")
            try:
                saved_paths.append(
                    download_to_disk(
                        games_sample_url(_game_year, _game_month),
                        RAW_DIR / f"lichess_{_label}.pgn.zst",
                        max_bytes=budget,
                    )
                )
            except LichessConnectionError as exc:
                _failures.append(f"`{_label}`: {exc}")
    message = (
        f"Guardados **{len(saved_paths)}** prefijos de ~{budget / 1_000_000:.1f} MB "
        f"({budget * len(saved_paths) / 1_000_000:.0f} MB en total) en `{RAW_DIR}`."
    )
    if saved_paths:
        message += "\n\n" + "\n".join(f"- `{path.name}`" for path in saved_paths)
    if _failures:
        message += "\n\nFallaron:\n" + "\n".join(f"- {failure}" for failure in _failures)
    mo.md(message)
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

    - `positions_games_sample.parquet` -> `02_model_scratch_and_finetune.py` (etapa 1)
      y `03_maia.py` (modelo A, LoRA con jugadas humanas).
    - `evals_clean_sample.parquet` -> `02_...` (etapa 2) y `03_maia.py` (modelo B,
      LoRA con jugadas de Stockfish).

    Para el servidor final se procesa por bloques (streaming -> Parquet) y se
    entrena con un dataset iterable; la lógica de parseo no cambia.
    """)
    return


if __name__ == "__main__":
    app.run()
