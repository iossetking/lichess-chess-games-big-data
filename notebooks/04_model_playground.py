import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    from datetime import datetime
    from functools import lru_cache
    from pathlib import Path

    def _project_dir() -> Path:
        """Find the repository when run from a checkout or a Colab session."""
        if "__file__" in globals():
            return Path(__file__).resolve().parents[1]
        override = os.environ.get("LICHESS_PROJECT_DIR")
        if override:
            return Path(override).expanduser().resolve()
        for candidate in (Path.cwd(), Path("/content/lichess-chess-games-big-data")):
            if (candidate / "src" / "lichess_pipeline").is_dir():
                return candidate
        return Path.cwd()

    PROJECT_DIR = _project_dir()
    if (PROJECT_DIR / "src" / "lichess_pipeline").is_dir():
        sys.path.insert(0, str(PROJECT_DIR / "src"))

    DATA_DIR = Path(
        os.environ.get("LICHESS_DATA_DIR", PROJECT_DIR / "data")
    ).expanduser()
    MODELS_DIR = Path(
        os.environ.get("LICHESS_MODELS_DIR", DATA_DIR / "models")
    ).expanduser()
    POSITIONS_PATH = Path(
        os.environ.get(
            "LICHESS_POSITIONS_PATH",
            Path(os.environ.get("LICHESS_PROCESSED_DIR", DATA_DIR / "processed"))
            / "positions_games_sample.parquet",
        )
    ).expanduser()

    import chess
    import chess.svg
    import marimo as mo
    import polars as pl

    from lichess_pipeline.model.checkpoints import (
        discover_policy_checkpoints,
        load_policy_checkpoint,
    )
    from lichess_pipeline.model.scratch_net import predict_top_moves
    load_cached = lru_cache(maxsize=None)(load_policy_checkpoint)
    return (
        MODELS_DIR,
        POSITIONS_PATH,
        PROJECT_DIR,
        chess,
        datetime,
        discover_policy_checkpoints,
        load_cached,
        mo,
        pl,
        predict_top_moves,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Prueba visual de los modelos de jugadas

    Detecta **todos** los archivos `.pt` y `.pth` dentro de `data/models/`,
    incluidas sus subcarpetas, y compara las jugadas de los checkpoints que
    elijas sin volver a entrenar.

    La tabla muestra cada archivo con su carpeta de origen. Usa el selector
    para decidir qué modelos entran en la comparación; solo esos se cargan en
    memoria.

    Hay dos formas de elegir la posición: escribir un **FEN** o recorrer una
    **partida guardada** del Parquet del cuaderno 1. En la segunda vista
    también se marca la jugada que se jugó realmente.

    **Aviso:** las posiciones de las partidas guardadas pueden haber
    participado en el entrenamiento. Sirven para inspección visual, no como
    una medida independiente de exactitud.
    """)
    return


@app.cell
def _(MODELS_DIR, PROJECT_DIR, datetime, discover_policy_checkpoints, mo, pl):
    checkpoints = discover_policy_checkpoints(MODELS_DIR)
    if checkpoints:
        checkpoint_table = pl.DataFrame(
            {
                "Checkpoint": list(checkpoints),
                "Archivo": [
                    str(path.relative_to(PROJECT_DIR)) for path in checkpoints.values()
                ],
                "Tamaño (MB)": [
                    round(path.stat().st_size / (1024 * 1024), 1)
                    for path in checkpoints.values()
                ],
                "Modificado": [
                    datetime.fromtimestamp(path.stat().st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    for path in checkpoints.values()
                ],
            }
        )
        checkpoint_view = mo.vstack(
            [
                mo.md("## Checkpoints encontrados"),
                mo.ui.table(checkpoint_table, pagination=False),
            ]
        )
    else:
        checkpoint_view = mo.callout(
            mo.md(
                "No se encontró ningún archivo `.pt` o `.pth` dentro de "
                f"`{MODELS_DIR}`. Guarda los modelos ahí o en una subcarpeta "
                "con el nombre que quieras."
            ),
            kind="warn",
        )
    checkpoint_view
    return (checkpoints,)


@app.cell
def _(POSITIONS_PATH, mo, pl):
    if POSITIONS_PATH.exists():
        positions_df = pl.read_parquet(POSITIONS_PATH)
        _positions_view = mo.md(
            f"{positions_df.height:,} posiciones cargadas desde "
            f"`{POSITIONS_PATH.name}`."
        )
    else:
        positions_df = pl.DataFrame(
            schema={
                "game_id": pl.String,
                "fen": pl.String,
                "move_uci": pl.String,
                "move_ply": pl.Int16,
                "avg_elo": pl.Float32,
            }
        )
        _positions_view = mo.callout(
            mo.md(
                "No existe "
                f"`{POSITIONS_PATH}`. Ejecuta el cuaderno 1 para generar la "
                "muestra de partidas; sin ese archivo solo se puede usar un FEN."
            ),
            kind="warn",
        )
    games_df = (
        positions_df.group_by("game_id")
        .agg(
            pl.len().alias("positions"),
            pl.col("move_ply").min().alias("first_ply"),
            pl.col("move_ply").max().alias("last_ply"),
            pl.col("avg_elo").mean().alias("avg_elo"),
        )
        .sort("game_id")
    )
    _tables = (
        [mo.ui.table(games_df.head(20), pagination=False)] if games_df.height else []
    )
    mo.vstack(
        [
            mo.md("## Partidas guardadas"),
            _positions_view,
            *_tables,
        ]
    )
    return games_df, positions_df


@app.cell
def _(checkpoints, chess, games_df, mo):
    if checkpoints:
        _labels = list(checkpoints)
        _default_model = next(
            (label for label in _labels if "lora" in label.lower()), _labels[0]
        )
        model_choice = mo.ui.dropdown(
            options=_labels,
            value=_default_model,
            label="Modelo cuya flecha se dibuja en el tablero",
        )
    else:
        model_choice = mo.ui.dropdown(
            options=["Sin checkpoints"],
            value="Sin checkpoints",
            label="Modelo cuya flecha se dibuja en el tablero",
        )

    source = mo.ui.radio(
        options=["FEN", "Partida guardada"],
        value="FEN",
        label="Origen de la posición",
    )
    presets = {
        "Personalizado (usar el FEN de abajo)": None,
        "Inicio": chess.STARTING_FEN,
        "Apertura italiana (1.e4 e5 2.Nf3 Nc6 3.Bc4)": (
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        ),
        "Apertura española (Ruy López)": (
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        ),
        "Siciliana Najdorf (6.Be3)": (
            "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N1B3/PPP2PPP/R2QKB1R b KQkq - 1 6"
        ),
        "Final de peones": "8/5k2/8/8/8/4K3/4P3/8 w - - 0 1",
        "Final de torres": "8/8/4k3/8/8/4K3/8/R7 w - - 0 1",
        "Peón a promoción": "8/1P3k2/8/8/8/8/6K1/8 w - - 0 1",
        "Mate en una (torre)": "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
    }
    preset = mo.ui.dropdown(
        options=list(presets),
        value="Inicio",
        label="Posición de ejemplo (solo para el origen FEN)",
    )
    fen_input = mo.ui.text_area(
        value=chess.STARTING_FEN,
        label="FEN escrito a mano (elige «Personalizado» para usarlo)",
        debounce=500,
    )

    if games_df.height:
        _game_options = {
            (
                f"{row['game_id']} · {row['positions']} posiciones · "
                f"Elo {row['avg_elo']:.0f}"
                if row["avg_elo"] is not None
                else f"{row['game_id']} · {row['positions']} posiciones"
            ): row["game_id"]
            for row in games_df.iter_rows(named=True)
        }
        _default_game = next(iter(_game_options))
    else:
        _game_options = {"Sin partidas guardadas": ""}
        _default_game = "Sin partidas guardadas"
    game_choice = mo.ui.dropdown(
        options=_game_options,
        value=_default_game,
        label="Partida guardada",
    )

    mo.vstack(
        [
            mo.md("## Posición y modelo"),
            mo.hstack([source, model_choice], justify="start"),
            mo.md("### FEN"),
            preset,
            fen_input,
        ]
    )
    return fen_input, game_choice, model_choice, preset, presets, source


@app.cell
def _(game_choice, games_df, mo, pl):
    _selected_game = games_df.filter(pl.col("game_id") == game_choice.value)
    if _selected_game.height:
        _first_ply = int(_selected_game["first_ply"][0])
        _last_ply = int(_selected_game["last_ply"][0])
    else:
        _first_ply = 1
        _last_ply = 1
    ply_slider = mo.ui.slider(
        start=_first_ply,
        stop=max(_first_ply, _last_ply),
        step=1,
        value=_first_ply,
        label="Posición dentro de la partida (ply)",
    )
    mo.vstack(
        [
            mo.md("### Partida guardada"),
            mo.hstack([game_choice, ply_slider], justify="start"),
        ]
    )
    return (ply_slider,)


@app.cell
def _(
    chess,
    fen_input,
    game_choice,
    mo,
    pl,
    ply_slider,
    positions_df,
    preset,
    presets,
    source,
):
    played_move = None
    if source.value == "Partida guardada":
        _frame = positions_df.filter(pl.col("game_id") == game_choice.value).sort(
            "move_ply"
        )
        _match = _frame.filter(pl.col("move_ply") == int(ply_slider.value))
        mo.stop(
            _match.height == 0,
            mo.callout(
                mo.md("Esa partida no tiene una posición en el ply seleccionado."),
                kind="warn",
            ),
        )
        _row = _match.row(0, named=True)
        _candidate = _row["fen"]
        played_move = _row["move_uci"]
    else:
        _selected_preset = presets[preset.value]
        _candidate = (
            _selected_preset
            if _selected_preset is not None
            else fen_input.value.strip()
        )

    mo.stop(
        not _candidate,
        mo.callout(
            mo.md("Escribe un FEN o elige un ejemplo de la lista."), kind="warn"
        ),
    )
    try:
        board = chess.Board(_candidate)
        _fen_error = None
    except ValueError as error:
        board = None
        _fen_error = str(error)
    mo.stop(
        _fen_error is not None,
        mo.callout(mo.md(f"FEN inválido: {_fen_error}"), kind="danger"),
    )
    mo.stop(
        not any(board.legal_moves),
        mo.callout(
            mo.md("La posición es terminal (mate o ahogado): no hay jugadas legales."),
            kind="warn",
        ),
    )
    fen = board.fen()
    return board, fen, played_move


@app.cell
def _(checkpoints, mo):
    if checkpoints:
        _labels = list(checkpoints)
        comparison_choice = mo.ui.multiselect(
            options=_labels,
            value=_labels[:3],
            label="Modelos en la tabla",
        )
    else:
        comparison_choice = mo.ui.multiselect(
            options=["Sin checkpoints"],
            value=[],
            label="Modelos en la tabla",
        )
    comparison_choice
    return (comparison_choice,)


@app.cell
def _(
    board,
    checkpoints,
    chess,
    comparison_choice,
    fen,
    load_cached,
    mo,
    model_choice,
    pl,
    played_move,
    predict_top_moves,
    source,
):
    compared = [label for label in comparison_choice.value if label in checkpoints]
    _to_load = list(dict.fromkeys([*compared, model_choice.value]))
    _to_load = [label for label in _to_load if label in checkpoints]
    mo.stop(
        not _to_load,
        mo.callout(
            mo.md(
                "No hay checkpoints disponibles. Guarda un `.pt` o `.pth` "
                "dentro de `data/models/` y recarga el cuaderno."
            ),
            kind="warn",
        ),
    )
    models = {}
    load_errors = []
    for label in _to_load:
        try:
            models[label] = load_cached(checkpoints[label])
        except (RuntimeError, FileNotFoundError) as error:
            load_errors.append(f"- **{label}**: {error}")
    mo.stop(
        not models,
        mo.callout(
            mo.md("Ningún checkpoint se pudo cargar:\n\n" + "\n".join(load_errors)),
            kind="danger",
        ),
    )
    predictions = {
        label: predict_top_moves(model, fen, top_k=3) for label, model in models.items()
    }
    table_labels = [label for label in compared if label in predictions]
    selected_label = model_choice.value
    if selected_label not in predictions:
        selected_label = next(iter(predictions))

    _arrows = []
    _best = chess.Move.from_uci(predictions[selected_label][0][0])
    _arrows.append(chess.svg.Arrow(_best.from_square, _best.to_square, color="#2a78d4"))
    if played_move:
        try:
            _played = chess.Move.from_uci(played_move)
        except ValueError:
            _played = None
        if _played is not None and board.is_legal(_played):
            _arrows.append(
                chess.svg.Arrow(_played.from_square, _played.to_square, color="#2e9e5b")
            )
    board_svg = chess.svg.board(board, arrows=_arrows, size=420, coordinates=True)

    _played_note = ""
    if played_move:
        _ranked = [move for move, _ in predictions[selected_label]]
        if played_move == _ranked[0]:
            _played_note = "fue la **primera** opción del modelo"
        elif played_move in _ranked:
            _played_note = (
                f"aparece en el top-3 (puesto {_ranked.index(played_move) + 1})"
            )
        else:
            _played_note = "no aparece en el top-3 del modelo"

    if table_labels:
        _rows = []
        for rank in range(3):
            row = {"#": rank + 1}
            for label in table_labels:
                moves = predictions[label]
                row[label] = (
                    f"{moves[rank][0]} · {100 * moves[rank][1]:.1f} %"
                    if rank < len(moves)
                    else "—"
                )
            _rows.append(row)
        _table_view = mo.ui.table(pl.DataFrame(_rows), pagination=False)
        _table_note = mo.md(
            "Cada columna muestra las tres jugadas legales más probables de un "
            "modelo. Las probabilidades se reparten solo entre jugadas legales."
        )
    else:
        _table_view = mo.callout(
            mo.md(
                "Selecciona al menos un modelo en "
                "«Modelos que aparecen en la tabla de comparación»."
            ),
            kind="warn",
        )
        _table_note = mo.md("")

    _board_column = [
        mo.Html(
            str(board_svg).replace(
                "<svg ", '<svg style="max-width:100%;height:auto" ', 1
            )
        ),
        mo.md(f"**Posición:** `{fen}`"),
        mo.md(f"**Flecha azul:** mejor jugada de *{selected_label}*"),
    ]
    if load_errors:
        _board_column.append(
            mo.callout(
                mo.md(
                    "Estos checkpoints no se pudieron cargar y se omiten:\n\n"
                    + "\n".join(load_errors)
                ),
                kind="warn",
            )
        )
    if source.value == "Partida guardada":
        _board_column.append(
            mo.md(
                f"**Jugada jugada:** `{played_move}` · {_played_note} "
                "(flecha verde cuando es legal)."
            )
        )

    mo.hstack(
        [
            mo.vstack(_board_column),
            mo.vstack([_table_view, _table_note]),
        ],
        justify="start",
        align="start",
        wrap=True,
    )
    return


if __name__ == "__main__":
    app.run()
