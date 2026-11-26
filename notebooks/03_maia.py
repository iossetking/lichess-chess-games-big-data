import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    def _project_dir() -> Path:
        """Find the repository when run from a checkout or a Colab session."""
        override = os.environ.get("LICHESS_PROJECT_DIR")
        if override:
            return Path(override).expanduser().resolve()
        candidates: list[Path] = []
        if "__file__" in globals():
            candidates.append(Path(__file__).resolve().parents[1])
        candidates.extend([Path.cwd(), Path("/content/lichess-chess-games-big-data")])
        for candidate in candidates:
            if (candidate / "src" / "lichess_pipeline").is_dir():
                return candidate
        return Path.cwd()

    PROJECT_DIR = _project_dir()
    if (PROJECT_DIR / "src" / "lichess_pipeline").is_dir():
        sys.path.insert(0, str(PROJECT_DIR / "src"))

    DATA_DIR = Path(
        os.environ.get("LICHESS_DATA_DIR", PROJECT_DIR / "data")
    ).expanduser()
    PROCESSED_DIR = Path(
        os.environ.get("LICHESS_PROCESSED_DIR", DATA_DIR / "processed")
    ).expanduser()
    MODELS_DIR = Path(
        os.environ.get("LICHESS_MODELS_DIR", DATA_DIR / "models")
    ).expanduser()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    import chess
    import chess.svg
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

    from lichess_pipeline.model.maia3_lora import (
        apply_lora,
        count_parameters,
        evaluate,
        load_maia3,
        make_maia_dataset,
        pick_device,
        predict_top_moves,
        save_adapter,
        train_lora,
    )
    from lichess_pipeline.pgn_parser import (
        clean_games_df,
        evals_url,
        extract_positions_df,
        games_sample_url,
        load_clean_evals_df,
        parse_pgn_sample_df,
    )

    MAIA_CACHE_DIR = MODELS_DIR / "maia3"

    POSITIONS_PATH = PROCESSED_DIR / "positions_games_sample.parquet"
    EVALS_PATH = PROCESSED_DIR / "evals_clean_sample.parquet"
    HUMAN_ADAPTER_PATH = MODELS_DIR / "maia3_lora_human.pt"
    STOCKFISH_ADAPTER_PATH = MODELS_DIR / "maia3_lora_stockfish.pt"
    SEED = 42

    def _env_int(name: str, fallback: int) -> int:
        try:
            return int(os.environ[name])
        except (KeyError, ValueError):
            return fallback

    def load_positions() -> pl.DataFrame:
        """Read the human-moves sample, or stream one from Lichess and cache it."""
        if POSITIONS_PATH.exists():
            return pl.read_parquet(POSITIONS_PATH)
        target = _env_int("LICHESS_MAIA_MAX_TRAIN", 8000)
        raw = parse_pgn_sample_df(
            games_sample_url(2025, 1), max_games=max(1200, target // 30)
        )
        frame = extract_positions_df(
            clean_games_df(raw, min_elo=1800), max_positions=target
        )
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(POSITIONS_PATH)
        return frame

    def load_evals() -> pl.DataFrame:
        """Read the Stockfish sample, or stream one from Lichess and cache it."""
        if EVALS_PATH.exists():
            return pl.read_parquet(EVALS_PATH)
        frame = load_clean_evals_df(
            evals_url(),
            max_evals=_env_int("LICHESS_MAIA_MAX_TRAIN", 8000),
            min_depth=20,
        )
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(EVALS_PATH)
        return frame

    def split_by_game(
        frame: pl.DataFrame, fraction: float = 0.1
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Return ``(train, validation, test)`` without sharing a game.

        Positions of one game are highly correlated, so whole games go to a
        single split; otherwise validation would contain near-copies of
        training rows.
        """
        games = (
            frame["game_id"]
            .unique(maintain_order=True)
            .sample(fraction=1.0, shuffle=True, seed=SEED)
        )
        size = max(1, int(fraction * len(games)))

        def rows_of(game_ids: pl.Series) -> pl.DataFrame:
            return frame.filter(pl.col("game_id").is_in(game_ids.implode()))

        return (
            rows_of(games.slice(2 * size)),
            rows_of(games.slice(0, size)),
            rows_of(games.slice(size, size)),
        )

    def split_rows(
        frame: pl.DataFrame, fraction: float = 0.1
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Return ``(train, validation, test)`` slices of a shuffled frame."""
        shuffled = frame.sample(fraction=1.0, shuffle=True, seed=SEED)
        size = max(1, int(fraction * shuffled.height))
        return (
            shuffled.slice(2 * size),
            shuffled.slice(0, size),
            shuffled.slice(size, size),
        )

    plt.rcParams.update(
        {"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100}
    )
    return (
        EVALS_PATH,
        HUMAN_ADAPTER_PATH,
        MAIA_CACHE_DIR,
        POSITIONS_PATH,
        STOCKFISH_ADAPTER_PATH,
        apply_lora,
        chess,
        count_parameters,
        evaluate,
        load_evals,
        load_maia3,
        load_positions,
        make_maia_dataset,
        mo,
        pick_device,
        pl,
        plt,
        predict_top_moves,
        save_adapter,
        split_by_game,
        split_rows,
        train_lora,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Maia-3 preentrenado + ajuste fino con LoRA

    **Enfoque 3.** En vez de entrenar una red desde cero (cuaderno 2), se parte
    de **Maia-3**, el modelo oficial del Computational Social Science Lab
    (Universidad de Toronto) entrenado con miles de millones de jugadas humanas
    de Lichess, y se ajusta con **LoRA**. Se crean **dos modelos**, cada uno a
    partir del mismo Maia-3 y con un dataset distinto del cuaderno 1:

    | Modelo | Dataset (`data/processed/`) | Aprende |
    |---|---|---|
    | **A · humano** | `positions_games_sample.parquet` | `FEN -> jugada` que hizo un humano de Elo alto |
    | **B · Stockfish** | `evals_clean_sample.parquet` | `FEN -> mejor jugada` según Stockfish |

    **Qué es LoRA.** Los pesos de Maia-3 se congelan. A cada capa lineal elegida
    se le suma una corrección de bajo rango `B·A` (dos matrices pequeñas) y solo
    esas matrices se entrenan. Con rango 8 son unos **100 mil parámetros
    entrenables frente a 5 millones**, y cada modelo ajustado se guarda en un
    archivo de menos de 1 MB. Aquí LoRA se aplica al MLP de los 8 bloques del
    transformer y a las dos proyecciones de la cabeza de jugadas.

    **Contrato del modelo**, igual que en el cuaderno 2: entra un FEN y sale una
    jugada UCI legal, porque las jugadas ilegales se enmascaran antes del
    `softmax`. Diferencias propias de Maia-3:

    - Ve el tablero **siempre desde el bando que mueve**: si mueven negras, el
      tablero y la jugada se reflejan.
    - Usa **4352 clases**: 4096 `origen -> destino` más 256 promociones, así que
      distingue coronar dama de coronar caballo.
    - Recibe el **Elo** de los jugadores. El modelo A usa el Elo medio de cada
      partida; el modelo B usa un Elo fijo, porque Stockfish no tiene rating.

    Este cuaderno funciona solo: si los Parquet no existen, genera una muestra
    pequeña desde `database.lichess.org`. La primera ejecución descarga los
    pesos de Maia-3 desde Hugging Face a `data/models/maia3/`.
    """)
    return


@app.cell
def _(EVALS_PATH, POSITIONS_PATH, load_evals, load_positions, mo, pl):
    positions_df = load_positions()
    evals_df = load_evals()
    _sources = pl.DataFrame(
        {
            "Modelo": ["A · humano", "B · Stockfish"],
            "Archivo": [POSITIONS_PATH.name, EVALS_PATH.name],
            "Columna objetivo": ["move_uci", "best_move"],
            "Posiciones": [positions_df.height, evals_df.height],
            "Partidas": [positions_df["game_id"].n_unique(), None],
        }
    )
    mo.vstack(
        [
            mo.md("## Datos de entrenamiento"),
            mo.ui.table(_sources, pagination=False),
            mo.md("#### Muestra del dataset humano"),
            mo.ui.table(positions_df.head(3), pagination=False),
            mo.md("#### Muestra del dataset de Stockfish"),
            mo.ui.table(evals_df.head(3), pagination=False),
        ]
    )
    return evals_df, positions_df


@app.cell
def _(mo):
    import os as _os_maia

    import torch as _torch_maia

    def _env_int(name: str, fallback: int) -> int:
        try:
            return int(_os_maia.environ[name])
        except (KeyError, ValueError):
            return fallback

    def _env_float(name: str, fallback: float) -> float:
        try:
            return float(_os_maia.environ[name])
        except (KeyError, ValueError):
            return fallback

    def _clamp(value: int, low: int, high: int) -> int:
        return max(low, min(high, value))

    _models = {"Maia-3 5M (rápido)": "maia3-5m", "Maia-3 23M": "maia3-23m"}
    _env_model = _os_maia.environ.get("LICHESS_MAIA_MODEL", "maia3-5m")
    _base_label = next(
        (label for label, name in _models.items() if name == _env_model),
        "Maia-3 5M (rápido)",
    )
    base_name = mo.ui.dropdown(
        options=_models,
        value=_base_label,
        label="Modelo base",
    )
    max_train = mo.ui.number(
        start=1000,
        stop=20_000_000,
        step=1000,
        value=_clamp(_env_int("LICHESS_MAIA_MAX_TRAIN", 20000), 1000, 20_000_000),
        label="Posiciones máximas por dataset",
    )
    epochs = mo.ui.slider(
        start=1,
        stop=20,
        step=1,
        value=_clamp(_env_int("LICHESS_MAIA_EPOCHS", 5), 1, 20),
        label="Épocas",
    )
    _rank_options = [2, 4, 8, 16, 32]
    rank = mo.ui.dropdown(
        options=_rank_options,
        value=(
            _env_int("LICHESS_MAIA_RANK", 8)
            if _env_int("LICHESS_MAIA_RANK", 8) in _rank_options
            else 8
        ),
        label="Rango LoRA",
    )
    _rates = {"1e-4": 1e-4, "2e-4": 2e-4, "5e-4": 5e-4, "1e-3": 1e-3}
    _env_rate = _env_float("LICHESS_MAIA_LR", 2e-4)
    _rate_label = min(_rates, key=lambda key: abs(_rates[key] - _env_rate))
    learning_rate = mo.ui.dropdown(
        options=_rates,
        value=_rate_label,
        label="Tasa de aprendizaje",
    )
    _batch_options = [128, 256, 512, 1024]
    _default_batch = 512 if _torch_maia.cuda.is_available() else 256
    _env_batch = _env_int("LICHESS_MAIA_BATCH_SIZE", _default_batch)
    batch_size = mo.ui.dropdown(
        options=_batch_options,
        value=_env_batch if _env_batch in _batch_options else _default_batch,
        label="Tamaño de lote (batch size)",
    )
    stockfish_elo = mo.ui.slider(
        start=1500,
        stop=3000,
        step=100,
        value=_clamp(_env_int("LICHESS_MAIA_ELO", 2500), 1500, 3000),
        label="Elo que se le indica a Maia para las posiciones de Stockfish",
    )
    train_button = mo.ui.run_button(label="Entrenar los dos modelos")
    mo.vstack(
        [
            mo.md("## Parámetros del ajuste fino"),
            mo.hstack([base_name, rank, learning_rate, batch_size], justify="start"),
            mo.hstack([max_train, epochs], justify="start"),
            stockfish_elo,
            train_button,
        ]
    )
    return (
        base_name,
        batch_size,
        epochs,
        learning_rate,
        max_train,
        rank,
        stockfish_elo,
        train_button,
    )


@app.cell
def _(
    base_name,
    batch_size,
    epochs,
    evals_df,
    learning_rate,
    make_maia_dataset,
    max_train,
    mo,
    pl,
    positions_df,
    rank,
    split_by_game,
    split_rows,
    stockfish_elo,
    train_button,
):
    # Run as a script (`uv run python notebooks/03_maia.py`) trains right away;
    # inside marimo it waits for the button.
    mo.stop(
        mo.running_in_notebook() and not train_button.value,
        mo.md("Pulsa **Entrenar los dos modelos** para empezar."),
    )
    mo.stop(
        positions_df["game_id"].n_unique() < 10 or evals_df.height < 100,
        mo.callout(
            mo.md(
                "Hacen falta al menos 10 partidas y 100 evaluaciones. Vuelve a "
                "ejecutar `01_eda_and_cleaning.py` con una muestra mayor."
            ),
            kind="warn",
        ),
    )
    # Every later cell reads this snapshot instead of the widgets, so changing
    # any control stops here until the button is pressed again.
    settings = {
        "base_name": base_name.value,
        "batch_size": int(batch_size.value),
        "rank": int(rank.value),
        "learning_rate": float(learning_rate.value),
        "epochs": int(epochs.value),
        "stockfish_elo": float(stockfish_elo.value),
    }
    _limit = int(max_train.value)
    _fixed_elo = settings["stockfish_elo"]
    _train, _validation, _test = split_by_game(positions_df.head(_limit))
    # Different games still share positions (openings above all). Those are
    # removed from validation and test so no evaluated FEN was seen in training.
    _seen = pl.col("fen").is_in(_train["fen"].implode())
    _repeated = _validation.filter(_seen).height + _test.filter(_seen).height
    _human_frames = (_train, _validation.filter(~_seen), _test.filter(~_seen))
    _stockfish_frames = split_rows(evals_df.head(_limit))

    human_train_ds, human_val_ds, human_test_ds = (
        make_maia_dataset(_frame, "fen", "move_uci", elo_col="avg_elo")[0]
        for _frame in _human_frames
    )
    stockfish_train_ds, stockfish_val_ds, stockfish_test_ds = (
        make_maia_dataset(_frame, "fen", "best_move", default_elo=_fixed_elo)[0]
        for _frame in _stockfish_frames
    )
    mo.callout(
        mo.md(
            "Entrenamiento / validación / prueba — "
            f"**humano:** {len(human_train_ds):,} / {len(human_val_ds):,} / "
            f"{len(human_test_ds):,} (partidas enteras en un solo conjunto; se "
            f"quitaron {_repeated:,} posiciones de validación y prueba que también "
            "aparecen en entrenamiento); "
            f"**Stockfish:** {len(stockfish_train_ds):,} / {len(stockfish_val_ds):,} / "
            f"{len(stockfish_test_ds):,}.\n\n"
            "La validación elige la mejor época; las métricas finales se miden "
            "en el conjunto de prueba, que el modelo nunca ve."
        ),
        kind="success",
    )
    return (
        human_test_ds,
        human_train_ds,
        human_val_ds,
        settings,
        stockfish_test_ds,
        stockfish_train_ds,
        stockfish_val_ds,
    )


@app.cell
def _(
    MAIA_CACHE_DIR,
    count_parameters,
    evaluate,
    human_test_ds,
    load_maia3,
    mo,
    pick_device,
    settings,
    stockfish_test_ds,
):
    device = pick_device()
    base_model = load_maia3(
        settings["base_name"], device=device, cache_dir=MAIA_CACHE_DIR
    )
    base_parameters = count_parameters(base_model)[1]
    base_metrics = {
        "humano": evaluate(base_model, human_test_ds, batch_size=settings["batch_size"]),
        "stockfish": evaluate(
            base_model, stockfish_test_ds, batch_size=settings["batch_size"]
        ),
    }
    mo.vstack(
        [
            mo.md("## Línea base: Maia-3 sin ajustar"),
            mo.callout(
                mo.md(
                    f"`{settings['base_name']}` cargado en **{device}** "
                    f"({base_parameters:,} parámetros). Sin ningún ajuste acierta "
                    f"**{100 * base_metrics['humano']['top1']:.1f} %** de las jugadas "
                    f"humanas y **{100 * base_metrics['stockfish']['top1']:.1f} %** de "
                    "las de Stockfish (Top-1 en prueba). Los dos modelos LoRA se "
                    "comparan contra estos números."
                ),
                kind="info",
            ),
        ]
    )
    return base_metrics, base_model, device


@app.cell
def _(
    MAIA_CACHE_DIR,
    apply_lora,
    count_parameters,
    device,
    load_maia3,
    settings,
    train_lora,
):
    def finetune(train_ds, val_ds):
        """Load a fresh Maia-3, add LoRA and train it on one dataset."""
        model = load_maia3(
            settings["base_name"], device=device, cache_dir=MAIA_CACHE_DIR
        )
        apply_lora(model, rank=settings["rank"], alpha=2.0 * settings["rank"])
        history = train_lora(
            model,
            train_ds,
            val_ds,
            epochs=settings["epochs"],
            lr=settings["learning_rate"],
            batch_size=settings["batch_size"],
            weight_decay=0.01,
        )
        return model, history, count_parameters(model)[0]

    return (finetune,)


@app.cell
def _(
    evaluate,
    finetune,
    human_test_ds,
    human_train_ds,
    human_val_ds,
    mo,
    pl,
    settings,
    stockfish_test_ds,
):
    human_model, _history, lora_parameters = finetune(human_train_ds, human_val_ds)
    human_frame = pl.DataFrame(_history)
    human_metrics = {
        "humano": evaluate(
            human_model, human_test_ds, batch_size=settings["batch_size"]
        ),
        "stockfish": evaluate(
            human_model, stockfish_test_ds, batch_size=settings["batch_size"]
        ),
    }
    mo.vstack(
        [
            mo.md("### Modelo A entrenado (LoRA con jugadas humanas)"),
            mo.md(
                f"Parámetros entrenables: **{lora_parameters:,}**. Se conserva la "
                f"época **{human_model.lora_best_epoch}** (la de menor pérdida de "
                "validación; 0 significa que ninguna época mejoró a Maia-3)."
            ),
            mo.ui.table(human_frame, pagination=False),
        ]
    )
    return human_frame, human_metrics, human_model, lora_parameters


@app.cell
def _(
    evaluate,
    finetune,
    human_test_ds,
    mo,
    pl,
    settings,
    stockfish_test_ds,
    stockfish_train_ds,
    stockfish_val_ds,
):
    stockfish_model, _history, _ = finetune(stockfish_train_ds, stockfish_val_ds)
    stockfish_frame = pl.DataFrame(_history)
    stockfish_metrics = {
        "humano": evaluate(
            stockfish_model, human_test_ds, batch_size=settings["batch_size"]
        ),
        "stockfish": evaluate(
            stockfish_model, stockfish_test_ds, batch_size=settings["batch_size"]
        ),
    }
    mo.vstack(
        [
            mo.md("### Modelo B entrenado (LoRA con jugadas de Stockfish)"),
            mo.md(
                f"Parte otra vez del Maia-3 original, no del modelo A. Se conserva "
                f"la época **{stockfish_model.lora_best_epoch}**."
            ),
            mo.ui.table(stockfish_frame, pagination=False),
        ]
    )
    return stockfish_frame, stockfish_metrics, stockfish_model


@app.cell
def _(
    base_metrics,
    base_model,
    count_parameters,
    human_frame,
    human_metrics,
    lora_parameters,
    mo,
    pl,
    plt,
    stockfish_frame,
    stockfish_metrics,
):
    _total = count_parameters(base_model)[1]
    _models = [
        ("Maia-3 sin ajustar", base_metrics, 0),
        ("A · LoRA humano", human_metrics, lora_parameters),
        ("B · LoRA Stockfish", stockfish_metrics, lora_parameters),
    ]
    comparison = pl.DataFrame(
        [
            {
                "Modelo": _name,
                "Prueba": _label,
                "Top-1 (%)": round(100 * _metrics[_key]["top1"], 2),
                "Top-3 (%)": round(100 * _metrics[_key]["top3"], 2),
                "Pérdida": round(_metrics[_key]["loss"], 4),
                "Parámetros entrenados": _trained,
            }
            for _name, _metrics, _trained in _models
            for _key, _label in (
                ("humano", "Jugadas humanas"),
                ("stockfish", "Stockfish"),
            )
        ]
    )

    _figure, _axes = plt.subplots(2, 2, figsize=(13, 8))
    for _row, (_title, _frame) in enumerate(
        (("Modelo A (humano)", human_frame), ("Modelo B (Stockfish)", stockfish_frame))
    ):
        _axes[_row, 0].plot(
            _frame["epoch"], _frame["train_loss"], marker="o", label="entrenamiento"
        )
        _axes[_row, 0].plot(
            _frame["epoch"], _frame["val_loss"], marker="s", label="validación"
        )
        _axes[_row, 0].set(
            title=f"{_title}: pérdida", xlabel="Época", ylabel="Cross-entropy"
        )
        _axes[_row, 0].legend()
        _axes[_row, 1].plot(
            _frame["epoch"], _frame["val_top1"], marker="o", label="Top-1"
        )
        _axes[_row, 1].plot(
            _frame["epoch"], _frame["val_top3"], marker="s", label="Top-3"
        )
        _axes[_row, 1].set(
            title=f"{_title}: exactitud en validación",
            xlabel="Época",
            ylabel="Fracción",
            ylim=(0, 1),
        )
        _axes[_row, 1].legend()
    _figure.tight_layout()

    _bars, _bar_axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    _names = [_name for _name, _, _ in _models]
    for _axis, (_key, _label) in zip(
        _bar_axes,
        (("humano", "jugadas humanas"), ("stockfish", "Stockfish")),
        strict=True,
    ):
        _values = [100 * _metrics[_key]["top1"] for _, _metrics, _ in _models]
        _drawn = _axis.bar(_names, _values, color=["#8a8a8a", "#2a78d4", "#d1603d"])
        _axis.bar_label(_drawn, fmt="%.1f")
        _axis.set(title=f"Top-1 en prueba: {_label}", ylabel="% de aciertos")
        _axis.tick_params(axis="x", rotation=12)
    _bars.tight_layout()

    _delta_a = 100 * (human_metrics["humano"]["top1"] - base_metrics["humano"]["top1"])
    _delta_b = 100 * (
        stockfish_metrics["stockfish"]["top1"] - base_metrics["stockfish"]["top1"]
    )
    mo.vstack(
        [
            mo.md("## Resultados en el conjunto de prueba"),
            mo.callout(
                mo.md(
                    f"Respecto a Maia-3 sin ajustar: el modelo A cambia "
                    f"**{_delta_a:+.2f}** puntos de Top-1 en jugadas humanas y el "
                    f"modelo B **{_delta_b:+.2f}** puntos en jugadas de Stockfish. "
                    f"Cada modelo entrena {lora_parameters:,} de {_total:,} parámetros "
                    f"({100 * lora_parameters / _total:.1f} %).\n\n"
                    "Cómo leerlo: Maia-3 ya se entrenó con miles de millones de "
                    "jugadas humanas, así que unas miles más apenas pueden moverlo "
                    "(modelo A). Stockfish sí es una señal nueva para él (modelo B). "
                    "Con muestras de este tamaño, diferencias de alrededor de un "
                    "punto están dentro del ruido estadístico; la ganancia real se "
                    "mide al repetirlo con más posiciones."
                ),
                kind="info",
            ),
            mo.ui.table(comparison, pagination=False),
            _bars,
            _figure,
        ]
    )
    return


@app.cell
def _(
    HUMAN_ADAPTER_PATH,
    STOCKFISH_ADAPTER_PATH,
    human_model,
    mo,
    save_adapter,
    settings,
    stockfish_model,
):
    save_adapter(
        human_model, HUMAN_ADAPTER_PATH, settings["base_name"], dataset="human"
    )
    save_adapter(
        stockfish_model,
        STOCKFISH_ADAPTER_PATH,
        settings["base_name"],
        dataset="stockfish",
        elo=settings["stockfish_elo"],
    )
    mo.callout(
        mo.md(
            "Adaptadores LoRA guardados (solo las matrices de bajo rango; el "
            "Maia-3 base se vuelve a descargar o leer de caché al cargarlos):\n\n"
            f"- `{HUMAN_ADAPTER_PATH}` ({HUMAN_ADAPTER_PATH.stat().st_size / 1024:.0f} KB)\n"
            f"- `{STOCKFISH_ADAPTER_PATH}` ({STOCKFISH_ADAPTER_PATH.stat().st_size / 1024:.0f} KB)"
        ),
        kind="success",
    )
    return


@app.cell
def _(chess, mo):
    presets = {
        "Personalizado (usar el FEN de abajo)": None,
        "Inicio": chess.STARTING_FEN,
        "Italiana tras 1.e4 e5 2.Nf3 Nc6 3.Bc4": (
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        ),
        "Final de peones": "8/5k2/8/8/8/4K3/4P3/8 w - - 0 1",
        "Coronación de peón": "8/4P1k1/8/8/8/8/8/4K3 w - - 0 1",
    }
    preset = mo.ui.dropdown(
        options=list(presets),
        value="Personalizado (usar el FEN de abajo)",
        label="Posición de ejemplo",
    )
    fen_input = mo.ui.text_area(
        value=chess.STARTING_FEN,
        label="FEN escrito a mano",
        debounce=500,
    )
    predict_elo = mo.ui.slider(
        start=1100, stop=3000, step=100, value=2500, label="Elo indicado al modelo"
    )
    mo.vstack(
        [
            mo.md("## Predicción interactiva"),
            mo.md(
                "Escribe cualquier FEN o elige un ejemplo (el ejemplo tiene "
                "prioridad). Se muestran las tres jugadas legales más probables "
                "de Maia-3 sin ajustar y de los dos modelos LoRA. El modelo B se "
                "entrenó con un Elo fijo (2500 por defecto); con otro Elo se le "
                "pregunta fuera de lo que vio al entrenar."
            ),
            preset,
            fen_input,
            predict_elo,
        ]
    )
    return fen_input, predict_elo, preset, presets


@app.cell
def _(
    base_model,
    chess,
    fen_input,
    human_model,
    mo,
    pl,
    predict_elo,
    predict_top_moves,
    preset,
    presets,
    stockfish_model,
):
    _selected = presets[preset.value]
    _fen = _selected if _selected is not None else fen_input.value.strip()
    try:
        _board = chess.Board(_fen)
        _fen_error = None
    except ValueError as _error:
        _board = None
        _fen_error = str(_error)
    mo.stop(
        _fen_error is not None,
        mo.callout(mo.md(f"FEN inválido: {_fen_error}"), kind="danger"),
    )
    mo.stop(
        not any(_board.legal_moves),
        mo.callout(
            mo.md("La posición es terminal (mate o ahogado): no hay jugadas legales."),
            kind="warn",
        ),
    )

    _elo = float(predict_elo.value)
    _columns = {
        "Maia-3 sin ajustar": predict_top_moves(base_model, _fen, top_k=3, elo=_elo),
        "A · LoRA humano": predict_top_moves(human_model, _fen, top_k=3, elo=_elo),
        "B · LoRA Stockfish": predict_top_moves(
            stockfish_model, _fen, top_k=3, elo=_elo
        ),
    }
    _best = chess.Move.from_uci(_columns["B · LoRA Stockfish"][0][0])
    _svg = chess.svg.board(
        _board,
        arrows=[chess.svg.Arrow(_best.from_square, _best.to_square)],
        size=380,
        coordinates=True,
    )
    _rows = [
        {
            "#": _position + 1,
            **{
                _name: f"{_moves[_position][0]} · {100 * _moves[_position][1]:.1f} %"
                for _name, _moves in _columns.items()
            },
        }
        for _position in range(min(len(_moves) for _moves in _columns.values()))
    ]
    mo.hstack(
        [
            mo.Html(str(_svg)),
            mo.vstack(
                [
                    mo.md(f"**Posición:** `{_fen}`"),
                    mo.md(
                        f"**Mejor jugada del modelo B (Stockfish):** `{_best.uci()}`"
                    ),
                    mo.ui.table(pl.DataFrame(_rows), pagination=False),
                    mo.md(
                        "Todas las jugadas mostradas son legales. La flecha marca "
                        "la primera opción del modelo B."
                    ),
                ]
            ),
        ],
        justify="start",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notas

    - **Licencias.** El código de Maia-3 es AGPL-3.0; la licencia de sus pesos
      no está declarada en el repositorio. Hay que citarlo en el reporte.
    - **Historia.** Maia-3 recibe las ocho últimas posiciones. Los datasets
      solo guardan el FEN actual, así que se repite ocho veces (lo mismo que
      hace Maia-3 cuando no conoce la partida).
    - **Qué no se ajusta.** Las matrices Q/K/V de la atención van fusionadas
      dentro de `nn.MultiheadAttention`; LoRA se aplica a los MLP y a la cabeza
      de jugadas.
    - **Escala.** Para usar más datos basta subir los límites del cuaderno 1 y
      el control «Posiciones máximas por dataset». Cada posición ocupa unos
      ~1.3 KB en memoria (máscaras empaquetadas en bits).
    """)
    return


if __name__ == "__main__":
    app.run()
