import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    from pathlib import Path

    import chess
    import chess.svg
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

    from lichess_pipeline.model.checkpoints import (
        compact_state_dict,
        load_policy_checkpoint,
        save_policy_checkpoint,
    )
    from lichess_pipeline.model.scratch_lora import (
        apply_head_lora,
        count_trainable_parameters,
        merge_head_lora,
        train_head_lora,
    )
    from lichess_pipeline.model.scratch_net import (
        ChessPolicyNet,
        evaluate_model,
        make_tensor_dataset,
        predict_top_moves,
        train_model,
    )
    from lichess_pipeline.model.splits import kfold, split_holdout
    from lichess_pipeline.pgn_parser import (
        clean_games_df,
        evals_url,
        extract_positions_df,
        games_sample_url,
        load_clean_evals_df,
        parse_pgn_sample_df,
    )

    PROJECT_DIR = Path(__file__).resolve().parents[1]
    PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
    MODELS_DIR = PROJECT_DIR / "data" / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    POSITIONS_PATH = PROCESSED_DIR / "positions_games_sample.parquet"
    EVALS_PATH = PROCESSED_DIR / "evals_clean_sample.parquet"

    def load_positions() -> pl.DataFrame:
        """Read the Stage 1 sample, or build a small one from Lichess directly."""
        if POSITIONS_PATH.exists():
            return pl.read_parquet(POSITIONS_PATH)
        raw = parse_pgn_sample_df(games_sample_url(2025, 1), max_games=1200)
        return extract_positions_df(
            clean_games_df(raw, min_elo=1800), max_positions=8000
        )

    def load_evals() -> pl.DataFrame:
        """Read the Stage 2 sample, or build a small one from Lichess directly."""
        if EVALS_PATH.exists():
            return pl.read_parquet(EVALS_PATH)
        return load_clean_evals_df(evals_url(), max_evals=8000, min_depth=20)

    plt.rcParams.update(
        {"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100}
    )
    return (
        ChessPolicyNet,
        EVALS_PATH,
        MODELS_DIR,
        POSITIONS_PATH,
        apply_head_lora,
        chess,
        compact_state_dict,
        copy,
        count_trainable_parameters,
        evaluate_model,
        kfold,
        load_evals,
        load_policy_checkpoint,
        load_positions,
        make_tensor_dataset,
        merge_head_lora,
        mo,
        pl,
        plt,
        predict_top_moves,
        save_policy_checkpoint,
        split_holdout,
        train_head_lora,
        train_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Modelo de mejor jugada: desde cero y dos ajustes finos

    Un único punto de partida (**Etapa 1**) y dos formas de especializarlo con
    Stockfish (**Etapa 2**). Los tres modelos se guardan por separado:

    | Modelo | Qué entrena | Archivo |
    |---|---|---|
    | **Desde cero** | Red convolucional con jugadas humanas de Lichess | `approach1_scratch.pt` |
    | **Ajuste de pesos** | Todos los pesos, con `FEN -> mejor jugada` de Stockfish | `approach1_weight_ftuning.pt` |
    | **Ajuste LoRA** | Adaptadores de bajo rango en el tronco residual y la cabeza | `approach1_lora_ftuning.pt` |

    **Por qué dos ajustes finos.** Con una muestra pequeña, actualizar todos los
    pesos puede deshacer lo aprendido en la etapa 1 y bajar el Top-1. LoRA
    congela la red y entrena únicamente correcciones de bajo rango `B·A` en las
    convoluciones del tronco, las proyecciones de la cabeza y su sesgo de
    geometría: unos 86 mil parámetros (≈7 % de la red). Así el modelo ajustado
    sigue siendo casi idéntico al de partida y solo se aparta de él en la
    dirección que mejora la validación.
    Además se conserva la época con la menor pérdida de validación, no la última.

    **Cómo leer las comparaciones.** La exactitud de la etapa 1 se mide con
    jugadas humanas y la de la etapa 2 con jugadas de Stockfish: son conjuntos
    distintos y sus porcentajes no se comparan entre sí. Las comparaciones
    válidas son las que usan la misma validación de Stockfish, es decir, la tabla
    final y los dos ajustes finos entre ellos.

    **Partición de los datos.** Primero se aparta un **20 % de prueba** que no se
    toca hasta el final: se usa para comprobar si el ajuste fino realmente
    generaliza. El 80 % restante (desarrollo) se reparte en **5 particiones de
    validación cruzada**: cada modelo se entrena 5 veces sobre 4/5 del desarrollo
    y se valida en el quinto restante, y se reporta media ± desviación. La etapa 1
    agrupa por `game_id` (todas las posiciones de una partida caen en el mismo
    lado, porque son casi copias entre sí); la etapa 2 no trae `game_id` y reparte
    filas. Después de la validación cruzada, cada modelo se reajusta con **todo**
    el desarrollo y se evalúa una sola vez en el conjunto de prueba.

    La validación cruzada multiplica el coste: cada pasada entrena los tres
    modelos. Con el dataset completo puede tardar bastante; baja las épocas o el
    número de particiones para una prueba rápida.

    Entra una cadena FEN y sale una jugada UCI. Las jugadas ilegales se enmascaran
    antes del `softmax`, así que la salida es legal por construcción: el modelo
    solo puede equivocarse de *elección*, no de *legalidad*. El tablero se codifica
    como `13 x 8 x 8` (6 planos de piezas blancas, 6 de negras y 1 para el turno) y
    la jugada como una de `64 x 64 = 4096` clases `origen -> destino`.

    Limitación: los enroques y la captura al paso del FEN no se codifican como
    planos. La máscara de legalidad los usa al predecir, pero el modelo no aprende
    a preferirlos. Añadir dos planos es una mejora futura.

    Este cuaderno funciona solo: si los Parquet no existen, genera una muestra
    pequeña directamente desde `database.lichess.org`.
    """)
    return


@app.cell
def _(EVALS_PATH, POSITIONS_PATH, load_evals, load_positions, mo, pl):
    positions_df = load_positions()
    evals_df = load_evals()
    sources = pl.DataFrame(
        {
            "Fuente": ["Etapa 1 · jugadas humanas", "Etapa 2 · Stockfish"],
            "Archivo": [str(POSITIONS_PATH), str(EVALS_PATH)],
            "Filas": [positions_df.height, evals_df.height],
            "Columnas": [positions_df.width, evals_df.width],
        }
    )
    mo.vstack(
        [
            mo.md("## Datos de entrenamiento"),
            mo.ui.table(sources, pagination=False),
            mo.md("#### Muestra de la etapa 1"),
            mo.ui.table(positions_df.head(3), pagination=False),
            mo.md("#### Muestra de la etapa 2"),
            mo.ui.table(evals_df.head(3), pagination=False),
        ]
    )
    return evals_df, positions_df


@app.cell
def _(evals_df, mo, positions_df):
    import os

    def _env_int(name: str, fallback: int) -> int:
        try:
            return int(os.environ[name])
        except (KeyError, ValueError):
            return fallback

    def _clamp(value: int, low: int, high: int) -> int:
        return max(low, min(high, value))

    def _step_for(total: int) -> int:
        for candidate in (100, 200, 500, 1000, 2000, 5000, 10000, 20000):
            if total / candidate <= 50:
                return candidate
        return max(1, total // 50)

    _human_total = positions_df.height
    _eval_total = evals_df.height
    _human_step = _step_for(_human_total)
    _eval_step = _step_for(_eval_total)
    _human_start = min(_human_step, _human_total) or 1
    _eval_start = min(_eval_step, _eval_total) or 1

    max_human_positions = mo.ui.slider(
        start=_human_start,
        stop=max(1, _human_total),
        step=_human_step,
        value=_clamp(
            _env_int("LICHESS_MAX_HUMAN", min(40000, _human_total)), _human_start, max(1, _human_total)
        ),
        label=f"Posiciones máximas etapa 1 (humanos · {_human_total:,})",
    )
    epochs_stage1 = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=_clamp(_env_int("LICHESS_EPOCHS_STAGE1", 5), 1, 10),
        label="Épocas etapa 1 (humanos)",
    )
    max_eval_positions = mo.ui.slider(
        start=_eval_start,
        stop=max(1, _eval_total),
        step=_eval_step,
        value=_clamp(
            _env_int("LICHESS_MAX_EVAL", min(40000, _eval_total)), _eval_start, max(1, _eval_total)
        ),
        label=f"Posiciones máximas etapa 2 (Stockfish · {_eval_total:,})",
    )
    epochs_stage2 = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=_clamp(_env_int("LICHESS_EPOCHS_STAGE2", 5), 1, 10),
        label="Épocas ajuste de pesos (Stockfish)",
    )
    epochs_lora = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=_clamp(_env_int("LICHESS_EPOCHS_LORA", 5), 1, 10),
        label="Épocas ajuste LoRA (Stockfish)",
    )
    _rank_options = [4, 8, 16, 32]
    lora_rank = mo.ui.dropdown(
        options=_rank_options,
        value=(
            _env_int("LICHESS_LORA_RANK", 8)
            if _env_int("LICHESS_LORA_RANK", 8) in _rank_options
            else 8
        ),
        label="Rango LoRA",
    )
    _fold_options = {"3 particiones": 3, "5 particiones": 5, "10 particiones": 10}
    _env_folds = _env_int("LICHESS_N_FOLDS", 5)
    n_folds = mo.ui.dropdown(
        options=_fold_options,
        value=f"{_env_folds} particiones" if _env_folds in _fold_options.values() else "5 particiones",
        label="Validación cruzada",
    )
    cv_weight = mo.ui.checkbox(
        value=bool(_env_int("LICHESS_CV_WEIGHT", 1)), label="Validar ajuste de pesos"
    )
    cv_lora = mo.ui.checkbox(
        value=bool(_env_int("LICHESS_CV_LORA", 1)), label="Validar ajuste LoRA"
    )
    train_button = mo.ui.run_button(label="Guardar parametros")
    mo.vstack(
        [
            mo.md("## Parametros de Entrenamiento"),
            mo.hstack([max_human_positions, epochs_stage1], justify="start"),
            mo.hstack([max_eval_positions, epochs_stage2], justify="start"),
            mo.hstack([epochs_lora, lora_rank], justify="start"),
            mo.hstack([n_folds, cv_weight, cv_lora], justify="start"),
            train_button,
        ]
    )
    return (
        cv_lora,
        cv_weight,
        epochs_lora,
        epochs_stage1,
        epochs_stage2,
        lora_rank,
        max_eval_positions,
        max_human_positions,
        n_folds,
        train_button,
    )


@app.cell
def _(
    evals_df,
    make_tensor_dataset,
    max_eval_positions,
    max_human_positions,
    mo,
    positions_df,
    split_holdout,
    train_button,
):
    mo.stop(
        mo.running_in_notebook() and not train_button.value,
        mo.md("Pulsa **Guardar parametros**."),
    )
    mo.stop(
        positions_df.height < 20 or evals_df.height < 20,
        mo.callout(
            mo.md(
                "Se necesitan al menos 20 posiciones en cada fuente para entrenar. "
                "Vuelve a ejecutar `01_eda_and_cleaning.py` con una muestra mayor."
            ),
            kind="warn",
        ),
    )
    mo.stop(
        positions_df["game_id"].n_unique() < 4,
        mo.callout(
            mo.md(
                "La validación agrupa partidas completas, así que la muestra de "
                "jugadas humanas necesita al menos 4 partidas distintas."
            ),
            kind="warn",
        ),
    )
    human_limit = int(max_human_positions.value)
    eval_limit = int(max_eval_positions.value)
    human_sample = positions_df.head(min(human_limit, positions_df.height))
    eval_sample = evals_df.head(min(eval_limit, evals_df.height))
    human_test_df, human_dev_df = split_holdout(
        human_sample, fraction=0.2, group_col="game_id"
    )
    eval_test_df, eval_dev_df = split_holdout(eval_sample, fraction=0.2)
    human_dev_ds = make_tensor_dataset(human_dev_df)
    eval_dev_ds = make_tensor_dataset(eval_dev_df, "fen", "best_move")
    human_test_ds = make_tensor_dataset(human_test_df)
    eval_test_ds = make_tensor_dataset(eval_test_df, "fen", "best_move")
    mo.callout(
        mo.md(
            f"**Prueba (20 %, reservado hasta el final):** "
            f"{human_test_df.height:,} posiciones humanas de "
            f"{human_test_df['game_id'].n_unique():,} partidas y "
            f"{eval_test_df.height:,} posiciones de Stockfish.\n\n"
            f"**Desarrollo (80 %):** {human_dev_df.height:,} posiciones humanas de "
            f"{human_dev_df['game_id'].n_unique():,} partidas y "
            f"{eval_dev_df.height:,} de Stockfish, repartidas en particiones de "
            "validación cruzada."
        ),
        kind="success",
    )
    return (
        eval_dev_df,
        eval_dev_ds,
        eval_test_ds,
        human_dev_df,
        human_dev_ds,
        human_test_ds,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validación cruzada

    Cada modelo tiene su propia celda. El modelo **desde cero** siempre se
    entrena en cada partición, porque es la base que necesitan los dos ajustes
    finos; los pesos de cada partición se conservan en memoria (a media
    precisión) y no se escriben en disco. Los dos
    ajustes finos se pueden desactivar por separado con las casillas de los
    parámetros: así se entrena y valida solo lo que interesa.
    """)
    return


@app.cell
def _(eval_dev_df, human_dev_df, kfold, make_tensor_dataset, mo, n_folds, pl):
    _folds = int(n_folds.value)
    mo.stop(
        min(human_dev_df["game_id"].n_unique(), eval_dev_df.height) < _folds,
        mo.callout(
            mo.md(
                f"No hay suficientes partidas para {_folds} particiones "
                f"(desarrollo: {human_dev_df['game_id'].n_unique():,} partidas "
                f"humanas y {eval_dev_df.height:,} posiciones de Stockfish). Baja "
                "el número de particiones o sube las posiciones máximas."
            ),
            kind="warn",
        ),
    )
    human_cv_folds = kfold(human_dev_df, n_folds=_folds, group_col="game_id")
    eval_cv_folds = kfold(eval_dev_df, n_folds=_folds)
    human_monitor_ds = make_tensor_dataset(human_cv_folds[0][1])
    eval_monitor_ds = make_tensor_dataset(
        eval_cv_folds[0][1], "fen", "best_move"
    )

    def cv_row(fold, model, evaluation, metrics, best_epoch):
        return {
            "Partición": fold,
            "Modelo": model,
            "Evaluación": evaluation,
            "Pérdida": metrics["loss"],
            "Top-1": metrics["top1"],
            "Top-3": metrics["top3"],
            "Mejor época": best_epoch,
        }

    def mean_history(frames):
        _stacked = pl.concat(
            [
                frame.with_columns(pl.lit(index + 1).alias("Partición"))
                for index, frame in enumerate(frames)
            ]
        )
        return (
            _stacked.group_by("epoch")
            .agg(
                pl.col("train_loss").mean(),
                pl.col("val_loss").mean(),
                pl.col("val_top1").mean(),
                pl.col("val_top3").mean(),
            )
            .sort("epoch")
        )

    mo.md(
        f"El desarrollo se reparte en **{_folds}** particiones. La primera "
        "partición de validación también se usa para monitorizar los reajustes "
        "finales."
    )
    return (
        cv_row,
        eval_cv_folds,
        eval_monitor_ds,
        human_cv_folds,
        human_monitor_ds,
        mean_history,
    )


@app.cell
def _(
    ChessPolicyNet,
    compact_state_dict,
    cv_row,
    epochs_stage1,
    eval_cv_folds,
    evaluate_model,
    human_cv_folds,
    make_tensor_dataset,
    mean_history,
    mo,
    pl,
    plt,
    train_model,
):
    _rows = []
    _histories = []
    scratch_fold_states = []
    for _fold, ((_h_train, _h_val), (_e_train, _e_val)) in enumerate(
        zip(human_cv_folds, eval_cv_folds, strict=True), start=1
    ):
        _h_val_ds = make_tensor_dataset(_h_val)
        _e_val_ds = make_tensor_dataset(_e_val, "fen", "best_move")
        _model = ChessPolicyNet()
        _history = train_model(
            _model,
            make_tensor_dataset(_h_train),
            _h_val_ds,
            epochs=int(epochs_stage1.value),
            lr=1e-3,
            batch_size=128,
        )
        _histories.append(pl.DataFrame(_history))
        # Kept in memory (half precision, like the saved checkpoints) so the
        # fine-tuning cells can reuse them without writing data/models/cv/.
        scratch_fold_states.append(compact_state_dict(_model.state_dict()))
        _rows.append(
            cv_row(
                _fold,
                "Desde cero",
                "Validación humana",
                evaluate_model(_model, _h_val_ds, batch_size=128),
                int(epochs_stage1.value),
            )
        )
        _rows.append(
            cv_row(
                _fold,
                "Desde cero",
                "Validación Stockfish",
                evaluate_model(_model, _e_val_ds, batch_size=128),
                int(epochs_stage1.value),
            )
        )
    scratch_cv_frame = pl.DataFrame(_rows)
    scratch_cv_curve = mean_history(_histories)
    _human_rows = scratch_cv_frame.filter(
        pl.col("Evaluación") == "Validación humana"
    )

    _table = scratch_cv_frame.select(
        "Partición",
        "Evaluación",
        pl.col("Pérdida").round(4).alias("Pérdida"),
        (100 * pl.col("Top-1")).round(2).alias("Top-1 (%)"),
        (100 * pl.col("Top-3")).round(2).alias("Top-3 (%)"),
    )
    _figure, _axes = plt.subplots(1, 2, figsize=(13, 3.8))
    _axes[0].plot(
        scratch_cv_curve["epoch"],
        scratch_cv_curve["train_loss"],
        marker="o",
        label="entrenamiento",
    )
    _axes[0].plot(
        scratch_cv_curve["epoch"],
        scratch_cv_curve["val_loss"],
        marker="s",
        label="validación",
    )
    _axes[0].set(
        title="Desde cero: pérdida media", xlabel="Época", ylabel="Cross-entropy"
    )
    _axes[0].legend()
    _axes[1].plot(
        scratch_cv_curve["epoch"], scratch_cv_curve["val_top1"], marker="o", label="Top-1"
    )
    _axes[1].plot(
        scratch_cv_curve["epoch"], scratch_cv_curve["val_top3"], marker="s", label="Top-3"
    )
    _axes[1].set(
        title="Desde cero: exactitud media",
        xlabel="Época",
        ylabel="Fracción",
        ylim=(0, 1),
    )
    _axes[1].legend()
    _figure.tight_layout()

    mo.vstack(
        [
            mo.md("### Desde cero"),
            mo.callout(
                mo.md(
                    "Media en validación humana: Top-1 "
                    f"**{100 * _human_rows['Top-1'].mean():.2f} %** "
                    f"(± {100 * _human_rows['Top-1'].std():.2f}) y Top-3 "
                    f"**{100 * _human_rows['Top-3'].mean():.2f} %** "
                    f"(± {100 * _human_rows['Top-3'].std():.2f}). Los pesos de cada "
                    "partición se conservan en memoria (a media precisión) para "
                    "los ajustes finos."
                ),
                kind="info",
            ),
            mo.ui.table(_table, pagination=False),
            _figure,
        ]
    )
    return (scratch_fold_states,)


@app.cell
def _(
    ChessPolicyNet,
    cv_row,
    cv_weight,
    epochs_stage2,
    eval_cv_folds,
    evaluate_model,
    make_tensor_dataset,
    mean_history,
    mo,
    pl,
    plt,
    scratch_fold_states,
    train_model,
):
    mo.stop(
        not cv_weight.value,
        mo.callout(
            mo.md(
                "Ajuste de pesos desactivado en la validación cruzada. Actívalo en "
                "los parámetros para entrenarlo."
            ),
            kind="info",
        ),
    )
    _rows = []
    _histories = []
    for _fold, ((_e_train, _e_val), _fold_state) in enumerate(
        zip(eval_cv_folds, scratch_fold_states, strict=True), start=1
    ):
        _e_val_ds = make_tensor_dataset(_e_val, "fen", "best_move")
        _model = ChessPolicyNet()
        _model.load_state_dict(_fold_state)
        _history = train_model(
            _model,
            make_tensor_dataset(_e_train, "fen", "best_move"),
            _e_val_ds,
            epochs=int(epochs_stage2.value),
            lr=1e-4,
            batch_size=128,
        )
        _histories.append(pl.DataFrame(_history))
        _rows.append(
            cv_row(
                _fold,
                "Ajuste de pesos",
                "Validación Stockfish",
                evaluate_model(_model, _e_val_ds, batch_size=128),
                int(epochs_stage2.value),
            )
        )
    weight_cv_frame = pl.DataFrame(_rows)
    weight_cv_curve = mean_history(_histories)

    _table = weight_cv_frame.select(
        "Partición",
        pl.col("Pérdida").round(4).alias("Pérdida"),
        (100 * pl.col("Top-1")).round(2).alias("Top-1 (%)"),
        (100 * pl.col("Top-3")).round(2).alias("Top-3 (%)"),
    )
    _figure, _axes = plt.subplots(1, 2, figsize=(13, 3.8))
    _axes[0].plot(
        weight_cv_curve["epoch"],
        weight_cv_curve["train_loss"],
        marker="o",
        label="entrenamiento",
    )
    _axes[0].plot(
        weight_cv_curve["epoch"],
        weight_cv_curve["val_loss"],
        marker="s",
        label="validación",
    )
    _axes[0].set(
        title="Ajuste de pesos: pérdida media",
        xlabel="Época",
        ylabel="Cross-entropy",
    )
    _axes[0].legend()
    _axes[1].plot(
        weight_cv_curve["epoch"], weight_cv_curve["val_top1"], marker="o", label="Top-1"
    )
    _axes[1].plot(
        weight_cv_curve["epoch"], weight_cv_curve["val_top3"], marker="s", label="Top-3"
    )
    _axes[1].set(
        title="Ajuste de pesos: exactitud media",
        xlabel="Época",
        ylabel="Fracción",
        ylim=(0, 1),
    )
    _axes[1].legend()
    _figure.tight_layout()

    mo.vstack(
        [
            mo.md("### Ajuste de pesos"),
            mo.callout(
                mo.md(
                    "Media en validación Stockfish: Top-1 "
                    f"**{100 * weight_cv_frame['Top-1'].mean():.2f} %** "
                    f"(± {100 * weight_cv_frame['Top-1'].std():.2f}) y Top-3 "
                    f"**{100 * weight_cv_frame['Top-3'].mean():.2f} %** "
                    f"(± {100 * weight_cv_frame['Top-3'].std():.2f})."
                ),
                kind="info",
            ),
            mo.ui.table(_table, pagination=False),
            _figure,
        ]
    )
    return


@app.cell
def _(
    ChessPolicyNet,
    apply_head_lora,
    copy,
    cv_lora,
    cv_row,
    epochs_lora,
    eval_cv_folds,
    evaluate_model,
    lora_rank,
    make_tensor_dataset,
    mean_history,
    merge_head_lora,
    mo,
    pl,
    plt,
    scratch_fold_states,
    train_head_lora,
):
    mo.stop(
        not cv_lora.value,
        mo.callout(
            mo.md(
                "Ajuste LoRA desactivado en la validación cruzada. Actívalo en los "
                "parámetros para entrenarlo."
            ),
            kind="info",
        ),
    )
    _rank = int(lora_rank.value)
    _rows = []
    _histories = []
    for _fold, ((_e_train, _e_val), _fold_state) in enumerate(
        zip(eval_cv_folds, scratch_fold_states, strict=True), start=1
    ):
        _e_val_ds = make_tensor_dataset(_e_val, "fen", "best_move")
        _base = ChessPolicyNet()
        _base.load_state_dict(_fold_state)
        _model = apply_head_lora(
            copy.deepcopy(_base), rank=_rank, alpha=2.0 * _rank
        )
        _history = train_head_lora(
            _model,
            make_tensor_dataset(_e_train, "fen", "best_move"),
            _e_val_ds,
            epochs=int(epochs_lora.value),
            lr=1e-3,
            batch_size=128,
        )
        _histories.append(pl.DataFrame(_history))
        _rows.append(
            cv_row(
                _fold,
                "Ajuste LoRA",
                "Validación Stockfish",
                evaluate_model(
                    merge_head_lora(_model), _e_val_ds, batch_size=128
                ),
                _model.best_epoch,
            )
        )
    lora_cv_frame = pl.DataFrame(_rows)
    lora_cv_curve = mean_history(_histories)

    _table = lora_cv_frame.select(
        "Partición",
        "Mejor época",
        pl.col("Pérdida").round(4).alias("Pérdida"),
        (100 * pl.col("Top-1")).round(2).alias("Top-1 (%)"),
        (100 * pl.col("Top-3")).round(2).alias("Top-3 (%)"),
    )
    _figure, _axes = plt.subplots(1, 2, figsize=(13, 3.8))
    _axes[0].plot(
        lora_cv_curve["epoch"],
        lora_cv_curve["train_loss"],
        marker="o",
        label="entrenamiento",
    )
    _axes[0].plot(
        lora_cv_curve["epoch"],
        lora_cv_curve["val_loss"],
        marker="s",
        label="validación",
    )
    _axes[0].set(title="LoRA: pérdida media", xlabel="Época", ylabel="Cross-entropy")
    _axes[0].legend()
    _axes[1].plot(
        lora_cv_curve["epoch"], lora_cv_curve["val_top1"], marker="o", label="Top-1"
    )
    _axes[1].plot(
        lora_cv_curve["epoch"], lora_cv_curve["val_top3"], marker="s", label="Top-3"
    )
    _axes[1].set(
        title="LoRA: exactitud media", xlabel="Época", ylabel="Fracción", ylim=(0, 1)
    )
    _axes[1].legend()
    _figure.tight_layout()

    mo.vstack(
        [
            mo.md("### Ajuste LoRA"),
            mo.callout(
                mo.md(
                    f"Rango {_rank}. Media en validación Stockfish: Top-1 "
                    f"**{100 * lora_cv_frame['Top-1'].mean():.2f} %** "
                    f"(± {100 * lora_cv_frame['Top-1'].std():.2f}) y Top-3 "
                    f"**{100 * lora_cv_frame['Top-3'].mean():.2f} %** "
                    f"(± {100 * lora_cv_frame['Top-3'].std():.2f})."
                ),
                kind="info",
            ),
            mo.ui.table(_table, pagination=False),
            _figure,
        ]
    )
    return


@app.cell
def _(
    ChessPolicyNet,
    epochs_stage1,
    human_dev_ds,
    human_monitor_ds,
    mo,
    pl,
    train_model,
):
    scratch_model = ChessPolicyNet()
    stage1_history = train_model(
        scratch_model,
        human_dev_ds,
        human_monitor_ds,
        epochs=int(epochs_stage1.value),
        lr=1e-3,
        batch_size=128,
    )
    stage1_frame = pl.DataFrame(stage1_history)
    mo.vstack(
        [
            mo.md("### Reajuste del modelo desde cero (todo el desarrollo)"),
            mo.callout(
                mo.md(
                    "Este es el modelo que se guarda. La partición de validación de "
                    "la primera corrida solo se usa para monitorizar el entrenamiento; "
                    "la puntuación honesta está en la validación cruzada y en el "
                    "conjunto de prueba."
                ),
                kind="info",
            ),
            mo.ui.table(
                stage1_frame.select("epoch", "train_loss"), pagination=False
            ),
        ]
    )
    return (scratch_model,)


@app.cell
def _(
    ChessPolicyNet,
    epochs_stage2,
    eval_dev_ds,
    eval_monitor_ds,
    mo,
    pl,
    scratch_model,
    train_model,
):
    weight_ft_model = ChessPolicyNet()
    weight_ft_model.load_state_dict(scratch_model.state_dict())
    stage2_history = train_model(
        weight_ft_model,
        eval_dev_ds,
        eval_monitor_ds,
        epochs=int(epochs_stage2.value),
        lr=1e-4,
        batch_size=128,
    )
    stage2_frame = pl.DataFrame(stage2_history)
    mo.vstack(
        [
            mo.md("### Reajuste del ajuste de pesos (todo el desarrollo)"),
            mo.ui.table(
                stage2_frame.select("epoch", "train_loss"), pagination=False
            ),
        ]
    )
    return (weight_ft_model,)


@app.cell
def _(
    apply_head_lora,
    copy,
    count_trainable_parameters,
    epochs_lora,
    eval_dev_ds,
    eval_monitor_ds,
    lora_rank,
    merge_head_lora,
    mo,
    pl,
    scratch_model,
    train_head_lora,
):
    _rank = int(lora_rank.value)
    _epochs = int(epochs_lora.value)
    lora_model = apply_head_lora(
        copy.deepcopy(scratch_model), rank=_rank, alpha=2.0 * _rank
    )
    lora_history = train_head_lora(
        lora_model,
        eval_dev_ds,
        eval_monitor_ds,
        epochs=_epochs,
        lr=1e-3,
        batch_size=128,
        keep_best=False,
    )
    lora_ft_model = merge_head_lora(lora_model)
    lora_trainable, lora_total = count_trainable_parameters(lora_model)
    lora_frame = pl.DataFrame(lora_history)
    mo.vstack(
        [
            mo.md("### Reajuste del ajuste LoRA (todo el desarrollo)"),
            mo.callout(
                mo.md(
                    f"Rango {_rank}: **{lora_trainable:,}** de {lora_total:,} "
                    f"parámetros ({100 * lora_trainable / lora_total:.2f} %). Se "
                    f"entrena durante **{_epochs}** épocas, las del control de "
                    "parámetros; la validación cruzada indica cuántas convienen."
                ),
                kind="info",
            ),
            mo.ui.table(lora_frame.select("epoch", "train_loss"), pagination=False),
        ]
    )
    return lora_ft_model, lora_total, lora_trainable


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparación final

    Una única evaluación de los modelos reajustados en el 20 % de prueba. Las
    medias y desviaciones de la validación cruzada están en cada celda de modelo.
    """)
    return


@app.cell
def _(
    eval_test_ds,
    evaluate_model,
    human_test_ds,
    lora_ft_model,
    lora_total,
    lora_trainable,
    mo,
    pl,
    plt,
    scratch_model,
    weight_ft_model,
):
    _scratch_human = evaluate_model(scratch_model, human_test_ds, batch_size=128)
    _scratch_stockfish = evaluate_model(scratch_model, eval_test_ds, batch_size=128)
    _weight_test = evaluate_model(weight_ft_model, eval_test_ds, batch_size=128)
    _lora_test = evaluate_model(lora_ft_model, eval_test_ds, batch_size=128)

    test_comparison = pl.DataFrame(
        [
            {
                "Modelo": "Desde cero",
                "Evaluación": "Prueba humana",
                "Top-1 (%)": round(100 * _scratch_human["top1"], 2),
                "Top-3 (%)": round(100 * _scratch_human["top3"], 2),
                "Pérdida": round(_scratch_human["loss"], 4),
                "Parámetros entrenados": "todos",
            },
            {
                "Modelo": "Desde cero",
                "Evaluación": "Prueba Stockfish",
                "Top-1 (%)": round(100 * _scratch_stockfish["top1"], 2),
                "Top-3 (%)": round(100 * _scratch_stockfish["top3"], 2),
                "Pérdida": round(_scratch_stockfish["loss"], 4),
                "Parámetros entrenados": "todos",
            },
            {
                "Modelo": "Ajuste de pesos",
                "Evaluación": "Prueba Stockfish",
                "Top-1 (%)": round(100 * _weight_test["top1"], 2),
                "Top-3 (%)": round(100 * _weight_test["top3"], 2),
                "Pérdida": round(_weight_test["loss"], 4),
                "Parámetros entrenados": "todos",
            },
            {
                "Modelo": "Ajuste LoRA",
                "Evaluación": "Prueba Stockfish",
                "Top-1 (%)": round(100 * _lora_test["top1"], 2),
                "Top-3 (%)": round(100 * _lora_test["top3"], 2),
                "Pérdida": round(_lora_test["loss"], 4),
                "Parámetros entrenados": f"{lora_trainable:,}",
            },
        ]
    )

    _model_names = ["Desde cero", "Ajuste de pesos", "Ajuste LoRA"]
    _model_metrics = [_scratch_stockfish, _weight_test, _lora_test]
    _metric_specs = (
        ("top1", "Top-1", "% de aciertos", 100.0, "%.1f"),
        ("top3", "Top-3", "% de aciertos", 100.0, "%.1f"),
        ("loss", "Pérdida", "Cross-entropy", 1.0, "%.2f"),
    )
    _bar_figure, _bar_axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for _axis, (_key, _title, _ylabel, _scale, _fmt) in zip(
        _bar_axes, _metric_specs, strict=True
    ):
        _values = [_scale * _metrics[_key] for _metrics in _model_metrics]
        _drawn = _axis.bar(
            _model_names, _values, color=["#8a8a8a", "#d1603d", "#2a78d4"]
        )
        _axis.bar_label(_drawn, fmt=_fmt)
        _axis.set(title=f"{_title} en prueba Stockfish", ylabel=_ylabel)
        _axis.tick_params(axis="x", rotation=8)
    _bar_figure.tight_layout()

    _delta_weight = 100 * (_weight_test["top1"] - _scratch_stockfish["top1"])
    _delta_lora = 100 * (_lora_test["top1"] - _scratch_stockfish["top1"])
    mo.vstack(
        [
            mo.md("### Conjunto de prueba (una sola evaluación)"),
            mo.callout(
                mo.md(
                    f"Frente al modelo desde cero en la misma prueba, el ajuste de "
                    f"pesos cambia **{_delta_weight:+.2f}** puntos de Top-1 y el "
                    f"ajuste LoRA **{_delta_lora:+.2f}**. LoRA entrena "
                    f"{lora_trainable:,} de {lora_total:,} parámetros "
                    f"({100 * lora_trainable / lora_total:.2f} %). Con este tamaño "
                    "de muestra, diferencias de uno o dos puntos están dentro del "
                    "ruido; compara también la desviación entre particiones de la "
                    "validación cruzada."
                ),
                kind="info",
            ),
            mo.ui.table(test_comparison, pagination=False),
            _bar_figure,
        ]
    )
    return


@app.cell
def _(
    MODELS_DIR,
    lora_ft_model,
    mo,
    save_policy_checkpoint,
    scratch_model,
    weight_ft_model,
):
    scratch_path = MODELS_DIR / "approach1_scratch.pt"
    weight_path = MODELS_DIR / "approach1_weight_ftuning.pt"
    lora_path = MODELS_DIR / "approach1_lora_ftuning.pt"
    # Media precisión: el mismo modelo ocupa la mitad y sigue cargando igual.
    save_policy_checkpoint(scratch_model, scratch_path)
    save_policy_checkpoint(weight_ft_model, weight_path)
    # El adaptador LoRA se fusiona en la cabeza, así que este archivo es un
    # ChessPolicyNet normal y se carga sin reconstruir el adaptador.
    save_policy_checkpoint(lora_ft_model, lora_path)
    _megabytes = scratch_path.stat().st_size / 1024**2
    mo.callout(
        mo.md(
            "Pesos guardados (media precisión, "
            f"{_megabytes:.1f} MB cada uno):\n\n"
            f"- `{scratch_path}` (desde cero)\n"
            f"- `{weight_path}` (ajuste de pesos)\n"
            f"- `{lora_path}` (ajuste LoRA, ya fusionado)"
        ),
        kind="success",
    )
    return lora_path, scratch_path, weight_path


@app.cell
def _(
    load_policy_checkpoint,
    lora_path,
    mo,
    scratch_path,
    weight_path,
):
    mo.stop(
        not (
            scratch_path.exists()
            and weight_path.exists()
            and lora_path.exists()
        ),
        mo.callout(
            mo.md(
                "No hay pesos guardados todavía. Pulsa **Guardar parametros** "
                "para entrenar y guardar los tres modelos."
            ),
            kind="warn",
        ),
    )

    scratch_eval_model = load_policy_checkpoint(scratch_path)
    weight_eval_model = load_policy_checkpoint(weight_path)
    lora_eval_model = load_policy_checkpoint(lora_path)
    mo.md(f"Pesos cargados desde `{scratch_path.parent}`.")
    return lora_eval_model, scratch_eval_model, weight_eval_model


@app.cell
def _(chess, mo):
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
        "Gambito de dama rechazado": (
            "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4"
        ),
        "Ataque indio de rey": (
            "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 5"
        ),
        "Final de peones": "8/5k2/8/8/8/4K3/4P3/8 w - - 0 1",
        "Final de torres": "8/8/4k3/8/8/4K3/8/R7 w - - 0 1",
        "Mate en una (torre)": "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
        "Peón a promoción": "8/1P3k2/8/8/8/8/6K1/8 w - - 0 1",
        "Dama contra rey": "8/8/4k3/8/8/4K3/8/Q7 w - - 0 1",
    }

    get_fen, set_fen = mo.state(chess.STARTING_FEN)

    def _apply_preset(selected: str) -> None:
        preset_fen = presets[selected]
        if preset_fen is not None and preset_fen != get_fen():
            set_fen(preset_fen)

    preset = mo.ui.dropdown(
        options=list(presets),
        value="Inicio",
        label="Posición de ejemplo",
        on_change=_apply_preset,
    )
    model_choice = mo.ui.dropdown(
        options={
            "Ajuste LoRA (Stockfish)": "lora",
            "Ajuste de pesos (Stockfish)": "weight",
            "Desde cero (etapa 1)": "scratch",
        },
        value="Ajuste LoRA (Stockfish)",
        label="Modelo cuya mejor jugada se marca en el tablero",
    )
    mo.vstack(
        [
            mo.md("## Predicción interactiva"),
            mo.md(
                "Elige un ejemplo y el FEN se actualiza solo; después puedes "
                "editarlo a mano. La tabla compara la mejor jugada de los tres "
                "modelos y la flecha del tablero muestra la del modelo elegido."
            ),
            preset,
            model_choice,
        ]
    )
    return get_fen, model_choice, set_fen


@app.cell
def _(get_fen, mo, set_fen):
    # The text area lives in its own cell: marimo does not re-run the cell that
    # calls a state setter, so the dropdown can only refresh this widget if the
    # getter is read from a different cell.
    fen_input = mo.ui.text_area(
        value=get_fen(),
        label="FEN (lo rellena el ejemplo; también se puede editar)",
        debounce=500,
        on_change=set_fen,
    )
    fen_input
    return (fen_input,)


@app.cell
def _(
    chess,
    fen_input,
    lora_eval_model,
    mo,
    model_choice,
    pl,
    predict_top_moves,
    scratch_eval_model,
    weight_eval_model,
):
    fen = fen_input.value.strip()
    mo.stop(
        not fen,
        mo.callout(
            mo.md("Escribe un FEN o elige un ejemplo de la lista."), kind="warn"
        ),
    )
    try:
        board = chess.Board(fen)
        fen_error = None
    except ValueError as error:
        board = None
        fen_error = str(error)
    mo.stop(
        fen_error is not None,
        mo.callout(mo.md(f"FEN inválido: {fen_error}"), kind="danger"),
    )
    mo.stop(
        not any(board.legal_moves),
        mo.callout(
            mo.md("La posición es terminal (mate o ahogado): no hay jugadas legales."),
            kind="warn",
        ),
    )

    models = {
        "scratch": ("Desde cero", scratch_eval_model),
        "weight": ("Ajuste de pesos", weight_eval_model),
        "lora": ("Ajuste LoRA", lora_eval_model),
    }
    predictions = {
        key: predict_top_moves(model, fen, top_k=3) for key, (_, model) in models.items()
    }
    selected_key = model_choice.value
    selected_label, _ = models[selected_key]
    best_move = chess.Move.from_uci(predictions[selected_key][0][0])
    board_svg = chess.svg.board(
        board,
        arrows=[chess.svg.Arrow(best_move.from_square, best_move.to_square)],
        size=380,
        coordinates=True,
    )

    rows = []
    for position in range(3):
        row = {"#": position + 1}
        for model_key, (model_label, _) in models.items():
            move_uci, probability = predictions[model_key][position]
            row[model_label] = f"{move_uci} · {100 * probability:.1f} %"
        rows.append(row)

    mo.hstack(
        [
            mo.Html(str(board_svg)),
            mo.vstack(
                [
                    mo.md(f"**Posición:** `{fen}`"),
                    mo.md(
                        f"**Flecha:** mejor jugada de *{selected_label}* "
                        f"(`{best_move.uci()}`)"
                    ),
                    mo.ui.table(pl.DataFrame(rows), pagination=False),
                    mo.md(
                        "Cada columna son las tres jugadas legales más probables de "
                        "un modelo, en orden de probabilidad."
                    ),
                ]
            ),
        ],
        justify="start",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
