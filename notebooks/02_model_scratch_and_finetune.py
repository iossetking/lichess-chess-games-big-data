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
    import torch

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
        copy,
        count_trainable_parameters,
        evaluate_model,
        kfold,
        load_evals,
        load_positions,
        make_tensor_dataset,
        merge_head_lora,
        mo,
        pl,
        plt,
        predict_top_moves,
        split_holdout,
        torch,
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
    | **Ajuste LoRA** | Solo un adaptador de bajo rango sobre la cabeza de jugadas | `approach1_lora_ftuning.pt` |

    **Por qué dos ajustes finos.** Con una muestra pequeña, actualizar todos los
    pesos puede deshacer lo aprendido en la etapa 1 y bajar el Top-1. LoRA
    congela la red y entrena únicamente una corrección de bajo rango `B·A` sobre
    la capa final, así que el modelo ajustado sigue siendo casi idéntico al de
    partida y solo se aparta de él en la dirección que mejora la validación.
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
    def _step_for(total: int) -> int:
        for candidate in (100, 200, 500, 1000, 2000, 5000, 10000, 20000):
            if total / candidate <= 50:
                return candidate
        return max(1, total // 50)

    _human_total = positions_df.height
    _eval_total = evals_df.height
    _human_step = _step_for(_human_total)
    _eval_step = _step_for(_eval_total)

    max_human_positions = mo.ui.slider(
        start=min(_human_step, _human_total) or 1,
        stop=max(1, _human_total),
        step=_human_step,
        value=min(40000, _human_total) or 1,
        label=f"Posiciones máximas etapa 1 (humanos · {_human_total:,})",
    )
    epochs_stage1 = mo.ui.slider(
        start=1, stop=10, step=1, value=5, label="Épocas etapa 1 (humanos)"
    )
    max_eval_positions = mo.ui.slider(
        start=min(_eval_step, _eval_total) or 1,
        stop=max(1, _eval_total),
        step=_eval_step,
        value=min(40000, _eval_total) or 1,
        label=f"Posiciones máximas etapa 2 (Stockfish · {_eval_total:,})",
    )
    epochs_stage2 = mo.ui.slider(
        start=1, stop=10, step=1, value=5, label="Épocas ajuste de pesos (Stockfish)"
    )
    epochs_lora = mo.ui.slider(
        start=1, stop=10, step=1, value=5, label="Épocas ajuste LoRA (Stockfish)"
    )
    lora_rank = mo.ui.dropdown(options=[4, 8, 16, 32], value=8, label="Rango LoRA")
    n_folds = mo.ui.dropdown(
        options={"3 particiones": 3, "5 particiones": 5, "10 particiones": 10},
        value="5 particiones",
        label="Validación cruzada",
    )
    train_button = mo.ui.run_button(label="Guardar parametros")
    mo.vstack(
        [
            mo.md("## Parametros de Entrenamiento"),
            mo.hstack([max_human_positions, epochs_stage1], justify="start"),
            mo.hstack([max_eval_positions, epochs_stage2], justify="start"),
            mo.hstack([epochs_lora, lora_rank], justify="start"),
            n_folds,
            train_button,
        ]
    )
    return (
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
        not train_button.value,
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


@app.cell
def _(
    ChessPolicyNet,
    apply_head_lora,
    copy,
    epochs_lora,
    epochs_stage1,
    epochs_stage2,
    eval_dev_df,
    evaluate_model,
    human_dev_df,
    kfold,
    lora_rank,
    make_tensor_dataset,
    merge_head_lora,
    mo,
    n_folds,
    pl,
    plt,
    train_head_lora,
    train_model,
):
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
    _human_folds = kfold(human_dev_df, n_folds=_folds, group_col="game_id")
    _eval_folds = kfold(eval_dev_df, n_folds=_folds)
    _rows = []
    _histories = {"Desde cero": [], "Ajuste de pesos": [], "Ajuste LoRA": []}
    _monitor = None

    def _row(fold, model, evaluation, metrics, best_epoch):
        return {
            "Partición": fold,
            "Modelo": model,
            "Evaluación": evaluation,
            "Pérdida": metrics["loss"],
            "Top-1": metrics["top1"],
            "Top-3": metrics["top3"],
            "Mejor época": best_epoch,
        }

    for _fold, ((_h_train, _h_val), (_e_train, _e_val)) in enumerate(
        zip(_human_folds, _eval_folds, strict=True), start=1
    ):
        _h_train_ds = make_tensor_dataset(_h_train)
        _h_val_ds = make_tensor_dataset(_h_val)
        _e_train_ds = make_tensor_dataset(_e_train, "fen", "best_move")
        _e_val_ds = make_tensor_dataset(_e_val, "fen", "best_move")

        _scratch = ChessPolicyNet()
        _scratch_history = train_model(
            _scratch,
            _h_train_ds,
            _h_val_ds,
            epochs=int(epochs_stage1.value),
            lr=1e-3,
            batch_size=128,
        )
        _histories["Desde cero"].append(pl.DataFrame(_scratch_history))
        _rows.append(
            _row(
                _fold,
                "Desde cero",
                "Validación humana",
                evaluate_model(_scratch, _h_val_ds, batch_size=128),
                int(epochs_stage1.value),
            )
        )
        _rows.append(
            _row(
                _fold,
                "Desde cero",
                "Validación Stockfish",
                evaluate_model(_scratch, _e_val_ds, batch_size=128),
                int(epochs_stage1.value),
            )
        )

        _weight = ChessPolicyNet()
        _weight.load_state_dict(_scratch.state_dict())
        _weight_history = train_model(
            _weight,
            _e_train_ds,
            _e_val_ds,
            epochs=int(epochs_stage2.value),
            lr=1e-4,
            batch_size=128,
        )
        _histories["Ajuste de pesos"].append(pl.DataFrame(_weight_history))
        _rows.append(
            _row(
                _fold,
                "Ajuste de pesos",
                "Validación Stockfish",
                evaluate_model(_weight, _e_val_ds, batch_size=128),
                int(epochs_stage2.value),
            )
        )

        _rank = int(lora_rank.value)
        _lora = apply_head_lora(
            copy.deepcopy(_scratch), rank=_rank, alpha=2.0 * _rank
        )
        _lora_history = train_head_lora(
            _lora,
            _e_train_ds,
            _e_val_ds,
            epochs=int(epochs_lora.value),
            lr=1e-3,
            batch_size=128,
        )
        _histories["Ajuste LoRA"].append(pl.DataFrame(_lora_history))
        _rows.append(
            _row(
                _fold,
                "Ajuste LoRA",
                "Validación Stockfish",
                evaluate_model(
                    merge_head_lora(_lora), _e_val_ds, batch_size=128
                ),
                _lora.best_epoch,
            )
        )

        if _fold == 1:
            _monitor = (_h_val_ds, _e_val_ds)

    cv_frame = pl.DataFrame(_rows)
    cv_summary = cv_frame.group_by(
        ["Modelo", "Evaluación"], maintain_order=True
    ).agg(
        pl.col("Pérdida").mean().round(4).alias("Pérdida media"),
        pl.col("Pérdida").std().round(4).alias("Pérdida desv."),
        (100 * pl.col("Top-1")).mean().round(2).alias("Top-1 media (%)"),
        (100 * pl.col("Top-1")).std().round(2).alias("Top-1 desv."),
        (100 * pl.col("Top-3")).mean().round(2).alias("Top-3 media (%)"),
        (100 * pl.col("Top-3")).std().round(2).alias("Top-3 desv."),
    )

    def _mean_history(frames):
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

    cv_curves = {name: _mean_history(frames) for name, frames in _histories.items()}
    _lora_epochs = [
        row["Mejor época"] for row in _rows if row["Modelo"] == "Ajuste LoRA"
    ]
    lora_epochs_refit = max(1, round(sum(_lora_epochs) / len(_lora_epochs)))
    human_monitor_ds, eval_monitor_ds = _monitor

    _cv_figure, _cv_axes = plt.subplots(1, 2, figsize=(13, 3.8))
    _cv_axes[0].plot(
        cv_curves["Desde cero"]["epoch"],
        cv_curves["Desde cero"]["val_top1"],
        marker="o",
        label="Top-1",
    )
    _cv_axes[0].plot(
        cv_curves["Desde cero"]["epoch"],
        cv_curves["Desde cero"]["val_top3"],
        marker="s",
        label="Top-3",
    )
    _cv_axes[0].set(
        title="Desde cero (validación humana)",
        xlabel="Época",
        ylabel="Fracción",
        ylim=(0, 1),
    )
    _cv_axes[0].legend()
    for _name, _color in (("Ajuste de pesos", "#d1603d"), ("Ajuste LoRA", "#2a78d4")):
        _cv_axes[1].plot(
            cv_curves[_name]["epoch"],
            cv_curves[_name]["val_top1"],
            marker="o",
            color=_color,
            label=f"{_name} · Top-1",
        )
    _cv_axes[1].set(
        title="Ajustes finos (validación Stockfish)",
        xlabel="Época",
        ylabel="Fracción",
        ylim=(0, 1),
    )
    _cv_axes[1].legend(fontsize=8)
    _cv_figure.tight_layout()

    mo.vstack(
        [
            mo.md(f"### Validación cruzada de {_folds} particiones"),
            mo.callout(
                mo.md(
                    "Cada partición entrena desde cero y mide en el quinto que quedó "
                    "fuera. Las cifras son media y desviación entre particiones; el "
                    "20 % de prueba no participa."
                ),
                kind="info",
            ),
            mo.ui.table(cv_summary, pagination=False),
            _cv_figure,
        ]
    )
    return (
        cv_curves,
        cv_frame,
        cv_summary,
        eval_monitor_ds,
        human_monitor_ds,
        lora_epochs_refit,
    )


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
    eval_dev_ds,
    eval_monitor_ds,
    lora_epochs_refit,
    lora_rank,
    merge_head_lora,
    mo,
    pl,
    scratch_model,
    train_head_lora,
):
    _rank = int(lora_rank.value)
    lora_model = apply_head_lora(
        copy.deepcopy(scratch_model), rank=_rank, alpha=2.0 * _rank
    )
    lora_history = train_head_lora(
        lora_model,
        eval_dev_ds,
        eval_monitor_ds,
        epochs=lora_epochs_refit,
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
                    f"entrena durante **{lora_epochs_refit}** épocas, el número "
                    "elegido por la validación cruzada; no se selecciona época con "
                    "el conjunto de prueba."
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
    ## Resultados de la validación cruzada

    Cada tabla muestra las particiones de un modelo y su curva media. La
    validación de la etapa 1 usa jugadas humanas; los dos ajustes finos y la fila
    de Stockfish de la etapa 1 usan jugadas de Stockfish.
    """)
    return


@app.cell
def _(cv_curves, cv_frame, mo, pl, plt):
    _scratch_cv = cv_frame.filter(
        (pl.col("Modelo") == "Desde cero")
        & (pl.col("Evaluación") == "Validación humana")
    )
    _scratch_table = _scratch_cv.select(
        "Partición",
        pl.col("Pérdida").round(4).alias("Pérdida"),
        (100 * pl.col("Top-1")).round(2).alias("Top-1 (%)"),
        (100 * pl.col("Top-3")).round(2).alias("Top-3 (%)"),
    )
    _curve = cv_curves["Desde cero"]

    _figure, _axes = plt.subplots(1, 2, figsize=(13, 3.8))
    _axes[0].plot(
        _curve["epoch"], _curve["train_loss"], marker="o", label="entrenamiento"
    )
    _axes[0].plot(_curve["epoch"], _curve["val_loss"], marker="s", label="validación")
    _axes[0].set(
        title="Desde cero: pérdida media", xlabel="Época", ylabel="Cross-entropy"
    )
    _axes[0].legend()
    _axes[1].plot(_curve["epoch"], _curve["val_top1"], marker="o", label="Top-1")
    _axes[1].plot(_curve["epoch"], _curve["val_top3"], marker="s", label="Top-3")
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
            mo.md("### Desde cero · jugadas humanas"),
            mo.ui.table(_scratch_table, pagination=False),
            _figure,
        ]
    )
    return


@app.cell
def _(cv_curves, cv_frame, mo, pl, plt):
    _weight_cv = cv_frame.filter(pl.col("Modelo") == "Ajuste de pesos")
    _weight_table = _weight_cv.select(
        "Partición",
        pl.col("Pérdida").round(4).alias("Pérdida"),
        (100 * pl.col("Top-1")).round(2).alias("Top-1 (%)"),
        (100 * pl.col("Top-3")).round(2).alias("Top-3 (%)"),
    )
    _curve = cv_curves["Ajuste de pesos"]

    _figure, _axes = plt.subplots(1, 2, figsize=(13, 3.8))
    _axes[0].plot(
        _curve["epoch"], _curve["train_loss"], marker="o", label="entrenamiento"
    )
    _axes[0].plot(_curve["epoch"], _curve["val_loss"], marker="s", label="validación")
    _axes[0].set(
        title="Ajuste de pesos: pérdida media",
        xlabel="Época",
        ylabel="Cross-entropy",
    )
    _axes[0].legend()
    _axes[1].plot(_curve["epoch"], _curve["val_top1"], marker="o", label="Top-1")
    _axes[1].plot(_curve["epoch"], _curve["val_top3"], marker="s", label="Top-3")
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
            mo.md("### Ajuste de pesos · jugadas de Stockfish"),
            mo.ui.table(_weight_table, pagination=False),
            _figure,
        ]
    )
    return


@app.cell
def _(cv_curves, cv_frame, mo, pl, plt):
    _lora_cv = cv_frame.filter(pl.col("Modelo") == "Ajuste LoRA")
    _lora_table = _lora_cv.select(
        "Partición",
        "Mejor época",
        pl.col("Pérdida").round(4).alias("Pérdida"),
        (100 * pl.col("Top-1")).round(2).alias("Top-1 (%)"),
        (100 * pl.col("Top-3")).round(2).alias("Top-3 (%)"),
    )
    _curve = cv_curves["Ajuste LoRA"]

    _figure, _axes = plt.subplots(1, 2, figsize=(13, 3.8))
    _axes[0].plot(
        _curve["epoch"], _curve["train_loss"], marker="o", label="entrenamiento"
    )
    _axes[0].plot(_curve["epoch"], _curve["val_loss"], marker="s", label="validación")
    _axes[0].set(title="LoRA: pérdida media", xlabel="Época", ylabel="Cross-entropy")
    _axes[0].legend()
    _axes[1].plot(_curve["epoch"], _curve["val_top1"], marker="o", label="Top-1")
    _axes[1].plot(_curve["epoch"], _curve["val_top3"], marker="s", label="Top-3")
    _axes[1].set(
        title="LoRA: exactitud media", xlabel="Época", ylabel="Fracción", ylim=(0, 1)
    )
    _axes[1].legend()
    _figure.tight_layout()

    mo.vstack(
        [
            mo.md("### Ajuste LoRA · jugadas de Stockfish"),
            mo.ui.table(_lora_table, pagination=False),
            _figure,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparación final

    Arriba, la validación cruzada sobre el desarrollo. Abajo, una única
    evaluación de los modelos reajustados en el 20 % de prueba.
    """)
    return


@app.cell
def _(
    cv_summary,
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
            mo.md("### Validación cruzada (desarrollo)"),
            mo.ui.table(cv_summary, pagination=False),
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
def _(MODELS_DIR, lora_ft_model, mo, scratch_model, torch, weight_ft_model):
    scratch_path = MODELS_DIR / "approach1_scratch.pt"
    weight_path = MODELS_DIR / "approach1_weight_ftuning.pt"
    lora_path = MODELS_DIR / "approach1_lora_ftuning.pt"
    torch.save(scratch_model.state_dict(), scratch_path)
    torch.save(weight_ft_model.state_dict(), weight_path)
    # El adaptador LoRA se fusiona en la cabeza, así que este archivo es un
    # ChessPolicyNet normal y se carga sin reconstruir el adaptador.
    torch.save(lora_ft_model.state_dict(), lora_path)
    mo.callout(
        mo.md(
            "Pesos guardados:\n\n"
            f"- `{scratch_path}` (desde cero)\n"
            f"- `{weight_path}` (ajuste de pesos)\n"
            f"- `{lora_path}` (ajuste LoRA, ya fusionado)"
        ),
        kind="success",
    )
    return lora_path, scratch_path, weight_path


@app.cell
def _(ChessPolicyNet, lora_path, mo, scratch_path, torch, weight_path):
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

    def _load(path):
        model = ChessPolicyNet()
        model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        return model.eval()

    scratch_eval_model = _load(scratch_path)
    weight_eval_model = _load(weight_path)
    lora_eval_model = _load(lora_path)
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
