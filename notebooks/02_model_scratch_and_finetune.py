import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import chess
    import chess.svg
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import torch

    from lichess_pipeline.model.scratch_net import (
        ChessPolicyNet,
        evaluate_model,
        make_tensor_dataset,
        predict_top_moves,
        train_model,
    )
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

    def split_train_val(
        frame: pl.DataFrame, validation_fraction: float = 0.1, seed: int = 42
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return ``(validation, train)`` disjoint slices of a shuffled frame."""
        shuffled = frame.sample(fraction=1.0, shuffle=True, seed=seed)
        n_validation = max(1, int(validation_fraction * shuffled.height))
        return shuffled.head(n_validation), shuffled.tail(shuffled.height - n_validation)

    plt.rcParams.update(
        {"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100}
    )
    return (
        ChessPolicyNet,
        EVALS_PATH,
        MODELS_DIR,
        POSITIONS_PATH,
        chess,
        evaluate_model,
        load_evals,
        load_positions,
        make_tensor_dataset,
        mo,
        pl,
        plt,
        predict_top_moves,
        split_train_val,
        torch,
        train_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Modelo de mejor jugada: entrenamiento desde cero + fine tuning

    **Enfoque 1 (principal).** Un único modelo, dos etapas:

    1. **Desde cero con jugadas humanas.** La red aprende `FEN -> jugada`, jugada
       a partir de partidas de Lichess con Elo alto (`positions_games_sample.parquet`,
       derivado de `lichess_db_standard_rated_*.pgn.zst`).
    2. **Ajuste fino con Stockfish.** Se cargan los mismos pesos y se entrenan
       con `FEN -> mejor jugada` de `lichess_db_eval.jsonl.zst`
       (`evals_clean_sample.parquet`), con una tasa de aprendizaje menor.

    Entra una cadena FEN y sale una jugada UCI. Las
    jugadas ilegales se enmascaran antes del `softmax`, así que la salida es
    legal por construcción: el modelo solo puede equivocarse de *elección*, no
    de *legalidad*. El tablero se codifica como `13 x 8 x 8` (6 planos de
    piezas blancas, 6 de negras y 1 para el turno) y la jugada como una de
    `64 x 64 = 4096` clases `origen -> destino`.

    Limitación: los enroques y la captura al paso del FEN no se
    codifican como planos. La máscara de legalidad los usa al predecir, pero el
    modelo no aprende a preferirlos. Añadir dos planos es una mejora futura.

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
def _(mo):
    max_train = mo.ui.slider(
        start=2000,
        stop=40000,
        step=2000,
        value=20000,
        label="Posiciones máximas por etapa",
    )
    epochs_stage1 = mo.ui.slider(
        start=1, stop=10, step=1, value=3, label="Épocas etapa 1 (humanos)"
    )
    epochs_stage2 = mo.ui.slider(
        start=1, stop=10, step=1, value=2, label="Épocas etapa 2 (Stockfish)"
    )
    train_button = mo.ui.run_button(label="Guardar parametros")
    mo.vstack(
        [
            mo.md("## Parametros de Entrenamiento para ambas etapas"),
            mo.hstack([max_train, epochs_stage1, epochs_stage2], justify="start"),
            train_button,
        ]
    )
    return epochs_stage1, epochs_stage2, max_train, train_button


@app.cell
def _(
    evals_df,
    make_tensor_dataset,
    max_train,
    mo,
    positions_df,
    split_train_val,
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
    limit = int(max_train.value)
    human_val, human_train = split_train_val(positions_df.head(min(limit, positions_df.height)))
    eval_val, eval_train = split_train_val(evals_df.head(min(limit, evals_df.height)))
    human_train_ds = make_tensor_dataset(human_train)
    human_val_ds = make_tensor_dataset(human_val)
    eval_train_ds = make_tensor_dataset(eval_train, "fen", "best_move")
    eval_val_ds = make_tensor_dataset(eval_val, "fen", "best_move")
    mo.callout(
        mo.md(
            f"Etapa 1: {human_train.height:,} entrenamiento / {human_val.height:,} validación. "
            f"Etapa 2: {eval_train.height:,} / {eval_val.height:,}."
        ),
        kind="success",
    )
    return eval_train_ds, eval_val_ds, human_train_ds, human_val_ds


@app.cell
def _(
    ChessPolicyNet,
    epochs_stage1,
    evaluate_model,
    human_train_ds,
    human_val_ds,
    mo,
    pl,
    train_model,
):
    stage1_model = ChessPolicyNet()
    stage1_history = train_model(
        stage1_model,
        human_train_ds,
        human_val_ds,
        epochs=int(epochs_stage1.value),
        lr=1e-3,
        batch_size=128,
    )
    stage1_metrics = evaluate_model(stage1_model, human_val_ds, batch_size=128)
    stage1_frame = pl.DataFrame(stage1_history)
    mo.vstack(
        [
            mo.md("### Etapa 1 completada (jugadas humanas)"),
            mo.ui.table(stage1_frame, pagination=False),
        ]
    )
    return stage1_frame, stage1_metrics, stage1_model


@app.cell
def _(
    ChessPolicyNet,
    epochs_stage2,
    eval_train_ds,
    eval_val_ds,
    evaluate_model,
    mo,
    pl,
    stage1_model,
    train_model,
):
    before_finetuning = evaluate_model(stage1_model, eval_val_ds, batch_size=128)
    stage2_model = ChessPolicyNet()
    stage2_model.load_state_dict(stage1_model.state_dict())
    stage2_history = train_model(
        stage2_model,
        eval_train_ds,
        eval_val_ds,
        epochs=int(epochs_stage2.value),
        lr=1e-4,
        batch_size=128,
    )
    after_finetuning = evaluate_model(stage2_model, eval_val_ds, batch_size=128)
    stage2_frame = pl.DataFrame(stage2_history)
    mo.vstack(
        [
            mo.md("### Etapa 2 completada (ajuste fino con Stockfish)"),
            mo.ui.table(stage2_frame, pagination=False),
        ]
    )
    return after_finetuning, before_finetuning, stage2_frame, stage2_model


@app.cell
def _(
    after_finetuning,
    before_finetuning,
    mo,
    pl,
    plt,
    stage1_frame,
    stage1_metrics,
    stage2_frame,
):
    comparison = pl.DataFrame(
        [
            {
                "Evaluación": "Etapa 1 · validación humana",
                "Top-1": stage1_metrics["top1"],
                "Top-3": stage1_metrics["top3"],
                "Pérdida": stage1_metrics["loss"],
            },
            {
                "Evaluación": "Etapa 1 sin ajustar · validación Stockfish",
                "Top-1": before_finetuning["top1"],
                "Top-3": before_finetuning["top3"],
                "Pérdida": before_finetuning["loss"],
            },
            {
                "Evaluación": "Etapa 2 ajustada · validación Stockfish",
                "Top-1": after_finetuning["top1"],
                "Top-3": after_finetuning["top3"],
                "Pérdida": after_finetuning["loss"],
            },
        ]
    ).with_columns(
        (100 * pl.col("Top-1")).round(2).alias("Top-1 (%)"),
        (100 * pl.col("Top-3")).round(2).alias("Top-3 (%)"),
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(
        stage1_frame["epoch"], stage1_frame["train_loss"], marker="o", label="entrenamiento"
    )
    axes[0, 0].plot(
        stage1_frame["epoch"], stage1_frame["val_loss"], marker="s", label="validación"
    )
    axes[0, 0].set(title="Etapa 1: pérdida", xlabel="Época", ylabel="Cross-entropy")
    axes[0, 0].legend()

    axes[0, 1].plot(stage1_frame["epoch"], stage1_frame["val_top1"], marker="o", label="Top-1")
    axes[0, 1].plot(stage1_frame["epoch"], stage1_frame["val_top3"], marker="s", label="Top-3")
    axes[0, 1].set(title="Etapa 1: exactitud", xlabel="Época", ylabel="Fracción", ylim=(0, 1))
    axes[0, 1].legend()

    axes[1, 0].plot(
        stage2_frame["epoch"], stage2_frame["train_loss"], marker="o", label="entrenamiento"
    )
    axes[1, 0].plot(
        stage2_frame["epoch"], stage2_frame["val_loss"], marker="s", label="validación"
    )
    axes[1, 0].set(title="Etapa 2: pérdida", xlabel="Época", ylabel="Cross-entropy")
    axes[1, 0].legend()

    axes[1, 1].plot(stage2_frame["epoch"], stage2_frame["val_top1"], marker="o", label="Top-1")
    axes[1, 1].plot(stage2_frame["epoch"], stage2_frame["val_top3"], marker="s", label="Top-3")
    axes[1, 1].set(title="Etapa 2: exactitud", xlabel="Época", ylabel="Fracción", ylim=(0, 1))
    axes[1, 1].legend()

    figure.tight_layout()

    delta_top1 = 100 * (after_finetuning["top1"] - before_finetuning["top1"])
    trend = "mejora" if delta_top1 >= 0 else "no mejora"
    comparison_figure, comparison_axes = plt.subplots(figsize=(6, 3.5))
    comparison_axes.bar(
        ["Etapa 1 sin ajustar", "Etapa 2 ajustada"],
        [100 * before_finetuning["top1"], 100 * after_finetuning["top1"]],
    )
    comparison_axes.set(
        title="Top-1 en la misma validación de Stockfish",
        ylabel="% de aciertos",
    )
    comparison_figure.tight_layout()

    mo.vstack(
        [
            mo.md("## Resultados"),
            mo.callout(
                mo.md(
                    f"El ajuste fino **{trend}** la coincidencia con la jugada de "
                    f"Stockfish: {delta_top1:+.2f} puntos de Top-1 (fila 3 frente a "
                    "fila 2). La exactitud es baja en términos absolutos porque hay "
                    "~30 jugadas legales por posición y el modelo se entrena con una "
                    "muestra mínima; crece al usar el dataset completo en el servidor."
                ),
                kind="info",
            ),
            mo.ui.table(comparison, pagination=False),
            mo.hstack([figure, comparison_figure], justify="start"),
        ]
    )
    return


@app.cell
def _(MODELS_DIR, mo, stage1_model, stage2_model, torch):
    stage1_path = MODELS_DIR / "approach1_stage1_scratch.pt"
    stage2_path = MODELS_DIR / "approach1_stage2_finetuned.pt"
    torch.save(stage1_model.state_dict(), stage1_path)
    torch.save(stage2_model.state_dict(), stage2_path)
    mo.callout(
        mo.md(f"Pesos guardados:\n\n- `{stage1_path}`\n- `{stage2_path}`"),
        kind="success",
    )
    return


@app.cell
def _(ChessPolicyNet, MODELS_DIR, mo, torch):
    stage1_weights = MODELS_DIR / "approach1_stage1_scratch.pt"
    stage2_weights = MODELS_DIR / "approach1_stage2_finetuned.pt"
    mo.stop(
        not (stage1_weights.exists() and stage2_weights.exists()),
        mo.callout(
            mo.md(
                "No hay pesos guardados todavía. Pulsa **Guardar parametros** "
                "para entrenar y guardar los dos modelos."
            ),
            kind="warn",
        ),
    )
    stage1_eval_model = ChessPolicyNet()
    stage1_eval_model.load_state_dict(
        torch.load(stage1_weights, map_location="cpu", weights_only=True)
    )
    stage1_eval_model.eval()
    stage2_eval_model = ChessPolicyNet()
    stage2_eval_model.load_state_dict(
        torch.load(stage2_weights, map_location="cpu", weights_only=True)
    )
    stage2_eval_model.eval()
    mo.md(f"Pesos cargados desde `{MODELS_DIR}`.")
    return stage1_eval_model, stage2_eval_model


@app.cell
def _(chess, mo):
    presets = {
        "Personalizado (usar el FEN de abajo)": None,
        "Inicio": chess.STARTING_FEN,
        "Italiana tras 1.e4 e5 2.Nf3 Nc6 3.Bc4": (
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        ),
        "Final de peones": "8/5k2/8/8/8/4K3/4P3/8 w - - 0 1",
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
    mo.vstack(
        [
            mo.md("## Predicción interactiva"),
            mo.md(
                "Escribe cualquier FEN o elige un ejemplo de la lista (el ejemplo "
                "tiene prioridad sobre el texto). La red devuelve sus tres jugadas "
                "legales más probables."
            ),
            preset,
            fen_input,
        ]
    )
    return fen_input, preset, presets


@app.cell
def _(
    chess,
    fen_input,
    mo,
    pl,
    predict_top_moves,
    preset,
    presets,
    stage1_eval_model,
    stage2_eval_model,
):
    selected_preset = presets[preset.value]
    fen = selected_preset if selected_preset is not None else fen_input.value.strip()
    source = (
        f"preset «{preset.value}»"
        if selected_preset is not None
        else "FEN escrito a mano"
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

    stage1_predictions = predict_top_moves(stage1_eval_model, fen, top_k=3)
    stage2_predictions = predict_top_moves(stage2_eval_model, fen, top_k=3)
    best_move = chess.Move.from_uci(stage2_predictions[0][0])
    board_svg = chess.svg.board(
        board,
        arrows=[chess.svg.Arrow(best_move.from_square, best_move.to_square)],
        size=380,
        coordinates=True,
    )

    rows = []
    for rank, ((first_move, first_probability), (second_move, second_probability)) in enumerate(
        zip(stage1_predictions, stage2_predictions, strict=True), start=1
    ):
        rows.append(
            {
                "#": rank,
                "Etapa 1 (humanos)": f"{first_move} · {100 * first_probability:.1f} %",
                "Etapa 2 (Stockfish)": f"{second_move} · {100 * second_probability:.1f} %",
            }
        )
    mo.hstack(
        [
            mo.Html(str(board_svg)),
            mo.vstack(
                [
                    mo.md(f"**Posición:** {source} · `{fen}`"),
                    mo.md(f"**Mejor jugada del modelo ajustado:** `{best_move.uci()}`"),
                    mo.ui.table(pl.DataFrame(rows), pagination=False),
                    mo.md(
                        "Las tres jugadas de cada columna son legales en esta posición. "
                        "La flecha del tablero muestra la mejor jugada de la etapa 2."
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
