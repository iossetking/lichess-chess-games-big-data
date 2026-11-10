import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    from pathlib import Path

    import chess
    import chess.svg
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import torch

    from lichess_pipeline.model.maia_adapter import MaiaResidualPolicyNet
    from lichess_pipeline.model.scratch_net import (
        ChessPolicyNet,
        evaluate_model,
        make_tensor_dataset,
        predict_top_moves,
        train_model,
    )
    from lichess_pipeline.pgn_parser import evals_url, load_clean_evals_df

    PROJECT_DIR = Path(__file__).resolve().parents[1]
    PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
    MODELS_DIR = PROJECT_DIR / "data" / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    EVALS_PATH = PROCESSED_DIR / "evals_clean_sample.parquet"
    STAGE2_PATH = MODELS_DIR / "approach1_stage2_finetuned.pt"
    MAIA_PATH = MODELS_DIR / "approach3_maia_finetuned.pt"

    def load_evals() -> pl.DataFrame:
        if EVALS_PATH.exists():
            return pl.read_parquet(EVALS_PATH)
        return load_clean_evals_df(evals_url(), max_evals=8000, min_depth=20)

    def split_train_val(
        frame: pl.DataFrame, validation_fraction: float = 0.1, seed: int = 42
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        shuffled = frame.sample(fraction=1.0, shuffle=True, seed=seed)
        n_validation = max(1, int(validation_fraction * shuffled.height))
        return shuffled.head(n_validation), shuffled.tail(shuffled.height - n_validation)

    def measure_latency_milliseconds(
        model: torch.nn.Module, fen: str, repeats: int = 20
    ) -> float:
        start = time.perf_counter()
        for _ in range(repeats):
            predict_top_moves(model, fen, top_k=3)
        return 1000 * (time.perf_counter() - start) / repeats

    def parameter_count(model: torch.nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters())

    plt.rcParams.update(
        {"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100}
    )
    return (
        EVALS_PATH,
        MAIA_PATH,
        MODELS_DIR,
        STAGE2_PATH,
        ChessPolicyNet,
        MaiaResidualPolicyNet,
        chess,
        evaluate_model,
        load_evals,
        make_tensor_dataset,
        measure_latency_milliseconds,
        mo,
        parameter_count,
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
    # 3. Enfoque 3: arquitectura Maia (red residual)

    Maia es la familia de modelos de ajedrez "humano" del Computational Chess
    Lab (Universidad de Toronto), entrenada con partidas de Lichess. Su red usa
    una torre convolucional residual con bloques de *squeeze-excitation* (la
    arquitectura `lk` de Leela/Maia).

    Este cuaderno construye esa misma arquitectura
    (`MaiaResidualPolicyNet`) y la entrena sobre las evaluaciones de Stockfish,
    para compararla **con el mismo conjunto de validación y las mismas reglas**
    contra el modelo del enfoque 1 (desde cero + ajuste fino).

    **Sobre los pesos oficiales:** los checkpoints publicados de Maia vienen en
    formato Leela (`*.pb.gz`) y sus anchos de canal no coinciden con esta
    versión reducida. El punto de extensión está documentado en
    `src/lichess_pipeline/model/maia_adapter.py`: cuando haya pesos compatibles,
    se cargan con `load_state_dict(..., strict=False)` en lugar de entrenar
    desde cero.

    Requiere haber ejecutado `02_model_scratch_and_finetune.py` para tener el
    checkpoint del enfoque 1.
    """)
    return


@app.cell
def _(EVALS_PATH, load_evals, mo, pl):
    evals_df = load_evals()
    mo.vstack(
        [
            mo.md("## Datos"),
            mo.ui.table(
                pl.DataFrame(
                    {
                        "Fuente": ["Evaluaciones de Stockfish"],
                        "Archivo": [str(EVALS_PATH)],
                        "Posiciones": [evals_df.height],
                    }
                ),
                pagination=False,
            ),
            mo.ui.table(evals_df.head(3), pagination=False),
        ]
    )
    return (evals_df,)


@app.cell
def _(mo):
    max_train = mo.ui.slider(
        start=2000,
        stop=40000,
        step=2000,
        value=20000,
        label="Posiciones máximas de entrenamiento",
    )
    epochs = mo.ui.slider(start=1, stop=10, step=1, value=3, label="Épocas")
    train_button = mo.ui.run_button(label="Entrenar la red Maia")
    mo.vstack(
        [
            mo.md("## Entrenamiento de la red Maia"),
            mo.hstack([max_train, epochs], justify="start"),
            train_button,
        ]
    )
    return epochs, max_train, train_button


@app.cell
def _(
    epochs,
    evals_df,
    make_tensor_dataset,
    max_train,
    mo,
    split_train_val,
    train_button,
):
    mo.stop(
        not train_button.value,
        mo.md("Pulsa **Entrenar la red Maia** (en CPU tarda uno o dos minutos)."),
    )
    limit = int(max_train.value)
    eval_val, eval_train = split_train_val(evals_df.head(min(limit, evals_df.height)))
    eval_train_ds = make_tensor_dataset(eval_train, "fen", "best_move")
    eval_val_ds = make_tensor_dataset(eval_val, "fen", "best_move")
    mo.callout(
        mo.md(
            f"Entrenamiento: {eval_train.height:,} posiciones; "
            f"validación: {eval_val.height:,}."
        ),
        kind="success",
    )
    return eval_train_ds, eval_val_ds, eval_val


@app.cell
def _(
    MaiaResidualPolicyNet,
    MAIA_PATH,
    epochs,
    eval_train_ds,
    eval_val_ds,
    evaluate_model,
    mo,
    pl,
    torch,
    train_model,
):
    maia_model = MaiaResidualPolicyNet()
    maia_history = train_model(
        maia_model,
        eval_train_ds,
        eval_val_ds,
        epochs=int(epochs.value),
        lr=1e-3,
        batch_size=128,
    )
    maia_metrics = evaluate_model(maia_model, eval_val_ds, batch_size=128)
    torch.save(maia_model.state_dict(), MAIA_PATH)
    maia_frame = pl.DataFrame(maia_history)
    mo.vstack(
        [
            mo.md("### Red Maia entrenada"),
            mo.ui.table(maia_frame, pagination=False),
            mo.callout(mo.md(f"Pesos guardados en `{MAIA_PATH}`."), kind="success"),
        ]
    )
    return maia_frame, maia_metrics, maia_model


@app.cell
def _(
    ChessPolicyNet,
    STAGE2_PATH,
    eval_val_ds,
    evaluate_model,
    mo,
    torch,
):
    mo.stop(
        not STAGE2_PATH.exists(),
        mo.callout(
            mo.md(
                f"No existe `{STAGE2_PATH}`. Ejecuta primero "
                "`02_model_scratch_and_finetune.py` y vuelve a este cuaderno."
            ),
            kind="warn",
        ),
    )
    approach1_model = ChessPolicyNet()
    approach1_model.load_state_dict(torch.load(STAGE2_PATH, weights_only=True))
    approach1_metrics = evaluate_model(approach1_model, eval_val_ds, batch_size=128)
    mo.md("Checkpoint del enfoque 1 cargado y evaluado en la misma validación.")
    return approach1_metrics, approach1_model


@app.cell
def _(
    approach1_metrics,
    approach1_model,
    chess,
    eval_val,
    maia_metrics,
    maia_model,
    measure_latency_milliseconds,
    mo,
    parameter_count,
    pl,
    plt,
):
    validation_fens = eval_val["fen"].to_list()[:50]
    approach1_latency = sum(
        measure_latency_milliseconds(approach1_model, fen, repeats=1)
        for fen in validation_fens
    ) / len(validation_fens)
    maia_latency = sum(
        measure_latency_milliseconds(maia_model, fen, repeats=1)
        for fen in validation_fens
    ) / len(validation_fens)

    comparison = pl.DataFrame(
        [
            {
                "Modelo": "Enfoque 1 · desde cero + ajuste fino",
                "Top-1": approach1_metrics["top1"],
                "Top-3": approach1_metrics["top3"],
                "Pérdida": approach1_metrics["loss"],
                "Parámetros": parameter_count(approach1_model),
                "Latencia (ms/FEN)": approach1_latency,
            },
            {
                "Modelo": "Enfoque 3 · red Maia",
                "Top-1": maia_metrics["top1"],
                "Top-3": maia_metrics["top3"],
                "Pérdida": maia_metrics["loss"],
                "Parámetros": parameter_count(maia_model),
                "Latencia (ms/FEN)": maia_latency,
            },
        ]
    ).with_columns(
        (100 * pl.col("Top-1")).round(2).alias("Top-1 (%)"),
        (100 * pl.col("Top-3")).round(2).alias("Top-3 (%)"),
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    models = comparison["Modelo"].to_list()
    axes[0].bar(models, comparison["Top-1 (%)"].to_list(), label="Top-1")
    axes[0].bar(models, comparison["Top-3 (%)"].to_list(), alpha=0.5, label="Top-3")
    axes[0].set(title="Exactitud en la misma validación", ylabel="%")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].legend()

    axes[1].bar(models, comparison["Latencia (ms/FEN)"].to_list())
    axes[1].set(title="Latencia de predicción", ylabel="milisegundos por FEN")
    axes[1].tick_params(axis="x", rotation=12)

    figure.tight_layout()
    mo.vstack(
        [
            mo.md("## Comparación de los dos enfoques"),
            mo.ui.table(comparison, pagination=False),
            figure,
            mo.md(
                "Ambos modelos usan la misma máscara de jugadas legales, así que la "
                "diferencia es de calidad de ranking y de coste, no de legalidad. "
                "Con una muestra mínima los dos quedan lejos de un motor; la comparación "
                "se repite con el dataset completo en el servidor."
            ),
        ]
    )
    return comparison, figure


@app.cell
def _(chess, mo):
    presets = {
        "Personalizado (usar el FEN de abajo)": None,
        "Inicio": chess.STARTING_FEN,
        "Gambito de dama declinado": (
            "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 2 3"
        ),
    }
    preset = mo.ui.dropdown(
        options=list(presets),
        value="Personalizado (usar el FEN de abajo)",
        label="Posición de ejemplo",
    )
    fen_input = mo.ui.text_area(value=chess.STARTING_FEN, label="FEN escrito a mano")
    mo.vstack(
        [
            mo.md("## Predicción interactiva: Maia frente al enfoque 1"),
            preset,
            fen_input,
        ]
    )
    return fen_input, preset, presets


@app.cell
def _(
    approach1_model,
    chess,
    fen_input,
    maia_model,
    mo,
    pl,
    predict_top_moves,
    preset,
    presets,
):
    selected_preset = presets[preset.value]
    fen = selected_preset if selected_preset is not None else fen_input.value.strip()
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
        not board.has_legal_moves(),
        mo.callout(mo.md("La posición es terminal: no hay jugadas legales."), kind="warn"),
    )

    approach1_predictions = predict_top_moves(approach1_model, fen, top_k=3)
    maia_predictions = predict_top_moves(maia_model, fen, top_k=3)
    best_move = chess.Move.from_uci(maia_predictions[0][0])
    board_svg = chess.svg.board(board, lastmove=best_move, size=380, coordinates=True)

    rows = []
    for rank, ((first_move, first_probability), (second_move, second_probability)) in enumerate(
        zip(approach1_predictions, maia_predictions, strict=True), start=1
    ):
        rows.append(
            {
                "#": rank,
                "Enfoque 1": f"{first_move} · {100 * first_probability:.1f} %",
                "Maia": f"{second_move} · {100 * second_probability:.1f} %",
            }
        )
    mo.hstack(
        [
            mo.Html(str(board_svg)),
            mo.vstack(
                [
                    mo.md(f"**Mejor jugada de Maia:** `{best_move.uci()}`"),
                    mo.ui.table(pl.DataFrame(rows), pagination=False),
                    mo.md(
                        "Las dos columnas son jugadas legales. La flecha del tablero "
                        "corresponde a la mejor jugada de la red Maia."
                    ),
                ]
            ),
        ],
        justify="start",
    )
    return


if __name__ == "__main__":
    app.run()
