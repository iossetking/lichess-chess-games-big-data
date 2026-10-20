"""Small, testable helpers shared by the Lichess EDA notebook."""

import polars as pl


DATA_DICTIONARY = {
    "Event": {
        "type": "String",
        "description": "Nombre o categoría del evento de la partida.",
        "classification": "Categórica nominal",
    },
    "Site": {
        "type": "String (URL)",
        "description": "URL única de la partida en Lichess; funciona como identificador.",
        "classification": "Categórica nominal (identificador)",
    },
    "White": {
        "type": "String",
        "description": "Nombre de usuario del jugador que condujo las piezas blancas.",
        "classification": "Categórica nominal (identificador)",
    },
    "Black": {
        "type": "String",
        "description": "Nombre de usuario del jugador que condujo las piezas negras.",
        "classification": "Categórica nominal (identificador)",
    },
    "Result": {
        "type": "String",
        "description": "Resultado PGN: victoria blanca, victoria negra o tablas.",
        "classification": "Categórica nominal",
    },
    "WhiteTitle": {
        "type": "String (nullable)",
        "description": "Título FIDE registrado para el jugador de blancas, si existe.",
        "classification": "Categórica nominal",
    },
    "BlackTitle": {
        "type": "String (nullable)",
        "description": "Título FIDE registrado para el jugador de negras, si existe.",
        "classification": "Categórica nominal",
    },
    "WhiteElo": {
        "type": "Int16 (nullable)",
        "description": "Rating Glicko-2 del jugador de blancas al inicio de la partida.",
        "classification": "Numérica discreta",
    },
    "BlackElo": {
        "type": "Int16 (nullable)",
        "description": "Rating Glicko-2 del jugador de negras al inicio de la partida.",
        "classification": "Numérica discreta",
    },
    "WhiteRatingDiff": {
        "type": "Int16 (nullable)",
        "description": "Cambio de rating del jugador de blancas tras la partida.",
        "classification": "Numérica discreta",
    },
    "BlackRatingDiff": {
        "type": "Int16 (nullable)",
        "description": "Cambio de rating del jugador de negras tras la partida.",
        "classification": "Numérica discreta",
    },
    "UTCDate": {
        "type": "Date",
        "description": "Fecha UTC en la que se jugó la partida.",
        "classification": "Temporal",
    },
    "UTCTime": {
        "type": "Time",
        "description": "Hora UTC de inicio de la partida.",
        "classification": "Temporal",
    },
    "ECO": {
        "type": "String",
        "description": "Código ECO de apertura de ajedrez.",
        "classification": "Categórica nominal",
    },
    "Opening": {
        "type": "String",
        "description": "Nombre de la apertura registrada para la partida.",
        "classification": "Categórica nominal",
    },
    "Termination": {
        "type": "String",
        "description": "Motivo de finalización, por ejemplo normal o caída de tiempo.",
        "classification": "Categórica nominal",
    },
    "TimeControl": {
        "type": "String",
        "description": "Control de tiempo PGN, normalmente segundos+incremento.",
        "classification": "Categórica nominal",
    },
    "movetext": {
        "type": "String (PGN)",
        "description": "Notación de jugadas y comentarios PGN de la partida.",
        "classification": "Categórica nominal (texto de alta cardinalidad)",
    },
}


def iqr_outlier_summary(series: pl.Series) -> dict[str, float | int | None]:
    """Summarize Tukey IQR fences and the observed values outside them."""
    values = series.drop_nulls()
    valid_count = len(values)

    if valid_count == 0:
        return {
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_fence": None,
            "upper_fence": None,
            "outlier_count": 0,
            "valid_count": 0,
            "outlier_percent": 0.0,
        }

    q1 = float(values.quantile(0.25, interpolation="linear"))
    q3 = float(values.quantile(0.75, interpolation="linear"))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_count = int(((values < lower_fence) | (values > upper_fence)).sum())

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "outlier_count": outlier_count,
        "valid_count": valid_count,
        "outlier_percent": 100 * outlier_count / valid_count,
    }


def utc_hour_counts(games: pl.LazyFrame) -> pl.DataFrame:
    """Count games by UTC hour, excluding records without a recorded time."""
    return (
        games.filter(pl.col("UTCTime").is_not_null())
        .with_columns(pl.col("UTCTime").dt.hour().alias("hora_utc"))
        .group_by("hora_utc")
        .len(name="partidas")
        .sort("hora_utc")
        .collect()
    )


def plot_top_categories(top_categories: dict[str, pl.DataFrame], plt):
    """Render top-category counts as readable horizontal bar charts."""
    import math

    if not top_categories:
        return plt.figure(figsize=(8, 3))

    column_count = min(2, len(top_categories))
    row_count = math.ceil(len(top_categories) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(16, 4.5 * row_count),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for axis, (column, categories) in zip(flat_axes, top_categories.items()):
        ordered_categories = categories.sort("partidas")
        labels = (
            ordered_categories[column]
            .cast(pl.String)
            .fill_null("Sin dato")
            .to_list()
        )
        counts = ordered_categories["partidas"].to_list()
        bars = axis.barh(labels, counts, color="tab:blue", alpha=0.85)
        axis.bar_label(bars, padding=3, fmt="{:,.0f}")
        axis.set_title(column)
        axis.set_xlabel("Partidas")
        axis.grid(axis="x", linestyle="--", alpha=0.25)
        axis.set_axisbelow(True)
        if counts:
            axis.set_xlim(0, max(counts) * 1.18)

    for axis in flat_axes[len(top_categories) :]:
        axis.remove()

    figure.tight_layout()
    return figure


def movetext_character_lengths(games: pl.LazyFrame) -> pl.Series:
    """Return PGN movetext lengths in Unicode characters, retaining nulls."""
    return games.select(
        pl.col("movetext").str.len_chars().alias("longitud_caracteres")
    ).collect()["longitud_caracteres"]


def movetext_length_summary(lengths: pl.Series) -> pl.DataFrame:
    """Format movetext length statistics as a display-ready table."""
    valid_lengths = lengths.drop_nulls()
    valid_count = len(valid_lengths)
    iqr_summary = iqr_outlier_summary(valid_lengths)

    def format_number(value, precision: int = 0) -> str:
        if value is None:
            return "Sin datos"
        return f"{value:,.{precision}f}"

    return pl.DataFrame(
        {
            "Métrica": [
                "Partidas con longitud disponible",
                "Partidas sin movetext",
                "Media (caracteres)",
                "Mediana (caracteres)",
                "Percentil 90 (caracteres)",
                "Máximo (caracteres)",
                "Atípicos IQR",
                "Atípicos IQR (%)",
            ],
            "Valor": [
                format_number(valid_count),
                format_number(len(lengths) - valid_count),
                format_number(valid_lengths.mean(), 2),
                format_number(valid_lengths.median(), 2),
                format_number(valid_lengths.quantile(0.9, interpolation="linear"), 2),
                format_number(valid_lengths.max()),
                format_number(iqr_summary["outlier_count"]),
                format_number(iqr_summary["outlier_percent"], 3),
            ],
        }
    )
