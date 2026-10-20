import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    from datetime import date
    from pathlib import Path
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from huggingface_hub import hf_hub_download

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eda_helpers import (
        DATA_DICTIONARY,
        iqr_outlier_summary,
        movetext_character_lengths,
        movetext_length_summary,
        plot_top_categories,
        utc_hour_counts,
    )

    return (
        DATA_DICTIONARY,
        Path,
        date,
        hf_hub_download,
        iqr_outlier_summary,
        mo,
        movetext_character_lengths,
        movetext_length_summary,
        np,
        pl,
        plot_top_categories,
        plt,
        utc_hour_counts,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploración de partidas estándar de Lichess

    Este análisis explora los tres shards de enero de 2015. Las estadísticas y
    conteos se calculan sobre todas las filas del mes; para mantener legibles las
    visualizaciones densas se emplean muestras reproducibles cuando corresponde.
    """)
    return


@app.cell
def _(Path, hf_hub_download, pl):
    REPOSITORY_ID = "Lichess/standard-chess-games"
    REPOSITORY_URL = "https://huggingface.co/datasets/Lichess/standard-chess-games"
    SOURCE_URL = "https://database.lichess.org/#standard_games"
    REPOSITORY_REVISION = "de4e636eddf568a9394cc01fb0b9e1da04a6babf"
    REPOSITORY_REVISION_DATE = "2025-10-16"
    SHARD_FILENAMES = [
        f"data/year=2015/month=01/train-{index:05d}-of-00003.parquet"
        for index in range(3)
    ]
    CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "huggingface"

    shard_paths = [
        hf_hub_download(
            repo_id=REPOSITORY_ID,
            filename=filename,
            repo_type="dataset",
            revision=REPOSITORY_REVISION,
            cache_dir=CACHE_DIR,
        )
        for filename in SHARD_FILENAMES
    ]
    games = pl.scan_parquet(shard_paths)
    return (
        REPOSITORY_ID,
        REPOSITORY_REVISION,
        REPOSITORY_REVISION_DATE,
        REPOSITORY_URL,
        SHARD_FILENAMES,
        SOURCE_URL,
        games,
        shard_paths,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Fuente, versión y alcance
    """)
    return


@app.cell(hide_code=True)
def _(
    Path,
    REPOSITORY_ID,
    REPOSITORY_REVISION,
    REPOSITORY_REVISION_DATE,
    REPOSITORY_URL,
    SHARD_FILENAMES,
    SOURCE_URL,
    date,
    games,
    mo,
    pl,
    shard_paths,
):
    source_stats = games.select(
        pl.len().alias("partidas"),
        pl.col("UTCDate").min().alias("fecha_minima"),
        pl.col("UTCDate").max().alias("fecha_maxima"),
    ).collect()
    total_size_mb = sum(Path(path).stat().st_size for path in shard_paths) / (1024**2)

    dataset_metadata = pl.DataFrame(
        {
            "Metadato": [
                "Dataset",
                "Repositorio",
                "Fuente original",
                "Licencia",
                "Fecha de consulta",
                "Revisión fijada",
                "Fecha de revisión",
                "Partición del repositorio",
                "Cobertura de particiones de esta revisión",
                "Ámbito analizado",
                "Shards incluidos",
                "Tamaño comprimido local",
            ],
            "Valor": [
                REPOSITORY_ID,
                REPOSITORY_URL,
                SOURCE_URL,
                "CC0-1.0",
                date.today().isoformat(),
                REPOSITORY_REVISION,
                REPOSITORY_REVISION_DATE,
                "Parquet particionado por año y mes; split train",
                "Años 2013–2025; última partición mensual disponible: 2025-09",
                (
                    f"Enero de 2015; {source_stats['partidas'][0]:,} partidas; "
                    f"UTCDate entre {source_stats['fecha_minima'][0]} y "
                    f"{source_stats['fecha_maxima'][0]}"
                ),
                "Tres archivos train-xxxxx-of-00003.parquet (partes del mismo mes)",
                f"{total_size_mb:,.1f} MiB en {len(SHARD_FILENAMES)} archivos",
            ],
        }
    )

    mo.vstack(
        [
            mo.md(
                "Los shards son fragmentos de almacenamiento del mismo mes, "
                "no meses distintos. La revisión se fija para que los resultados "
                "puedan reproducirse aunque la rama `main` avance."
            ),
            mo.ui.table(dataset_metadata, pagination=False)
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Esquema y diccionario de datos
    """)
    return


@app.cell
def _(DATA_DICTIONARY, games, mo, pl):
    actual_schema = games.collect_schema()
    schema_table = pl.DataFrame(
        {
            "Variable": actual_schema.names(),
            "Tipo": [str(actual_schema[name]) for name in actual_schema.names()],
            "Descripción": [DATA_DICTIONARY[name]["description"] for name in actual_schema.names()],
            "Clasificación": [
                DATA_DICTIONARY[name]["classification"] for name in actual_schema.names()
            ],
        }
    )
    # schema_table
    mo.ui.table(schema_table, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Calidad y estructura
    """)
    return


@app.cell
def _(games, mo, pl):
    source_columns = games.collect_schema().names()
    quality_stats = games.select(
        pl.len().alias("filas"),
        pl.col("Site").n_unique().alias("URLs_unicas"),
        pl.col("Site").is_duplicated().sum().alias("URLs_duplicadas"),
        *[
            pl.col(column).null_count().alias(f"{column}_nulos")
            for column in source_columns
        ],
    ).collect()
    row_count = quality_stats["filas"][0]
    null_summary = pl.DataFrame(
        {
            "Variable": source_columns,
            "Nulos": [quality_stats[f"{column}_nulos"][0] for column in source_columns],
        }
    ).with_columns(
        (100 * pl.col("Nulos") / row_count).round(3).alias("Nulos (%)")
    ).sort("Nulos", descending=True)
    valid_results = ["1-0", "0-1", "1/2-1/2"]
    invalid_result_count = games.filter(~pl.col("Result").is_in(valid_results)).select(
        pl.len()
    ).collect().item()
    quality_overview = pl.DataFrame(
        {
            "Indicador": [
                "Filas",
                "Columnas",
                "URLs únicas de partida",
                "Filas con URL duplicada",
                "Resultados fuera de las tres categorías esperadas",
            ],
            "Valor": [
                row_count,
                len(source_columns),
                quality_stats["URLs_unicas"][0],
                quality_stats["URLs_duplicadas"][0],
                invalid_result_count,
            ],
        }
    )
    mo.vstack([
        quality_overview,
        mo.md("#### Valores faltantes por variable"),
        mo.ui.table(null_summary, pagination=False)
    ])
    return null_summary, quality_stats


@app.cell
def _(games):
    sample_rows = games.head(5).collect()
    sample_rows
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Distribuciones categóricas y temporales
    """)
    return


@app.cell
def _(games, pl, plot_top_categories, plt, utc_hour_counts):
    categorical_columns = [
        "Result",
        "Termination",
        "Event",
        "TimeControl",
        "ECO",
        "Opening",
        "WhiteTitle",
        "BlackTitle",
    ]
    top_categories = {
        column: games.group_by(column)
        .len(name="partidas")
        .sort("partidas", descending=True)
        .head(10)
        .collect()
        for column in categorical_columns
    }
    daily_counts = (
        games.group_by("UTCDate")
        .len(name="partidas")
        .sort("UTCDate")
        .collect()
    )
    weekday_counts = (
        games.with_columns(pl.col("UTCDate").dt.weekday().alias("dia_semana"))
        .group_by("dia_semana")
        .len(name="partidas")
        .sort("dia_semana")
        .collect()
    )
    hourly_counts = utc_hour_counts(games)
    category_figure = plot_top_categories(top_categories, plt)
    category_figure
    return daily_counts, hourly_counts, weekday_counts


@app.cell
def _(daily_counts, hourly_counts, plt, weekday_counts):
    weekday_names = ["", "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    temporal_figure, temporal_axes = plt.subplots(3, 1, figsize=(12, 12))

    temporal_axes[0].plot(
        daily_counts["UTCDate"].to_list(),
        daily_counts["partidas"].to_list(),
        linewidth=1,
    )
    temporal_axes[0].set(title="Partidas por fecha UTC", xlabel="Fecha", ylabel="Partidas")
    temporal_axes[0].tick_params(axis="x", rotation=30)

    temporal_axes[1].bar(
        [weekday_names[day] for day in weekday_counts["dia_semana"].to_list()],
        weekday_counts["partidas"].to_list(),
    )
    temporal_axes[1].set(title="Partidas por día de la semana UTC", ylabel="Partidas")
    temporal_axes[1].tick_params(axis="x", rotation=20)

    temporal_axes[2].bar(hourly_counts["hora_utc"].to_list(), hourly_counts["partidas"].to_list())
    temporal_axes[2].set(
        title="Partidas por hora UTC",
        xlabel="Hora del día",
        ylabel="Partidas",
        xticks=range(0, 24),
    )
    temporal_figure.tight_layout()
    temporal_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Longitud de `movetext`
    """)
    return


@app.cell
def _(games, mo, movetext_character_lengths, movetext_length_summary, plt):
    movetext_lengths_all = movetext_character_lengths(games)
    movetext_lengths = movetext_lengths_all.drop_nulls()
    movetext_summary = movetext_length_summary(movetext_lengths_all)

    movetext_figure, movetext_axes = plt.subplots(1, 2, figsize=(13, 4.5))
    movetext_axes[0].hist(movetext_lengths.to_numpy(), bins=60, color="tab:blue", alpha=0.85)
    movetext_axes[0].set(
        title="Distribución de longitud de movetext",
        xlabel="Caracteres de notación PGN",
        ylabel="Partidas",
    )
    movetext_axes[1].boxplot(
        movetext_lengths.to_numpy(),
        orientation="horizontal",
        tick_labels=["movetext"],
        showfliers=False,
    )
    movetext_axes[1].set(
        title="Caja y bigotes (sin dibujar cada atípico)",
        xlabel="Caracteres de notación PGN",
    )
    movetext_figure.tight_layout()

    mo.vstack(
        [
            mo.md(
                "La longitud cuenta caracteres Unicode de la notación PGN; "
                "sirve como aproximación al tamaño de la partida, no como conteo exacto de jugadas."
            ),
            mo.ui.table(movetext_summary, pagination=False),
            movetext_figure,
        ]
    )
    return movetext_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Ratings, relaciones y valores atípicos
    """)
    return


@app.cell
def _(games, iqr_outlier_summary, mo, pl):
    numeric_columns = ["WhiteElo", "BlackElo", "WhiteRatingDiff", "BlackRatingDiff"]
    numeric_data = games.select(numeric_columns).collect()
    descriptive_rows = []
    iqr_rows = []
    for column in numeric_columns:
        values = numeric_data[column].drop_nulls()
        descriptive_rows.append(
            {
                "Variable": column,
                "N válido": len(values),
                "Media": values.mean(),
                "Mediana": values.median(),
                "Desv. estándar": values.std(),
                "Mínimo": values.min(),
                "Máximo": values.max(),
            }
        )
        iqr_rows.append({"Variable": column, **iqr_outlier_summary(numeric_data[column])})

    descriptive_summary = pl.DataFrame(descriptive_rows)
    iqr_summary = pl.DataFrame(iqr_rows).select(
        "Variable",
        "q1",
        "q3",
        "iqr",
        "lower_fence",
        "upper_fence",
        "outlier_count",
        "valid_count",
        "outlier_percent",
    ).rename(
        {
            "q1": "Q1",
            "q3": "Q3",
            "iqr": "IQR",
            "lower_fence": "Límite inferior",
            "upper_fence": "Límite superior",
            "outlier_count": "Atípicos IQR",
            "valid_count": "N válido",
            "outlier_percent": "Atípicos (%)",
        }
    )
    mo.vstack(
        [
            mo.md("### Estadísticos descriptivos de las variables numéricas"),
            descriptive_summary,
            mo.md(
                "### Detección de atípicos con IQR\n"
                "Se señalan observaciones fuera de Q1 − 1.5 × IQR y Q3 + 1.5 × IQR. "
                "Son casos para inspección, no registros descartados automáticamente."
            ),
            iqr_summary,
        ]
    )
    return iqr_summary, numeric_columns, numeric_data


@app.cell
def _(iqr_summary, np, numeric_columns, numeric_data, pl, plt):
    boxplot_figure, boxplot_axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, columns, title in [
        (boxplot_axes[0], numeric_columns[:2], "Ratings iniciales"),
        (boxplot_axes[1], numeric_columns[2:], "Cambios de rating"),
    ]:
        axis.boxplot(
            [numeric_data[column].drop_nulls().to_numpy() for column in columns],
            tick_labels=columns,
            showfliers=False,
        )
        axis.set_title(title)
        axis.set_ylabel("Puntos de rating")
        axis.tick_params(axis="x", rotation=15)
    boxplot_figure.suptitle("Diagramas de caja (bigotes hasta 1.5 × IQR; fliers resumidos en tabla)")
    boxplot_figure.tight_layout()

    sample_size = min(20_000, numeric_data.height)
    scatter_sample = numeric_data.sample(n=sample_size, seed=42) if sample_size else numeric_data
    rating_rows = iqr_summary.filter(pl.col("Variable").is_in(["WhiteElo", "BlackElo"]))
    rating_fences = {
        row["Variable"]: (row["Límite inferior"], row["Límite superior"])
        for row in rating_rows.to_dicts()
    }
    sampled_white = scatter_sample["WhiteElo"].to_numpy()
    sampled_black = scatter_sample["BlackElo"].to_numpy()
    valid_mask = ~(np.isnan(sampled_white) | np.isnan(sampled_black))
    rating_outlier_mask = (
        (sampled_white < rating_fences["WhiteElo"][0])
        | (sampled_white > rating_fences["WhiteElo"][1])
        | (sampled_black < rating_fences["BlackElo"][0])
        | (sampled_black > rating_fences["BlackElo"][1])
    )

    relation_figure, relation_axes = plt.subplots(1, 2, figsize=(13, 5))
    relation_axes[0].scatter(
        sampled_white[valid_mask & ~rating_outlier_mask],
        sampled_black[valid_mask & ~rating_outlier_mask],
        s=8,
        alpha=0.25,
        label="Muestra sin rating extremo",
    )
    relation_axes[0].scatter(
        sampled_white[valid_mask & rating_outlier_mask],
        sampled_black[valid_mask & rating_outlier_mask],
        s=12,
        alpha=0.65,
        color="tab:red",
        label="Muestra con rating extremo (IQR)",
    )
    relation_axes[0].set(
        title="Rating de blancas vs. negras (muestra reproducible)",
        xlabel="WhiteElo",
        ylabel="BlackElo",
    )
    relation_axes[0].legend()

    sampled_white_diff = scatter_sample["WhiteRatingDiff"].to_numpy()
    sampled_black_diff = scatter_sample["BlackRatingDiff"].to_numpy()
    diff_mask = ~(np.isnan(sampled_white_diff) | np.isnan(sampled_black_diff))
    relation_axes[1].scatter(
        sampled_white_diff[diff_mask],
        sampled_black_diff[diff_mask],
        s=8,
        alpha=0.25,
    )
    relation_axes[1].axhline(0, color="black", linewidth=0.7)
    relation_axes[1].axvline(0, color="black", linewidth=0.7)
    relation_axes[1].set(
        title="Cambio de rating por color (muestra reproducible)",
        xlabel="WhiteRatingDiff",
        ylabel="BlackRatingDiff",
    )
    relation_figure.tight_layout()
    boxplot_figure, relation_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Consideraciones

    - `WhiteElo` y `BlackElo` son ratings Glicko-2, según la ficha del dataset.
    - Los puntos fuera de los límites IQR se reportan como candidatos a revisión;
      en partidas reales pueden ser valores válidos.
    - La longitud de `movetext` mide caracteres de notación PGN, no equivale
      directamente al número de jugadas.
    - Las frecuencias, resúmenes y conteos de atípicos se calculan con las tres
      partes completas del mes; la muestra de los gráficos de dispersión usa semilla fija.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Métodos de limpieza propuestos

    Las siguientes propuestas corresponden a las variables y controles del
    análisis. No se aplican cambios al dataset.

    | Problema | Variables | Método de limpieza propuesto | Criterio |
    |---|---|---|---|
    | Títulos nulos | `WhiteTitle`, `BlackTitle` | Mostrar como **Sin título registrado**. | Conservar las partidas; no asumir que el jugador carece de título. |
    | Ratings nulos | `WhiteElo`, `BlackElo` | Excluir del análisis de ratings. Para modelos que requieran datos completos, imputar la mediana por control de tiempo y marcar la imputación. | Calcular la mediana solo con datos de entrenamiento. |
    | Cambios de rating nulos | `WhiteRatingDiff`, `BlackRatingDiff` | Mantener como desconocidos y excluir del análisis de cambios. | No sustituir por cero: cero significa que no hubo cambio. |
    | Apertura o categorías nulas | `ECO`, `Opening`, `Event`, `Termination`, `TimeControl` | Recuperar de la fuente o etiquetar como **Desconocido**. | Evitar rellenar con la categoría más frecuente porque altera la distribución. |
    | Campos esenciales nulos | `Site`, `Result`, `UTCDate`, `UTCTime` | Recuperar de la fuente; si no es posible, excluir únicamente del análisis que necesite ese campo. | No inventar identificadores, resultados ni fechas. |
    | Notación nula | `movetext` | Recuperar el PGN o excluir de análisis de jugadas y longitud. | La partida puede conservarse para otros análisis. |
    | Filas completamente duplicadas | Todas las columnas | Conservar una sola copia. | Confirmar que todos los campos coinciden. |
    | URL repetida con datos diferentes | `Site` | Comparar los registros con la fuente y conservar la versión validada. | No eliminar automáticamente; no agrupar las URL nulas como una misma partida. |
    | Ratings atípicos | `WhiteElo`, `BlackElo` | Marcar mediante IQR y revisar; conservar los extremos válidos. | Un rating fuera de los límites IQR no demuestra un error. |
    | Cambios de rating atípicos | `WhiteRatingDiff`, `BlackRatingDiff` | Validar contra la fuente; corregir errores confirmados o convertirlos a nulos si no pueden recuperarse. | Los cambios extremos pueden ser reales; no eliminarlos solo por IQR. |
    | Longitud atípica de notación | `movetext` | Validar la sintaxis PGN y revisar truncamientos o contenido incorrecto. | Conservar partidas largas válidas; no recortar por cantidad de caracteres. |

    ### Conteos del análisis y tratamientos por variable

    Ejecuta las secciones 3, 4 y 5 para obtener los conteos de nulos,
    duplicados y valores atípicos en la tabla complementaria siguiente.

    Los conteos son por variable y pueden corresponder a una misma partida,
    por lo que no deben sumarse. Cuando el conteo es cero, el método queda
    como regla preventiva. Tras aplicar una limpieza, compara las filas antes
    y después, repite los controles de calidad y registra las correcciones
    y exclusiones.
    """)
    return


@app.cell(hide_code=True)
def _(iqr_summary, mo, movetext_summary, null_summary, pl, quality_stats):
    cleaning_rows = []
    for null_row in null_summary.to_dicts():
        variable = null_row["Variable"]
        if variable in ("WhiteTitle", "BlackTitle"):
            treatment = (
                "Conservar el nulo como título no registrado; para visualización, usar "
                "'Sin título registrado'. No inferir que el jugador carece de título."
            )
        elif variable in ("WhiteElo", "BlackElo"):
            treatment = (
                "Conservar y marcar el faltante; excluir solo del análisis que requiere rating. "
                "Si un modelo exige imputación, usar la mediana por control de tiempo "
                "calculada en entrenamiento y añadir un indicador de imputación."
            )
        elif variable in ("WhiteRatingDiff", "BlackRatingDiff"):
            treatment = (
                "Conservar y marcar como cambio desconocido; excluir del análisis de cambios. "
                "No rellenar con cero, porque cero representa un cambio observado."
            )
        elif variable in ("Site", "Result", "UTCDate", "UTCTime", "movetext"):
            treatment = (
                "Intentar recuperar de la fuente original; si no es posible, conservar "
                "con una marca de calidad y excluir solo de los análisis que requieren "
                "este campo. No inventar identificadores, resultados, fechas ni jugadas."
            )
        else:
            treatment = (
                "Conservar el faltante; usar 'Desconocido' en resúmenes categóricos. "
                "Recuperar de la fuente cuando sea posible, sin sustituir por la moda."
            )
        cleaning_rows.append({
            "Problema": "Nulos",
            "Variable": variable,
            "Resultado del análisis": f"{null_row['Nulos']:,} ({null_row['Nulos (%)']:.3f} %)",
            "Método de limpieza propuesto": treatment,
        })

    cleaning_rows.append({
        "Problema": "Duplicados por identificador",
        "Variable": "Site",
        "Resultado del análisis": (
            f"{quality_stats['URLs_duplicadas'][0]:,} filas con URL repetida "
            "(incluye todas las apariciones de cada grupo)"
        ),
        "Método de limpieza propuesto": (
            "Revisar grupos de Site no nulo: si todas las columnas coinciden, conservar "
            "una fila; si hay diferencias, contrastar con la fuente y resolver el conflicto "
            "antes de eliminar. No deduplicar las URL nulas como si fueran una misma partida."
        ),
    })
    for outlier_row in iqr_summary.to_dicts():
        cleaning_rows.append({
            "Problema": "Valores atípicos (IQR)",
            "Variable": outlier_row["Variable"],
            "Resultado del análisis": (
                f"{outlier_row['Atípicos IQR']:,} ({outlier_row['Atípicos (%)']:.3f} %); "
                f"límites: {outlier_row['Límite inferior']} a {outlier_row['Límite superior']}"
            ),
            "Método de limpieza propuesto": (
                "Marcar los valores fuera de Q1 − 1.5 × IQR y Q3 + 1.5 × IQR; "
                "validar contra la fuente y conservar los extremos reales. Corregir errores "
                "confirmados o convertirlos a nulos si no son recuperables. Para modelos "
                "sensibles, evaluar escalado robusto o winsorización en una copia, con "
                "umbrales calculados solo en entrenamiento."
            ),
        })
    length_metrics = dict(movetext_summary.iter_rows())
    cleaning_rows.append({
        "Problema": "Valores atípicos de longitud (IQR)",
        "Variable": "movetext",
        "Resultado del análisis": (
            f"{length_metrics['Atípicos IQR']} ({length_metrics['Atípicos IQR (%)']} %); "
            f"máximo: {length_metrics['Máximo (caracteres)']} caracteres"
        ),
        "Método de limpieza propuesto": (
            "Revisar la sintaxis PGN y posibles truncamientos; conservar partidas largas "
            "válidas. Si se necesita contar jugadas, analizar el PGN con un parser. "
            "No recortar el texto por superar el umbral IQR."
        ),
    })
    cleaning_methods = pl.DataFrame(cleaning_rows)
    mo.ui.table(cleaning_methods, pagination=False)
    return


if __name__ == "__main__":
    app.run()
