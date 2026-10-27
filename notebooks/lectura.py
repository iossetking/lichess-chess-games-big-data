import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    from functools import cache, lru_cache

    import marimo as mo
    import polars as pl

    from lichess_pipeline.data_ingestion import (
        DEFAULT_COLUMNS,
        EXPECTED_SCHEMA,
        REPOSITORY_ID,
        REPOSITORY_REVISION,
        REPOSITORY_REVISION_DATE,
        list_repo_parquet_files,
        month_download_size,
        month_shard_files,
        parse_month_partitions,
        remote_month_urls,
        scan_month,
    )

    WARN_GB = 5.0

    @lru_cache(maxsize=1)
    def cached_files() -> list[str]:
        # One HF API call per session, whatever the dropdown does.
        return list_repo_parquet_files(REPOSITORY_REVISION)

    @cache
    def cached_size(year: int, month: int) -> int:
        return month_download_size(year, month, REPOSITORY_REVISION, cached_files())

    return (
        DEFAULT_COLUMNS,
        EXPECTED_SCHEMA,
        REPOSITORY_ID,
        REPOSITORY_REVISION,
        REPOSITORY_REVISION_DATE,
        WARN_GB,
        cached_files,
        cached_size,
        mo,
        month_shard_files,
        parse_month_partitions,
        pl,
        remote_month_urls,
        scan_month,
    )


@app.cell(hide_code=True)
def _(REPOSITORY_ID, REPOSITORY_REVISION, REPOSITORY_REVISION_DATE, mo):
    mo.md(rf"""
    # Lectura de partidas de Lichess


    Fuente: `{REPOSITORY_ID}` en la revisión fijada
    `{REPOSITORY_REVISION[:8]}` ({REPOSITORY_REVISION_DATE}).
    """)
    return


@app.cell
def _(
    DEFAULT_COLUMNS,
    EXPECTED_SCHEMA,
    cached_files,
    mo,
    parse_month_partitions,
):
    months = parse_month_partitions(cached_files())
    month_selector = mo.ui.dropdown(
        options=months, value=months[-1], label="Mes (AAAA-MM)"
    )
    column_selector = mo.ui.multiselect(
        options=list(EXPECTED_SCHEMA),
        value=DEFAULT_COLUMNS,
        label="Columnas",
    )
    mo.vstack([month_selector, column_selector])
    return column_selector, month_selector


@app.cell
def _(
    WARN_GB,
    cached_files,
    cached_size,
    mo,
    month_selector,
    month_shard_files,
):
    year, month = (int(part) for part in month_selector.value.split("-"))
    n_shards = len(month_shard_files(cached_files(), year, month))
    size_gb = cached_size(year, month) / 1e9
    preview = mo.md(
        f"**{month_selector.value}**: {n_shards} shard(s), {size_gb:.2f} GB en Hugging Face."
    )
    if size_gb >= WARN_GB:
        preview = mo.vstack(
            [
                preview,
                mo.callout(
                    mo.md(
                        f"Mes grande (>= {WARN_GB:.0f} GB): contar filas y ver la "
                        "muestra es rápido, pero procesarlo completo transfiere todo el mes."
                    ),
                    kind="warn",
                ),
            ]
        )
    run_button = mo.ui.run_button(label="Leer")
    mo.vstack([preview, run_button])
    return month, run_button, year


@app.cell
def _(
    cached_files,
    column_selector,
    mo,
    month,
    remote_month_urls,
    run_button,
    scan_month,
    year,
):
    mo.stop(
        not run_button.value,
        mo.md("Pulsa **Leer** para continuar (también tras cambiar las columnas)."),
    )
    shard_paths = remote_month_urls(year, month, files=cached_files())
    try:
        # scan_month validates the schema and raises on mismatch.
        games = scan_month(shard_paths, column_selector.value)
        validation = mo.callout(mo.md("Esquema válido."), kind="success")
    except ValueError as error:
        mo.stop(True, mo.callout(mo.md(str(error)), kind="danger"))
    return games, shard_paths, validation


@app.cell
def _(EXPECTED_SCHEMA, games, mo, pl, shard_paths, validation):
    n_rows = games.select(pl.len()).collect().item()
    read_schema = games.collect_schema()
    schema_table = pl.DataFrame(
        {
            "columna": read_schema.names(),
            "tipo_leido": [str(t) for t in read_schema.dtypes()],
            "tipo_esperado": [str(EXPECTED_SCHEMA[c]) for c in read_schema.names()],
        }
    )
    mo.vstack(
        [
            mo.md(f"**{n_rows:,}** filas en {len(shard_paths)} shard(s)."),
            validation,
            mo.md("### Esquema leído vs esperado"),
            schema_table,
            mo.md("### Primeras 10 filas"),
            games.head(10).collect(),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
