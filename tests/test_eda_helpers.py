import ast
import unittest
from datetime import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from notebooks.eda_helpers import (
    DATA_DICTIONARY,
    iqr_outlier_summary,
    movetext_character_lengths,
    movetext_length_summary,
    plot_top_categories,
    utc_hour_counts,
)


class EdaHelpersTest(unittest.TestCase):
    def test_data_dictionary_covers_all_source_columns(self):
        expected_columns = {
            "Event",
            "Site",
            "White",
            "Black",
            "Result",
            "WhiteTitle",
            "BlackTitle",
            "WhiteElo",
            "BlackElo",
            "WhiteRatingDiff",
            "BlackRatingDiff",
            "UTCDate",
            "UTCTime",
            "ECO",
            "Opening",
            "Termination",
            "TimeControl",
            "movetext",
        }

        self.assertEqual(set(DATA_DICTIONARY), expected_columns)
        self.assertTrue(
            all(
                {"type", "description", "classification"} <= set(metadata)
                for metadata in DATA_DICTIONARY.values()
            )
        )

    def test_iqr_summary_reports_fences_and_outlier_rate(self):
        summary = iqr_outlier_summary(pl.Series("rating", [0, 1, 2, 3, 4, 5, 100]))

        self.assertEqual(summary["q1"], 1.5)
        self.assertEqual(summary["q3"], 4.5)
        self.assertEqual(summary["lower_fence"], -3.0)
        self.assertEqual(summary["upper_fence"], 9.0)
        self.assertEqual(summary["outlier_count"], 1)
        self.assertEqual(summary["valid_count"], 7)
        self.assertAlmostEqual(summary["outlier_percent"], 100 / 7)

    def test_iqr_summary_handles_null_and_constant_values(self):
        summary = iqr_outlier_summary(pl.Series("rating", [None, 1200, 1200, 1200]))

        self.assertEqual(summary["valid_count"], 3)
        self.assertEqual(summary["outlier_count"], 0)
        self.assertEqual(summary["lower_fence"], 1200.0)
        self.assertEqual(summary["upper_fence"], 1200.0)
        self.assertEqual(summary["outlier_percent"], 0.0)

    def test_utc_hour_counts_excludes_missing_times(self):
        frame = pl.DataFrame(
            {"UTCTime": [time(0, 5), None, time(23, 40), time(0, 35)]}
        ).lazy()

        counts = utc_hour_counts(frame)

        self.assertEqual(counts.to_dict(as_series=False), {"hora_utc": [0, 23], "partidas": [2, 1]})

    def test_numeric_visualization_cell_declares_polars_dependency(self):
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "eda.py"
        notebook = ast.parse(notebook_path.read_text())
        plot_cells = [
            node
            for node in ast.walk(notebook)
            if isinstance(node, ast.FunctionDef)
            and any(isinstance(child, ast.Name) and child.id == "iqr_summary" for child in ast.walk(node))
            and any(isinstance(child, ast.Name) and child.id == "rating_rows" for child in ast.walk(node))
            and any(isinstance(child, ast.Name) and child.id == "pl" for child in ast.walk(node))
        ]

        self.assertEqual(len(plot_cells), 1)
        self.assertIn("pl", {argument.arg for argument in plot_cells[0].args.args})

    def test_top_categories_are_rendered_as_horizontal_bars(self):
        categories = {
            "Result": pl.DataFrame(
                {"Result": ["1-0", "0-1"], "partidas": [5, 3]}
            )
        }

        figure = plot_top_categories(categories, plt)

        self.assertEqual(len(figure.axes), 1)
        self.assertEqual(len(figure.axes[0].patches), 2)
        self.assertEqual(figure.axes[0].get_xlabel(), "Partidas")
        plt.close(figure)

    def test_movetext_character_lengths_preserve_empty_and_missing_values(self):
        games = pl.DataFrame(
            {"movetext": ["e4 e5", "", None, "♟"]}
        ).lazy()

        lengths = movetext_character_lengths(games)

        self.assertEqual(lengths.to_list(), [5, 0, None, 1])

    def test_movetext_summary_formats_mixed_statistics_without_losing_values(self):
        summary = movetext_length_summary(
            pl.Series("longitud_caracteres", [5, 0, None, 1])
        )
        values = dict(summary.iter_rows())

        self.assertEqual(summary["Valor"].dtype, pl.String)
        self.assertEqual(values["Partidas con longitud disponible"], "3")
        self.assertEqual(values["Partidas sin movetext"], "1")
        self.assertEqual(values["Media (caracteres)"], "2.00")
        self.assertEqual(values["Máximo (caracteres)"], "5")


if __name__ == "__main__":
    unittest.main()
