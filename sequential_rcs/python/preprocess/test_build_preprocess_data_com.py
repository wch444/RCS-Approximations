"""Tests for the raw Expedia CSV-to-HDF preprocessing stage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from .build_preprocess_data_com import (
    OUTPUT_COLUMNS,
    build_preprocess_h5,
    preprocess_frame,
)


def make_raw() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for srch_id in range(1, 8):
        row = {column: 1.0 for column in OUTPUT_COLUMNS}
        row.update(
            {
                "srch_id": srch_id,
                "prop_id": 100 + srch_id,
                "site_id": 5,
                "position": srch_id,
                "date_time": "2013-01-01 00:00:00",
                "booking_bool": 0,
                "random_bool": 0,
                "click_bool": 0,
                "price_usd": 100.0,
                "srch_length_of_stay": 2.0,
                "srch_room_count": 1,
            }
        )
        rows.append(row)
    rows[1]["price_usd"] = 10.0
    rows[2]["price_usd"] = 5000.0
    rows[3]["srch_room_count"] = 2
    rows[4]["srch_length_of_stay"] = 11.0
    rows[5]["prop_review_score"] = np.nan
    return pd.DataFrame(rows)


class BuildPreprocessDataComTests(unittest.TestCase):
    def test_historical_filters_and_column_order(self) -> None:
        processed = preprocess_frame(make_raw())
        self.assertEqual(processed["srch_id"].tolist(), [1, 7])
        self.assertEqual(tuple(processed.columns), OUTPUT_COLUMNS)
        self.assertNotIn("srch_room_count", processed.columns)

    def test_writes_fixed_hdf_and_refuses_implicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_csv = root / "train.csv"
            output_h5 = root / "preprocess_data_com.h5"
            make_raw().to_csv(input_csv, index=False)

            summary = build_preprocess_h5(input_csv, output_h5)
            result = pd.read_hdf(output_h5, key="data")
            self.assertEqual(result["srch_id"].tolist(), [1, 7])
            self.assertEqual(tuple(result.columns), OUTPUT_COLUMNS)
            self.assertEqual(summary["input_rows"], 7)
            self.assertEqual(summary["output_rows"], 2)
            with self.assertRaises(FileExistsError):
                build_preprocess_h5(input_csv, output_h5)

    def test_missing_required_column_is_rejected(self) -> None:
        raw = make_raw().drop(columns="random_bool")
        with self.assertRaisesRegex(ValueError, "random_bool"):
            preprocess_frame(raw)


if __name__ == "__main__":
    unittest.main()
