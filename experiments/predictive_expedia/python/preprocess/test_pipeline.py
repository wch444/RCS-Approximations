"""Focused tests for the manuscript-aligned Expedia preprocessing pipeline."""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .pipeline import (
    BASE_COLUMNS,
    MANIFEST_NAME,
    RAW_COLUMNS,
    STANDARDIZED_COLUMNS,
    add_outside_rows,
    build_from_h5,
    build_trial,
    prepare_site,
    split_query_ids,
    standardize_split,
)


def make_raw(query_count: int = 12, site: int = 5) -> pd.DataFrame:
    """Create queries with within-query variation in every standardized field."""

    rows: list[dict[str, float]] = []
    for query_id in range(1, query_count + 1):
        for position in range(6):
            value = float(10 * query_id + position)
            rows.append(
                {
                    "srch_id": query_id,
                    "prop_id": 1000 * query_id + position,
                    "site_id": site,
                    "position": position + 1,
                    "booking_bool": float(query_id % 2 == 0 and position == 1),
                    "click_bool": float(position == 0),
                    "price_usd": 50.0 + value,
                    "prop_starrating": 1.0 + (position % 5),
                    "prop_location_score1": value / 10.0,
                    "prop_review_score": 1.0 + value / 100.0,
                    "srch_length_of_stay": 1.0 + (position % 3),
                    "prop_log_historical_price": 2.0 + value / 1000.0,
                    "promotion_flag": float(position % 2),
                    "srch_booking_window": value,
                    "srch_adults_count": 1.0 + (position % 2),
                    "srch_children_count": float(position % 3),
                    "prop_brand_bool": float((position + 1) % 2),
                    "srch_saturday_night_bool": float(position % 2),
                }
            )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


class ExpediaPreprocessTests(unittest.TestCase):
    def test_click_count_filters_complete_queries_and_adds_position_bins(self) -> None:
        raw = make_raw(query_count=10_001)
        high_click = make_raw(query_count=1).copy()
        extra = high_click.iloc[:5].copy()
        extra["position"] = np.arange(7, 12)
        extra["prop_id"] = np.arange(2001, 2006)
        high_click = pd.concat([high_click, extra], ignore_index=True)
        high_click["srch_id"] = 99_999
        high_click["click_bool"] = 0.0
        high_click.loc[:4, "click_bool"] = 1.0
        raw = pd.concat([raw, high_click], ignore_index=True)

        prepared, columns = prepare_site(raw, 5)
        retained = prepared.loc[prepared["srch_id"] == 99_999]
        self.assertTrue(retained.empty)
        self.assertFalse(prepared["srch_id"].isna().any())
        self.assertEqual(columns[: len(BASE_COLUMNS)], BASE_COLUMNS)
        self.assertEqual(
            columns[len(BASE_COLUMNS) :], ("position_bin_0", "position_bin_1")
        )
        self.assertEqual(prepared.shape[1], 22)
        self.assertTrue((prepared.loc[:, columns[20:]].sum(axis=1) == 1).all())

        train_ids, test_ids = split_query_ids(prepared, seed=7)
        self.assertEqual((len(train_ids), len(test_ids)), (6000, 4000))
        self.assertEqual(len(set(train_ids) | set(test_ids)), 10_000)

    def test_split_uses_historical_seeded_shuffle(self) -> None:
        frame = pd.DataFrame({"srch_id": np.arange(1, 10_006)})
        expected = list(range(1, 10_006))
        random.Random(3).shuffle(expected)
        expected = expected[:10_000]

        train_ids, test_ids = split_query_ids(frame, seed=3)
        self.assertEqual(list(train_ids), expected[:6000])
        self.assertEqual(list(test_ids), expected[6000:])

    def test_split_refuses_fewer_than_requested_queries(self) -> None:
        frame = pd.DataFrame({"srch_id": np.arange(1, 10_000)})
        with self.assertRaisesRegex(ValueError, "at least 10000"):
            split_query_ids(frame, seed=1)

    def test_standardization_uses_split_specific_sample_sd(self) -> None:
        prepared, _ = prepare_site(make_raw(), 5)
        first = prepared.loc[prepared["srch_id"] <= 6].copy()
        second = prepared.loc[prepared["srch_id"] > 6].copy()
        first_standardized = standardize_split(first)
        second_standardized = standardize_split(second)

        for standardized in (first_standardized, second_standardized):
            for column in STANDARDIZED_COLUMNS:
                self.assertAlmostEqual(
                    float(standardized[column].mean()), 0.0, places=12
                )
                self.assertAlmostEqual(
                    float(standardized[column].std(ddof=1)), 1.0, places=12
                )
        expected = (first["price_usd"] - first["price_usd"].mean()) / (
            first["price_usd"].var(ddof=1) ** 0.5
        )
        np.testing.assert_allclose(first_standardized["price_usd"], expected)

    def test_outside_row_and_matrix_column_contract(self) -> None:
        prepared, columns = prepare_site(make_raw(query_count=4), 5)
        standardized = standardize_split(prepared)
        result = add_outside_rows(standardized, columns)
        outside = result.loc[result["position"] == 100].copy()
        products = result.loc[result["position"] != 100].copy()

        self.assertEqual(tuple(columns[: len(BASE_COLUMNS)]), BASE_COLUMNS)
        self.assertEqual(result.shape[1], len(columns))
        self.assertEqual(len(outside), 4)
        self.assertEqual(outside["srch_id"].nunique(), 4)
        for column in set(columns).difference({"srch_id", "position", "booking_bool"}):
            self.assertTrue((outside[column] == 0).all(), column)
        self.assertEqual(
            outside.set_index("srch_id")["booking_bool"].to_dict(),
            {1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0},
        )
        self.assertTrue((outside.loc[:, columns[20:]] == 0).all().all())
        self.assertTrue((products.loc[:, columns[20:]].sum(axis=1) == 1).all())

    def test_trial_standardizes_train_and_test_separately(self) -> None:
        prepared, columns = prepare_site(make_raw(query_count=10_005), 5)
        train, test, metadata = build_trial(prepared, columns, trial=2)
        self.assertEqual(train.shape[1], 22)
        self.assertEqual(test.shape[1], 22)
        self.assertEqual(
            (metadata["train_queries"], metadata["test_queries"]), (6000, 4000)
        )
        for matrix in (train, test):
            products = matrix[:, 3] != 100
            np.testing.assert_allclose(
                matrix[products, 8:20].mean(axis=0), 0.0, atol=1e-12
            )
            np.testing.assert_allclose(
                matrix[products, 8:20].std(axis=0, ddof=1), 1.0, atol=1e-12
            )
            np.testing.assert_array_equal(matrix[products, 20:].sum(axis=1), 1.0)
        self.assertEqual(len(metadata["train_query_ids_sha256"]), 64)
        self.assertEqual(len(metadata["test_query_ids_sha256"]), 64)

    def test_explicit_h5_cli_contract_writes_mat_and_plain_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            h5_path = root / "input.h5"
            output_dir = root / "stage"
            raw = make_raw(query_count=10_001)
            raw.loc[raw["srch_id"] == 9_999, "click_bool"] = 0.0
            raw.loc[raw["srch_id"] == 10_000, "click_bool"] = 1.0
            raw.to_hdf(h5_path, key="data", mode="w")

            manifest = build_from_h5(h5_path, output_dir, sites=(5,), trials=(1,))
            mat_path = output_dir / "10k_sample_5_1.mat"
            data = loadmat(mat_path)
            self.assertEqual(set(("data_train", "data_test")).difference(data), set())
            self.assertEqual(
                (
                    len(np.unique(data["data_train"][:, 0])),
                    len(np.unique(data["data_test"][:, 0])),
                ),
                (6000, 4000),
            )
            self.assertEqual(data["data_train"].shape[1], 22)
            self.assertEqual(data["data_test"].shape[1], 22)
            self.assertEqual(
                manifest["standardization"],
                "separate train/test sample SD (ddof=1)",
            )
            self.assertEqual(manifest["max_query_clicks"], 4)
            self.assertTrue(manifest["click_count_filter_enabled"])
            self.assertTrue(manifest["position_encoding_enabled"])
            self.assertEqual(manifest["position_bin_width"], 5)
            self.assertEqual(manifest["base_matrix_column_count"], 20)
            self.assertEqual(tuple(manifest["base_matrix_columns"]), BASE_COLUMNS)
            self.assertEqual(manifest["model_feature_column_start_1_based"], 8)
            self.assertEqual(
                manifest["input_sha256"],
                hashlib.sha256(h5_path.read_bytes()).hexdigest(),
            )
            record = manifest["files"][0]
            self.assertEqual(record["columns"], 22)
            self.assertEqual(record["train_queries_click_count_gt_4"], 0)
            self.assertEqual(record["test_queries_click_count_gt_4"], 0)
            self.assertEqual(
                record["train_queries_click_count_eq_0"]
                + record["test_queries_click_count_eq_0"],
                1,
            )
            self.assertEqual(
                manifest["source_sites"][0]["source_queries_click_count_gt_4"], 1
            )
            self.assertEqual(
                record["file_sha256"], hashlib.sha256(mat_path.read_bytes()).hexdigest()
            )
            self.assertTrue(
                mat_path.read_bytes().startswith(
                    b"MATLAB 5.0 MAT-file, generated by the SCM reproducibility pipeline"
                )
            )
            on_disk = json.loads((output_dir / MANIFEST_NAME).read_text())
            self.assertEqual(on_disk, manifest)
            with self.assertRaises(FileExistsError):
                build_from_h5(h5_path, output_dir, sites=(5,), trials=(1,))


if __name__ == "__main__":
    unittest.main()
