#!/usr/bin/env python3
"""Build deterministic item features from Trivago item_metadata.csv.

Only an exact ``1 Star`` through ``5 Star`` property is a star rating.  In
particular, range descriptors such as ``From 2 Stars`` are not ratings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXACT_STAR_TAGS = {f"{value} Star": value for value in range(1, 6)}
RATING_ORDER = [
    "Satisfactory Rating",
    "Good Rating",
    "Very Good Rating",
    "Excellent Rating",
]
INDICATOR_TAGS = {
    "is_luxury": "Luxury Hotel",
    "is_business": "Business Hotel",
    "has_pool": "Swimming Pool (Outdoor)",
    "has_gym": "Gym",
    # The official metadata has no exact ``Spa`` property.  Use the amenity
    # tag and deliberately do not treat the distinct ``Spa Hotel`` category
    # as evidence that a spa facility is present.
    "has_spa": "Spa (Wellness Facility)",
}


def parse_properties(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    return tuple(property_.strip() for property_ in value.split("|"))


def extract_star(properties: tuple[str, ...]) -> float:
    matches = [EXACT_STAR_TAGS[value] for value in properties if value in EXACT_STAR_TAGS]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous exact star tags: {matches}")
    return float(matches[0]) if matches else np.nan


def extract_rating_tier(properties: set[str]) -> float:
    matches = [index for index, tag in enumerate(RATING_ORDER, start=1) if tag in properties]
    return float(max(matches)) if matches else np.nan


def extract_indicators(properties: set[str]) -> dict[str, int]:
    """Return exact-tag category and amenity indicators."""

    return {column: int(tag in properties) for column, tag in INDICATOR_TAGS.items()}


def build(source: Path, output: Path) -> None:
    raw = pd.read_csv(source, usecols=["item_id", "properties"])
    rows = []
    for item_id, value in raw.itertuples(index=False):
        ordered = parse_properties(value)
        property_set = set(ordered)
        record: dict[str, float | int] = {
            "item_id": int(item_id),
            "star_rating": extract_star(ordered),
            "rating_tier": extract_rating_tier(property_set),
        }
        record.update(extract_indicators(property_set))
        rows.append(record)

    frame = pd.DataFrame.from_records(rows)
    observed_star = int(frame["star_rating"].notna().sum())
    observed_tier = int(frame["rating_tier"].notna().sum())
    mean_star = float(frame["star_rating"].mean())
    mean_tier = float(frame["rating_tier"].mean())
    frame["star_rating"] = frame["star_rating"].fillna(mean_star)
    frame["rating_tier"] = frame["rating_tier"].fillna(mean_tier)
    for column in frame.columns:
        if column != "item_id":
            frame[column] = frame[column].astype(np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    manifest = {
        "schema_version": 1,
        "star_rule": "exact_tag_1_Star_through_5_Star",
        "spa_rule": "exact_tag_Spa_(Wellness_Facility);_Spa_Hotel_excluded",
        "n_items": len(frame),
        "observed_star": observed_star,
        "observed_rating_tier": observed_tier,
        "mean_star_before_imputation": mean_star,
        "mean_rating_tier_before_imputation": mean_tier,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.item_metadata.is_file():
        raise FileNotFoundError(args.item_metadata)
    build(args.item_metadata.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
