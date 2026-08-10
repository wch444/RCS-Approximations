#!/usr/bin/env python3
"""Convert the Trivago action log to one ordered choice observation per session."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PLATFORMS = ("BR", "US", "JP", "UK", "DE")
PLATFORM_SET = set(PLATFORMS)
INTERACTIONS = {
    "interaction item image",
    "interaction item info",
    "interaction item rating",
    "interaction item deals",
    "search for item",
}
USE_COLUMNS = [
    "session_id",
    "step",
    "action_type",
    "reference",
    "platform",
    "device",
    "impressions",
    "prices",
]


def to_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def build(train_file: Path, price_file: Path, output_dir: Path, chunksize: int) -> None:
    start = time.time()
    price_frame = pd.read_parquet(price_file, columns=["item_id", "mean_price"])
    item_price = dict(zip(price_frame["item_id"], price_frame["mean_price"]))

    # session -> [platform, first-device, interactions, clickouts]
    # clickout tuple: (step, reference, impressions, prices)
    sessions: dict[str, list[object]] = {}
    skipped: set[str] = set()
    reader = pd.read_csv(
        train_file, usecols=USE_COLUMNS, dtype=str, chunksize=chunksize
    )
    for chunk_index, chunk in enumerate(reader, start=1):
        for row in chunk.itertuples(index=False):
            session_id = row.session_id
            if session_id in skipped:
                continue
            if session_id not in sessions:
                if row.platform not in PLATFORM_SET:
                    skipped.add(session_id)
                    continue
                sessions[session_id] = [row.platform, row.device, [], []]
            entry = sessions[session_id]
            step = to_int(row.step)
            if step < 0:
                continue
            if row.action_type == "clickout item":
                entry[3].append((step, row.reference, row.impressions, row.prices))
            elif row.action_type in INTERACTIONS:
                item = to_int(row.reference)
                if item > 0:
                    entry[2].append((step, item))
        print(
            f"query pass chunk {chunk_index}: sessions={len(sessions):,}, "
            f"elapsed={time.time() - start:.1f}s",
            flush=True,
        )

    output_rows: dict[str, list[tuple[object, ...]]] = {p: [] for p in PLATFORMS}
    clickout_count: defaultdict[str, int] = defaultdict(int)
    abandoned_count: defaultdict[str, int] = defaultdict(int)
    next_search_id = 0
    for platform, device, interactions, clickouts in sessions.values():
        next_search_id += 1
        search_id = next_search_id
        if clickouts:
            last = max(clickouts, key=lambda value: value[0])
            last_step = last[0]
            try:
                items = [int(value) for value in last[2].split("|")]
                prices = [float(value) for value in last[3].split("|")]
                chosen = int(last[1])
            except (AttributeError, TypeError, ValueError):
                continue
            if len(items) != len(prices) or len(items) < 5 or chosen not in items:
                continue
            considered = {item for step, item in interactions if step < last_step}
            considered.update(
                to_int(reference)
                for step, reference, _, _ in clickouts
                if step < last_step and to_int(reference) > 0
            )
            considered.add(chosen)
            for position, (item, price) in enumerate(zip(items, prices), start=1):
                output_rows[platform].append(
                    (
                        search_id,
                        item,
                        position,
                        int(item == chosen),
                        int(item in considered),
                        price,
                        device,
                        0,
                    )
                )
            clickout_count[platform] += 1
        else:
            seen: set[int] = set()
            ordered: list[int] = []
            for _, item in sorted(interactions, key=lambda value: value[0]):
                if item not in seen:
                    seen.add(item)
                    ordered.append(item)
            ordered = [item for item in ordered if item in item_price][:25]
            if not ordered:
                continue
            for position, item in enumerate(ordered, start=1):
                output_rows[platform].append(
                    (
                        search_id,
                        item,
                        position,
                        0,
                        1,
                        float(item_price[item]),
                        device,
                        1,
                    )
                )
            abandoned_count[platform] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "srch_id",
        "prop_id",
        "position",
        "booking_bool",
        "click_bool",
        "origin_price",
        "device",
        "is_abandoned",
    ]
    for platform in PLATFORMS:
        frame = pd.DataFrame.from_records(output_rows[platform], columns=columns)
        frame = frame.astype(
            {
                "srch_id": np.int64,
                "prop_id": np.int64,
                "position": np.int32,
                "booking_bool": np.int8,
                "click_bool": np.int8,
                "origin_price": np.float32,
                "is_abandoned": np.int8,
            }
        )
        path = output_dir / f"queries_long_session_{platform}.parquet"
        frame.to_parquet(path, index=False)
        print(
            f"{platform}: clickout={clickout_count[platform]:,}, "
            f"abandoned={abandoned_count[platform]:,}, rows={len(frame):,}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--item-prices", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    for path in (args.train, args.item_prices):
        if not path.is_file():
            raise FileNotFoundError(path)
    build(
        args.train.resolve(),
        args.item_prices.resolve(),
        args.output_dir.resolve(),
        args.chunksize,
    )


if __name__ == "__main__":
    main()
