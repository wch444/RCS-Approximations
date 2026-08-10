#!/usr/bin/env python3
"""Compute each item's global mean displayed price from click-out rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def build(source: Path, output: Path, chunksize: int = 500_000) -> None:
    sums: defaultdict[int, float] = defaultdict(float)
    counts: defaultdict[int, int] = defaultdict(int)
    reader = pd.read_csv(
        source,
        usecols=["action_type", "impressions", "prices"],
        dtype=str,
        chunksize=chunksize,
    )
    for chunk_index, chunk in enumerate(reader, start=1):
        clickouts = chunk.loc[chunk["action_type"] == "clickout item"]
        for impressions, prices in zip(
            clickouts["impressions"].to_numpy(), clickouts["prices"].to_numpy()
        ):
            if not isinstance(impressions, str) or not isinstance(prices, str):
                continue
            try:
                items = [int(value) for value in impressions.split("|")]
                values = [float(value) for value in prices.split("|")]
            except ValueError:
                continue
            if len(items) != len(values):
                continue
            for item, price in zip(items, values):
                sums[item] += price
                counts[item] += 1
        print(f"price pass chunk {chunk_index}", flush=True)

    rows = [(item, sums[item] / counts[item], counts[item]) for item in sorted(sums)]
    frame = pd.DataFrame(rows, columns=["item_id", "mean_price", "n_obs"])
    frame["item_id"] = frame["item_id"].astype(np.int64)
    frame["mean_price"] = frame["mean_price"].astype(np.float32)
    frame["n_obs"] = frame["n_obs"].astype(np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    print(f"saved {output} ({len(frame):,} items)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    if not args.train.is_file():
        raise FileNotFoundError(args.train)
    build(args.train.resolve(), args.output.resolve(), args.chunksize)


if __name__ == "__main__":
    main()
