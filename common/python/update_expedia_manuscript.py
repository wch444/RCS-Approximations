#!/usr/bin/env python3
"""Check or update the Expedia result blocks in the v2 online appendix."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from common.python.predictive_reporting import (
    aggregate as aggregate_predictive,
    latex_table as predictive_latex_table,
    validate_tasks as validate_predictive_tasks,
)
from experiments.mixed_order_expedia.python import report as mixed_report
from experiments.predictive_expedia.python import report as predictive_report


ROOT = _REPOSITORY_ROOT
APPENDIX = ROOT / "omitted analysis_v2.tex"
MAIN_TEX = ROOT / "main_msom_v2.tex"
PREDICTIVE = ROOT / "experiments" / "predictive_expedia" / "results"
MIXED = ROOT / "experiments" / "mixed_order_expedia" / "results"
PAPER_FIGURES = ROOT / "new" / "figures"

SITES = ("5", "14", "15", "24", "32")
MODELS = ("CMNL", "MNL", "MMNL", "SeqINV", "NSMNL", "SCM")
HITS = ("Hit1", "Hit2", "Hit3")
ORDERINGS = ("Default", "PriceLowHigh", "PriceHighLow", "Location", "StarRating")
ORDERING_PROSE = {
    "Default": "default position",
    "PriceLowHigh": "price low-to-high",
    "PriceHighLow": "price high-to-low",
    "Location": "location score",
    "StarRating": "star rating",
}

PREDICTIVE_TRIALS = "predictive_expedia_trials.csv"
PREDICTIVE_MEANS = "predictive_expedia_means.csv"
PREDICTIVE_TABLE = "predictive_expedia_table.tex"
MIXED_RAW = "mixed_order_results.csv"
MIXED_SUMMARY = "mixed_order_summary.csv"
MIXED_TABLE = "mixed_order_table.tex"
MIXED_MANIFEST = "mixed_order_manifest.json"
FIGURE_NAMES = ("a", "b")
_FLOAT_RTOL = 5e-15
_FLOAT_ATOL = 5e-12


@dataclass(frozen=True)
class _Generation:
    predictive_means: pd.DataFrame
    predictive_table: str
    mixed_summary: pd.DataFrame
    mixed_table: str
    mixed_manifest: dict[str, object]
    mixed_raw: pd.DataFrame
    source_figures: Mapping[str, bytes]
    input_identities: Mapping[Path, tuple[int, int, int, int]]


def _identity(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _require_regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular file: {path}")


def _capture_identities(
    paths: Iterable[Path],
) -> dict[Path, tuple[int, int, int, int]]:
    snapshots: dict[Path, tuple[int, int, int, int]] = {}
    for path in paths:
        _require_regular_file(path, "Expedia manuscript input")
        snapshots[path] = _identity(path)
    return snapshots


def _revalidate_identities(
    snapshots: Mapping[Path, tuple[int, int, int, int]],
    *,
    exclude: Iterable[Path] = (),
) -> None:
    excluded = set(exclude)
    for path, expected in snapshots.items():
        if path in excluded:
            continue
        try:
            _require_regular_file(path, "Expedia manuscript input")
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(
                f"Expedia manuscript input changed during update: {path}"
            ) from error
        if _identity(path) != expected:
            raise ValueError(f"Expedia manuscript input changed during update: {path}")


@contextmanager
def _publication_locks() -> Iterator[None]:
    """Share the two reporters' publication locks while reading one generation."""

    expected_mixed_root = MIXED.parent.resolve()
    actual_mixed_root = Path(mixed_report.EXPERIMENT_ROOT).resolve()
    if expected_mixed_root != actual_mixed_root:
        raise ValueError(
            "Mixed-order updater and reporter do not share a publication-lock root."
        )
    with predictive_report._lock(PREDICTIVE):
        with mixed_report._exclusive_publication_lock():
            yield


def _assert_frame_matches(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    label: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=_FLOAT_RTOL,
            atol=_FLOAT_ATOL,
        )
    except AssertionError as error:
        raise ValueError(
            f"{label} is not derived from the current raw generation."
        ) from error


def _predictive_generation() -> tuple[pd.DataFrame, str]:
    trials = pd.read_csv(PREDICTIVE / PREDICTIVE_TRIALS, dtype={"Market": str})
    checked = validate_predictive_tasks(trials, "expedia")
    expected_means = aggregate_predictive(checked, "expedia")
    published_means = pd.read_csv(PREDICTIVE / PREDICTIVE_MEANS, dtype={"Market": str})
    if published_means.columns.tolist() != expected_means.columns.tolist():
        raise ValueError("Predictive means columns do not match the canonical schema.")
    if list(
        published_means[["Market", "Model"]].itertuples(index=False, name=None)
    ) != list(expected_means[["Market", "Model"]].itertuples(index=False, name=None)):
        raise ValueError(
            "Predictive means axes do not match the canonical 30-row grid."
        )
    _assert_frame_matches(published_means, expected_means, "Predictive means")

    expected_table = predictive_latex_table(expected_means, "expedia")
    published_table = (PREDICTIVE / PREDICTIVE_TABLE).read_text(encoding="utf-8")
    if published_table != expected_table:
        raise ValueError(
            "Predictive table is not derived from the current 300 trial rows."
        )
    return expected_means, expected_table


def _mixed_headline(
    overall: Mapping[str, np.ndarray | float | int | bool | str],
) -> dict[str, float]:
    rng = np.random.default_rng(mixed_report.DIRICHLET_SEED)
    theta = rng.dirichlet(
        np.ones(len(mixed_report.ORDERINGS)), size=mixed_report.DIRICHLET_DRAWS
    )
    market_h = theta @ np.asarray(overall["rH"], dtype=float)
    return {
        "H_worst": float(overall["worstH"]),
        "H_uniform": float(overall["unifH"]),
        "G_uniform": float(overall["unifG"]),
        "ALL_worst": float(overall["worstA"]),
        "H_fraction_of_G_uniform": float(overall["unifH"]) / float(overall["unifG"]),
        "dirichlet_H_mean": float(market_h.mean()),
        "dirichlet_H_p05": float(np.percentile(market_h, 5)),
        "dirichlet_H_p95": float(np.percentile(market_h, 95)),
    }


def _validate_manifest(
    manifest: dict[str, object],
    headline: Mapping[str, float],
    source_paths: Mapping[str, Path],
) -> None:
    expected_core: dict[str, object] = {
        "schema_version": 3,
        "artifact_type": "mixed_order_analysis",
        "mode": "full",
        "sites": list(mixed_report.SITES),
        "trials": list(mixed_report.FULL_TRIALS),
        "queries_per_task": mixed_report.FULL_QUERIES_PER_TASK,
        "expected_rows": (
            len(mixed_report.SITES)
            * len(mixed_report.FULL_TRIALS)
            * mixed_report.FULL_QUERIES_PER_TASK
        ),
        "actual_rows": (
            len(mixed_report.SITES)
            * len(mixed_report.FULL_TRIALS)
            * mixed_report.FULL_QUERIES_PER_TASK
        ),
        "orderings": list(mixed_report.ORDERINGS),
        "dirichlet_seed": mixed_report.DIRICHLET_SEED,
        "dirichlet_draws": mixed_report.DIRICHLET_DRAWS,
    }
    for key, expected in expected_core.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Mixed-order manifest has an invalid {key} field.")

    published_headline = manifest.get("headline")
    if not isinstance(published_headline, dict):
        raise ValueError("Mixed-order manifest headline is missing.")
    try:
        observed = np.array(
            [float(published_headline[key]) for key in headline], dtype=float
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Mixed-order manifest headline is incomplete.") from error
    target = np.array(list(headline.values()), dtype=float)
    if not np.allclose(
        observed, target, rtol=_FLOAT_RTOL, atol=_FLOAT_ATOL, equal_nan=False
    ):
        raise ValueError(
            "Mixed-order manifest headline is not derived from the current raw rows."
        )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("Mixed-order manifest output records are missing.")
    by_basename: dict[str, list[dict[str, object]]] = {}
    for record in outputs:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("Mixed-order manifest contains an invalid output record.")
        by_basename.setdefault(Path(record["path"]).name, []).append(record)
    for source in source_paths.values():
        basename = source.name
        records = by_basename.get(basename, [])
        if len(records) != 1 or records[0].get("bytes") != source.stat().st_size:
            raise ValueError(
                f"Mixed-order manifest does not identify the current {basename}."
            )


def _mixed_generation() -> tuple[pd.DataFrame, str, dict[str, object], pd.DataFrame]:
    raw = pd.read_csv(MIXED / MIXED_RAW)
    mixed_report.validate_results(
        raw,
        sites=mixed_report.SITES,
        trials=mixed_report.FULL_TRIALS,
        queries_per_task=mixed_report.FULL_QUERIES_PER_TASK,
    )
    overall = mixed_report.aggregate(raw, pooled=True)
    by_site = {
        int(site): mixed_report.aggregate(raw.loc[raw["site"].eq(site)], pooled=False)
        for site in mixed_report.SITES
    }
    expected_summary = mixed_report.build_summary(overall, by_site)
    published_summary = pd.read_csv(MIXED / MIXED_SUMMARY, dtype={"Site": str})
    expected_summary = expected_summary.copy()
    expected_summary["Site"] = expected_summary["Site"].astype(str)
    _assert_frame_matches(published_summary, expected_summary, "Mixed-order summary")

    expected_table = mixed_report.build_table(overall, by_site)
    if (MIXED / MIXED_TABLE).read_text(encoding="utf-8") != expected_table:
        raise ValueError(
            "Mixed-order table is not derived from the current 25,000 raw rows."
        )

    manifest = json.loads((MIXED / MIXED_MANIFEST).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Mixed-order manifest must contain one JSON object.")
    sources = {name: MIXED / f"fig_mixed_order_{name}.pdf" for name in FIGURE_NAMES}
    _validate_manifest(manifest, _mixed_headline(overall), sources)
    return expected_summary, expected_table, manifest, raw


def _load_generation() -> _Generation:
    source_paths = {
        name: MIXED / f"fig_mixed_order_{name}.pdf" for name in FIGURE_NAMES
    }
    inputs = (
        MAIN_TEX,
        APPENDIX,
        PREDICTIVE / PREDICTIVE_TRIALS,
        PREDICTIVE / PREDICTIVE_MEANS,
        PREDICTIVE / PREDICTIVE_TABLE,
        MIXED / MIXED_RAW,
        MIXED / MIXED_SUMMARY,
        MIXED / MIXED_TABLE,
        MIXED / MIXED_MANIFEST,
        *source_paths.values(),
    )
    snapshots = _capture_identities(inputs)
    predictive_means, predictive_table = _predictive_generation()
    mixed_summary, mixed_table, manifest, raw = _mixed_generation()
    source_figures = {name: path.read_bytes() for name, path in source_paths.items()}
    _revalidate_identities(snapshots)
    return _Generation(
        predictive_means=predictive_means,
        predictive_table=predictive_table,
        mixed_summary=mixed_summary,
        mixed_table=mixed_table,
        mixed_manifest=manifest,
        mixed_raw=raw,
        source_figures=source_figures,
        input_identities=snapshots,
    )


def _norm(text: str) -> str:
    return " ".join(text.split())


def _replace_span(
    text: str, start: str, end: str, replacement: str
) -> tuple[str, bool]:
    if text.count(start) != 1:
        raise ValueError(f"Expected one start token: {start}")
    left = text.index(start)
    right = text.find(end, left + len(start))
    if right < 0:
        raise ValueError(f"Missing end token: {end}")
    if _norm(text[left:right]) == _norm(replacement):
        return text, False
    return text[:left] + replacement.strip() + "\n\n" + text[right:], True


def _replace_table(text: str, label: str, replacement: str) -> tuple[str, bool]:
    token = rf"\label{{{label}}}"
    if text.count(token) != 1:
        raise ValueError(f"Expected one table label: {label}")
    label_at = text.index(token)
    left = text.rfind(r"\begin{tabular}", 0, label_at)
    end_at = text.rfind(r"\end{tabular}", left, label_at)
    if left < 0 or end_at < left:
        raise ValueError(f"Cannot locate tabular block: {label}")
    right = end_at + len(r"\end{tabular}")
    if text[left:right].strip() == replacement.strip():
        return text, False
    return text[:left] + replacement.strip() + text[right:], True


def _predictive_claim(means: pd.DataFrame) -> str:
    blocks = {
        site: means.loc[means["Market"].eq(site)].set_index("Model") for site in SITES
    }
    rmse_winners = [block["RMSE"].idxmin() for block in blocks.values()]
    mmnl_minus_mnl = np.array(
        [
            block.loc["MNL", "RMSE"] - block.loc["MMNL", "RMSE"]
            for block in blocks.values()
        ]
    )
    scm_rmse_gap = max(
        (block.loc["SCM", "RMSE"] / block["RMSE"].min() - 1) * 100
        for block in blocks.values()
    )
    hit_winners = {
        metric: [block[metric].idxmax() for block in blocks.values()] for metric in HITS
    }
    scm_hit1_gain = np.array(
        [
            (block.loc["SCM", "Hit1"] / block.drop(index="SCM")["Hit1"].max() - 1) * 100
            for block in blocks.values()
        ]
    )

    if set(rmse_winners) == {"MMNL"}:
        rmse = "MMNL attains the lowest RMSE in all five sites."
    else:
        rmse = "The site-wise lowest-RMSE models are " + ", ".join(rmse_winners) + "."
    if (mmnl_minus_mnl > 0).all():
        comparison = (
            "Its RMSE advantage over MNL is small, ranging from approximately "
            f"${mmnl_minus_mnl.min():.4f}$ to ${mmnl_minus_mnl.max():.4f}$"
        )
    else:
        difference = np.abs(mmnl_minus_mnl)
        comparison = (
            "The absolute MMNL--MNL RMSE difference ranges from approximately "
            f"${difference.min():.4f}$ to ${difference.max():.4f}$"
        )
    if all(set(winners) == {"SCM"} for winners in hit_winners.values()):
        hits = (
            "By contrast, SCM achieves the highest $1$-hit, $2$-hit, and $3$-hit "
            "score in every site."
        )
    else:
        counts = [hit_winners[metric].count("SCM") for metric in HITS]
        hits = (
            "SCM has the highest $1$-hit, $2$-hit, and $3$-hit score in "
            f"{counts[0]}, {counts[1]}, and {counts[2]} of the five sites, respectively."
        )
    direction = "improvement" if (scm_hit1_gain >= 0).all() else "relative difference"
    return (
        r"As shown in Table~\ref{Tab:compare_expedia}, "
        + rmse
        + " "
        + comparison
        + f", and the SCM RMSE remains within ${scm_rmse_gap:.1f}\\%$ of the best value. "
        + hits
        + f" Its $1$-hit {direction} over the strongest competing model ranges from "
        f"approximately ${scm_hit1_gain.min():.1f}\\%$ to "
        f"${scm_hit1_gain.max():.1f}\\%$."
    )


def _mixed_blocks(
    summary: pd.DataFrame, manifest: dict, raw: pd.DataFrame
) -> tuple[str, ...]:
    def values(site: str, field: str) -> np.ndarray:
        scope = "all" if site == "All" else "site"
        block = summary.loc[
            summary["Scope"].eq(scope) & summary["Site"].astype(str).eq(site)
        ].set_index("Ordering")
        return block.loc[list(ORDERINGS), field].to_numpy(float)

    h_all = values("All", "H_Mean")
    g_all = values("All", "G_Mean")
    worst = float(h_all.min())
    uniform = float(h_all.mean())
    oracle = float(g_all.mean())
    site_worst = np.array([values(site, "H_Mean").min() for site in SITES])
    worst_name = ORDERING_PROSE[ORDERINGS[int(np.argmin(h_all))]]
    best_name = ORDERING_PROSE[ORDERINGS[int(np.argmax(h_all))]]
    middle = h_all[[0, 3, 4]]
    headline = manifest["headline"]

    result_claim = (
        r"Table~\ref{tab: realistic_orders} shows that the heuristic performs well "
        "across all five realistic orderings. In the pooled summary, the worst ordering "
        f"is {worst_name}, with ratio ${worst:.3f}$, while the best is {best_name}, "
        f"with ratio ${h_all.max():.3f}$. The default, location-score, and star-rating "
        f"orders lie in a narrow middle range, between ${middle.min():.3f}$ and "
        f"${middle.max():.3f}$. Hence, for any unknown mixture over the five orderings, "
        f"the pooled market-level ratio lies between ${worst:.3f}$ and "
        f"${h_all.max():.3f}$; under the uniform mixture it is ${uniform:.3f}$. The "
        "site-level results are similarly stable, with worst-order ratios ranging from "
        f"${site_worst.min():.3f}$ to ${site_worst.max():.3f}$."
    )
    draws = f"{int(manifest['dirichlet_draws']):,}".replace(",", "{,}")
    dirichlet_claim = (
        r"Figure~\ref{fig:realistic_orders_b} illustrates the same point by drawing "
        f"${draws}$ mixtures from $\\mathrm{{Dir}}(1,\\ldots,1)$. The resulting "
        "$W$-heuristic ratios "
        f"are tightly concentrated around the uniform-mixture value ${uniform:.3f}$, "
        f"with 5th--95th percentile interval "
        f"$[{float(headline['dirichlet_H_p05']):.3f},"
        f"{float(headline['dirichlet_H_p95']):.3f}]$. All these values are well above "
        r"the $0.375$ worst-case guarantee in Corollary~\ref{thm: scm_worst}. "
        "Moreover, because the "
        "denominator is a universal upper bound, the ratios relative to the true optimum "
        "are at least as large as the reported values."
    )
    benchmark_claim = (
        r"The last column of Table~\ref{tab: realistic_orders} provides an order-aware "
        r"benchmark. For each realized ordering $\pi_k$, this benchmark applies the "
        r"$G$-heuristic tailored to that ordering and then averages the resulting revenues "
        "under the uniform mixture over the five orderings. Thus, the comparison with the "
        "last column measures the value of knowing and optimizing for the realized "
        "evaluation order, while using the same normalization as the order-agnostic "
        "heuristic. Relative to this benchmark, the cost of not observing the realized "
        "order is small: the order-agnostic heuristic's pooled normalized ratio is "
        f"${100*uniform/oracle:.1f}\\%$ of the order-aware benchmark's pooled "
        f"normalized ratio, with a difference of ${100*(oracle-uniform):.1f}$ "
        "percentage points."
    )

    all_worst = float(
        summary.loc[summary["Scope"].eq("all"), "ALL_WorstOrdering"].iloc[0]
    )
    hidden_comment = (
        "% The order-agnostic heuristic also improves upon the offer-all policy in the "
        f"worst mixture. Its pooled worst-case ratio is ${worst:.3f}$, compared with "
        f"${all_worst:.3f}$ for offering the full set $\\mathcal N$, while selecting "
        f"fewer products, with a median assortment size of ${raw['card_SqL'].median():.0f}$ "
        f"out of a median ${raw['n_prod'].median():.0f}$ displayed products. These results "
        "indicate that the order-agnostic heuristic remains effective when customers "
        "choose among a small number of realistic sorting rules under an arbitrary and "
        "unknown mixture."
    )

    table = [
        r"\begin{tabular}{lccccc|cc|c}",
        r"\toprule",
        r"Site & \shortstack{Default\\position} & \shortstack{Price\\low-to-high}",
        r"& \shortstack{Price\\high-to-low} & \shortstack{Location\\score}",
        r"& \shortstack{Star\\rating} & \shortstack{Worst\\$\theta$}",
        r"& \shortstack{Uniform\\$\theta$} & \shortstack{Order-aware\\G, unif. $\theta$} \\",
        r"\midrule",
    ]
    for site in SITES:
        h = values(site, "H_Mean")
        g = values(site, "G_Mean")
        table.append(
            f"{site} & "
            + " & ".join(f"{x:.3f}" for x in h)
            + f" & {h.min():.3f} & {h.mean():.3f} & {g.mean():.3f} \\\\"
        )
    table.extend(
        [
            r"\midrule",
            r"\textbf{All} & "
            + " & ".join(rf"\textbf{{{x:.3f}}}" for x in h_all)
            + rf" & \textbf{{{worst:.3f}}} & \textbf{{{uniform:.3f}}} & {oracle:.3f} \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return (
        result_claim,
        dirichlet_claim,
        benchmark_claim,
        hidden_comment,
        "\n".join(table),
    )


def _expected_appendix(generation: _Generation) -> tuple[str, list[str]]:
    if r"\input{omitted analysis_v2}" not in MAIN_TEX.read_text(encoding="utf-8"):
        raise ValueError("main_msom_v2.tex does not input omitted analysis_v2.")
    blocks = _mixed_blocks(
        generation.mixed_summary,
        generation.mixed_manifest,
        generation.mixed_raw,
    )

    text = APPENDIX.read_text(encoding="utf-8")
    changes: list[str] = []
    operations = (
        (
            "predictive claim",
            _replace_span,
            (
                r"As shown in Table~\ref{Tab:compare_expedia}",
                "The two metrics emphasize different aspects of prediction.",
                _predictive_claim(generation.predictive_means),
            ),
        ),
        (
            "predictive table",
            _replace_table,
            (
                "Tab:compare_expedia",
                generation.predictive_table,
            ),
        ),
        (
            "mixed-order headline",
            _replace_span,
            (
                r"Table~\ref{tab: realistic_orders} shows that",
                r"Figure~\ref{fig:realistic_orders_b} illustrates",
                blocks[0],
            ),
        ),
        (
            "mixed-order Dirichlet claim",
            _replace_span,
            (
                r"Figure~\ref{fig:realistic_orders_b} illustrates",
                r"The last column of Table~\ref{tab: realistic_orders}",
                blocks[1],
            ),
        ),
        (
            "mixed-order benchmark claim",
            _replace_span,
            (
                r"The last column of Table~\ref{tab: realistic_orders}",
                "% The order-agnostic heuristic also improves",
                blocks[2],
            ),
        ),
        (
            "mixed-order hidden comment",
            _replace_span,
            (
                "% The order-agnostic heuristic also improves",
                r"\begin{table}[htb]",
                blocks[3],
            ),
        ),
        ("mixed-order table", _replace_table, ("tab: realistic_orders", blocks[4])),
    )
    for name, function, arguments in operations:
        text, changed = function(text, *arguments)
        if changed:
            changes.append(name)
    return text, changes


_PDF_DATE = re.compile(rb"/(?:CreationDate|ModDate) \(D:[^)]+\)")


def _normalized_pdf(content: bytes) -> bytes:
    return _PDF_DATE.sub(b"/Date (normalized)", content)


def _pdf_body(path: Path) -> bytes:
    return _normalized_pdf(path.read_bytes())


def _temporary_sibling(target: Path, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=prefix,
        suffix=target.suffix,
    )
    os.close(descriptor)
    return Path(name)


def _validate_target(target: Path) -> None:
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError(f"Publication target parent is invalid: {target.parent}")
    if target.is_symlink():
        raise ValueError(f"Publication target cannot be a symbolic link: {target}")
    if target.exists() and not target.is_file():
        raise ValueError(f"Publication target must be a regular file: {target}")


def _stage_generation(contents: Mapping[Path, bytes]) -> dict[Path, Path]:
    staged: dict[Path, Path] = {}
    try:
        for target, content in contents.items():
            _validate_target(target)
            stage = _temporary_sibling(target, f".{target.name}.stage-")
            staged[target] = stage
            stage.write_bytes(content)
            if target.is_file():
                shutil.copymode(target, stage)
            else:
                stage.chmod(0o644)
        return staged
    except BaseException:
        for stage in staged.values():
            stage.unlink(missing_ok=True)
        raise


def _backup_generation(
    targets: Sequence[Path],
) -> tuple[dict[Path, bool], dict[Path, Path]]:
    originals: dict[Path, bool] = {}
    backups: dict[Path, Path] = {}
    try:
        for target in targets:
            _validate_target(target)
            originals[target] = target.is_file()
            if originals[target]:
                backup = _temporary_sibling(target, f".{target.name}.backup-")
                backups[target] = backup
                shutil.copy2(target, backup)
        return originals, backups
    except BaseException:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
        raise


def _commit_stage(stage: Path, target: Path) -> None:
    """Replace one target; kept separate so transaction failures are testable."""

    os.replace(stage, target)


def _restore_generation(
    targets: Sequence[Path],
    originals: Mapping[Path, bool],
    backups: Mapping[Path, Path],
) -> list[str]:
    errors: list[str] = []
    for target in targets:
        restore: Path | None = None
        try:
            if originals[target]:
                restore = _temporary_sibling(target, f".{target.name}.restore-")
                shutil.copy2(backups[target], restore)
                os.replace(restore, target)
                _require_regular_file(target, "Restored publication target")
                if target.read_bytes() != backups[target].read_bytes():
                    raise OSError(f"Restored publication target changed: {target}")
            elif target.exists() or target.is_symlink():
                target.unlink()
            if originals[target] != target.is_file():
                raise OSError(f"Restored publication target has wrong state: {target}")
        except BaseException as error:  # pragma: no cover - fatal filesystem failure
            errors.append(f"{target}: {error}")
        finally:
            if restore is not None:
                restore.unlink(missing_ok=True)
    return errors


def _postverify_generation(expected_appendix: str, generation: _Generation) -> None:
    if APPENDIX.read_text(encoding="utf-8") != expected_appendix:
        raise ValueError("Committed Expedia appendix changed during publication.")
    rendered, remaining = _expected_appendix(generation)
    if remaining or rendered != expected_appendix:
        raise RuntimeError(f"Post-update TeX remains stale: {remaining}")
    for name in FIGURE_NAMES:
        target = PAPER_FIGURES / f"mixed_order_{name}.pdf"
        _require_regular_file(target, "Committed mixed-order paper figure")
        if target.read_bytes() != generation.source_figures[name]:
            raise ValueError(f"Committed Expedia paper figure changed: {target.name}")


def _publish_generation(expected_appendix: str, generation: _Generation) -> None:
    contents = {
        APPENDIX: expected_appendix.encode("utf-8"),
        **{
            PAPER_FIGURES / f"mixed_order_{name}.pdf": generation.source_figures[name]
            for name in FIGURE_NAMES
        },
    }
    targets = list(contents)
    if len(set(targets)) != 3:
        raise ValueError("Expedia manuscript publication targets must be distinct.")
    staged = _stage_generation(contents)
    backups: dict[Path, Path] = {}
    try:
        originals, backups = _backup_generation(targets)
        try:
            _revalidate_identities(generation.input_identities)
            for target in targets:
                _commit_stage(staged[target], target)
            _postverify_generation(expected_appendix, generation)
            _revalidate_identities(
                generation.input_identities,
                exclude=(APPENDIX,),
            )
        except BaseException as publication_error:
            rollback_errors = _restore_generation(
                targets,
                originals,
                backups,
            )
            if rollback_errors:
                publication_error.add_note(
                    "Expedia manuscript publication rollback was incomplete: "
                    + " | ".join(rollback_errors)
                )
            raise
    finally:
        for path in (*staged.values(), *backups.values()):
            path.unlink(missing_ok=True)


def run(write: bool) -> None:
    with _publication_locks():
        generation = _load_generation()
        expected, changes = _expected_appendix(generation)
        stale_figures = [
            name
            for name in FIGURE_NAMES
            if (PAPER_FIGURES / f"mixed_order_{name}.pdf").is_symlink()
            or not (PAPER_FIGURES / f"mixed_order_{name}.pdf").is_file()
            or _normalized_pdf(generation.source_figures[name])
            != _pdf_body(PAPER_FIGURES / f"mixed_order_{name}.pdf")
        ]
        _revalidate_identities(generation.input_identities)
        if not write:
            problems = changes + [
                f"stale figure mixed_order_{name}.pdf" for name in stale_figures
            ]
            if problems:
                raise SystemExit(
                    "Expedia manuscript outputs are stale: " + ", ".join(problems)
                )
            print("Expedia manuscript result blocks are current.")
            return

        if changes or stale_figures:
            _publish_generation(expected, generation)
        else:
            _postverify_generation(expected, generation)
            _revalidate_identities(generation.input_identities)
        print(
            f"updated {len(changes)} TeX block(s) and "
            f"{len(stale_figures)} figure(s)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    run(parser.parse_args().write)


if __name__ == "__main__":
    main()
