"""
Temporal Fairness Monitoring
============================

Builds a chronological fairness-monitoring dataset from recommendation
snapshots and historical MovieLens interactions.

The module is deliberately separated from forecasting:
    Temporal monitoring -> produces the historical metric trajectory
    Forecasting         -> predicts future fairness deterioration

Metrics:
    - Gini exposure inequality
    - Catalogue coverage
    - Overall recommendation diversity
    - Niche-item proportion
    - Fairness gap
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_TOP_K = 10

# MovieLens 32M item popularity groups:
# bottom 50% = niche
# 50%-80% = mid
# top 20% = popular
POPULARITY_GROUPS = ["niche", "mid", "popular"]


# ---------------------------------------------------------------------
# Gini
# ---------------------------------------------------------------------

def gini(values: np.ndarray) -> float:
    """
    Calculate the Gini coefficient.

    0 = equal exposure
    1 = maximum concentration
    """

    x = np.asarray(values, dtype=float)

    if x.size == 0:
        return 0.0

    if np.any(x < 0):
        raise ValueError("Gini requires non-negative values.")

    total = x.sum()

    if total == 0:
        return 0.0

    x = np.sort(x)
    n = len(x)

    index = np.arange(1, n + 1)

    return float(
        ((2 * index - n - 1) * x).sum()
        / (n * total)
    )


# ---------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------

def exposure_weight(rank: pd.Series) -> pd.Series:
    """
    Position-discounted exposure.

    Rank 1 receives the greatest exposure.
    """

    return 1.0 / np.log2(rank.astype(float) + 1.0)


# ---------------------------------------------------------------------
# Popularity groups
# ---------------------------------------------------------------------

def build_period_popularity_groups(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign each item to a popularity group for one historical period.

    Popularity is based only on interactions available within the
    supplied period.

    Groups:
        bottom 50% -> niche
        50%-80%    -> mid
        top 20%    -> popular
    """

    item_counts = (
        interactions
        .groupby("item_id")
        .size()
        .reset_index(name="interaction_count")
    )

    if item_counts.empty:
        item_counts["item_group"] = pd.Series(dtype=str)
        return item_counts

    # Rank rather than qcut avoids failures caused by tied counts.
    item_counts["popularity_rank"] = (
        item_counts["interaction_count"]
        .rank(method="first", pct=True)
    )

    item_counts["item_group"] = np.select(
        [
            item_counts["popularity_rank"] <= 0.50,
            item_counts["popularity_rank"] <= 0.80,
        ],
        [
            "niche",
            "mid",
        ],
        default="popular",
    )

    return item_counts[
        ["item_id", "interaction_count", "item_group"]
    ]


# ---------------------------------------------------------------------
# Period metrics
# ---------------------------------------------------------------------

def calculate_period_metrics(
    recommendations: pd.DataFrame,
    item_groups: pd.DataFrame,
    total_catalogue_items: int,
    period: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Calculate monitoring metrics for one temporal period.
    """

    required_recommendation_columns = {
        "user_id",
        "item_id",
        "rank",
    }

    missing = (
        required_recommendation_columns
        - set(recommendations.columns)
    )

    if missing:
        raise ValueError(
            f"Recommendations missing columns: {sorted(missing)}"
        )

    df = recommendations.copy()

    # Only monitor the requested Top-K.
    df = df[df["rank"] <= top_k].copy()

    if df.empty:
        return {
            "period": period,
            "gini_index": 0.0,
            "catalog_coverage_pct": 0.0,
            "overall_diversity": 0.0,
            "niche_item_proportion": 0.0,
            "fairness_gap": 0.0,
            "recommendation_count": 0,
            "unique_recommended_items": 0,
        }

    # Attach period-specific popularity groups.
    df = df.merge(
        item_groups[
            ["item_id", "item_group"]
        ],
        on="item_id",
        how="left",
        validate="many_to_one",
    )

    missing_groups = int(df["item_group"].isna().sum())

    if missing_groups:
        raise ValueError(
            f"{missing_groups} recommendations have no "
            "period-specific popularity group."
        )

    # Position-discounted exposure.
    df["exposure"] = exposure_weight(df["rank"])

    # --------------------------------------------------------------
    # 1. Gini exposure inequality
    # --------------------------------------------------------------

    exposure_by_item = (
        df.groupby("item_id")["exposure"]
        .sum()
    )

    # Include zero-exposure catalogue items.
    all_items = item_groups["item_id"].drop_duplicates()

    exposure_vector = (
        all_items
        .to_frame()
        .merge(
            exposure_by_item.rename("exposure"),
            on="item_id",
            how="left",
        )["exposure"]
        .fillna(0.0)
        .to_numpy()
    )

    gini_index = gini(exposure_vector)

    # --------------------------------------------------------------
    # 2. Catalogue coverage
    # --------------------------------------------------------------

    unique_recommended_items = df["item_id"].nunique()

    coverage_pct = (
        unique_recommended_items
        / total_catalogue_items
        * 100.0
        if total_catalogue_items > 0
        else 0.0
    )

    # --------------------------------------------------------------
    # 3. Overall diversity
    # --------------------------------------------------------------

    # In the monitoring dataset this is represented as the proportion
    # of recommendation rows that correspond to distinct items.
    overall_diversity = (
        unique_recommended_items / len(df)
        if len(df) > 0
        else 0.0
    )

    # --------------------------------------------------------------
    # 4. Niche-item proportion
    # --------------------------------------------------------------

    niche_item_proportion = (
        (df["item_group"] == "niche").mean()
    )

    # --------------------------------------------------------------
    # 5. Fairness gap
    # --------------------------------------------------------------

    group_exposure = (
        df.groupby("item_group")["exposure"]
        .sum()
    )

    total_exposure = group_exposure.sum()

    if total_exposure > 0:
        exposure_share = (
            group_exposure / total_exposure
        )
    else:
        exposure_share = pd.Series(dtype=float)

    catalogue_share = (
        item_groups["item_group"]
        .value_counts(normalize=True)
    )

    # Compare observed exposure share with catalogue availability.
    group_amplification = {}

    for group in POPULARITY_GROUPS:

        catalogue_prop = float(
            catalogue_share.get(group, 0.0)
        )

        exposure_prop = float(
            exposure_share.get(group, 0.0)
        )

        if catalogue_prop > 0:
            group_amplification[group] = (
                exposure_prop / catalogue_prop
            )
        else:
            group_amplification[group] = 0.0

    amplification_values = list(
        group_amplification.values()
    )

    fairness_gap = (
        max(amplification_values)
        - min(amplification_values)
        if amplification_values
        else 0.0
    )

    return {
        "period": period,
        "gini_index": round(gini_index, 6),
        "catalog_coverage_pct": round(coverage_pct, 4),
        "overall_diversity": round(overall_diversity, 6),
        "niche_item_proportion": round(
            float(niche_item_proportion),
            6,
        ),
        "fairness_gap": round(fairness_gap, 6),
        "recommendation_count": int(len(df)),
        "unique_recommended_items": int(
            unique_recommended_items
        ),
    }


# ---------------------------------------------------------------------
# Temporal monitor
# ---------------------------------------------------------------------

def run_temporal_monitor(
    recommendations: pd.DataFrame,
    interactions: pd.DataFrame,
    period_column: str = "year",
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """
    Run temporal monitoring across historical periods.

    Expected recommendation columns:
        period/year
        user_id
        item_id
        rank

    Expected interaction columns:
        user_id
        item_id
        timestamp/year
    """

    if period_column not in recommendations.columns:
        raise ValueError(
            f"Recommendation data requires '{period_column}'."
        )

    if period_column not in interactions.columns:
        raise ValueError(
            f"Interaction data requires '{period_column}'."
        )

    periods = sorted(
        recommendations[period_column]
        .dropna()
        .unique()
    )

    results = []

    for period in periods:

        period_recommendations = recommendations[
            recommendations[period_column] == period
        ].copy()

        period_interactions = interactions[
            interactions[period_column] == period
        ].copy()

        if period_interactions.empty:
            continue

        period_item_groups = (
            build_period_popularity_groups(
                period_interactions
            )
        )

        total_catalogue_items = (
            period_item_groups["item_id"].nunique()
        )

        metrics = calculate_period_metrics(
            recommendations=period_recommendations,
            item_groups=period_item_groups,
            total_catalogue_items=total_catalogue_items,
            period=str(period),
            top_k=top_k,
        )

        results.append(metrics)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Run temporal fairness monitoring."
    )

    parser.add_argument(
        "--recommendations",
        required=True,
        help="CSV containing period-specific recommendations.",
    )

    parser.add_argument(
        "--interactions",
        required=True,
        help="CSV containing historical interactions.",
    )

    parser.add_argument(
        "--period-column",
        default="year",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output",
        default="outputs/temporal/temporal_metrics.csv",
    )

    args = parser.parse_args()

    recommendations = pd.read_csv(
        args.recommendations
    )

    interactions = pd.read_csv(
        args.interactions
    )

    results = run_temporal_monitor(
        recommendations=recommendations,
        interactions=interactions,
        period_column=args.period_column,
        top_k=args.top_k,
    )

    output_dir = os.path.dirname(args.output)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    results.to_csv(
        args.output,
        index=False,
    )

    print(
        "\n=========================================="
    )
    print(
        "       TEMPORAL MONITOR COMPLETE"
    )
    print(
        "=========================================="
    )

    print(
        f"\nPeriods analysed: {len(results)}"
    )

    if not results.empty:
        print(
            results.to_string(index=False)
        )

    print(
        f"\nSaved to: {os.path.abspath(args.output)}"
    )


if __name__ == "__main__":
    main()