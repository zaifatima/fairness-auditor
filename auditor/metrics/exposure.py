"""
Exposure Audit Module
=====================

Calculates recommendation-exposure diagnostics from:

1. A standardised recommendation file:
       model, timestamp, user_id, item_id, rank, score

2. An item-group file:
       item_id, interaction_count, item_group

The module deliberately reports exposure concentration as an
AUDIT SIGNAL rather than automatically labelling it "unfair".

Metrics produced
----------------
- Catalogue share
- Recommendation share
- Exposure share
- Exposure amplification
- Mean rank
- Median rank
- Best rank
- Top-K exposure concentration
- Gini exposure inequality

Position-discounted exposure:
    E(r) = 1 / log2(r + 1)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

RECOMMENDATION_COLUMNS = {
    "model",
    "timestamp",
    "user_id",
    "item_id",
    "rank",
}

ITEM_GROUP_COLUMNS = {
    "item_id",
    "item_group",
}


def validate_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    """Ensure required columns exist."""

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{sorted(missing)}"
        )


# ---------------------------------------------------------------------
# Gini
# ---------------------------------------------------------------------

def gini(values: np.ndarray) -> float:
    """
    Calculate the Gini coefficient for a non-negative vector.

    Returns:
        0.0 = perfectly equal exposure
        1.0 = maximally concentrated exposure
    """

    x = np.asarray(values, dtype=float)

    if x.size == 0:
        return np.nan

    if np.any(x < 0):
        raise ValueError(
            "Gini calculation requires non-negative values."
        )

    total = x.sum()

    if total == 0:
        return 0.0

    x = np.sort(x)

    n = len(x)

    index = np.arange(1, n + 1)

    return float(
        (
            (2 * index - n - 1) * x
        ).sum()
        / (n * total)
    )


# ---------------------------------------------------------------------
# Exposure weights
# ---------------------------------------------------------------------

def exposure_weight(rank: pd.Series) -> pd.Series:
    """
    Position-discounted exposure.

    Rank 1 receives the highest weight.
    """

    return 1.0 / np.log2(rank.astype(float) + 1.0)


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------

def calculate_exposure_audit(
    recommendations: pd.DataFrame,
    item_groups: pd.DataFrame,
    top_k_values: List[int] | None = None,
) -> Dict[str, pd.DataFrame]:

    if top_k_values is None:
        top_k_values = [5, 10, 20, 50]

    validate_columns(
        recommendations,
        RECOMMENDATION_COLUMNS,
        "Recommendation data",
    )

    validate_columns(
        item_groups,
        ITEM_GROUP_COLUMNS,
        "Item-group data",
    )

    recommendations = recommendations.copy()
    item_groups = item_groups.copy()

    # ---------------------------------------------------------------
    # Standardise join keys.
    # ---------------------------------------------------------------

    recommendations["item_id"] = (
        recommendations["item_id"].astype(str)
    )

    item_groups["item_id"] = (
        item_groups["item_id"].astype(str)
    )

    # ---------------------------------------------------------------
    # Prevent accidental duplicate item-group mappings.
    # ---------------------------------------------------------------

    if item_groups["item_id"].duplicated().any():
        raise ValueError(
            "item_groups contains duplicate item_id values."
        )

    # ---------------------------------------------------------------
    # Join historical item popularity information.
    # ---------------------------------------------------------------

    df = recommendations.merge(
        item_groups[
            [
                "item_id",
                "interaction_count",
                "item_group",
            ]
        ],
        on="item_id",
        how="left",
        validate="many_to_one",
    )

    missing_groups = int(
        df["item_group"].isna().sum()
    )

    if missing_groups:
        raise ValueError(
            f"{missing_groups} recommendation rows have "
            "no item-group assignment."
        )

    # ---------------------------------------------------------------
    # Rank validation.
    # ---------------------------------------------------------------

    if (df["rank"] <= 0).any():
        raise ValueError(
            "Rank values must be positive."
        )

    # ---------------------------------------------------------------
    # Position-discounted exposure.
    # ---------------------------------------------------------------

    df["exposure_weight"] = exposure_weight(
        df["rank"]
    )

    # ===============================================================
    # 1. CATALOGUE DISTRIBUTION
    # ===============================================================

    catalogue_by_group = (
        item_groups
        .groupby("item_group")
        .agg(
            catalogue_items=("item_id", "nunique")
        )
        .reset_index()
    )

    total_catalogue_items = (
        catalogue_by_group["catalogue_items"].sum()
    )

    catalogue_by_group["catalogue_share"] = (
        catalogue_by_group["catalogue_items"]
        / total_catalogue_items
    )

    # ===============================================================
    # 2. RECOMMENDATION + EXPOSURE DISTRIBUTION
    # ===============================================================

    exposure_by_group = (
        df
        .groupby("item_group")
        .agg(
            recommendation_count=("item_id", "size"),
            unique_recommended_items=("item_id", "nunique"),
            total_exposure=("exposure_weight", "sum"),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            best_rank=("rank", "min"),
        )
        .reset_index()
    )

    total_recommendations = len(df)

    total_exposure = (
        exposure_by_group["total_exposure"].sum()
    )

    exposure_by_group["recommendation_share"] = (
        exposure_by_group["recommendation_count"]
        / total_recommendations
    )

    exposure_by_group["exposure_share"] = (
        exposure_by_group["total_exposure"]
        / total_exposure
    )

    # ===============================================================
    # 3. JOIN CATALOGUE SHARE
    # ===============================================================

    group_summary = catalogue_by_group.merge(
        exposure_by_group,
        on="item_group",
        how="outer",
    )

    # Missing values are possible if a group exists in catalogue
    # but receives zero recommendation exposure.
    numeric_columns = [
        "recommendation_count",
        "unique_recommended_items",
        "total_exposure",
    ]

    for column in numeric_columns:
        group_summary[column] = (
            group_summary[column]
            .fillna(0)
        )

    # ===============================================================
    # 4. EXPOSURE AMPLIFICATION
    # ===============================================================

    group_summary["exposure_amplification"] = (
        group_summary["exposure_share"]
        / group_summary["catalogue_share"]
    )

    # ===============================================================
    # 5. EXPOSURE / RECOMMENDATION OVER-REPRESENTATION
    # ===============================================================

    group_summary["exposure_minus_recommendation_share"] = (
        group_summary["exposure_share"]
        - group_summary["recommendation_share"]
    )

    # ===============================================================
    # 6. ITEM-LEVEL EXPOSURE
    # ===============================================================

    item_exposure = (
        df
        .groupby(
            [
                "item_id",
                "interaction_count",
                "item_group",
            ]
        )
        .agg(
            recommendation_count=("item_id", "size"),
            total_exposure=("exposure_weight", "sum"),
            mean_rank=("rank", "mean"),
            best_rank=("rank", "min"),
        )
        .reset_index()
    )

    item_exposure = item_exposure.sort_values(
        "total_exposure",
        ascending=False,
    )

    item_exposure["exposure_share"] = (
        item_exposure["total_exposure"]
        / item_exposure["total_exposure"].sum()
    )

    item_exposure["cumulative_exposure_share"] = (
        item_exposure["exposure_share"]
        .cumsum()
    )

    # ===============================================================
    # 7. GINI EXPOSURE INEQUALITY
    # ===============================================================

    # IMPORTANT:
    # Include zero-exposure catalogue items. Otherwise the Gini
    # ignores the thousands of catalogue items receiving no exposure.
    #
    # This is critical for an exposure-concentration audit.

    all_catalogue = item_groups[
        [
            "item_id",
            "item_group",
        ]
    ].copy()

    all_catalogue = all_catalogue.drop_duplicates(
        subset=["item_id"]
    )

    item_exposure_values = (
        item_exposure[
            [
                "item_id",
                "total_exposure",
            ]
        ]
    )

    gini_df = all_catalogue.merge(
        item_exposure_values,
        on="item_id",
        how="left",
    )

    gini_df["total_exposure"] = (
        gini_df["total_exposure"]
        .fillna(0.0)
    )

    overall_gini = gini(
        gini_df["total_exposure"].to_numpy()
    )

    # Gini by item group is also useful diagnostically.
    group_gini_rows = []

    for group_name, group_data in (
        gini_df.groupby("item_group")
    ):

        group_gini_rows.append(
            {
                "item_group": group_name,
                "gini_exposure": gini(
                    group_data[
                        "total_exposure"
                    ].to_numpy()
                ),
            }
        )

    gini_by_group = pd.DataFrame(
        group_gini_rows
    )

    # ===============================================================
    # 8. TOP-K CONCENTRATION
    # ===============================================================

    concentration_rows = []

    for k in top_k_values:

        available_k = min(
            k,
            len(item_exposure),
        )

        concentration = (
            item_exposure
            .head(available_k)[
                "total_exposure"
            ]
            .sum()
            / total_exposure
        )

        concentration_rows.append(
            {
                "top_k": k,
                "items_included": available_k,
                "exposure_share": concentration,
            }
        )

    top_k_concentration = pd.DataFrame(
        concentration_rows
    )

    # ===============================================================
    # 9. METADATA / SUMMARY
    # ===============================================================

    metadata = pd.DataFrame(
        [
            {
                "model": (
                    df["model"]
                    .dropna()
                    .astype(str)
                    .iloc[0]
                ),
                "timestamp": (
                    df["timestamp"]
                    .dropna()
                    .astype(str)
                    .iloc[0]
                ),
                "users_audited": (
                    df["user_id"].nunique()
                ),
                "recommendation_rows": len(df),
                "catalogue_items": (
                    item_groups["item_id"]
                    .nunique()
                ),
                "recommended_items": (
                    df["item_id"].nunique()
                ),
                "overall_gini": overall_gini,
            }
        ]
    )

    return {
        "group_summary": group_summary,
        "item_exposure": item_exposure,
        "gini_by_group": gini_by_group,
        "top_k_concentration": top_k_concentration,
        "metadata": metadata,
        "joined_data": df,
    }


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Run exposure audit."
    )

    parser.add_argument(
        "--recommendations",
        required=True,
    )

    parser.add_argument(
        "--item-groups",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/exposure_audit",
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    recommendations = pd.read_csv(
        args.recommendations
    )

    item_groups = pd.read_csv(
        args.item_groups
    )

    results = calculate_exposure_audit(
        recommendations,
        item_groups,
    )

    # ---------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------

    results["group_summary"].to_csv(
        os.path.join(
            args.output_dir,
            "exposure_by_group.csv",
        ),
        index=False,
    )

    results["item_exposure"].to_csv(
        os.path.join(
            args.output_dir,
            "item_exposure.csv",
        ),
        index=False,
    )

    results["gini_by_group"].to_csv(
        os.path.join(
            args.output_dir,
            "gini_by_group.csv",
        ),
        index=False,
    )

    results["top_k_concentration"].to_csv(
        os.path.join(
            args.output_dir,
            "top_k_concentration.csv",
        ),
        index=False,
    )

    results["metadata"].to_csv(
        os.path.join(
            args.output_dir,
            "audit_metadata.csv",
        ),
        index=False,
    )

    # The joined data is useful during development but can become
    # large in the final experiment, so keep it as an optional output.
    results["joined_data"].to_csv(
        os.path.join(
            args.output_dir,
            "audited_recommendations.csv",
        ),
        index=False,
    )

    # ---------------------------------------------------------------
    # Console report
    # ---------------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "       EXPOSURE AUDIT COMPLETE"
    )

    print(
        "=========================================="
    )

    metadata = results["metadata"].iloc[0]

    print(
        f"\nModel: {metadata['model']}"
    )

    print(
        f"Users audited: {metadata['users_audited']:,}"
    )

    print(
        f"Recommendation rows: "
        f"{metadata['recommendation_rows']:,}"
    )

    print(
        f"Catalogue items: "
        f"{metadata['catalogue_items']:,}"
    )

    print(
        f"Recommended items: "
        f"{metadata['recommended_items']:,}"
    )

    print(
        f"Overall exposure Gini: "
        f"{metadata['overall_gini']:.6f}"
    )

    print(
        "\n--- Exposure by item group ---"
    )

    display_columns = [
        "item_group",
        "catalogue_items",
        "catalogue_share",
        "recommendation_count",
        "recommendation_share",
        "total_exposure",
        "exposure_share",
        "exposure_amplification",
        "mean_rank",
        "best_rank",
    ]

    print(
        results["group_summary"][
            display_columns
        ]
        .sort_values("item_group")
        .to_string(index=False)
    )

    print(
        "\n--- Top-K exposure concentration ---"
    )

    print(
        results["top_k_concentration"]
        .to_string(index=False)
    )

    print(
        "\n--- Highest-exposure items ---"
    )

    print(
        results["item_exposure"]
        .head(20)
        .to_string(index=False)
    )

    print(
        "\nOutputs saved to:"
    )

    print(
        os.path.abspath(args.output_dir)
    )


if __name__ == "__main__":
    main()
Exposure Audit Module
=====================

Calculates recommendation-exposure diagnostics from:

1. A standardised recommendation file:
       model, timestamp, user_id, item_id, rank, score

2. An item-group file:
       item_id, interaction_count, item_group

The module deliberately reports exposure concentration as an
AUDIT SIGNAL rather than automatically labelling it "unfair".

Metrics produced
----------------
- Catalogue share
- Recommendation share
- Exposure share
- Exposure amplification
- Mean rank
- Median rank
- Best rank
- Top-K exposure concentration
- Gini exposure inequality

Position-discounted exposure:
    E(r) = 1 / log2(r + 1)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

RECOMMENDATION_COLUMNS = {
    "model",
    "timestamp",
    "user_id",
    "item_id",
    "rank",
}

ITEM_GROUP_COLUMNS = {
    "item_id",
    "item_group",
}


def validate_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    """Ensure required columns exist."""

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{sorted(missing)}"
        )


# ---------------------------------------------------------------------
# Gini
# ---------------------------------------------------------------------

def gini(values: np.ndarray) -> float:
    """
    Calculate the Gini coefficient for a non-negative vector.

    Returns:
        0.0 = perfectly equal exposure
        1.0 = maximally concentrated exposure
    """

    x = np.asarray(values, dtype=float)

    if x.size == 0:
        return np.nan

    if np.any(x < 0):
        raise ValueError(
            "Gini calculation requires non-negative values."
        )

    total = x.sum()

    if total == 0:
        return 0.0

    x = np.sort(x)

    n = len(x)

    index = np.arange(1, n + 1)

    return float(
        (
            (2 * index - n - 1) * x
        ).sum()
        / (n * total)
    )


# ---------------------------------------------------------------------
# Exposure weights
# ---------------------------------------------------------------------

def exposure_weight(rank: pd.Series) -> pd.Series:
    """
    Position-discounted exposure.

    Rank 1 receives the highest weight.
    """

    return 1.0 / np.log2(rank.astype(float) + 1.0)


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------

def calculate_exposure_audit(
    recommendations: pd.DataFrame,
    item_groups: pd.DataFrame,
    top_k_values: List[int] | None = None,
) -> Dict[str, pd.DataFrame]:

    if top_k_values is None:
        top_k_values = [5, 10, 20, 50]

    validate_columns(
        recommendations,
        RECOMMENDATION_COLUMNS,
        "Recommendation data",
    )

    validate_columns(
        item_groups,
        ITEM_GROUP_COLUMNS,
        "Item-group data",
    )

    recommendations = recommendations.copy()
    item_groups = item_groups.copy()

    # ---------------------------------------------------------------
    # Standardise join keys.
    # ---------------------------------------------------------------

    recommendations["item_id"] = (
        recommendations["item_id"].astype(str)
    )

    item_groups["item_id"] = (
        item_groups["item_id"].astype(str)
    )

    # ---------------------------------------------------------------
    # Prevent accidental duplicate item-group mappings.
    # ---------------------------------------------------------------

    if item_groups["item_id"].duplicated().any():
        raise ValueError(
            "item_groups contains duplicate item_id values."
        )

    # ---------------------------------------------------------------
    # Join historical item popularity information.
    # ---------------------------------------------------------------

    df = recommendations.merge(
        item_groups[
            [
                "item_id",
                "interaction_count",
                "item_group",
            ]
        ],
        on="item_id",
        how="left",
        validate="many_to_one",
    )

    missing_groups = int(
        df["item_group"].isna().sum()
    )

    if missing_groups:
        raise ValueError(
            f"{missing_groups} recommendation rows have "
            "no item-group assignment."
        )

    # ---------------------------------------------------------------
    # Rank validation.
    # ---------------------------------------------------------------

    if (df["rank"] <= 0).any():
        raise ValueError(
            "Rank values must be positive."
        )

    # ---------------------------------------------------------------
    # Position-discounted exposure.
    # ---------------------------------------------------------------

    df["exposure_weight"] = exposure_weight(
        df["rank"]
    )

    # ===============================================================
    # 1. CATALOGUE DISTRIBUTION
    # ===============================================================

    catalogue_by_group = (
        item_groups
        .groupby("item_group")
        .agg(
            catalogue_items=("item_id", "nunique")
        )
        .reset_index()
    )

    total_catalogue_items = (
        catalogue_by_group["catalogue_items"].sum()
    )

    catalogue_by_group["catalogue_share"] = (
        catalogue_by_group["catalogue_items"]
        / total_catalogue_items
    )

    # ===============================================================
    # 2. RECOMMENDATION + EXPOSURE DISTRIBUTION
    # ===============================================================

    exposure_by_group = (
        df
        .groupby("item_group")
        .agg(
            recommendation_count=("item_id", "size"),
            unique_recommended_items=("item_id", "nunique"),
            total_exposure=("exposure_weight", "sum"),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            best_rank=("rank", "min"),
        )
        .reset_index()
    )

    total_recommendations = len(df)

    total_exposure = (
        exposure_by_group["total_exposure"].sum()
    )

    exposure_by_group["recommendation_share"] = (
        exposure_by_group["recommendation_count"]
        / total_recommendations
    )

    exposure_by_group["exposure_share"] = (
        exposure_by_group["total_exposure"]
        / total_exposure
    )

    # ===============================================================
    # 3. JOIN CATALOGUE SHARE
    # ===============================================================

    group_summary = catalogue_by_group.merge(
        exposure_by_group,
        on="item_group",
        how="outer",
    )

    # Missing values are possible if a group exists in catalogue
    # but receives zero recommendation exposure.
    numeric_columns = [
        "recommendation_count",
        "unique_recommended_items",
        "total_exposure",
    ]

    for column in numeric_columns:
        group_summary[column] = (
            group_summary[column]
            .fillna(0)
        )

    # ===============================================================
    # 4. EXPOSURE AMPLIFICATION
    # ===============================================================

    group_summary["exposure_amplification"] = (
        group_summary["exposure_share"]
        / group_summary["catalogue_share"]
    )

    # ===============================================================
    # 5. EXPOSURE / RECOMMENDATION OVER-REPRESENTATION
    # ===============================================================

    group_summary["exposure_minus_recommendation_share"] = (
        group_summary["exposure_share"]
        - group_summary["recommendation_share"]
    )

    # ===============================================================
    # 6. ITEM-LEVEL EXPOSURE
    # ===============================================================

    item_exposure = (
        df
        .groupby(
            [
                "item_id",
                "interaction_count",
                "item_group",
            ]
        )
        .agg(
            recommendation_count=("item_id", "size"),
            total_exposure=("exposure_weight", "sum"),
            mean_rank=("rank", "mean"),
            best_rank=("rank", "min"),
        )
        .reset_index()
    )

    item_exposure = item_exposure.sort_values(
        "total_exposure",
        ascending=False,
    )

    item_exposure["exposure_share"] = (
        item_exposure["total_exposure"]
        / item_exposure["total_exposure"].sum()
    )

    item_exposure["cumulative_exposure_share"] = (
        item_exposure["exposure_share"]
        .cumsum()
    )

    # ===============================================================
    # 7. GINI EXPOSURE INEQUALITY
    # ===============================================================

    # IMPORTANT:
    # Include zero-exposure catalogue items. Otherwise the Gini
    # ignores the thousands of catalogue items receiving no exposure.
    #
    # This is critical for an exposure-concentration audit.

    all_catalogue = item_groups[
        [
            "item_id",
            "item_group",
        ]
    ].copy()

    all_catalogue = all_catalogue.drop_duplicates(
        subset=["item_id"]
    )

    item_exposure_values = (
        item_exposure[
            [
                "item_id",
                "total_exposure",
            ]
        ]
    )

    gini_df = all_catalogue.merge(
        item_exposure_values,
        on="item_id",
        how="left",
    )

    gini_df["total_exposure"] = (
        gini_df["total_exposure"]
        .fillna(0.0)
    )

    overall_gini = gini(
        gini_df["total_exposure"].to_numpy()
    )

    # Gini by item group is also useful diagnostically.
    group_gini_rows = []

    for group_name, group_data in (
        gini_df.groupby("item_group")
    ):

        group_gini_rows.append(
            {
                "item_group": group_name,
                "gini_exposure": gini(
                    group_data[
                        "total_exposure"
                    ].to_numpy()
                ),
            }
        )

    gini_by_group = pd.DataFrame(
        group_gini_rows
    )

    # ===============================================================
    # 8. TOP-K CONCENTRATION
    # ===============================================================

    concentration_rows = []

    for k in top_k_values:

        available_k = min(
            k,
            len(item_exposure),
        )

        concentration = (
            item_exposure
            .head(available_k)[
                "total_exposure"
            ]
            .sum()
            / total_exposure
        )

        concentration_rows.append(
            {
                "top_k": k,
                "items_included": available_k,
                "exposure_share": concentration,
            }
        )

    top_k_concentration = pd.DataFrame(
        concentration_rows
    )

    # ===============================================================
    # 9. METADATA / SUMMARY
    # ===============================================================

    metadata = pd.DataFrame(
        [
            {
                "model": (
                    df["model"]
                    .dropna()
                    .astype(str)
                    .iloc[0]
                ),
                "timestamp": (
                    df["timestamp"]
                    .dropna()
                    .astype(str)
                    .iloc[0]
                ),
                "users_audited": (
                    df["user_id"].nunique()
                ),
                "recommendation_rows": len(df),
                "catalogue_items": (
                    item_groups["item_id"]
                    .nunique()
                ),
                "recommended_items": (
                    df["item_id"].nunique()
                ),
                "overall_gini": overall_gini,
            }
        ]
    )

    return {
        "group_summary": group_summary,
        "item_exposure": item_exposure,
        "gini_by_group": gini_by_group,
        "top_k_concentration": top_k_concentration,
        "metadata": metadata,
        "joined_data": df,
    }


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Run exposure audit."
    )

    parser.add_argument(
        "--recommendations",
        required=True,
    )

    parser.add_argument(
        "--item-groups",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/exposure_audit",
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    recommendations = pd.read_csv(
        args.recommendations
    )

    item_groups = pd.read_csv(
        args.item_groups
    )

    results = calculate_exposure_audit(
        recommendations,
        item_groups,
    )

    # ---------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------

    results["group_summary"].to_csv(
        os.path.join(
            args.output_dir,
            "exposure_by_group.csv",
        ),
        index=False,
    )

    results["item_exposure"].to_csv(
        os.path.join(
            args.output_dir,
            "item_exposure.csv",
        ),
        index=False,
    )

    results["gini_by_group"].to_csv(
        os.path.join(
            args.output_dir,
            "gini_by_group.csv",
        ),
        index=False,
    )

    results["top_k_concentration"].to_csv(
        os.path.join(
            args.output_dir,
            "top_k_concentration.csv",
        ),
        index=False,
    )

    results["metadata"].to_csv(
        os.path.join(
            args.output_dir,
            "audit_metadata.csv",
        ),
        index=False,
    )

    # The joined data is useful during development but can become
    # large in the final experiment, so keep it as an optional output.
    results["joined_data"].to_csv(
        os.path.join(
            args.output_dir,
            "audited_recommendations.csv",
        ),
        index=False,
    )

    # ---------------------------------------------------------------
    # Console report
    # ---------------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "       EXPOSURE AUDIT COMPLETE"
    )

    print(
        "=========================================="
    )

    metadata = results["metadata"].iloc[0]

    print(
        f"\nModel: {metadata['model']}"
    )

    print(
        f"Users audited: {metadata['users_audited']:,}"
    )

    print(
        f"Recommendation rows: "
        f"{metadata['recommendation_rows']:,}"
    )

    print(
        f"Catalogue items: "
        f"{metadata['catalogue_items']:,}"
    )

    print(
        f"Recommended items: "
        f"{metadata['recommended_items']:,}"
    )

    print(
        f"Overall exposure Gini: "
        f"{metadata['overall_gini']:.6f}"
    )

    print(
        "\n--- Exposure by item group ---"
    )

    display_columns = [
        "item_group",
        "catalogue_items",
        "catalogue_share",
        "recommendation_count",
        "recommendation_share",
        "total_exposure",
        "exposure_share",
        "exposure_amplification",
        "mean_rank",
        "best_rank",
    ]

    print(
        results["group_summary"][
            display_columns
        ]
        .sort_values("item_group")
        .to_string(index=False)
    )

    print(
        "\n--- Top-K exposure concentration ---"
    )

    print(
        results["top_k_concentration"]
        .to_string(index=False)
    )

    print(
        "\n--- Highest-exposure items ---"
    )

    print(
        results["item_exposure"]
        .head(20)
        .to_string(index=False)
    )

    print(
        "\nOutputs saved to:"
    )

    print(
        os.path.abspath(args.output_dir)
    )


if __name__ == "__main__":
    main()
