import pandas as pd
import numpy as np


RECOMMENDATIONS = (
    "outputs/NeuMF-ml-32m-50k-top10-recommendations.csv"
)

ITEM_GROUPS = (
    "outputs/item_groups.csv"
)


def main():

    recommendations = pd.read_csv(
        RECOMMENDATIONS
    )

    item_groups = pd.read_csv(
        ITEM_GROUPS
    )

    # --------------------------------------------------------------
    # Join recommendation output to historical popularity group.
    # --------------------------------------------------------------

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

    # Check that every recommended item has a group.
    missing_groups = df["item_group"].isna().sum()

    print(
        f"Missing item groups: {missing_groups}"
    )

    if missing_groups > 0:
        raise ValueError(
            "Some recommended items could not be matched "
            "to an item popularity group."
        )

    # --------------------------------------------------------------
    # Position-discounted exposure.
    #
    # Rank 1 receives more exposure than rank 10.
    # --------------------------------------------------------------

    df["exposure_weight"] = (
        1
        / np.log2(df["rank"] + 1)
    )

    # --------------------------------------------------------------
    # Total exposure by popularity group.
    # --------------------------------------------------------------

    group_exposure = (
        df.groupby("item_group")
        .agg(
            recommendation_count=("item_id", "size"),
            total_exposure=("exposure_weight", "sum"),
            unique_items=("item_id", "nunique"),
        )
        .reset_index()
    )

    total_exposure = (
        group_exposure["total_exposure"].sum()
    )

    group_exposure["exposure_share"] = (
        group_exposure["total_exposure"]
        / total_exposure
    )

    group_exposure["recommendation_share"] = (
        group_exposure["recommendation_count"]
        / len(df)
    )

    print("\n=== Exposure by item popularity ===")
    print(
        group_exposure
        .sort_values("item_group")
        .to_string(index=False)
    )

    # --------------------------------------------------------------
    # Average rank by group.
    # --------------------------------------------------------------

    rank_summary = (
        df.groupby("item_group")
        .agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            best_rank=("rank", "min"),
        )
        .reset_index()
    )

    print(
        "\n=== Rank summary ==="
    )

    print(
        rank_summary
        .sort_values("item_group")
        .to_string(index=False)
    )

    # --------------------------------------------------------------
    # Top recommended items.
    # --------------------------------------------------------------

    item_summary = (
        df.groupby(
            [
                "item_id",
                "interaction_count",
                "item_group",
            ]
        )
        .agg(
            recommendation_count=("item_id", "size"),
            total_exposure=("exposure_weight", "sum"),
            best_rank=("rank", "min"),
        )
        .reset_index()
        .sort_values(
            "total_exposure",
            ascending=False,
        )
    )

    print(
        "\n=== Top 20 items by exposure ==="
    )

    print(
        item_summary.head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------------
    # Catalogue concentration.
    #
    # How much of the total exposure is captured by the
    # most-exposed items?
    # --------------------------------------------------------------

    total_item_exposure = (
        item_summary["total_exposure"].sum()
    )

    item_summary["exposure_share"] = (
        item_summary["total_exposure"]
        / total_item_exposure
    )

    for n in [5, 10, 20, 50]:

        share = (
            item_summary
            .head(n)["exposure_share"]
            .sum()
        )

        print(
            f"Top {n} items receive "
            f"{share:.2%} of recommendation exposure."
        )

    # --------------------------------------------------------------
    # Save diagnostic result.
    # --------------------------------------------------------------

    group_exposure.to_csv(
        "outputs/exposure_by_item_group.csv",
        index=False,
    )

    item_summary.to_csv(
        "outputs/item_exposure_summary.csv",
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        "  outputs/exposure_by_item_group.csv"
    )

    print(
        "  outputs/item_exposure_summary.csv"
    )


if __name__ == "__main__":
    main()
