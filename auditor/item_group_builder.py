import argparse
import os

import pandas as pd


def load_interactions(path: str) -> pd.DataFrame:
    """Load a RecBole .inter interaction file."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Interaction file not found: {path}"
        )

    df = pd.read_csv(
        path,
        sep="\t",
        engine="python",
    )

    # Convert RecBole column names such as:
    # user_id:token -> user_id
    # item_id:token -> item_id
    rename_map = {
        column: column.split(":")[0]
        for column in df.columns
        if ":" in column
    }

    df = df.rename(columns=rename_map)

    required = {"user_id", "item_id"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df


def build_item_groups(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate item popularity based on historical interaction count.

    Items are divided into empirical thirds:
        niche    = lowest interaction counts
        medium   = middle interaction counts
        popular  = highest interaction counts
    """

    counts = (
        interactions
        .groupby("item_id")
        .size()
        .reset_index(name="interaction_count")
    )

    counts["item_group"] = pd.qcut(
        counts["interaction_count"],
        q=3,
        labels=[
            "niche",
            "medium",
            "popular",
        ],
        duplicates="drop",
    )

    return counts


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    interactions = load_interactions(
        args.input
    )

    groups = build_item_groups(
        interactions
    )

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True,
    )

    groups.to_csv(
        args.output,
        index=False,
    )

    print(
        "\n=== Item grouping complete ==="
    )

    print(
        f"Items: {len(groups):,}"
    )

    print(
        "\nGroup distribution:"
    )

    print(
        groups["item_group"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nInteraction statistics:"
    )

    print(
        groups["interaction_count"].describe()
    )

    print(
        "\nGroup statistics:"
    )

    print(
        groups.groupby(
            "item_group",
            observed=True,
        )["interaction_count"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
    )

    print(
        "\nSample:"
    )

    print(
        groups.head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
