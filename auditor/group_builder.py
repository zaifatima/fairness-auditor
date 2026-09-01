import argparse
import os
import pandas as pd


def load_interactions(path: str) -> pd.DataFrame:
    """Load a RecBole .inter file."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Interaction file not found: {path}"
        )

    df = pd.read_csv(
        path,
        sep="\t",
        engine="python"
    )

    # Convert RecBole names such as:
    # user_id:token -> user_id
    rename_map = {
        col: col.split(":")[0]
        for col in df.columns
        if ":" in col
    }

    df = df.rename(columns=rename_map)

    required = {"user_id", "item_id"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df


def build_user_groups(
    interactions: pd.DataFrame
) -> pd.DataFrame:
    """
    Divide users into approximately equal low/medium/high
    interaction groups using empirical thirds.
    """

    counts = (
        interactions
        .groupby("user_id")
        .size()
        .reset_index(name="interaction_count")
    )

    counts["user_group"] = pd.qcut(
        counts["interaction_count"],
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop"
    )

    return counts


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    interactions = load_interactions(args.input)

    groups = build_user_groups(interactions)

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True
    )

    groups.to_csv(
        args.output,
        index=False
    )

    print("\n=== User grouping complete ===")
    print(f"Users: {len(groups):,}")

    print("\nGroup distribution:")
    print(
        groups["user_group"]
        .value_counts()
        .sort_index()
    )

    print("\nInteraction statistics:")
    print(
        groups["interaction_count"]
        .describe()
    )

    print("\nGroup statistics:")
    print(
        groups.groupby("user_group", observed=True)[
            "interaction_count"
        ].agg(
            ["count", "mean", "median", "min", "max"]
        )
    )

    print("\nSample:")
    print(
        groups.head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
