from pathlib import Path
import pandas as pd


# ============================================================
# Temporal Dataset Builder
# Builds leakage-safe historical train/test datasets
# for the temporal fairness auditing experiment.
#
# For each evaluation year t:
#   Train = interactions from years < t
#   Test  = interactions from year t
#
# Evaluation period: 1999-2023
# ============================================================

PROJECT_ROOT = Path(r"C:\mscproject_code")

INPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "ml-32m-50k"
    / "ml-32m-50k.inter"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "temporal_datasets"

START_YEAR = 1999
END_YEAR = 2023


def load_interactions():
    """Load the RecBole .inter file into a pandas DataFrame."""

    print("=" * 70)
    print("Loading interaction data")
    print("=" * 70)
    print(f"Input: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find interaction file:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        dtype={
            "user_id:token": "int32",
            "item_id:token": "int32",
            "rating:float": "float32",
            "timestamp:float": "float64",
        },
    )

    df = df.rename(
        columns={
            "user_id:token": "user_id",
            "item_id:token": "item_id",
            "rating:float": "rating",
            "timestamp:float": "timestamp",
        }
    )

    # MovieLens timestamps are Unix timestamps.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        errors="coerce",
    )

    df["year"] = df["timestamp"].dt.year

    if df["year"].isna().any():
        bad_rows = df["year"].isna().sum()
        raise ValueError(
            f"{bad_rows} rows have invalid timestamps."
        )

    print(f"Rows loaded: {len(df):,}")
    print(
        f"Year range: "
        f"{int(df['year'].min())} - {int(df['year'].max())}"
    )

    return df


def save_temporal_dataset(df, year):
    """
    Save one historical checkpoint.

    Training data contains only interactions before the
    evaluation year.

    Test data contains interactions from the evaluation year.
    """

    train_df = df[df["year"] < year].copy()
    test_df = df[df["year"] == year].copy()

    year_dir = OUTPUT_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    train_file = year_dir / "train.csv"
    test_file = year_dir / "test.csv"

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    return train_df, test_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_interactions()

    print()
    print("=" * 70)
    print(
        f"Building temporal datasets: "
        f"{START_YEAR}-{END_YEAR}"
    )
    print("=" * 70)

    summary = []

    for year in range(START_YEAR, END_YEAR + 1):

        train_df, test_df = save_temporal_dataset(df, year)

        summary.append(
            {
                "year": year,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_users": train_df["user_id"].nunique(),
                "test_users": test_df["user_id"].nunique(),
                "train_items": train_df["item_id"].nunique(),
                "test_items": test_df["item_id"].nunique(),
            }
        )

        print(
            f"{year}: "
            f"train={len(train_df):,} rows, "
            f"test={len(test_df):,} rows, "
            f"train_users={train_df['user_id'].nunique():,}, "
            f"test_users={test_df['user_id'].nunique():,}"
        )

    summary_df = pd.DataFrame(summary)

    summary_file = OUTPUT_DIR / "temporal_dataset_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print()
    print("=" * 70)
    print("Temporal dataset construction complete")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Summary file: {summary_file}")

    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()