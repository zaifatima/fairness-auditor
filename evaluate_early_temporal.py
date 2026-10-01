import os
import numpy as np
import pandas as pd

ROOT = r"C:\mscproject_code"

REC_DIR = os.path.join(
    ROOT, "outputs", "temporal_recommendations"
)

INTER_FILE = os.path.join(
    ROOT, "dataset", "ml-32m-50k", "ml-32m-50k.inter"
)

YEARS = [1999, 2000, 2001]

MODELS = ["NeuMF", "LightGCN", "SASRec"]


def gini(values):
    x = np.asarray(values, dtype=float)

    if len(x) == 0:
        return np.nan

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


def exposure_weight(rank):
    return 1.0 / np.log2(rank + 1)


def load_items():
    df = pd.read_csv(
        INTER_FILE,
        sep="\t",
        engine="python"
    )

    df.columns = [
        str(c).split(":")[0].strip()
        for c in df.columns
    ]

    counts = (
        df.groupby("item_id")
        .size()
        .reset_index(name="interaction_count")
    )

    counts["item_group"] = pd.qcut(
        counts["interaction_count"],
        q=3,
        labels=["niche", "medium", "popular"],
        duplicates="drop"
    )

    return counts


def evaluate(model, year, item_groups):

    filename = (
        f"{model}-temporal-{year}"
        "-top10-recommendations.csv"
    )

    path = os.path.join(REC_DIR, filename)

    if not os.path.exists(path):
        print(f"Missing: {path}")
        return None

    recs = pd.read_csv(path)

    recs = recs.merge(
        item_groups[
            ["item_id", "item_group"]
        ],
        on="item_id",
        how="left"
    )

    recs["exposure"] = recs["rank"].apply(
        exposure_weight
    )

    # Exposure accumulated by recommended item
    exposure_by_item = (
        recs
        .groupby("item_id")["exposure"]
        .sum()
    )

    # Include zero-exposure catalogue items
    all_items = item_groups["item_id"].unique()

    exposure_by_item = exposure_by_item.reindex(
        all_items,
        fill_value=0.0
    )

    total_exposure = exposure_by_item.sum()

    # Core metrics
    g = gini(exposure_by_item.values)

    unique_items = recs["item_id"].nunique()

    coverage = (
        unique_items / len(all_items)
    )

    popular_items = set(
        item_groups.loc[
            item_groups["item_group"] == "popular",
            "item_id"
        ]
    )

    popular_catalogue_share = (
        len(popular_items) / len(all_items)
    )

    popular_exposure = exposure_by_item[
        exposure_by_item.index.isin(popular_items)
    ].sum()

    popular_exposure_share = (
        popular_exposure / total_exposure
    )

    amplification = (
        popular_exposure_share
        / popular_catalogue_share
    )

    return {
        "year": year,
        "model": model,
        "recommendation_rows": len(recs),
        "users": recs["user_id"].nunique(),
        "unique_recommended_items": unique_items,
        "gini": g,
        "catalogue_coverage": coverage,
        "popular_exposure_share": popular_exposure_share,
        "exposure_amplification": amplification,
    }


def main():

    print("=" * 70)
    print("EARLY TEMPORAL FAIRNESS EVALUATION: 1999-2001")
    print("=" * 70)

    item_groups = load_items()

    results = []

    for year in YEARS:

        for model in MODELS:

            print(
                f"Evaluating {model} - {year}..."
            )

            result = evaluate(
                model,
                year,
                item_groups
            )

            if result:
                results.append(result)

    df = pd.DataFrame(results)

    out_dir = os.path.join(
        ROOT,
        "outputs",
        "three_model_evaluation"
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    output = os.path.join(
        out_dir,
        "early_temporal_1999_2001.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print("\n" + "=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    print("\nSaved to:")
    print(output)


if __name__ == "__main__":
    main()