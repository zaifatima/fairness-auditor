import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


ROOT = r"C:\mscproject_code"

MODEL_FILES = {
    "NeuMF": os.path.join(
        ROOT, "outputs", "NeuMF-ml-32m-50k-top10-recommendations.csv"
    ),
    "LightGCN": os.path.join(
        ROOT, "outputs", "LightGCN-ml-32m-50k-top10-recommendations.csv"
    ),
    "SASRec": os.path.join(
        ROOT, "outputs", "SASRec-ml-32m-50k-top10-recommendations.csv"
    ),
}

USER_GROUP_FILE = os.path.join(
    ROOT, "outputs", "user_groups.csv"
)

SOURCE_INTER = os.path.join(
    ROOT, "dataset", "ml-32m-50k", "ml-32m-50k.inter"
)

OUT_DIR = os.path.join(
    ROOT, "outputs", "three_model_evaluation"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_interactions():
    df = pd.read_csv(
        SOURCE_INTER,
        sep="\t",
        engine="python"
    )

    df.columns = [
        str(c).split(":")[0].strip()
        for c in df.columns
    ]

    return df


def load_user_groups():
    return pd.read_csv(USER_GROUP_FILE)


def load_recommendations():
    frames = []

    for model, path in MODEL_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        required = {
            "model",
            "user_id",
            "item_id",
            "rank",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{model} missing columns: {sorted(missing)}"
            )

        df["model"] = model
        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True
    )


# ---------------------------------------------------------
# ITEM POPULARITY GROUPS
# ---------------------------------------------------------

def build_item_groups(interactions):
    counts = (
        interactions
        .groupby("item_id")
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


# ---------------------------------------------------------
# GINI
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# EXPOSURE
# ---------------------------------------------------------

def exposure_weight(rank):
    return 1.0 / np.log2(rank + 1)


def model_exposure_metrics(recs, item_groups):
    merged = recs.merge(
        item_groups[["item_id", "item_group"]],
        on="item_id",
        how="left"
    )

    # Exposure per item, INCLUDING zero-exposure catalogue items
    item_exp = (
        merged
        .assign(
            exposure=merged["rank"].apply(exposure_weight)
        )
        .groupby("item_id")["exposure"]
        .sum()
    )

    all_items = item_groups["item_id"].unique()

    item_exp = item_exp.reindex(
        all_items,
        fill_value=0.0
    )

    total_exposure = item_exp.sum()

    g = gini(item_exp.values)

    unique_recommended = merged["item_id"].nunique()
    catalogue_size = len(all_items)

    coverage = (
        unique_recommended / catalogue_size
        if catalogue_size
        else np.nan
    )

    popular_mask = (
        item_groups["item_group"] == "popular"
    )

    popular_items = set(
        item_groups.loc[
            popular_mask, "item_id"
        ]
    )

    catalogue_share = (
        len(popular_items) / catalogue_size
    )

    popular_exposure = item_exp[
        item_exp.index.isin(popular_items)
    ].sum()

    exposure_share = (
        popular_exposure / total_exposure
        if total_exposure
        else np.nan
    )

    amplification = (
        exposure_share / catalogue_share
        if catalogue_share
        else np.nan
    )

    return {
        "gini": g,
        "catalogue_coverage": coverage,
        "catalogue_size": catalogue_size,
        "unique_recommended_items": unique_recommended,
        "popular_catalogue_share": catalogue_share,
        "popular_exposure_share": exposure_share,
        "exposure_amplification": amplification,
    }


# ---------------------------------------------------------
# USER GROUP DISPARITY
# ---------------------------------------------------------

def group_metrics(recs, user_groups):

    df = recs.merge(
        user_groups[[
            "user_id",
            "user_group"
        ]],
        on="user_id",
        how="left"
    )

    df["exposure"] = df["rank"].apply(
        exposure_weight
    )

    rows = []

    for group in ["low", "medium", "high"]:

        sub = df[df["user_group"] == group]

        if len(sub) == 0:
            continue

        rows.append({
            "user_group": group,
            "users": sub["user_id"].nunique(),
            "recommendation_rows": len(sub),
            "mean_rank": sub["rank"].mean(),
            "mean_exposure": sub["exposure"].mean(),
            "popular_share": (
                sub["item_group"] == "popular"
            ).mean()
            if "item_group" in sub.columns
            else np.nan,
        })

    result = pd.DataFrame(rows)

    return result


# ---------------------------------------------------------
# KS
# ---------------------------------------------------------

def ks_popular_vs_nonpopular(recs):

    popular_ranks = recs.loc[
        recs["item_group"] == "popular",
        "rank"
    ].astype(float)

    nonpopular_ranks = recs.loc[
        recs["item_group"] != "popular",
        "rank"
    ].astype(float)

    if len(popular_ranks) == 0 or len(nonpopular_ranks) == 0:
        return np.nan, np.nan

    result = ks_2samp(
        popular_ranks,
        nonpopular_ranks
    )

    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------
# rND / rKL / rRD
# Adapted to per-user Top-10 lists
# Popular vs non-popular
# ---------------------------------------------------------

def ranking_fairness_metrics(recs, popular_catalogue_share):

    results = []

    for user_id, user_df in recs.groupby("user_id"):

        user_df = user_df.sort_values("rank")

        n = len(user_df)

        if n == 0:
            continue

        group = (
            user_df["item_group"] == "popular"
        ).astype(int).values

        discounted_nd = 0.0
        discounted_kl = 0.0
        discounted_rd = 0.0

        for k in range(1, n + 1):

            top = group[:k]

            p_pop = top.mean()
            p_non = 1.0 - p_pop

            q_pop = popular_catalogue_share
            q_non = 1.0 - q_pop

            discount = 1.0 / np.log2(k + 1)

            # rND component
            discounted_nd += (
                discount *
                abs(p_pop - q_pop)
            )

            # KL component
            eps = 1e-12

            p1 = max(p_pop, eps)
            p2 = max(p_non, eps)

            q1 = max(q_pop, eps)
            q2 = max(q_non, eps)

            kl = (
                p1 * np.log(p1 / q1)
                +
                p2 * np.log(p2 / q2)
            )

            discounted_kl += (
                discount * kl
            )

            # rRD component
            if p_non == 0 or q_non == 0:
                rd_component = 0.0
            else:
                rd_component = abs(
                    (p_pop / p_non)
                    -
                    (q_pop / q_non)
                )

            discounted_rd += (
                discount * rd_component
            )

        results.append({
            "user_id": user_id,
            "rND_adapted": discounted_nd,
            "rKL_adapted": discounted_kl,
            "rRD_adapted": discounted_rd,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("THREE-MODEL FAIRNESS EVALUATION")
    print("=" * 70)

    interactions = load_interactions()
    user_groups = load_user_groups()
    recs = load_recommendations()

    item_groups = build_item_groups(interactions)

    recs = recs.merge(
        item_groups[[
            "item_id",
            "item_group"
        ]],
        on="item_id",
        how="left"
    )

    all_results = []

    for model in ["NeuMF", "LightGCN", "SASRec"]:

        print(f"\nEvaluating {model}...")

        model_recs = recs[
            recs["model"] == model
        ].copy()

        metrics = model_exposure_metrics(
            model_recs,
            item_groups
        )

        ks_stat, ks_p = ks_popular_vs_nonpopular(
            model_recs
        )

        ranking = ranking_fairness_metrics(
            model_recs,
            metrics["popular_catalogue_share"]
        )

        metrics["KS_statistic"] = ks_stat
        metrics["KS_pvalue"] = ks_p

        metrics["rND_adapted_mean"] = (
            ranking["rND_adapted"].mean()
        )

        metrics["rKL_adapted_mean"] = (
            ranking["rKL_adapted"].mean()
        )

        metrics["rRD_adapted_mean"] = (
            ranking["rRD_adapted"].mean()
        )

        metrics["model"] = model

        all_results.append(metrics)

        group_table = model_recs.merge(
            user_groups[[
                "user_id",
                "user_group"
            ]],
            on="user_id",
            how="left"
        )

        group_table["exposure"] = (
            group_table["rank"]
            .apply(exposure_weight)
        )

        group_summary = (
            group_table
            .groupby("user_group")
            .agg(
                users=("user_id", "nunique"),
                recommendation_rows=("item_id", "size"),
                mean_rank=("rank", "mean"),
                mean_exposure=("exposure", "mean"),
                popular_share=(
                    "item_group",
                    lambda x: (x == "popular").mean()
                )
            )
            .reset_index()
        )

        group_summary.to_csv(
            os.path.join(
                OUT_DIR,
                f"{model}-group-metrics.csv"
            ),
            index=False
        )

    summary = pd.DataFrame(all_results)[[
        "model",
        "gini",
        "catalogue_coverage",
        "unique_recommended_items",
        "popular_catalogue_share",
        "popular_exposure_share",
        "exposure_amplification",
        "KS_statistic",
        "KS_pvalue",
        "rND_adapted_mean",
        "rKL_adapted_mean",
        "rRD_adapted_mean",
    ]]

    summary.to_csv(
        os.path.join(
            OUT_DIR,
            "three-model-fairness-summary.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print(summary.to_string(index=False))
    print("=" * 70)

    print(
        "\nSaved to:",
        OUT_DIR
    )


if __name__ == "__main__":
    main()