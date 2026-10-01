import argparse
import os

import pandas as pd


def load_test_interactions(path):
    test = pd.read_csv(path)

    required = {"user_id", "item_id"}

    missing = required - set(test.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return test


def load_recommendations(path):
    recs = pd.read_csv(path)

    required = {
        "user_id",
        "item_id",
        "rank",
    }

    missing = required - set(recs.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return recs


def evaluate_model(
    recommendations,
    test,
    groups,
    top_k=10,
):
    test_items = (
        test
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    results = []

    for user_id, true_items in test_items.items():

        user_recs = recommendations[
            recommendations["user_id"] == user_id
        ].copy()

        user_recs = user_recs[
            user_recs["rank"] <= top_k
        ].sort_values("rank")

        recommended_items = set(
            user_recs["item_id"]
        )

        hits = true_items.intersection(
            recommended_items
        )

        hit = 1 if len(hits) > 0 else 0

        recall = (
            len(hits) / len(true_items)
            if len(true_items) > 0
            else 0.0
        )

        reciprocal_rank = 0.0

        if len(hits) > 0:

            hit_rows = user_recs[
                user_recs["item_id"].isin(hits)
            ]

            first_rank = hit_rows["rank"].min()

            reciprocal_rank = (
                1.0 / float(first_rank)
            )

        results.append(
            {
                "user_id": int(user_id),
                "num_test_items": len(true_items),
                "hits": len(hits),
                "hit": hit,
                "recall": recall,
                "reciprocal_rank": reciprocal_rank,
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.merge(
        groups[
            [
                "user_id",
                "interaction_count",
                "user_group",
            ]
        ],
        on="user_id",
        how="left",
    )

    return results_df


def print_results(
    results,
    model_name,
    top_k,
):
    print()
    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    print(
        f"Users evaluated: "
        f"{len(results):,}"
    )

    print(
        f"Overall Hit@{top_k}: "
        f"{results['hit'].mean():.6f}"
    )

    print(
        f"Overall Recall@{top_k}: "
        f"{results['recall'].mean():.6f}"
    )

    print(
        f"Overall MRR: "
        f"{results['reciprocal_rank'].mean():.6f}"
    )

    print()
    print("--- Results by activity group ---")

    grouped = (
        results
        .groupby("user_group", observed=True)
        .agg(
            users=("user_id", "count"),
            hit_at_10=("hit", "mean"),
            recall_at_10=("recall", "mean"),
            mrr=("reciprocal_rank", "mean"),
            mean_test_items=("num_test_items", "mean"),
            mean_interactions=("interaction_count", "mean"),
        )
        .reindex(
            ["low", "medium", "high"]
        )
    )

    print(
        grouped.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print()
    print("--- Group disparities ---")

    for metric in [
        "hit_at_10",
        "recall_at_10",
        "mrr",
    ]:

        value_range = (
            grouped[metric].max()
            - grouped[metric].min()
        )

        std = grouped[metric].std()

        print(
            f"{metric}: "
            f"range={value_range:.6f}, "
            f"std={std:.6f}"
        )

    return grouped


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--recommendations",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--test-interactions",
        required=True,
    )

    parser.add_argument(
        "--user-groups",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/preference_alignment",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    print("=== ML-32M PREFERENCE ALIGNMENT ===")

    print("Loading held-out test interactions...")

    test = load_test_interactions(
        args.test_interactions
    )

    print(
        f"Held-out rows: "
        f"{len(test):,}"
    )

    print(
        f"Held-out users: "
        f"{test['user_id'].nunique():,}"
    )

    print("Loading user groups...")

    groups = pd.read_csv(
        args.user_groups
    )

    print(
        groups["user_group"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    all_summaries = []

    for recommendation_path in args.recommendations:

        print()
        print(
            f"Loading: "
            f"{recommendation_path}"
        )

        recommendations = load_recommendations(
            recommendation_path
        )

        if "model" in recommendations.columns:
            model_name = str(
                recommendations["model"].iloc[0]
            )
        else:
            model_name = os.path.basename(
                recommendation_path
            )

        results = evaluate_model(
            recommendations,
            test,
            groups,
            top_k=args.top_k,
        )

        grouped = print_results(
            results,
            model_name,
            args.top_k,
        )

        results_path = os.path.join(
            args.output_dir,
            f"{model_name}-per-user.csv",
        )

        results.to_csv(
            results_path,
            index=False,
        )

        summary = {
            "model": model_name,
            "users": len(results),
            "hit_at_10": results["hit"].mean(),
            "recall_at_10": results["recall"].mean(),
            "mrr": results["reciprocal_rank"].mean(),
            "hit_disparity": (
                grouped["hit_at_10"].max()
                - grouped["hit_at_10"].min()
            ),
            "recall_disparity": (
                grouped["recall_at_10"].max()
                - grouped["recall_at_10"].min()
            ),
            "mrr_disparity": (
                grouped["mrr"].max()
                - grouped["mrr"].min()
            ),
        }

        all_summaries.append(summary)

    summary_df = pd.DataFrame(
        all_summaries
    )

    summary_path = os.path.join(
        args.output_dir,
        "model_comparison.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(
        f"Saved comparison: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()
