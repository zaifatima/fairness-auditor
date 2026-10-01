import pandas as pd
import numpy as np
from pathlib import Path

RECOMMENDATIONS = Path(
    "outputs/temporal_recommendations/NeuMF-temporal-common-years.csv"
)

INTERACTIONS = Path(
    "outputs/ml32m_interactions_with_year.csv"
)

OUTPUT = Path(
    "outputs/temporal_neumf/temporal_fairness_trend.csv"
)

START_YEAR = 2003
END_YEAR = 2023
STEP = 2
HISTORICAL_START_YEAR = 1996
TOP_K = 10
POPULAR_SHARE = 0.20


def gini(values):
    x = np.asarray(values, dtype=float)

    if x.size == 0:
        return 0.0

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
    return 1.0 / np.log2(rank.astype(float) + 1.0)


print("Loading recommendations...")
recs = pd.read_csv(RECOMMENDATIONS)

print("Loading interactions...")
interactions = pd.read_csv(INTERACTIONS)

required_recs = {"year", "user_id", "item_id", "rank"}
required_int = {"year", "user_id", "item_id"}

missing_recs = required_recs - set(recs.columns)
missing_int = required_int - set(interactions.columns)

if missing_recs:
    raise ValueError(
        f"Recommendation file missing columns: {sorted(missing_recs)}"
    )

if missing_int:
    raise ValueError(
        f"Interaction file missing columns: {sorted(missing_int)}"
    )

recs["year"] = pd.to_numeric(recs["year"])
recs["item_id"] = pd.to_numeric(recs["item_id"])
recs["rank"] = pd.to_numeric(recs["rank"])

interactions["year"] = pd.to_numeric(interactions["year"])
interactions["item_id"] = pd.to_numeric(interactions["item_id"])

years = list(
    range(
        START_YEAR,
        END_YEAR + 1,
        STEP
    )
)

results = []

for year in years:

    print()
    print("=" * 70)
    print(f"NeuMF temporal fairness: {year}")
    print("=" * 70)

    historical = interactions[
        (interactions["year"] >= HISTORICAL_START_YEAR)
        & (interactions["year"] < year)
    ].copy()

    year_recs = recs[
        (recs["year"] == year)
        & (recs["rank"] <= TOP_K)
    ].copy()

    if historical.empty:
        print("No historical interactions. Skipping.")
        continue

    if year_recs.empty:
        print("No recommendations. Skipping.")
        continue

    historical_catalogue = (
        historical["item_id"]
        .drop_duplicates()
        .tolist()
    )

    catalogue_size = len(historical_catalogue)

    item_popularity = (
        historical
        .groupby("item_id")
        .size()
        .sort_values(ascending=False)
    )

    n_popular = max(
        1,
        int(
            len(item_popularity)
            * POPULAR_SHARE
        )
    )

    popular_items = set(
        item_popularity
        .head(n_popular)
        .index
    )

    year_recs["exposure"] = (
        exposure_weight(
            year_recs["rank"]
        )
    )

    item_exposure = (
        year_recs
        .groupby("item_id")["exposure"]
        .sum()
    )

    exposure_vector = np.array(
        [
            float(
                item_exposure.get(
                    item_id,
                    0.0
                )
            )
            for item_id in historical_catalogue
        ]
    )

    gini_exposure = gini(
        exposure_vector
    )

    unique_items = (
        year_recs["item_id"]
        .nunique()
    )

    catalogue_coverage = (
        unique_items / catalogue_size
        if catalogue_size
        else 0.0
    )

    total_exposure = float(
        item_exposure.sum()
    )

    popular_exposure = float(
        sum(
            item_exposure.get(
                item_id,
                0.0
            )
            for item_id in popular_items
        )
    )

    popular_catalogue_share = (
        len(popular_items)
        / len(item_popularity)
        if len(item_popularity)
        else 0.0
    )

    popular_exposure_share = (
        popular_exposure
        / total_exposure
        if total_exposure
        else 0.0
    )

    exposure_amplification = (
        popular_exposure_share
        / popular_catalogue_share
        if popular_catalogue_share
        else 0.0
    )

    results.append(
        {
            "year": year,
            "historical_start_year":
                HISTORICAL_START_YEAR,
            "historical_end_year":
                year - 1,
            "historical_rows":
                len(historical),
            "historical_catalogue_items":
                catalogue_size,
            "recommendation_rows":
                len(year_recs),
            "unique_items_recommended":
                unique_items,
            "catalogue_coverage":
                catalogue_coverage,
            "gini_exposure":
                gini_exposure,
            "popular_item_catalogue_share":
                popular_catalogue_share,
            "popular_item_exposure_share":
                popular_exposure_share,
            "exposure_amplification":
                exposure_amplification,
        }
    )

    print(
        f"Historical rows: {len(historical):,}"
    )
    print(
        f"Historical catalogue: {catalogue_size:,}"
    )
    print(
        f"Recommendations: {len(year_recs):,}"
    )
    print(
        f"Unique recommended: {unique_items:,}"
    )
    print(
        f"Gini exposure: {gini_exposure:.6f}"
    )
    print(
        f"Catalogue coverage: "
        f"{catalogue_coverage:.4%}"
    )
    print(
        f"Popular exposure share: "
        f"{popular_exposure_share:.4%}"
    )
    print(
        f"Exposure amplification: "
        f"{exposure_amplification:.4f}"
    )


output_df = pd.DataFrame(results)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

output_df.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 70)
print("NeuMF temporal fairness monitoring COMPLETE")
print("=" * 70)
print()
print(output_df.to_string(index=False))
print()
print(f"Saved to: {OUTPUT.resolve()}")
