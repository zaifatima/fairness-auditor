"""
Preference Alignment module -- Auditor Engine component E.

Measures whether recommendations genuinely match what a user
demonstrably likes (their held-out ground-truth item), and whether that
alignment differs across demographic groups (gender, age). This is the
"true preference alignment" concept from Deldjoo & Di Noia's CFaiRLLM
(2025), adapted here as a fairness-audit metric feeding the Group
Outcome (A) layer, rather than replicating their neutral-vs-sensitive
prompt methodology directly.

Two per-group disparity metrics are reported, following the same
naming/spirit as CFaiRLLM's SNSR/SNSV (cite accordingly in the
write-up -- do not confuse with this project's own PRAG metric, which
is unrelated and comes from Desai et al.):
  - Range  : max(group score) - min(group score)
  - StdDev : standard deviation of group scores

Reads only from recommendations.db (the shared, model-agnostic
interface) plus the ground-truth candidate files and user profile
demographics -- no dependency on RecBole or the LLM-calling code.

Run from your project root:
    python preference_alignment.py
"""
import json
import sqlite3
import pandas as pd

DB_PATH = "recommendations.db"
CANDIDATES_PATH = "dataset/ml-1m-demographics/llm_full_candidates.json"
PROFILES_PATH = "dataset/ml-1m-demographics/user_profiles.csv"
MODEL_NAME = "gpt-4o-mini"
TOP_K = 10

# ---- Load ground truth and demographics ----
with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
    candidates = json.load(f)
true_items = {c["user_id"]: c["true_item"] for c in candidates}

profiles = pd.read_csv(PROFILES_PATH)[["user_id", "gender", "age_label"]]

# ---- Load recommendations for this model ----
conn = sqlite3.connect(DB_PATH)
recs = pd.read_sql_query(
    "SELECT user_id, item_id, rank FROM recommendations WHERE model = ?",
    conn, params=(MODEL_NAME,),
)
conn.close()

print(f"Loaded {recs['user_id'].nunique():,} users' recommendations for model={MODEL_NAME}")

# ---- Per-user preference alignment score ----
# Hit@K: 1 if the true item appears anywhere in the top-K, else 0.
# Reciprocal rank: 1/rank if hit, else 0 -- rewards hitting near rank 1
# more than hitting at rank 10, giving a finer-grained alignment signal
# than Hit@K alone.
scores = []
for uid, true_item in true_items.items():
    user_recs = recs[recs["user_id"] == uid]
    if user_recs.empty:
        continue  # user not yet processed by the LLM run
    hit_row = user_recs[user_recs["item_id"] == true_item]
    hit = 1 if not hit_row.empty else 0
    reciprocal_rank = (1.0 / hit_row["rank"].iloc[0]) if hit else 0.0
    scores.append({"user_id": uid, "hit": hit, "reciprocal_rank": reciprocal_rank})

scores_df = pd.DataFrame(scores)
merged = scores_df.merge(profiles, on="user_id", how="left")

print(f"\nScored {len(merged):,} users with a completed recommendation.")
print(f"Overall Hit@{TOP_K}: {merged['hit'].mean():.4f}")
print(f"Overall Mean Reciprocal Rank: {merged['reciprocal_rank'].mean():.4f}")

# ---- Group Outcome: preference alignment by demographic group ----
def report_group_disparity(df, group_col, metric_col):
    group_scores = df.groupby(group_col)[metric_col].mean()
    print(f"\n--- {metric_col} by {group_col} ---")
    print(group_scores.to_string())
    disparity_range = group_scores.max() - group_scores.min()
    disparity_std = group_scores.std()
    print(f"Range (max-min): {disparity_range:.4f}")
    print(f"StdDev across groups: {disparity_std:.4f}")
    return group_scores, disparity_range, disparity_std

for group_col in ["gender", "age_label"]:
    for metric_col in ["hit", "reciprocal_rank"]:
        report_group_disparity(merged, group_col, metric_col)
