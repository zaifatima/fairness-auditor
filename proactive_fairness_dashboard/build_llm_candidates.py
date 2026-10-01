"""
Builds a uni100-style test candidate set per user: their most recent
rated movie (ground truth) plus 99 random negatives they have not rated,
shuffled together. This matches the same evaluation protocol used for
NCF/LightGCN/SASRec, so the LLM's results are directly comparable.

Run from your project root (with prepare_ml1m.py already run):
    python build_llm_candidates.py
"""
import json
import random
import pandas as pd

random.seed(2026)

OUT_DIR = "dataset/ml-1m-demographics"
N_NEGATIVES = 99
PILOT_SIZE = 30  # start small to validate before scaling

ratings = pd.read_csv(f"{OUT_DIR}/ratings.csv")
movies = pd.read_csv(f"{OUT_DIR}/movies.csv")
profiles = pd.read_csv(f"{OUT_DIR}/user_profiles.csv")

movie_lookup = movies.set_index("movie_id")[["title", "genres"]].to_dict("index")
all_movie_ids = set(movies["movie_id"].tolist())

# ground-truth test item = each user's most recent rating (by timestamp)
ratings_sorted = ratings.sort_values("timestamp")
test_items = ratings_sorted.groupby("user_id").tail(1).set_index("user_id")["movie_id"].to_dict()
rated_by_user = ratings.groupby("user_id")["movie_id"].apply(set).to_dict()

pilot_user_ids = profiles["user_id"].sample(n=PILOT_SIZE, random_state=2026).tolist()

records = []
for uid in pilot_user_ids:
    if uid not in test_items:
        continue
    true_item = test_items[uid]
    already_rated = rated_by_user.get(uid, set())
    candidate_pool = list(all_movie_ids - already_rated - {true_item})
    negatives = random.sample(candidate_pool, N_NEGATIVES)

    candidates = negatives + [true_item]
    random.shuffle(candidates)

    profile_row = profiles[profiles["user_id"] == uid].iloc[0]

    records.append({
        "user_id": int(uid),
        "true_item": int(true_item),
        "candidates": [int(c) for c in candidates],
        "candidate_titles": [movie_lookup[c]["title"] for c in candidates],
        "passion_profile": profile_row["passion_profile"],
        "consumption_profile": profile_row["consumption_profile"],
    })

with open(f"{OUT_DIR}/llm_pilot_candidates.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print(f"Built candidate sets for {len(records)} users -> {OUT_DIR}/llm_pilot_candidates.json")
print("\nExample record (first user):")
print(f"  User {records[0]['user_id']}, true item: {movie_lookup[records[0]['true_item']]['title']}")
print(f"  {len(records[0]['candidates'])} total candidates")
