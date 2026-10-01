"""
Runs GPT-4o-mini as a fifth recommender, audited against the exact
same 100 users, ground truth, and catalogue as NCF/LightGCN/SASRec/AR-LM
-- so it plugs directly into preference_alignment.py and the exposure
scripts unchanged.

Design constraint (state explicitly in the dissertation): GPT cannot
score all ~58,195 catalogue items per user the way the trained models
do. This uses a candidate-constrained protocol instead: each user's
true held-out items plus random negatives, totalling 100 candidates,
with GPT asked to rank its top-10 from that list. This is a real
methodological difference from the other models' full-catalogue
scoring, not a hidden shortcut.

Resumable (skips users already written) and retries transient API
errors with backoff, same pattern already proven for the ML-1M pilot.

Run from your project root:
    python run_gpt_audit.py
"""
import json
import random
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd
from openai import OpenAI

random.seed(2026)

INTER_PATH = "dataset/ml-32m-50k/ml-32m-50k.inter"
ITEM_PATH = "dataset/ml-32m-50k/ml-32m-50k.item"
TEST_INTERACTIONS_PATH = "outputs/audited_test_interactions.csv"
USER_GROUPS_PATH = "outputs/user_groups.csv"
OUTPUT_CSV = "outputs/GPT4o-mini-ml-32m-50k-top10-recommendations.csv"

MODEL_NAME = "GPT4o-mini"
N_CANDIDATES_TOTAL = 100
TOP_K = 10
MAX_RETRIES = 3

with open("openai_key.txt", "r", encoding="utf-8") as f:
    api_key = f.read().strip()
client = OpenAI(api_key=api_key)

print("Loading data ...", flush=True)
inter = pd.read_csv(INTER_PATH, sep="\t")
inter.columns = [c.split(":")[0] for c in inter.columns]
items = pd.read_csv(ITEM_PATH, sep="\t")
items.columns = [c.split(":")[0] for c in items.columns]
item_lookup = items.set_index("item_id")[["movie_title", "genre"]].to_dict("index")

test_inter = pd.read_csv(TEST_INTERACTIONS_PATH)
user_groups = pd.read_csv(USER_GROUPS_PATH)

audited_users = sorted(test_inter["user_id"].unique().tolist())
print(f"Audited users: {len(audited_users)}", flush=True)

all_item_ids = set(inter["item_id"].unique())

# Historical interactions EXCLUDE anything in the held-out test set,
# to avoid leaking the ground truth into the profile shown to GPT.
test_pairs = set(zip(test_inter["user_id"], test_inter["item_id"]))
history_only = inter[~inter.apply(lambda r: (r["user_id"], r["item_id"]) in test_pairs, axis=1)]

def build_profile(uid):
    hist = history_only[history_only["user_id"] == uid].sort_values("timestamp")
    if hist.empty:
        return "The user has limited prior rating history available."
    genres = []
    for iid in hist["item_id"]:
        g = item_lookup.get(iid, {}).get("genre", "")
        if isinstance(g, str):
            genres.extend(g.split())
    top_genres = pd.Series(genres).value_counts().head(3).index.tolist() if genres else []
    top_rated = hist.sort_values("rating", ascending=False).head(5) if "rating" in hist.columns else hist.head(5)
    items_text = "; ".join(
        f"{item_lookup.get(r.item_id, {}).get('movie_title', 'Unknown')} "
        f"(Genres: {item_lookup.get(r.item_id, {}).get('genre', '')})"
        for r in top_rated.itertuples()
    )
    genre_text = ", ".join(top_genres) if top_genres else "varied genres"
    return f"The user mostly enjoys {genre_text}. Notable past preferences: {items_text}."

def build_candidates(uid):
    true_items = test_inter[test_inter["user_id"] == uid]["item_id"].tolist()
    rated = set(history_only[history_only["user_id"] == uid]["item_id"])
    pool = list(all_item_ids - rated - set(true_items))
    n_negatives = max(0, N_CANDIDATES_TOTAL - len(true_items))
    negatives = random.sample(pool, min(n_negatives, len(pool)))
    candidates = true_items + negatives
    random.shuffle(candidates)
    return candidates, set(true_items)

SYSTEM_PROMPT = (
    "You are a movie recommendation engine. Given a user's taste profile "
    "and a numbered list of candidate movies, select and rank your top "
    f"{TOP_K} picks for that user, most recommended first. "
    "Respond ONLY with valid JSON in this exact format: "
    '{"ranked_candidate_numbers": [7, 42, 3, ...]} '
    "using the candidate numbers shown, not movie titles."
)

def build_prompt(profile_text, candidates):
    lines = "\n".join(
        f"{i+1}. {item_lookup.get(c, {}).get('movie_title', f'Item {c}')}"
        for i, c in enumerate(candidates)
    )
    return f"{profile_text}\n\nCandidate movies:\n{lines}\n\nSelect and rank your top {TOP_K} from the numbered list above."

conn = sqlite3.connect("recommendations.db")
existing_users = set()
try:
    existing_df = pd.read_csv(OUTPUT_CSV)
    existing_users = set(existing_df["user_id"].unique())
    print(f"Resuming: {len(existing_users)} users already done", flush=True)
except FileNotFoundError:
    existing_df = pd.DataFrame()

remaining = [u for u in audited_users if u not in existing_users]
print(f"Remaining: {len(remaining)}", flush=True)

run_ts = datetime.now(timezone.utc).isoformat()
new_rows = []

for i, uid in enumerate(remaining):
    profile_text = build_profile(uid)
    candidates, true_items = build_candidates(uid)

    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(profile_text, candidates)},
                ],
            )
            parsed = json.loads(response.choices[0].message.content)
            ranked_numbers = parsed["ranked_candidate_numbers"][:TOP_K]

            for rank, cand_num in enumerate(ranked_numbers, start=1):
                idx = cand_num - 1
                if 0 <= idx < len(candidates):
                    item_id = candidates[idx]
                    score = 1.0 - (rank - 1) / TOP_K
                    new_rows.append({
                        "model": MODEL_NAME, "timestamp": run_ts,
                        "user_id": uid, "item_id": item_id,
                        "rank": rank, "score": score,
                    })
            break
        except Exception as e:
            attempt += 1
            wait = 2 ** attempt
            print(f"  User {uid} attempt {attempt} failed: {e} -- retrying in {wait}s", flush=True)
            time.sleep(wait)
    else:
        print(f"  User {uid}: gave up after {MAX_RETRIES} attempts", flush=True)

    if (i + 1) % 10 == 0:
        combined = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
        combined.to_csv(OUTPUT_CSV, index=False)
        print(f"[{i+1}/{len(remaining)}] saved progress ({len(new_rows)} new rows so far)", flush=True)

    time.sleep(0.3)

final = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
final.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone. Total rows: {len(final)}, users: {final['user_id'].nunique()}")
print(f"Output: {OUTPUT_CSV}")
