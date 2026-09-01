"""
Calls GPT-4o-mini for each user in the pilot candidate set, asking it to
rank its top-10 picks from the 100 candidates (1 true held-out item +
99 negatives). Writes results into a SQLite recommendations table using
the same schema (model, run_timestamp, user_id, item_id, rank, score)
planned for every recommender in this project, so the LLM plugs into
the fairness engine identically to NCF/LightGCN/SASRec.

Reads the API key from openai_key.txt (not an env var, to sidestep
PowerShell quoting issues) -- this file must contain only the key,
nothing else, and must never be committed to git.

Run from your project root:
    python run_llm_pilot.py
"""
import json
import sqlite3
import time
from datetime import datetime, timezone

from openai import OpenAI

CANDIDATES_PATH = "dataset/ml-1m-demographics/llm_pilot_candidates.json"
DB_PATH = "recommendations.db"
MODEL_NAME = "gpt-4o-mini"
TOP_K = 10

with open("openai_key.txt", "r", encoding="utf-8") as f:
    api_key = f.read().strip()
client = OpenAI(api_key=api_key)

with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
    pilot_users = json.load(f)

conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TABLE IF NOT EXISTS recommendations (
        model TEXT,
        run_timestamp TEXT,
        user_id INTEGER,
        item_id INTEGER,
        rank INTEGER,
        score REAL
    )
""")
conn.commit()

run_timestamp = datetime.now(timezone.utc).isoformat()

SYSTEM_PROMPT = (
    "You are a movie recommendation engine. Given a user's taste profile "
    "and a numbered list of candidate movies, select and rank your top "
    f"{TOP_K} picks for that user, most recommended first. "
    "Respond ONLY with valid JSON in this exact format: "
    '{"ranked_candidate_numbers": [7, 42, 3, ...]} '
    "using the candidate numbers shown, not movie titles."
)

def build_user_prompt(user):
    candidate_lines = "\n".join(
        f"{i+1}. {title}" for i, title in enumerate(user["candidate_titles"])
    )
    return (
        f"{user['passion_profile']}\n"
        f"{user['consumption_profile']}\n\n"
        f"Candidate movies:\n{candidate_lines}\n\n"
        f"Select and rank your top {TOP_K} recommendations for this user "
        f"from the numbered list above."
    )

success_count = 0
fail_count = 0

for i, user in enumerate(pilot_users):
    print(f"[{i+1}/{len(pilot_users)}] User {user['user_id']} ...", flush=True)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(user)},
            ],
        )
        parsed = json.loads(response.choices[0].message.content)
        ranked_numbers = parsed["ranked_candidate_numbers"][:TOP_K]

        rows = []
        for rank, cand_num in enumerate(ranked_numbers, start=1):
            idx = cand_num - 1  # candidates were shown 1-indexed
            if 0 <= idx < len(user["candidates"]):
                item_id = user["candidates"][idx]
                score = 1.0 - (rank - 1) / TOP_K  # simple rank-based score
                rows.append((MODEL_NAME, run_timestamp, user["user_id"], item_id, rank, score))

        conn.executemany(
            "INSERT INTO recommendations (model, run_timestamp, user_id, item_id, rank, score) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        success_count += 1
        print(f"  -> wrote {len(rows)} recommendations", flush=True)

    except Exception as e:
        fail_count += 1
        print(f"  -> FAILED: {e}", flush=True)

    time.sleep(0.3)  # gentle pacing, well within rate limits

conn.close()
print(f"\nDone. {success_count} succeeded, {fail_count} failed out of {len(pilot_users)}.")
print(f"Results written to {DB_PATH}")
