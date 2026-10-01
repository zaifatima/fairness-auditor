"""
Full-scale LLM pilot runner. Resumable: on each run, skips any user
who already has recommendations written for this model in the
database, so an interrupted run (sleep, closed window, network drop)
never needs to restart from zero. Retries transient API errors with
backoff instead of aborting the whole run on one failure.

Run from your project root -- safe to re-run repeatedly, it will just
pick up where it left off:
    python run_llm_full.py
"""
import json
import sqlite3
import time
from datetime import datetime, timezone

from openai import OpenAI

CANDIDATES_PATH = "dataset/ml-1m-demographics/llm_full_candidates.json"
DB_PATH = "recommendations.db"
MODEL_NAME = "gpt-4o-mini"
TOP_K = 10
MAX_RETRIES = 3

with open("openai_key.txt", "r", encoding="utf-8") as f:
    api_key = f.read().strip()
client = OpenAI(api_key=api_key)

with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
    all_users = json.load(f)

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

cur = conn.cursor()
cur.execute("SELECT DISTINCT user_id FROM recommendations WHERE model = ?", (MODEL_NAME,))
already_done = {row[0] for row in cur.fetchall()}
remaining_users = [u for u in all_users if u["user_id"] not in already_done]

print(f"Total users: {len(all_users)}, already done: {len(already_done)}, remaining: {len(remaining_users)}")

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
start_time = time.time()

for i, user in enumerate(remaining_users):
    attempt = 0
    while attempt < MAX_RETRIES:
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
                idx = cand_num - 1
                if 0 <= idx < len(user["candidates"]):
                    item_id = user["candidates"][idx]
                    score = 1.0 - (rank - 1) / TOP_K
                    rows.append((MODEL_NAME, run_timestamp, user["user_id"], item_id, rank, score))

            conn.executemany(
                "INSERT INTO recommendations (model, run_timestamp, user_id, item_id, rank, score) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            success_count += 1
            break

        except Exception as e:
            attempt += 1
            wait = 2 ** attempt
            print(f"  User {user['user_id']} attempt {attempt} failed: {e} -- retrying in {wait}s", flush=True)
            time.sleep(wait)
    else:
        fail_count += 1
        print(f"  User {user['user_id']}: gave up after {MAX_RETRIES} attempts", flush=True)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        remaining_est = (len(remaining_users) - i - 1) / rate / 60
        print(f"[{i+1}/{len(remaining_users)}] {success_count} ok, {fail_count} failed. "
              f"~{remaining_est:.1f} min remaining.", flush=True)

    time.sleep(0.3)

conn.close()
print(f"\nDone this run. {success_count} succeeded, {fail_count} failed out of {len(remaining_users)} attempted.")
