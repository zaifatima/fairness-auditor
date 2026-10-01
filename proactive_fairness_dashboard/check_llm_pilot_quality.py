import json
import sqlite3

conn = sqlite3.connect("recommendations.db")
cur = conn.cursor()

with open("dataset/ml-1m-demographics/llm_pilot_candidates.json", "r", encoding="utf-8") as f:
    pilot_users = json.load(f)

true_items = {u["user_id"]: u["true_item"] for u in pilot_users}

hits = 0
for uid, true_item in true_items.items():
    cur.execute(
        "SELECT rank FROM recommendations WHERE user_id = ? AND item_id = ? ORDER BY rank LIMIT 1",
        (uid, true_item),
    )
    row = cur.fetchone()
    if row:
        hits += 1
        print(f"User {uid}: HIT at rank {row[0]}")
    else:
        print(f"User {uid}: not in top 10")

print(f"\nHit@10: {hits}/{len(true_items)} = {hits/len(true_items):.3f}")
conn.close()
