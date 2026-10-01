import sqlite3
conn = sqlite3.connect("recommendations.db")
cur = conn.cursor()
cur.execute("ALTER TABLE recommendations ADD COLUMN dataset TEXT")
cur.execute("UPDATE recommendations SET dataset = 'ml-1m' WHERE dataset IS NULL")
conn.commit()
cur.execute("SELECT model, dataset, COUNT(DISTINCT user_id) FROM recommendations GROUP BY model, dataset")
for row in cur.fetchall():
    print(row)
conn.close()
