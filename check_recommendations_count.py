import sqlite3
conn = sqlite3.connect("recommendations.db")
cur = conn.cursor()
cur.execute("SELECT model, COUNT(DISTINCT user_id) FROM recommendations GROUP BY model")
for row in cur.fetchall():
    print(row)
conn.close()
