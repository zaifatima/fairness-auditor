"""
Converts raw MovieLens-32M CSVs into RecBole atomic file format.

Expected input (after extracting ml-32m.zip):
    raw_data/ml-32m/ratings.csv
    raw_data/ml-32m/movies.csv

Output (RecBole atomic files):
    dataset/ml-32m/ml-32m.inter
    dataset/ml-32m/ml-32m.item

Run from your project root (fairness-auditor/) with the venv activated:
    python convert_movielens.py
"""

import os
import pandas as pd

RAW_DIR = os.path.join("raw_data", "ml-32m")
OUT_DIR = os.path.join("dataset", "ml-32m")
os.makedirs(OUT_DIR, exist_ok=True)


def convert_interactions():
    print("Reading ratings.csv ...")
    ratings = pd.read_csv(
        os.path.join(RAW_DIR, "ratings.csv"),
        dtype={"userId": "int64", "movieId": "int64", "rating": "float32", "timestamp": "int64"},
    )

    # RecBole atomic file column naming convention: field:type
    ratings = ratings.rename(
        columns={
            "userId": "user_id:token",
            "movieId": "item_id:token",
            "rating": "rating:float",
            "timestamp": "timestamp:float",
        }
    )
    ratings = ratings[["user_id:token", "item_id:token", "rating:float", "timestamp:float"]]

    out_path = os.path.join(OUT_DIR, "ml-32m.inter")
    print(f"Writing {len(ratings):,} interactions to {out_path} ...")
    ratings.to_csv(out_path, sep="\t", index=False)
    print("Done with .inter file.")


def convert_items():
    print("Reading movies.csv ...")
    movies = pd.read_csv(os.path.join(RAW_DIR, "movies.csv"))

    # genres in raw file are pipe-separated, e.g. "Action|Adventure|Sci-Fi"
    # RecBole token_seq fields are space-separated
    movies["genres"] = movies["genres"].apply(
        lambda g: " ".join(g.split("|")) if isinstance(g, str) else ""
    )

    movies = movies.rename(
        columns={
            "movieId": "item_id:token",
            "title": "movie_title:token_seq",
            "genres": "genre:token_seq",
        }
    )
    movies = movies[["item_id:token", "movie_title:token_seq", "genre:token_seq"]]

    out_path = os.path.join(OUT_DIR, "ml-32m.item")
    print(f"Writing {len(movies):,} items to {out_path} ...")
    movies.to_csv(out_path, sep="\t", index=False)
    print("Done with .item file.")


if __name__ == "__main__":
    convert_interactions()
    convert_items()
    print("\nAll done. Atomic files are in:", OUT_DIR)
    print("Quick check — run these to confirm:")
    print(f"  head -5 {os.path.join(OUT_DIR, 'ml-32m.inter')}")
    print(f"  head -5 {os.path.join(OUT_DIR, 'ml-32m.item')}")
