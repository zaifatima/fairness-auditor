"""
Converts the raw Last.fm Dataset - 1K users TSV into RecBole atomic file format.

Expected input (after extracting lastfm-dataset-1K.tar / .tar.gz):
    raw_data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv

Output (RecBole atomic files):
    dataset/lastfm-1k/lastfm-1k.inter
    dataset/lastfm-1k/lastfm-1k.item

Design choice: items are ARTISTS, not individual tracks. The raw file has
~961K unique tracks vs ~177K unique artists — artist-level items keep the
item space comparable in scale to MovieLens-32M's ~87K movies, and matches
how this dataset is used in most published recommender work (e.g. the
Latent Collaborative Retrieval paper). Track MBIDs are also missing for a
large share of rows, whereas artist name is almost always present.

The raw file is known to have some malformed rows (bad quoting, stray
tabs) — on_bad_lines="skip" drops those rather than crashing.

Run from your project root with the venv activated:
    python convert_lastfm.py
"""

import os
import pandas as pd

RAW_DIR = os.path.join("raw_data", "lastfm-dataset-1K")
OUT_DIR = os.path.join("dataset", "lastfm-1k")
os.makedirs(OUT_DIR, exist_ok=True)

RAW_FILE = "userid-timestamp-artid-artname-traid-traname.tsv"
COLS = ["user_id", "timestamp", "artist_mbid", "artist_name", "track_mbid", "track_name"]


def convert_interactions():
    print("Reading interactions file (~19M rows, this will take a few minutes) ...")
    df = pd.read_csv(
        os.path.join(RAW_DIR, RAW_FILE),
        sep="\t",
        header=None,
        names=COLS,
        usecols=["user_id", "timestamp", "artist_name"],
        quoting=3,  # QUOTE_NONE — this file has unbalanced quote characters
        on_bad_lines="skip",
    )
    print(f"Loaded {len(df):,} raw rows")

    df = df.dropna(subset=["artist_name", "timestamp", "user_id"])

    print("Parsing timestamps ...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].astype("int64") // 10**9  # -> unix seconds

    print("Building artist -> item_id mapping ...")
    artist_cat = df["artist_name"].astype("category")
    df["item_id"] = artist_cat.cat.codes + 1  # RecBole token ids start at 1, not 0
    artist_lookup = dict(enumerate(artist_cat.cat.categories, start=1))

    inter = df.rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "timestamp": "timestamp:float",
        }
    )
    inter = inter[["user_id:token", "item_id:token", "timestamp:float"]]

    out_path = os.path.join(OUT_DIR, "lastfm-1k.inter")
    print(f"Writing {len(inter):,} interactions to {out_path} ...")
    inter.to_csv(out_path, sep="\t", index=False)
    print("Done with .inter file.")

    return artist_lookup


def convert_items(artist_lookup):
    items = pd.DataFrame(
        {
            "item_id:token": list(artist_lookup.keys()),
            "artist_name:token": list(artist_lookup.values()),
        }
    )
    out_path = os.path.join(OUT_DIR, "lastfm-1k.item")
    print(f"Writing {len(items):,} items to {out_path} ...")
    items.to_csv(out_path, sep="\t", index=False)
    print("Done with .item file.")


if __name__ == "__main__":
    lookup = convert_interactions()
    convert_items(lookup)
    print("\nAll done. Atomic files are in:", OUT_DIR)
