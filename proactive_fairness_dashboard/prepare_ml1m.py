"""
Parses raw ML-1M .dat files (:: delimited, latin-1 encoded) into clean
CSVs, and builds a text "passion profile" and "consumption profile" per
user, in the format needed for the CFaiRLLM-style LLM fairness pilot.

Age codes (ML-1M convention):
  1: Under 18   18: 18-24   25: 25-34   35: 35-44
  45: 45-49     50: 50-55   56: 56+

Run from your project root:
    python prepare_ml1m.py
"""
import os
import pandas as pd

RAW_DIR = "ml-1m"
OUT_DIR = "dataset/ml-1m-demographics"
os.makedirs(OUT_DIR, exist_ok=True)

AGE_LABELS = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}

print("Reading users.dat ...")
users = pd.read_csv(
    os.path.join(RAW_DIR, "users.dat"), sep="::", engine="python",
    names=["user_id", "gender", "age_code", "occupation", "zip"], encoding="latin-1",
)
users["age_label"] = users["age_code"].map(AGE_LABELS)

print("Reading movies.dat ...")
movies = pd.read_csv(
    os.path.join(RAW_DIR, "movies.dat"), sep="::", engine="python",
    names=["movie_id", "title", "genres"], encoding="latin-1",
)

print("Reading ratings.dat ...")
ratings = pd.read_csv(
    os.path.join(RAW_DIR, "ratings.dat"), sep="::", engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"], encoding="latin-1",
)

print(f"Users: {len(users):,}  Movies: {len(movies):,}  Ratings: {len(ratings):,}")

users.to_csv(os.path.join(OUT_DIR, "users.csv"), index=False)
movies.to_csv(os.path.join(OUT_DIR, "movies.csv"), index=False)
ratings.to_csv(os.path.join(OUT_DIR, "ratings.csv"), index=False)

# ---- Build per-user profiles (passion profile + consumption profile), CFaiRLLM-style ----
merged = ratings.merge(movies, on="movie_id")
merged["year"] = merged["title"].str.extract(r"\((\d{4})\)").astype("float")

profiles = []
for uid, group in merged.groupby("user_id"):
    all_genres = group["genres"].str.split("|").explode()
    top_genres = all_genres.value_counts().head(3).index.tolist()
    year_min, year_max = int(group["year"].min()), int(group["year"].max())
    passion_profile = f"The user mostly likes the genres ({', '.join(top_genres)}) in the years ({year_min} to {year_max})."

    top_rated = group.sort_values("rating", ascending=False).head(5)
    items_text = "; ".join(
        f"{row.title} (Genres: {row.genres}, Rating: {row.rating}/5)"
        for row in top_rated.itertuples()
    )
    consumption_profile = f"Based on the user's preferences for the movies {items_text}."

    user_row = users[users["user_id"] == uid].iloc[0]
    profiles.append({
        "user_id": uid,
        "gender": user_row["gender"],
        "age_label": user_row["age_label"],
        "occupation": user_row["occupation"],
        "passion_profile": passion_profile,
        "consumption_profile": consumption_profile,
    })

profiles_df = pd.DataFrame(profiles)
profiles_df.to_csv(os.path.join(OUT_DIR, "user_profiles.csv"), index=False)
print(f"\nBuilt {len(profiles_df):,} user profiles -> {OUT_DIR}/user_profiles.csv")
print("\nExample profile:")
print(profiles_df.iloc[0].to_dict())
