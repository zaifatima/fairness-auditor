import os
import pandas as pd

SRC_DIR = os.path.join("dataset", "ml-32m")
OUT_DIR = os.path.join("dataset", "ml-32m-smoke")
os.makedirs(OUT_DIR, exist_ok=True)

N_USERS = 3000
SEED = 2026

print("Reading full ml-32m.inter ...")
inter = pd.read_csv(os.path.join(SRC_DIR, "ml-32m.inter"), sep="\t")

all_users = inter["user_id:token"].unique()
sampled_users = set(pd.Series(all_users).sample(n=N_USERS, random_state=SEED).values)

subset = inter[inter["user_id:token"].isin(sampled_users)]
print(f"Sampled {len(sampled_users)} users -> {len(subset):,} interactions")

subset.to_csv(os.path.join(OUT_DIR, "ml-32m-smoke.inter"), sep="\t", index=False)

item_src = os.path.join(SRC_DIR, "ml-32m.item")
item_dst = os.path.join(OUT_DIR, "ml-32m-smoke.item")
with open(item_src, "r", encoding="utf-8") as f_in, open(item_dst, "w", encoding="utf-8") as f_out:
    f_out.write(f_in.read())

print("Smoke-test dataset written to", OUT_DIR)
