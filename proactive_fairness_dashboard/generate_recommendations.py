"""
Generates top-K recommendations for a trained model and writes them into
the shared recommendations table, using the SAME candidate-set protocol
(1 held-out true item + 99 random negatives) as the LLM pilot.

Adds the scipy dok_matrix._update compatibility patch (same fix used
for the local LightGCN evaluation) -- this local machine''s scipy
version removed an internal method name LightGCN''s graph construction
still calls by its old name.

Run from your project root:
    python generate_recommendations.py --model LightGCN --dataset ml-32m-50k --checkpoint saved/LightGCN-ml-32m-50k-latest.pth
"""
import argparse
import random
import sqlite3
import sys
import types
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import scipy.sparse

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

if not hasattr(scipy.sparse.dok_matrix, "_update"):
    scipy.sparse.dok_matrix._update = scipy.sparse.dok_matrix.update

if "ray" not in sys.modules:
    fake_ray = types.ModuleType("ray")
    fake_ray_tune = types.ModuleType("ray.tune")
    fake_ray.tune = fake_ray_tune
    sys.modules["ray"] = fake_ray
    sys.modules["ray.tune"] = fake_ray_tune

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import init_seed, init_logger, get_model

CONFIG_MAP = {"NCF": "ncf.yaml", "NeuMF": "ncf.yaml", "LightGCN": "lightgcn.yaml", "SASRec": "sasrec.yaml"}
MODEL_NAME_MAP = {"NCF": "NeuMF", "NeuMF": "NeuMF", "LightGCN": "LightGCN", "SASRec": "SASRec"}

TOP_K = 10
N_NEGATIVES = 99
SEED = 2026


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    model_name = MODEL_NAME_MAP[args.model]
    config = Config(
        model=model_name, dataset=args.dataset,
        config_file_list=[CONFIG_MAP[args.model]],
        config_dict={"data_path": "dataset/"},
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    ckpt = torch.load(args.checkpoint, map_location=config["device"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    uid_field = config["USER_ID_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]

    uid_map = dataset.field2token_id[uid_field]
    iid_map = dataset.field2token_id[iid_field]

    valid_item_ids = {int(k) for k in iid_map.keys() if str(k) != "[PAD]"}
    valid_user_ids = {int(k) for k in uid_map.keys() if str(k) != "[PAD]"}
    print(f"Valid items: {len(valid_item_ids)}, valid users: {len(valid_user_ids)}")

    print("Building candidate sets from raw interaction file ...")
    inter_path = f"dataset/{args.dataset}/{args.dataset}.inter"
    inter = pd.read_csv(inter_path, sep="\t")
    inter.columns = [c.split(":")[0] for c in inter.columns]

    inter_sorted = inter.sort_values("timestamp")
    test_items = inter_sorted.groupby("user_id").tail(1).set_index("user_id")["item_id"].to_dict()
    rated_by_user = inter.groupby("user_id")["item_id"].apply(set).to_dict()

    random.seed(SEED)

    conn = sqlite3.connect("recommendations.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            model TEXT, run_timestamp TEXT, user_id INTEGER,
            item_id INTEGER, rank INTEGER, score REAL, dataset TEXT
        )
    """)
    conn.commit()

    run_timestamp = datetime.now(timezone.utc).isoformat()
    all_users = list(test_items.keys())
    print(f"Scoring {len(all_users)} users ...")

    written = 0
    skipped = 0
    errors_shown = 0

    for i, uid in enumerate(all_users):
        if uid not in valid_user_ids:
            skipped += 1
            continue
        true_item = test_items[uid]
        if true_item not in valid_item_ids:
            skipped += 1
            continue

        already_rated = rated_by_user.get(uid, set())
        pool = list(valid_item_ids - already_rated - {true_item})
        if len(pool) < N_NEGATIVES:
            skipped += 1
            continue
        negatives = random.sample(pool, N_NEGATIVES)
        candidates = negatives + [true_item]

        try:
            uid_internal = uid_map[str(uid)]
            cand_internal = [iid_map[str(c)] for c in candidates]
        except KeyError as e:
            skipped += 1
            if errors_shown < 5:
                print(f"  SKIP user {uid}: key not found -> {e}")
                errors_shown += 1
            continue

        user_tensor = torch.tensor([uid_internal] * len(cand_internal))
        item_tensor = torch.tensor(cand_internal)
        interaction = Interaction({uid_field: user_tensor, iid_field: item_tensor}).to(config["device"])

        try:
            with torch.no_grad():
                scores = model.predict(interaction).cpu().numpy()
        except Exception as e:
            skipped += 1
            if errors_shown < 5:
                print(f"  SKIP user {uid}: predict() failed -> {e}")
                errors_shown += 1
            continue

        ranked_idx = scores.argsort()[::-1][:TOP_K]
        rows = []
        for rank, idx in enumerate(ranked_idx, start=1):
            rows.append((model_name, run_timestamp, int(uid), int(candidates[idx]), rank, float(scores[idx]), args.dataset))

        conn.executemany(
            "INSERT INTO recommendations (model, run_timestamp, user_id, item_id, rank, score, dataset) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        written += 1

        if (i + 1) % 5000 == 0:
            conn.commit()
            print(f"[{i+1}/{len(all_users)}] {written} written, {skipped} skipped", flush=True)

    conn.commit()
    conn.close()
    print(f"\nDone. {written} users written, {skipped} skipped, dataset={args.dataset}, model={model_name}")


if __name__ == "__main__":
    main()
