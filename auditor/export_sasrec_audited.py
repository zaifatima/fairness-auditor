import argparse
import os
import sys
import types
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

if not hasattr(np, "float_"):
    np.float_ = np.float64

if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

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


def load_model(dataset_name, checkpoint_path):

    config = Config(
        model="SASRec",
        dataset=dataset_name,
        config_file_list=["sasrec.yaml"],
        config_dict={
            "data_path": "dataset/",
        },
    )

    init_seed(
        config["seed"],
        config["reproducibility"],
    )

    init_logger(config)

    dataset = create_dataset(config)

    train_data, valid_data, test_data = data_preparation(
        config,
        dataset,
    )

    model = get_model(config["model"])(
        config,
        train_data._dataset,
    ).to(config["device"])

    checkpoint = torch.load(
        checkpoint_path,
        map_location=config["device"],
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    return config, dataset, train_data, model


def load_test_pairs(path):

    test = pd.read_csv(
        path,
        usecols=["user_id", "item_id"],
    )

    test["user_id"] = test["user_id"].astype(int)
    test["item_id"] = test["item_id"].astype(int)

    return set(
        zip(
            test["user_id"],
            test["item_id"],
        )
    )


def build_histories(
    interactions,
    audited_users,
    test_pairs,
):

    interactions = interactions.sort_values(
        ["user_id", "timestamp"]
    )

    audited_users = set(
        int(x) for x in audited_users
    )

    histories = {}

    for user_id, group in interactions.groupby(
        "user_id",
        sort=False,
    ):

        user_id = int(user_id)

        if user_id not in audited_users:
            continue

        history = []

        for row in group.itertuples(index=False):

            pair = (
                int(row.user_id),
                int(row.item_id),
            )

            if pair in test_pairs:
                continue

            history.append(
                int(row.item_id)
            )

        if history:
            histories[user_id] = history

    return histories


def score_user_catalogue(
    config,
    dataset,
    model,
    user_idx,
    history_indices,
    item_batch_size,
    topk,
):

    device = config["device"]

    user_field = config["USER_ID_FIELD"]
    item_field = config["ITEM_ID_FIELD"]

    item_seq_field = "item_id_list"
    item_seq_len_field = "item_length"

    max_length = int(
        config["MAX_ITEM_LIST_LENGTH"]
    )

    history_indices = history_indices[
        -max_length:
    ]

    seq_len = len(history_indices)

    padded_history = (
        [0] * (max_length - seq_len)
        + history_indices
    )

    item_indices = torch.arange(
        1,
        dataset.item_num,
        dtype=torch.long,
    )

    all_scores = []

    with torch.no_grad():

        for start in range(
            0,
            len(item_indices),
            item_batch_size,
        ):

            batch_items = item_indices[
                start:start + item_batch_size
            ].to(device)

            batch_size = len(
                batch_items
            )

            batch_users = torch.full(
                (batch_size,),
                user_idx,
                dtype=torch.long,
                device=device,
            )

            batch_sequences = torch.tensor(
                [padded_history] * batch_size,
                dtype=torch.long,
                device=device,
            )

            batch_lengths = torch.full(
                (batch_size,),
                seq_len,
                dtype=torch.long,
                device=device,
            )

            interaction = Interaction(
                {
                    user_field: batch_users,
                    item_field: batch_items,
                    item_seq_field: batch_sequences,
                    item_seq_len_field: batch_lengths,
                }
            ).to(device)

            scores = model.predict(
                interaction
            )

            all_scores.append(
                scores.detach().cpu()
            )

    scores = torch.cat(
        all_scores
    )

    k = min(
        topk,
        len(scores)
    )

    top_scores, top_positions = torch.topk(
        scores,
        k=k,
        largest=True,
        sorted=True,
    )

    selected_items = item_indices[
        top_positions
    ].numpy()

    selected_scores = (
        top_scores.numpy()
    )

    return (
        selected_items,
        selected_scores,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--interactions",
        required=True,
    )

    parser.add_argument(
        "--test-interactions",
        required=True,
    )

    parser.add_argument(
        "--audited-users",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--item-batch-size",
        type=int,
        default=2048,
    )

    args = parser.parse_args()

    print(
        "=== SASRec AUDITED EXPORT ==="
    )

    audited_test = pd.read_csv(
        args.audited_users
    )

    audited_users = sorted(
        audited_test["user_id"]
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        f"Audited users: "
        f"{len(audited_users):,}"
    )

    print(
        "Loading interactions..."
    )

    interactions = pd.read_csv(
        args.interactions,
        sep="\t",
        engine="python",
        usecols=[
            "user_id:token",
            "item_id:token",
            "rating:float",
            "timestamp:float",
        ],
    )

    interactions.columns = [
        "user_id",
        "item_id",
        "rating",
        "timestamp",
    ]

    interactions["user_id"] = (
        interactions["user_id"].astype(int)
    )

    interactions["item_id"] = (
        interactions["item_id"].astype(int)
    )

    interactions["timestamp"] = (
        interactions["timestamp"].astype(int)
    )

    test_pairs = load_test_pairs(
        args.test_interactions
    )

    print(
        "Loading SASRec model..."
    )

    (
        config,
        dataset,
        train_data,
        model,
    ) = load_model(
        args.dataset,
        args.checkpoint,
    )

    print(
        f"Model loaded: {config['model']}"
    )

    print(
        f"RecBole catalogue: "
        f"{dataset.item_num - 1:,}"
    )

    print(
        f"Maximum sequence length: "
        f"{config['MAX_ITEM_LIST_LENGTH']}"
    )

    print(
        "Building audited user histories..."
    )

    histories = build_histories(
        interactions,
        audited_users,
        test_pairs,
    )

    print(
        f"Usable histories: "
        f"{len(histories):,}"
    )

    user_field = config[
        "USER_ID_FIELD"
    ]

    item_field = config[
        "ITEM_ID_FIELD"
    ]

    user_to_index = {}

    for user_idx in range(
        1,
        dataset.user_num,
    ):

        raw_user_id = dataset.id2token(
            user_field,
            user_idx,
        )

        user_to_index[
            int(raw_user_id)
        ] = user_idx

    run_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    rows = []

    for count, raw_user_id in enumerate(
        audited_users,
        start=1,
    ):

        if raw_user_id not in user_to_index:
            print(
                f"WARNING: user "
                f"{raw_user_id} not found."
            )
            continue

        if raw_user_id not in histories:
            print(
                f"WARNING: no history for "
                f"user {raw_user_id}."
            )
            continue

        raw_history = histories[
            raw_user_id
        ]

        history_indices = []

        for raw_item_id in raw_history:

            try:
                item_idx = dataset.token2id(
                    item_field,
                    str(raw_item_id),
                )

                history_indices.append(
                    int(item_idx)
                )

            except Exception:
                continue

        if not history_indices:
            continue

        user_idx = user_to_index[
            raw_user_id
        ]

        (
            selected_items,
            selected_scores,
        ) = score_user_catalogue(
            config=config,
            dataset=dataset,
            model=model,
            user_idx=user_idx,
            history_indices=history_indices,
            item_batch_size=args.item_batch_size,
            topk=args.topk,
        )

        for rank, (
            item_idx,
            score,
        ) in enumerate(
            zip(
                selected_items,
                selected_scores,
            ),
            start=1,
        ):

            raw_item_id = dataset.id2token(
                item_field,
                int(item_idx),
            )

            rows.append(
                {
                    "model": "SASRec",
                    "timestamp": run_timestamp,
                    "user_id": raw_user_id,
                    "item_id": int(raw_item_id),
                    "rank": rank,
                    "score": float(score),
                }
            )

        if (
            count % 10 == 0
            or count == len(audited_users)
        ):
            print(
                f"Processed "
                f"{count}/{len(audited_users)} users",
                flush=True,
            )

    recommendations = pd.DataFrame(
        rows,
        columns=[
            "model",
            "timestamp",
            "user_id",
            "item_id",
            "rank",
            "score",
        ],
    )

    if recommendations.empty:
        raise RuntimeError(
            "No recommendations generated."
        )

    os.makedirs(
        os.path.dirname(args.output)
        or ".",
        exist_ok=True,
    )

    recommendations.to_csv(
        args.output,
        index=False,
    )

    print()
    print(
        "=== EXPORT COMPLETE ==="
    )
    print(
        f"Rows: "
        f"{len(recommendations):,}"
    )
    print(
        f"Users: "
        f"{recommendations['user_id'].nunique():,}"
    )
    print(
        f"Output: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
