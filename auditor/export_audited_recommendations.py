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


CONFIG_MAP = {
    "NeuMF": "ncf.yaml",
    "LightGCN": "lightgcn.yaml",
    "SASRec": "sasrec.yaml",
}

MODEL_NAME_MAP = {
    "NeuMF": "NeuMF",
    "LightGCN": "LightGCN",
    "SASRec": "SASRec",
}


def load_model(model_name, dataset_name, checkpoint_path):

    config = Config(
        model=MODEL_NAME_MAP[model_name],
        dataset=dataset_name,
        config_file_list=[
            CONFIG_MAP[model_name]
        ],
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

    return (
        config,
        dataset,
        model,
    )


def score_user_catalogue(
    config,
    dataset,
    model,
    user_idx,
    item_batch_size,
    topk,
):

    device = config["device"]

    user_field = config["USER_ID_FIELD"]
    item_field = config["ITEM_ID_FIELD"]

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

            batch_users = torch.full(
                (len(batch_items),),
                user_idx,
                dtype=torch.long,
                device=device,
            )

            interaction = Interaction(
                {
                    user_field: batch_users,
                    item_field: batch_items,
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
        "--model",
        required=True,
        choices=[
            "NeuMF",
            "LightGCN",
            "SASRec",
        ],
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
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

    if not os.path.exists(
        args.checkpoint
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    audited = pd.read_csv(
        args.audited_users
    )

    audited_users = sorted(
        audited["user_id"]
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        f"Audited users: "
        f"{len(audited_users):,}"
    )

    print(
        "Loading model..."
    )

    (
        config,
        dataset,
        model,
    ) = load_model(
        args.model,
        args.dataset,
        args.checkpoint,
    )

    print(
        f"Model loaded: "
        f"{args.model}"
    )

    print(
        f"Catalogue items: "
        f"{dataset.item_num - 1:,}"
    )

    print(
        f"Generating Top-{args.topk}..."
    )

    user_field = config[
        "USER_ID_FIELD"
    ]

    item_field = config[
        "ITEM_ID_FIELD"
    ]

    run_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    rows = []

    # Map raw user IDs to RecBole internal IDs.
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
                    "model": args.model,
                    "timestamp": run_timestamp,
                    "user_id": int(raw_user_id),
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
