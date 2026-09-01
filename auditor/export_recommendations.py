import argparse
import os
import sys
import types
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------------
# Compatibility patches used by your existing project
# ------------------------------------------------------------------

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
    "NCF": "ncf.yaml",
    "NeuMF": "ncf.yaml",
    "LightGCN": "lightgcn.yaml",
    "SASRec": "sasrec.yaml",
}

MODEL_NAME_MAP = {
    "NCF": "NeuMF",
    "NeuMF": "NeuMF",
    "LightGCN": "LightGCN",
    "SASRec": "SASRec",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["NCF", "NeuMF", "LightGCN", "SASRec"],
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
        "--topk",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-users",
        type=int,
        default=10,
        help="Number of users to export. 0 = all users.",
    )

    parser.add_argument(
        "--item-batch-size",
        type=int,
        default=2048,
        help="Number of candidate items scored per batch.",
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    return parser.parse_args()


def load_model(args):
    model_name = MODEL_NAME_MAP[args.model]

    config = Config(
        model=model_name,
        dataset=args.dataset,
        config_file_list=[CONFIG_MAP[args.model]],
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
        args.checkpoint,
        map_location=config["device"],
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return config, dataset, train_data, model


def score_user_catalogue(
    config,
    dataset,
    model,
    user_idx,
    item_batch_size,
    topk,
):
    """
    Score every catalogue item for one user using model.predict().

    We process the item catalogue in batches to control memory use.
    """

    device = config["device"]

    user_field = config["USER_ID_FIELD"]
    item_field = config["ITEM_ID_FIELD"]

    # RecBole index 0 is the padding token.
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

            scores = model.predict(interaction)

            scores = scores.detach().cpu()

            all_scores.append(scores)

    scores = torch.cat(all_scores)

    k = min(topk, len(scores))

    top_scores, top_positions = torch.topk(
        scores,
        k=k,
        largest=True,
        sorted=True,
    )

    selected_items = item_indices[
        top_positions
    ].numpy()

    selected_scores = top_scores.numpy()

    return selected_items, selected_scores


def export_recommendations(
    config,
    dataset,
    model,
    topk,
    max_users,
    item_batch_size,
):
    user_field = config["USER_ID_FIELD"]
    item_field = config["ITEM_ID_FIELD"]

    # RecBole reserves index 0.
    user_indices = list(
        range(1, dataset.user_num)
    )

    if max_users > 0:
        user_indices = user_indices[:max_users]

    run_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    rows = []

    print(
        f"Users selected: {len(user_indices)}",
        flush=True,
    )

    print(
        f"Catalogue items: {dataset.item_num - 1:,}",
        flush=True,
    )

    print(
        f"Top-K: {topk}",
        flush=True,
    )

    print(
        f"Item batch size: {item_batch_size}",
        flush=True,
    )

    for count, user_idx in enumerate(
        user_indices,
        start=1,
    ):

        selected_items, selected_scores = (
            score_user_catalogue(
                config=config,
                dataset=dataset,
                model=model,
                user_idx=user_idx,
                item_batch_size=item_batch_size,
                topk=topk,
            )
        )

        raw_user_id = dataset.id2token(
            user_field,
            user_idx,
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
                    "model": config["model"],
                    "timestamp": run_timestamp,
                    "user_id": str(raw_user_id),
                    "item_id": str(raw_item_id),
                    "rank": rank,
                    "score": float(score),
                }
            )

        if (
            count % 5 == 0
            or count == len(user_indices)
        ):
            print(
                f"Processed "
                f"{count}/{len(user_indices)} users",
                flush=True,
            )

    return pd.DataFrame(rows)


def main():

    args = parse_args()

    if args.topk <= 0:
        raise ValueError(
            "--topk must be greater than zero"
        )

    if args.max_users < 0:
        raise ValueError(
            "--max-users must be zero or greater"
        )

    if args.item_batch_size <= 0:
        raise ValueError(
            "--item-batch-size must be greater "
            "than zero"
        )

    if not os.path.exists(
        args.checkpoint
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    print(
        "Loading model...",
        flush=True,
    )

    (
        config,
        dataset,
        train_data,
        model,
    ) = load_model(args)

    print(
        f"Model loaded: {config['model']}",
        flush=True,
    )

    recommendations = (
        export_recommendations(
            config=config,
            dataset=dataset,
            model=model,
            topk=args.topk,
            max_users=args.max_users,
            item_batch_size=args.item_batch_size,
        )
    )

    if recommendations.empty:
        raise RuntimeError(
            "No recommendations were generated."
        )

    os.makedirs(
        "outputs",
        exist_ok=True,
    )

    if args.output is None:
        output_path = (
            f"outputs/"
            f"{config['model']}-"
            f"{args.dataset}-"
            f"top{args.topk}-"
            f"recommendations.csv"
        )
    else:
        output_path = args.output

    recommendations.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n=== Recommendation export complete ===",
        flush=True,
    )

    print(
        f"Output: {output_path}",
        flush=True,
    )

    print(
        f"Rows: {len(recommendations):,}",
        flush=True,
    )

    print(
        f"Users: "
        f"{recommendations['user_id'].nunique():,}",
        flush=True,
    )

    print(
        f"Unique recommended items: "
        f"{recommendations['item_id'].nunique():,}",
        flush=True,
    )

    print(
        "\nSample:",
        flush=True,
    )

    print(
        recommendations.head(20).to_string(
            index=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
