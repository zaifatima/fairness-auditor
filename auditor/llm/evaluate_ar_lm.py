import argparse
import os
from datetime import datetime

import pandas as pd
import torch

from auditor.llm.ar_lm import ARMovieLM
from auditor.llm.train_ar_lm import (
    load_interactions,
    load_test_pairs,
    build_user_sequences,
)


def load_checkpoint(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    model = ARMovieLM(
        num_items=checkpoint["num_items"],
        hidden_size=checkpoint["hidden_size"],
        n_layers=checkpoint["n_layers"],
        n_heads=checkpoint["n_heads"],
        ff_size=checkpoint["ff_size"],
        max_seq_length=checkpoint["max_seq_length"],
        dropout=checkpoint.get("dropout", 0.1),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interactions",
        required=True,
    )

    parser.add_argument(
        "--test-interactions",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    print("=== AR-LM EVALUATION ===")

    print("Loading interactions...")

    interactions = load_interactions(
        args.interactions
    )

    print(
        f"Interactions: "
        f"{len(interactions):,}"
    )

    print("Loading held-out test set...")

    test_df = pd.read_csv(
        args.test_interactions
    )

    test_pairs = load_test_pairs(
        args.test_interactions
    )

    print(
        f"Held-out interactions: "
        f"{len(test_df):,}"
    )

    print(
        f"Audited users: "
        f"{test_df['user_id'].nunique():,}"
    )

    print("Loading checkpoint...")

    model, checkpoint = load_checkpoint(
        args.checkpoint
    )

    item_to_token = checkpoint[
        "item_to_token"
    ]

    token_to_item = checkpoint[
        "token_to_item"
    ]

    max_seq_length = checkpoint[
        "max_seq_length"
    ]

    print(
        f"Model items: "
        f"{checkpoint['num_items']:,}"
    )

    print(
        f"Best training loss: "
        f"{checkpoint['best_loss']:.6f}"
    )

    print("Building user histories...")

    sequences = build_user_sequences(
        interactions,
        test_pairs,
        item_to_token,
    )

    audited_users = sorted(
        test_df["user_id"].unique()
    )

    # build_user_sequences returns sequences in user-group order,
    # so create an explicit user -> sequence mapping instead.
    interactions_sorted = interactions.sort_values(
        ["user_id", "timestamp"]
    )

    user_histories = {}

    for user_id, group in interactions_sorted.groupby(
        "user_id",
        sort=False,
    ):

        user_id = int(user_id)

        if user_id not in set(
            int(x) for x in audited_users
        ):
            continue

        history = []

        for row in group.itertuples(index=False):

            pair = (
                int(row.user_id),
                int(row.item_id),
            )

            if pair in test_pairs:
                continue

            raw_item = int(row.item_id)

            if raw_item in item_to_token:
                history.append(
                    item_to_token[raw_item]
                )

        if history:
            user_histories[user_id] = history

    print(
        f"Usable audited histories: "
        f"{len(user_histories):,}"
    )

    rows = []

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    print("Generating recommendations...")

    with torch.no_grad():

        for count, user_id in enumerate(
            audited_users,
            start=1,
        ):

            user_id = int(user_id)

            history = user_histories.get(
                user_id
            )

            if not history:
                continue

            history = history[
                -max_seq_length:
            ]

            padded = (
                [0]
                * (
                    max_seq_length
                    - len(history)
                )
            )

            padded.extend(history)

            inputs = torch.tensor(
                [padded],
                dtype=torch.long,
            )

            logits = model.predict_next(
                inputs
            )

            scores = logits[0]

            # Never recommend the padding token.
            scores[0] = float("-inf")

            # Never recommend items already seen
            # in the user's training history.
            seen_tokens = set(history)

            for token in seen_tokens:
                if token < len(scores):
                    scores[token] = float("-inf")

            top_scores, top_tokens = torch.topk(
                scores,
                k=args.top_k,
            )

            for rank, (
                token,
                score,
            ) in enumerate(
                zip(
                    top_tokens.tolist(),
                    top_scores.tolist(),
                ),
                start=1,
            ):

                raw_item_id = token_to_item[
                    int(token)
                ]

                rows.append(
                    {
                        "model": "AR-LM",
                        "timestamp": timestamp,
                        "user_id": user_id,
                        "item_id": int(raw_item_id),
                        "rank": rank,
                        "score": float(score),
                    }
                )

            if count % 20 == 0:
                print(
                    f"  Users processed: "
                    f"{count:,}",
                    flush=True,
                )

    output_df = pd.DataFrame(
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

    os.makedirs(
        os.path.dirname(args.output)
        or ".",
        exist_ok=True,
    )

    output_df.to_csv(
        args.output,
        index=False,
    )

    print()
    print("=== COMPLETE ===")
    print(
        f"Recommendations: "
        f"{len(output_df):,}"
    )
    print(
        f"Users: "
        f"{output_df['user_id'].nunique():,}"
    )
    print(
        f"Output: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
