import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from auditor.llm.ar_lm import ARMovieLM


SEED = 2026


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_interactions(path):
    df = pd.read_csv(
        path,
        sep="\t",
        engine="python",
        usecols=[
            "user_id:token",
            "item_id:token",
            "rating:float",
            "timestamp:float",
        ],
    )

    df.columns = [
        "user_id",
        "item_id",
        "rating",
        "timestamp",
    ]

    df["user_id"] = df["user_id"].astype(np.int32)
    df["item_id"] = df["item_id"].astype(np.int32)
    df["rating"] = df["rating"].astype(np.float32)
    df["timestamp"] = df["timestamp"].astype(np.int64)

    return df


def load_test_pairs(path):
    test = pd.read_csv(
        path,
        usecols=["user_id", "item_id"],
    )

    test["user_id"] = test["user_id"].astype(np.int32)
    test["item_id"] = test["item_id"].astype(np.int32)

    return set(
        zip(
            test["user_id"],
            test["item_id"],
        )
    )


def build_item_mapping(interactions):
    """
    Map raw MovieLens item IDs to contiguous model token IDs.

    Token 0 is reserved for padding.
    Model tokens therefore run from 1 to num_items.
    """

    raw_item_ids = sorted(
        interactions["item_id"].unique().tolist()
    )

    item_to_token = {
        int(raw_item_id): token_id
        for token_id, raw_item_id
        in enumerate(raw_item_ids, start=1)
    }

    token_to_item = {
        token_id: raw_item_id
        for raw_item_id, token_id
        in item_to_token.items()
    }

    return item_to_token, token_to_item


def build_user_sequences(
    interactions,
    test_pairs,
    item_to_token,
    max_users=None,
):
    """
    Build one chronological training sequence per user.

    Held-out test interactions are removed completely.
    Raw item IDs are converted to contiguous model tokens.
    """

    interactions = interactions.sort_values(
        ["user_id", "timestamp"]
    )

    sequences = []

    for user_index, (user_id, group) in enumerate(
        interactions.groupby("user_id", sort=False)
    ):

        if max_users is not None and user_index >= max_users:
            break

        items = []

        for row in group.itertuples(index=False):

            pair = (
                int(row.user_id),
                int(row.item_id),
            )

            if pair in test_pairs:
                continue

            raw_item_id = int(row.item_id)

            if raw_item_id not in item_to_token:
                continue

            items.append(
                item_to_token[raw_item_id]
            )

        if len(items) >= 2:
            sequences.append(items)

    return sequences


class SequenceDataset(Dataset):
    """
    Generates next-item training examples.

    Each example predicts the next item from the preceding
    max_seq_length items.
    """

    def __init__(
        self,
        sequences,
        max_seq_length,
        max_examples=None,
    ):

        self.sequences = sequences
        self.max_seq_length = max_seq_length

        self.index = []

        for sequence_id, sequence in enumerate(sequences):

            for end in range(1, len(sequence)):

                self.index.append(
                    (
                        sequence_id,
                        end,
                    )
                )

                if (
                    max_examples is not None
                    and len(self.index) >= max_examples
                ):
                    return

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):

        sequence_id, end = self.index[index]

        sequence = self.sequences[sequence_id]

        start = max(
            0,
            end - self.max_seq_length,
        )

        history = sequence[start:end]

        target = sequence[end]

        padded = (
            [0]
            * (
                self.max_seq_length
                - len(history)
            )
        )

        padded.extend(history)

        return (
            torch.tensor(
                padded,
                dtype=torch.long,
            ),
            torch.tensor(
                target,
                dtype=torch.long,
            ),
        )


def train_epoch(
    model,
    loader,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch_index, (
        inputs,
        targets,
    ) in enumerate(loader):

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model.predict_next(
            inputs
        )

        loss = F.cross_entropy(
            logits,
            targets,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = inputs.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_examples += batch_size

        if (
            batch_index + 1
        ) % 100 == 0:

            print(
                f"  Batch "
                f"{batch_index + 1:,} "
                f"| loss={loss.item():.6f}",
                flush=True,
            )

    if total_examples == 0:
        return float("nan")

    return (
        total_loss
        / total_examples
    )


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
        "--output",
        default=(
            "saved/"
            "AR-LM-ml-32m-50k-best.pth"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--heads",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--ff-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device("cpu")

    print("=== AR-LM SETUP ===")
    print(f"Device: {device}")

    print("Loading interactions...")

    interactions = load_interactions(
        args.interactions
    )

    print(
        f"Interactions: "
        f"{len(interactions):,}"
    )

    print(
        f"Users: "
        f"{interactions.user_id.nunique():,}"
    )

    print(
        f"Raw items: "
        f"{interactions.item_id.nunique():,}"
    )

    print(
        f"Maximum raw item ID: "
        f"{interactions.item_id.max():,}"
    )

    print("Loading held-out test pairs...")

    test_pairs = load_test_pairs(
        args.test_interactions
    )

    print(
        f"Held-out test pairs: "
        f"{len(test_pairs):,}"
    )

    print("Building contiguous item mapping...")

    item_to_token, token_to_item = build_item_mapping(
        interactions
    )

    num_items = len(item_to_token)

    print(
        f"Model item tokens: "
        f"{num_items:,}"
    )

    print(
        "Token 0 reserved for padding."
    )

    print("Building user sequences...")

    sequences = build_user_sequences(
        interactions,
        test_pairs,
        item_to_token,
        max_users=args.max_users,
    )

    print(
        f"Training user sequences: "
        f"{len(sequences):,}"
    )

    print("Building training dataset...")

    dataset = SequenceDataset(
        sequences,
        max_seq_length=args.max_seq_length,
        max_examples=args.max_examples,
    )

    print(
        f"Training examples: "
        f"{len(dataset):,}"
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "No training examples were created."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    print("Creating AR-LM model...")

    model = ARMovieLM(
        num_items=num_items,
        hidden_size=args.hidden_size,
        n_layers=args.layers,
        n_heads=args.heads,
        ff_size=args.ff_size,
        max_seq_length=args.max_seq_length,
        dropout=0.1,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    os.makedirs(
        os.path.dirname(args.output)
        or ".",
        exist_ok=True,
    )

    best_loss = float("inf")

    print("=== TRAINING ===")

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print(
            f"Epoch {epoch}/{args.epochs}"
        )

        loss = train_epoch(
            model,
            loader,
            optimizer,
            device,
        )

        print(
            f"Epoch {epoch} "
            f"| loss={loss:.6f}",
            flush=True,
        )

        if loss < best_loss:

            best_loss = loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "num_items": num_items,
                "hidden_size": args.hidden_size,
                "n_layers": args.layers,
                "n_heads": args.heads,
                "ff_size": args.ff_size,
                "max_seq_length": args.max_seq_length,
                "dropout": 0.1,
                "item_to_token": item_to_token,
                "token_to_item": token_to_item,
                "best_loss": best_loss,
                "epoch": epoch,
                "seed": SEED,
            }

            torch.save(
                checkpoint,
                args.output,
            )

            print(
                f"Saved best checkpoint: "
                f"{args.output}",
                flush=True,
            )

    print("=== TRAINING COMPLETE ===")
    print(
        f"Best loss: "
        f"{best_loss:.6f}"
    )
    print(
        f"Checkpoint: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
