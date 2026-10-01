from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.model.general_recommender.neumf import NeuMF
from recbole.trainer import Trainer


# ============================================================
# TEMPORAL NEUMF PILOT
#
# Evaluation year: 1999
#
# Historical model-fitting period:
#   1996-1997
#
# Historical external period:
#   1998
#
# Final temporal evaluation:
#   1999
#
# IMPORTANT:
#   1998 and 1999 are NOT supplied to the RecBole dataset.
#
# RecBole performs its internal train/validation/test split
# only within the 1996-1997 historical training pool.
#
# 1998 is retained as a genuinely later historical period
# but is not used to fit or select the model.
#
# 1999 is completely untouched until recommendation generation.
# ============================================================


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(r"C:\mscproject_code")

TEMPORAL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "temporal_datasets"
    / "1999"
)

TRAIN_FILE = TEMPORAL_DIR / "train.csv"
TEST_FILE = TEMPORAL_DIR / "test.csv"

RECBole_DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "ml32m_temporal_1999"
)

RECBole_INTER_FILE = (
    RECBole_DATASET_DIR
    / "ml32m_temporal_1999.inter"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "temporal_recommendations"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "saved"
)

YEAR = 1999


# ============================================================
# NeuMF settings
# Same core settings as ncf.yaml
# ============================================================

SEED = 2026

MF_EMBEDDING_SIZE = 64
MLP_EMBEDDING_SIZE = 64
MLP_HIDDEN_SIZE = [128, 64, 32]
DROPOUT_PROB = 0.1

EPOCHS = 20

TRAIN_BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 1024

LEARNING_RATE = 0.001

TOP_K = 10


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# File checks
# ============================================================

def check_files():

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Historical temporal training file not found:\n"
            f"{TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Temporal evaluation file not found:\n"
            f"{TEST_FILE}"
        )


# ============================================================
# Load temporal data
# ============================================================

def load_temporal_data():

    print()
    print("=" * 70)
    print("Loading temporal data")
    print("=" * 70)

    historical_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    historical_df["timestamp"] = pd.to_datetime(
        historical_df["timestamp"]
    )

    test_df["timestamp"] = pd.to_datetime(
        test_df["timestamp"]
    )

    historical_df["year"] = (
        historical_df["timestamp"].dt.year
    )

    test_df["year"] = (
        test_df["timestamp"].dt.year
    )

    print(
        f"Historical rows: "
        f"{len(historical_df):,}"
    )

    print(
        f"1999 test rows: "
        f"{len(test_df):,}"
    )

    print(
        f"Historical users: "
        f"{historical_df['user_id'].nunique():,}"
    )

    print(
        f"1999 test users: "
        f"{test_df['user_id'].nunique():,}"
    )

    print(
        f"Historical items: "
        f"{historical_df['item_id'].nunique():,}"
    )

    print(
        f"1999 test items: "
        f"{test_df['item_id'].nunique():,}"
    )

    return historical_df, test_df


# ============================================================
# Build model-fitting dataset
# ============================================================

def build_model_training_data(historical_df):

    print()
    print("=" * 70)
    print("Building historical model-fitting dataset")
    print("=" * 70)

    # --------------------------------------------------------
    # Model-fitting period:
    #
    # 1996-1997 only.
    #
    # 1995 is excluded because it contains only a handful of
    # observations.
    # --------------------------------------------------------

    train_df = historical_df[
        (historical_df["year"] >= 1996)
        & (historical_df["year"] <= 1997)
    ].copy()

    # --------------------------------------------------------
    # 1998 is deliberately kept separate.
    #
    # It is NOT supplied to RecBole.
    # --------------------------------------------------------

    external_1998_df = historical_df[
        historical_df["year"] == 1998
    ].copy()

    print(
        f"Model-fitting period: "
        f"{train_df['year'].min()}-"
        f"{train_df['year'].max()}"
    )

    print(
        f"Model-fitting rows: "
        f"{len(train_df):,}"
    )

    print(
        f"1998 external historical rows: "
        f"{len(external_1998_df):,}"
    )

    print(
        f"1998 external users: "
        f"{external_1998_df['user_id'].nunique():,}"
    )

    print(
        f"1998 external items: "
        f"{external_1998_df['item_id'].nunique():,}"
    )

    return train_df, external_1998_df


# ============================================================
# Prepare RecBole .inter file
# ============================================================

def prepare_recbole_file(train_df):

    print()
    print("=" * 70)
    print("Preparing RecBole historical dataset")
    print("=" * 70)

    RECBole_DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = train_df[
        [
            "user_id",
            "item_id",
            "rating",
            "timestamp",
        ]
    ].copy()

    # --------------------------------------------------------
    # RecBole token fields
    # --------------------------------------------------------

    output["user_id"] = (
        output["user_id"]
        .astype(str)
    )

    output["item_id"] = (
        output["item_id"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Convert timestamp to Unix seconds.
    # --------------------------------------------------------

    output["timestamp"] = (
        pd.to_datetime(
            output["timestamp"]
        )
        .astype("int64")
        // 10**9
    )

    # --------------------------------------------------------
    # RecBole field names
    # --------------------------------------------------------

    output = output.rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "rating": "rating:float",
            "timestamp": "timestamp:float",
        }
    )

    output.to_csv(
        RECBole_INTER_FILE,
        sep="\t",
        index=False
    )

    print(
        f"RecBole dataset written to:\n"
        f"{RECBole_INTER_FILE}"
    )

    print(
        f"Rows written: "
        f"{len(output):,}"
    )

    print(
        f"Users written: "
        f"{train_df['user_id'].nunique():,}"
    )

    print(
        f"Items written: "
        f"{train_df['item_id'].nunique():,}"
    )

    return output


# ============================================================
# Build RecBole configuration
# ============================================================

def build_config():

    config_dict = {

        # ----------------------------------------------------
        # Fields
        # ----------------------------------------------------

        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": "rating",
        "TIME_FIELD": "timestamp",

        "load_col": {
            "inter": [
                "user_id",
                "item_id",
                "rating",
                "timestamp",
            ]
        },

        # ----------------------------------------------------
        # Rating filter
        # Same as ncf.yaml
        # ----------------------------------------------------

        "val_interval": {
            "rating": "[3,inf)"
        },

        # ----------------------------------------------------
        # Minimum interaction filters
        # Same as ncf.yaml
        # ----------------------------------------------------

        "min_user_inter_num": 5,
        "min_item_inter_num": 5,

        # ----------------------------------------------------
        # Internal RecBole validation split.
        #
        # IMPORTANT:
        # This split happens ONLY inside the 1996-1997
        # historical dataset.
        #
        # Therefore neither 1998 nor 1999 can leak into
        # model training or model selection.
        # ----------------------------------------------------

        "eval_args": {
            "split": {
                "RS": [0.8, 0.1, 0.1]
            },

            "group_by": "user",

            # Random split, matching the original ncf.yaml.
            "order": "RO",

            "mode": {
                "valid": "uni100",
                "test": "uni100",
            },
        },

        # ----------------------------------------------------
        # NeuMF architecture
        # ----------------------------------------------------

        "mf_embedding_size": (
            MF_EMBEDDING_SIZE
        ),

        "mlp_embedding_size": (
            MLP_EMBEDDING_SIZE
        ),

        "mlp_hidden_size": (
            MLP_HIDDEN_SIZE
        ),

        "dropout_prob": (
            DROPOUT_PROB
        ),

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        "epochs": EPOCHS,

        "train_batch_size": (
            TRAIN_BATCH_SIZE
        ),

        "eval_batch_size": (
            EVAL_BATCH_SIZE
        ),

        "learning_rate": (
            LEARNING_RATE
        ),

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        "metrics": [
            "Recall",
            "NDCG",
            "Hit",
            "Precision",
        ],

        "topk": [
            10,
            20,
        ],

        "valid_metric": "NDCG@10",

        # ----------------------------------------------------
        # Hardware
        # ----------------------------------------------------

        "device": "cpu",

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        "seed": SEED,

        # ----------------------------------------------------
        # Checkpoints
        # ----------------------------------------------------

        "checkpoint_dir": str(
            CHECKPOINT_DIR
        ),

        "eval_step": 5,

        # ----------------------------------------------------
        # Dataset location
        # ----------------------------------------------------

        "data_path": str(
            PROJECT_ROOT / "dataset"
        ),

        "dataset": "ml32m_temporal_1999",

        "model": "NeuMF",
    }

    config = Config(
        model="NeuMF",
        dataset="ml32m_temporal_1999",
        config_dict=config_dict,
    )

    return config


# ============================================================
# Train NeuMF
# ============================================================

def train_model(config):

    print()
    print("=" * 70)
    print("Creating RecBole dataset")
    print("=" * 70)

    dataset = create_dataset(config)

    print(dataset)

    print()
    print("=" * 70)
    print(
        "Preparing RecBole train/validation/test objects"
    )
    print("=" * 70)

    train_data, valid_data, test_data = (
        data_preparation(
            config,
            dataset
        )
    )

    print(
        "RecBole train/validation/test objects "
        "created successfully."
    )

    # --------------------------------------------------------
    # Train NeuMF
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Training NeuMF")
    print("=" * 70)

    model = NeuMF(
        config,
        train_data.dataset
    ).to(config["device"])

    trainer = Trainer(
        config,
        model
    )

    best_valid_score, best_valid_result = (
        trainer.fit(
            train_data,
            valid_data
        )
    )

    print()
    print("=" * 70)
    print("Training complete")
    print("=" * 70)

    print(
        f"Best validation score: "
        f"{best_valid_score}"
    )

    print(
        f"Best validation result: "
        f"{best_valid_result}"
    )

    # --------------------------------------------------------
    # Save temporal checkpoint
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_file = (
        CHECKPOINT_DIR
        / f"NeuMF-temporal-{YEAR}.pth"
    )

    torch.save(
        {
            "config": config,
            "state_dict": model.state_dict(),
        },
        checkpoint_file
    )

    print()
    print(
        f"Temporal checkpoint saved:\n"
        f"{checkpoint_file}"
    )

    return model, dataset


# ============================================================
# Generate Top-10 recommendations
# ============================================================

def generate_recommendations(
    model,
    dataset,
    train_df,
    test_df,
    config,
):

    print()
    print("=" * 70)
    print("Generating temporal Top-10 recommendations")
    print("=" * 70)

    model.eval()

    # --------------------------------------------------------
    # RecBole field names
    # --------------------------------------------------------

    user_field = dataset.uid_field
    item_field = dataset.iid_field

    print(
        f"RecBole user field: {user_field}"
    )

    print(
        f"RecBole item field: {item_field}"
    )

    # --------------------------------------------------------
    # RecBole mappings
    #
    # token -> internal integer ID
    # internal integer ID -> token
    # --------------------------------------------------------

    user_token_to_id = (
        dataset.field2token_id[user_field]
    )

    item_token_to_id = (
        dataset.field2token_id[item_field]
    )

    item_id_to_token = (
        dataset.field2id_token[item_field]
    )

    # --------------------------------------------------------
    # Users who have:
    #
    # 1. historical interactions in 1996-1997
    # 2. at least one interaction in 1999
    #
    # Then we additionally require the user to survive
    # RecBole's minimum-interaction filtering.
    # --------------------------------------------------------

    train_users = set(
        train_df["user_id"].unique()
    )

    test_users = set(
        test_df["user_id"].unique()
    )

    historical_test_users = (
        train_users.intersection(
            test_users
        )
    )

    eligible_users = []

    for user_id in sorted(
        historical_test_users
    ):

        user_token = str(user_id)

        if user_token in user_token_to_id:
            eligible_users.append(
                user_id
            )

    print(
        f"1999 test users: "
        f"{len(test_users):,}"
    )

    print(
        f"Users with pre-1999 history: "
        f"{len(historical_test_users):,}"
    )

    print(
        f"Users retained by RecBole: "
        f"{len(eligible_users):,}"
    )

    # --------------------------------------------------------
    # Historical catalogue
    #
    # Only items known during the model-fitting period.
    # --------------------------------------------------------

    historical_catalogue = set(
        train_df["item_id"].unique()
    )

    # Restrict catalogue to items actually represented
    # in the trained RecBole dataset.
    #
    # This is essential because items filtered out by
    # min_item_inter_num cannot be scored by the model.
    # --------------------------------------------------------

    model_catalogue_items = []

    for item_id in sorted(
        historical_catalogue
    ):

        item_token = str(item_id)

        if item_token in item_token_to_id:
            model_catalogue_items.append(
                item_id
            )

    print(
        f"Historical catalogue items: "
        f"{len(historical_catalogue):,}"
    )

    print(
        f"Items retained by RecBole: "
        f"{len(model_catalogue_items):,}"
    )

    # --------------------------------------------------------
    # Items consumed by each user before 1999.
    # --------------------------------------------------------

    rated_by_user = (
        train_df
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    # --------------------------------------------------------
    # Generate recommendations.
    # --------------------------------------------------------

    rows = []

    total_users = len(
        eligible_users
    )

    for counter, user_id in enumerate(
        eligible_users,
        start=1
    ):

        user_token = str(user_id)

        # ----------------------------------------------------
        # Map user to RecBole internal ID.
        # ----------------------------------------------------

        user_inner = user_token_to_id[
            user_token
        ]

        # ----------------------------------------------------
        # Remove items the user has already consumed.
        # ----------------------------------------------------

        rated_items = rated_by_user.get(
            user_id,
            set()
        )

        candidate_raw_items = [
            item_id
            for item_id in model_catalogue_items
            if item_id not in rated_items
        ]

        if not candidate_raw_items:
            continue

        # ----------------------------------------------------
        # Convert raw item IDs to RecBole internal IDs.
        # ----------------------------------------------------

        candidate_inner_items = [
            item_token_to_id[
                str(item_id)
            ]
            for item_id in candidate_raw_items
        ]

        # ----------------------------------------------------
        # Create a prediction Interaction.
        #
        # This is the correct RecBole mechanism.
        # ----------------------------------------------------

        user_tensor = torch.full(
            (
                len(candidate_inner_items),
            ),
            user_inner,
            dtype=torch.long,
            device=config["device"],
        )

        item_tensor = torch.tensor(
            candidate_inner_items,
            dtype=torch.long,
            device=config["device"],
        )

        interaction = Interaction(
            {
                user_field: user_tensor,
                item_field: item_tensor,
            }
        )

        # ----------------------------------------------------
        # Score every candidate item.
        # ----------------------------------------------------

        with torch.no_grad():

            scores = model.predict(
                interaction
            )

        # ----------------------------------------------------
        # Ensure scores are one-dimensional.
        # ----------------------------------------------------

        scores = scores.reshape(-1)

        # ----------------------------------------------------
        # Select Top-K.
        # ----------------------------------------------------

        top_n = min(
            TOP_K,
            len(candidate_inner_items)
        )

        top_indices = (
            torch.topk(
                scores,
                k=top_n
            )
            .indices
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Convert recommendations back to raw MovieLens IDs.
        # ----------------------------------------------------

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            internal_item_id = (
                candidate_inner_items[index]
            )

            raw_item_id = (
                item_id_to_token[
                    internal_item_id
                ]
            )

            rows.append(
                {
                    "model": "NeuMF",
                    "year": YEAR,
                    "timestamp": (
                        f"{YEAR}-12-31"
                    ),
                    "user_id": int(
                        user_id
                    ),
                    "item_id": int(
                        raw_item_id
                    ),
                    "rank": rank,
                    "score": float(
                        scores[index]
                        .cpu()
                        .item()
                    ),
                }
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if counter % 100 == 0:

            print(
                f"Processed "
                f"{counter:,}/"
                f"{total_users:,} "
                f"eligible users"
            )

    # ========================================================
    # Build output DataFrame
    # ========================================================

    output_df = pd.DataFrame(
        rows,
        columns=[
            "model",
            "year",
            "timestamp",
            "user_id",
            "item_id",
            "rank",
            "score",
        ]
    )

    # ========================================================
    # Save output
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / f"NeuMF-temporal-{YEAR}-top10-recommendations.csv"
    )

    output_df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # Final diagnostics
    # ========================================================

    print()
    print("=" * 70)
    print("Recommendation generation complete")
    print("=" * 70)

    print(
        f"Output file:\n"
        f"{output_file}"
    )

    print(
        f"Recommendation rows: "
        f"{len(output_df):,}"
    )

    if len(output_df) > 0:

        print(
            f"Recommendation users: "
            f"{output_df['user_id'].nunique():,}"
        )

        print(
            f"Average recommendations per user: "
            f"{len(output_df) / output_df['user_id'].nunique():.2f}"
        )

        print()
        print(
            "First recommendations:"
        )

        print(
            output_df.head(20).to_string(
                index=False
            )
        )

    else:

        print()
        print(
            "WARNING: No recommendations were generated."
        )

    return output_df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    check_files()

    # --------------------------------------------------------
    # Load temporal data
    # --------------------------------------------------------

    historical_df, test_df = (
        load_temporal_data()
    )

    # --------------------------------------------------------
    # Build model-fitting period
    # --------------------------------------------------------

    train_df, external_1998_df = (
        build_model_training_data(
            historical_df
        )
    )

    # --------------------------------------------------------
    # Create RecBole dataset.
    #
    # ONLY 1996-1997 enters this file.
    # --------------------------------------------------------

    prepare_recbole_file(
        train_df
    )

    # --------------------------------------------------------
    # Build configuration
    # --------------------------------------------------------

    config = build_config()

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model, dataset = (
        train_model(
            config
        )
    )

    # --------------------------------------------------------
    # Generate 1999 recommendations
    # --------------------------------------------------------

    recommendations = (
        generate_recommendations(
            model,
            dataset,
            train_df,
            test_df,
            config,
        )
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("1999 TEMPORAL NEUMF PILOT COMPLETE")
    print("=" * 70)

    print()
    print(
        "Model-fitting period: 1996-1997"
    )

    print(
        "External historical period: 1998 "
        "(not used for model fitting)"
    )

    print(
        "Evaluation year: 1999"
    )

    print(
        "1999 data was not supplied to RecBole."
    )