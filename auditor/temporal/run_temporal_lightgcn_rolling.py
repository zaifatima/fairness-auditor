from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.trainer import Trainer


# ============================================================
# ROLLING TEMPORAL LIGHTGCN EXPERIMENT
#
# Evaluation years:
#     1999 -> 2023
#
# Rolling-origin design:
#     Evaluation year t
#     -----------------
#     Training:
#         1996 ... t-1
#
#     Evaluation:
#         t
#
# IMPORTANT:
#     Evaluation year t is NEVER supplied to RecBole.
#
#     RecBole performs its internal train/validation/test
#     split only inside the historical data available before t.
#
# Outputs:
#     saved/
#         LightGCN-temporal-1999.pth
#         ...
#         LightGCN-temporal-2023.pth
#
#     outputs/temporal_recommendations/
#         LightGCN-temporal-1999-top10-recommendations.csv
#         ...
#         LightGCN-temporal-2023-top10-recommendations.csv
#         LightGCN-temporal-1999-2023-all-recommendations.csv
# ============================================================


# ============================================================
# COMPATIBILITY PATCHES
# ============================================================

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

try:
    import scipy.sparse

    if not hasattr(scipy.sparse.dok_matrix, "_update"):
        scipy.sparse.dok_matrix._update = scipy.sparse.dok_matrix.update
except Exception:
    pass


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\mscproject_code")

SOURCE_INTER_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "ml-32m-50k"
    / "ml-32m-50k.inter"
)

TEMPORAL_DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "temporal_lightgcn"
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


# ============================================================
# EXPERIMENT RANGE
# ============================================================

START_YEAR = 2003
END_YEAR = 2023
HISTORICAL_START_YEAR = 1996


# ============================================================
# LIGHTGCN SETTINGS
# Same core settings as lightgcn.yaml
# ============================================================

SEED = 2026

EMBEDDING_SIZE = 64
N_LAYERS = 1
REG_WEIGHT = 1e-5

EPOCHS = 20

TRAIN_BATCH_SIZE = 16384
EVAL_BATCH_SIZE = 16384

LEARNING_RATE = 0.001

TOP_K = 10


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CHECK REQUIRED FILE
# ============================================================

def check_source_file():
    if not SOURCE_INTER_FILE.exists():
        raise FileNotFoundError(
            "Source MovieLens interaction file not found:\n"
            f"{SOURCE_INTER_FILE}"
        )


# ============================================================
# LOAD FULL DATASET
# ============================================================

def load_source_data():
    print("\n" + "=" * 70)
    print("Loading MovieLens interaction data")
    print("=" * 70)

    df = pd.read_csv(
        SOURCE_INTER_FILE,
        sep="\t",
        low_memory=False,
    )

    # RecBole .inter files use headers such as:
    # user_id:token, item_id:token, rating:float, timestamp:float
    # Strip the type suffix so pandas columns become:
    # user_id, item_id, rating, timestamp
    df.columns = [str(col).split(":")[0].strip() for col in df.columns]

    required_columns = [
        "user_id",
        "item_id",
        "rating",
        "timestamp",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    # Make sure timestamp is numeric.
    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "user_id",
            "item_id",
            "rating",
            "timestamp",
        ]
    ).copy()

    # Convert Unix timestamp to calendar year.
    df["date"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        errors="coerce",
    )

    df = df.dropna(
        subset=["date"]
    ).copy()

    df["year"] = (
        df["date"]
        .dt.year
        .astype(int)
    )

    # Keep only the historical period used by the experiment.
    df = df[
        (df["year"] >= HISTORICAL_START_YEAR)
        & (df["year"] <= END_YEAR)
    ].copy()

    # Ensure identifiers are integers.
    df["user_id"] = pd.to_numeric(
        df["user_id"],
        errors="coerce",
    )

    df["item_id"] = pd.to_numeric(
        df["item_id"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "user_id",
            "item_id",
        ]
    ).copy()

    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)

    print(f"Rows loaded: {len(df):,}")
    print(f"Users: {df['user_id'].nunique():,}")
    print(f"Items: {df['item_id'].nunique():,}")
    print(
        f"Years: {df['year'].min()}-"
        f"{df['year'].max()}"
    )
    print(
        f"Timestamp dtype: "
        f"{df['timestamp'].dtype}"
    )

    return df


# ============================================================
# BUILD ROLLING DATA
# ============================================================

def build_year_data(
    full_df,
    evaluation_year,
):
    print()
    print("=" * 70)
    print(
        f"Building rolling dataset for "
        f"{evaluation_year}"
    )
    print("=" * 70)

    # Historical training pool:
    # all observations from 1996 up to year-1.
    historical_df = full_df[
        full_df["year"] < evaluation_year
    ].copy()

    # Evaluation year:
    # completely separate.
    test_df = full_df[
        full_df["year"] == evaluation_year
    ].copy()

    if historical_df.empty:
        raise ValueError(
            f"No historical data available before "
            f"{evaluation_year}."
        )

    if test_df.empty:
        raise ValueError(
            f"No evaluation data found for "
            f"{evaluation_year}."
        )

    print(
        f"Historical period: "
        f"{historical_df['year'].min()}-"
        f"{historical_df['year'].max()}"
    )

    print(
        f"Evaluation period: "
        f"{evaluation_year}"
    )

    print(
        f"Historical rows: "
        f"{len(historical_df):,}"
    )

    print(
        f"Evaluation rows: "
        f"{len(test_df):,}"
    )

    print(
        f"Historical users: "
        f"{historical_df['user_id'].nunique():,}"
    )

    print(
        f"Evaluation users: "
        f"{test_df['user_id'].nunique():,}"
    )

    print(
        f"Historical items: "
        f"{historical_df['item_id'].nunique():,}"
    )

    print(
        f"Evaluation items: "
        f"{test_df['item_id'].nunique():,}"
    )

    return historical_df, test_df


# ============================================================
# WRITE TEMPORAL RECBole DATASET
# ============================================================

def prepare_recbole_dataset(
    historical_df,
    evaluation_year,
):
    print()
    print("=" * 70)
    print(
        f"Preparing RecBole dataset for "
        f"{evaluation_year}"
    )
    print("=" * 70)

    dataset_name = (
        f"ml32m_temporal_lightgcn_{evaluation_year}"
    )

    dataset_dir = (
        TEMPORAL_DATASET_ROOT
        / dataset_name
    )

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    inter_file = (
        dataset_dir
        / f"{dataset_name}.inter"
    )

    # Only historical observations enter RecBole.
    # No evaluation-year rows are written here.
    output = historical_df[
        [
            "user_id",
            "item_id",
            "rating",
            "timestamp",
        ]
    ].copy()

    # RecBole token fields.
    output["user_id"] = (
        output["user_id"].astype(str)
    )

    output["item_id"] = (
        output["item_id"].astype(str)
    )

    # Keep timestamps as Unix seconds.
    output["timestamp"] = (
        pd.to_numeric(
            output["timestamp"],
            errors="coerce",
        )
        .astype("int64")
    )

    # RecBole column names.
    output = output.rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "rating": "rating:float",
            "timestamp": "timestamp:float",
        }
    )

    output.to_csv(
        inter_file,
        sep="\t",
        index=False,
    )

    print(
        f"RecBole dataset:\n"
        f"{inter_file}"
    )

    print(
        f"Historical rows written: "
        f"{len(output):,}"
    )

    return dataset_name


# ============================================================
# BUILD RECBole CONFIGURATION
# ============================================================

def build_config(
    dataset_name,
):
    config_dict = {
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

        "val_interval": {
            "rating": "[3,inf)"
        },

        "min_user_inter_num": 5,
        "min_item_inter_num": 5,

        "eval_args": {
            "split": {
                "RS": [
                    0.8,
                    0.1,
                    0.1,
                ]
            },
            "group_by": "user",
            "order": "RO",
            "mode": {
                "valid": "uni100",
                "test": "uni100",
            },
        },

        # LightGCN architecture.
        "embedding_size": EMBEDDING_SIZE,
        "n_layers": N_LAYERS,
        "reg_weight": REG_WEIGHT,

        # Training.
        "epochs": EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,

        # Evaluation.
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

        # CPU.
        "device": "cpu",

        # Reproducibility.
        "seed": SEED,

        # Checkpoints.
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "eval_step": 5,

        # Dataset.
        "data_path": str(TEMPORAL_DATASET_ROOT),
        "dataset": dataset_name,
        "model": "LightGCN",
    }

    config = Config(
        model="LightGCN",
        dataset=dataset_name,
        config_dict=config_dict,
    )

    return config


# ============================================================
# TRAIN ONE TEMPORAL LIGHTGCN
# ============================================================

def train_temporal_model(
    config,
    evaluation_year,
):
    print()
    print("=" * 70)
    print(
        f"Creating RecBole dataset: "
        f"{evaluation_year}"
    )
    print("=" * 70)

    dataset = create_dataset(config)
    print(dataset)

    print()
    print("=" * 70)
    print(
        "Preparing RecBole train/validation/test "
        "objects"
    )
    print("=" * 70)

    train_data, valid_data, test_data = (
        data_preparation(
            config,
            dataset,
        )
    )

    print(
        "RecBole train/validation/test objects "
        "created successfully."
    )

    print()
    print("=" * 70)
    print(
        f"Training LightGCN for "
        f"{evaluation_year}"
    )
    print("=" * 70)

    model = LightGCN(
        config,
        train_data.dataset,
    ).to(config["device"])

    trainer = Trainer(
        config,
        model,
    )

    start_time = time.time()

    best_valid_score, best_valid_result = (
        trainer.fit(
            train_data,
            valid_data,
        )
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 70)
    print(
        f"Training complete: "
        f"{evaluation_year}"
    )
    print("=" * 70)

    print(
        f"Training time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"Best validation score: "
        f"{best_valid_score}"
    )

    print(
        f"Best validation result: "
        f"{best_valid_result}"
    )

    # Save year-specific temporal checkpoint.
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_file = (
        CHECKPOINT_DIR
        / f"LightGCN-temporal-{evaluation_year}.pth"
    )

    torch.save(
        {
            "config": config,
            "state_dict": model.state_dict(),
        },
        checkpoint_file,
    )

    print()
    print(
        f"Temporal checkpoint saved:\n"
        f"{checkpoint_file}"
    )

    return model, dataset


# ============================================================
# GENERATE TEMPORAL TOP-10
# ============================================================

def generate_recommendations(
    model,
    dataset,
    historical_df,
    test_df,
    config,
    evaluation_year,
):
    print()
    print("=" * 70)
    print(
        f"Generating Top-10 recommendations: "
        f"{evaluation_year}"
    )
    print("=" * 70)

    model.eval()

    # RecBole fields.
    user_field = dataset.uid_field
    item_field = dataset.iid_field

    # RecBole token mappings.
    user_token_to_id = (
        dataset.field2token_id[user_field]
    )

    item_token_to_id = (
        dataset.field2token_id[item_field]
    )

    item_id_to_token = (
        dataset.field2id_token[item_field]
    )

    # Users with historical information AND
    # an interaction in the evaluation year.
    historical_users = set(
        historical_df["user_id"].unique()
    )

    evaluation_users = set(
        test_df["user_id"].unique()
    )

    overlapping_users = (
        historical_users
        .intersection(evaluation_users)
    )

    # RecBole may have removed users through
    # minimum interaction filtering.
    eligible_users = []

    for user_id in sorted(overlapping_users):
        if str(user_id) in user_token_to_id:
            eligible_users.append(user_id)

    print(
        f"{evaluation_year} test users: "
        f"{len(evaluation_users):,}"
    )

    print(
        f"Users with prior history: "
        f"{len(overlapping_users):,}"
    )

    print(
        f"Users retained by RecBole: "
        f"{len(eligible_users):,}"
    )

    # Historical catalogue:
    # only items known before the evaluation year.
    historical_catalogue = set(
        historical_df["item_id"].unique()
    )

    # Restrict to items represented by the model.
    model_catalogue_items = []

    for item_id in sorted(historical_catalogue):
        if str(item_id) in item_token_to_id:
            model_catalogue_items.append(item_id)

    print(
        f"Historical catalogue items: "
        f"{len(historical_catalogue):,}"
    )

    print(
        f"Items retained by RecBole: "
        f"{len(model_catalogue_items):,}"
    )

    # Map model item IDs to raw item IDs.
    model_catalogue_inner = np.array(
        [
            item_token_to_id[str(item_id)]
            for item_id in model_catalogue_items
        ],
        dtype=np.int64,
    )

    # Items already consumed by each user.
    rated_by_user = (
        historical_df
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    rows = []
    total_users = len(eligible_users)

    for counter, user_id in enumerate(
        eligible_users,
        start=1,
    ):
        user_token = str(user_id)
        user_inner = user_token_to_id[user_token]

        # Score the complete model catalogue for this user.
        # LightGCN provides full_sort_predict for all items.
        user_tensor = torch.tensor(
            [user_inner],
            dtype=torch.long,
            device=config["device"],
        )

        interaction = Interaction(
            {
                user_field: user_tensor,
            }
        ).to(config["device"])

        with torch.no_grad():
            scores = model.full_sort_predict(
                interaction
            )

        scores = scores.reshape(-1)

        # Keep only historical catalogue items.
        catalogue_inner_tensor = torch.tensor(
            model_catalogue_inner,
            dtype=torch.long,
            device=config["device"],
        )

        catalogue_scores = (
            scores[catalogue_inner_tensor]
            .clone()
        )

        # Remove previously consumed items.
        rated_items = rated_by_user.get(
            user_id,
            set(),
        )

        if rated_items:
            rated_inner = [
                item_token_to_id[str(item_id)]
                for item_id in rated_items
                if str(item_id) in item_token_to_id
            ]

            if rated_inner:
                rated_inner_tensor = torch.tensor(
                    rated_inner,
                    dtype=torch.long,
                    device=config["device"],
                )

                # Find those inner IDs inside the historical
                # catalogue and invalidate their scores.
                rated_mask = torch.isin(
                    catalogue_inner_tensor,
                    rated_inner_tensor,
                )

                catalogue_scores[rated_mask] = -float("inf")

        available_items = torch.isfinite(
            catalogue_scores
        )

        if not available_items.any():
            continue

        available_indices = torch.where(
            available_items
        )[0]

        top_n = min(
            TOP_K,
            len(available_indices),
        )

        top_local_indices = torch.topk(
            catalogue_scores[available_indices],
            k=top_n,
        ).indices

        top_indices = (
            available_indices[top_local_indices]
            .cpu()
            .numpy()
        )

        for rank, local_index in enumerate(
            top_indices,
            start=1,
        ):
            internal_item_id = int(
                model_catalogue_inner[local_index]
            )

            raw_item_id = item_id_to_token[
                internal_item_id
            ]

            score = scores[
                internal_item_id
            ].item()

            rows.append(
                {
                    "model": "LightGCN",
                    "year": evaluation_year,
                    "timestamp": f"{evaluation_year}-12-31",
                    "user_id": int(user_id),
                    "item_id": int(raw_item_id),
                    "rank": rank,
                    "score": float(score),
                }
            )

        if counter % 100 == 0:
            print(
                f"Processed "
                f"{counter:,}/"
                f"{total_users:,} "
                f"eligible users"
            )

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
        ],
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    yearly_file = (
        OUTPUT_DIR
        / (
            f"LightGCN-temporal-"
            f"{evaluation_year}-"
            f"top10-recommendations.csv"
        )
    )

    output_df.to_csv(
        yearly_file,
        index=False,
    )

    print()
    print("=" * 70)
    print(
        f"Recommendation generation complete: "
        f"{evaluation_year}"
    )
    print("=" * 70)

    print(
        f"Output file:\n"
        f"{yearly_file}"
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
            output_df.head(10).to_string(
                index=False
            )
        )
    else:
        print()
        print(
            "WARNING: No recommendations generated."
        )

    return output_df


# ============================================================
# RUN ONE YEAR
# ============================================================

def run_one_year(
    full_df,
    evaluation_year,
):
    year_start = time.time()

    print()
    print()
    print("#" * 70)
    print(
        f"STARTING TEMPORAL CHECKPOINT: "
        f"{evaluation_year}"
    )
    print("#" * 70)

    # Build rolling historical/evaluation datasets.
    historical_df, test_df = build_year_data(
        full_df,
        evaluation_year,
    )

    # Prepare RecBole historical dataset.
    dataset_name = prepare_recbole_dataset(
        historical_df,
        evaluation_year,
    )

    # Configuration.
    config = build_config(
        dataset_name
    )

    # Train LightGCN.
    model, dataset = train_temporal_model(
        config,
        evaluation_year,
    )

    # Generate Top-10 recommendations for
    # the evaluation year.
    recommendations = generate_recommendations(
        model,
        dataset,
        historical_df,
        test_df,
        config,
        evaluation_year,
    )

    elapsed = (
        time.time()
        - year_start
    )

    print()
    print("#" * 70)
    print(
        f"CHECKPOINT {evaluation_year} COMPLETE"
    )
    print(
        f"Total elapsed time: "
        f"{elapsed / 60:.2f} minutes"
    )
    print("#" * 70)

    return recommendations


# ============================================================
# CONSOLIDATE YEARLY OUTPUTS
# ============================================================

def consolidate_outputs(
    all_results,
):
    print()
    print("=" * 70)
    print(
        "Consolidating temporal recommendation outputs"
    )
    print("=" * 70)

    valid_results = [
        df
        for df in all_results
        if df is not None
        and len(df) > 0
    ]

    if not valid_results:
        print(
            "No recommendation outputs available "
            "for consolidation."
        )
        return pd.DataFrame()

    combined_df = pd.concat(
        valid_results,
        ignore_index=True,
    )

    combined_df = (
        combined_df
        .sort_values(
            [
                "year",
                "user_id",
                "rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_file = (
        OUTPUT_DIR
        / (
            f"LightGCN-temporal-"
            f"{START_YEAR}-"
            f"{END_YEAR}-"
            f"all-recommendations.csv"
        )
    )

    combined_df.to_csv(
        combined_file,
        index=False,
    )

    print(
        f"Combined output:\n"
        f"{combined_file}"
    )

    print(
        f"Total recommendation rows: "
        f"{len(combined_df):,}"
    )

    print(
        f"Years represented: "
        f"{combined_df['year'].nunique():,}"
    )

    print(
        f"Users represented: "
        f"{combined_df['user_id'].nunique():,}"
    )

    yearly_summary = (
        combined_df
        .groupby("year")
        .agg(
            recommendation_rows=(
                "item_id",
                "count",
            ),
            users=(
                "user_id",
                "nunique",
            ),
            mean_score=(
                "score",
                "mean",
            ),
        )
        .reset_index()
    )

    print()
    print(
        "Yearly recommendation summary:"
    )

    print(
        yearly_summary.to_string(
            index=False
        )
    )

    summary_file = (
        OUTPUT_DIR
        / (
            f"LightGCN-temporal-"
            f"{START_YEAR}-"
            f"{END_YEAR}-"
            f"summary.csv"
        )
    )

    yearly_summary.to_csv(
        summary_file,
        index=False,
    )

    print()
    print(
        f"Summary saved:\n"
        f"{summary_file}"
    )

    return combined_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print(
        "ROLLING TEMPORAL LIGHTGCN EXPERIMENT"
    )
    print("=" * 70)

    print(
        f"Evaluation years: "
        f"{START_YEAR}-{END_YEAR}"
    )

    print(
        f"Historical start year: "
        f"{HISTORICAL_START_YEAR}"
    )

    print(
        "Evaluation-year leakage prevention: "
        "ENABLED"
    )

    print(
        "Device: CPU"
    )

    print("=" * 70)

    # Reproducibility.
    set_seed(SEED)

    # Check source data.
    check_source_file()

    # Load source data once.
    full_df = load_source_data()

    # Validate requested years.
    available_years = set(
        full_df["year"].unique()
    )

    for year in range(
        START_YEAR,
        END_YEAR + 1,
        2,
    ):
        if year not in available_years:
            raise ValueError(
                f"Evaluation year {year} "
                f"is not present in the dataset."
            )

    # Run rolling checkpoints.
    all_results = []

    for evaluation_year in range(
        START_YEAR,
        END_YEAR + 1,
        2,
    ):
        checkpoint_file = (
            CHECKPOINT_DIR
            / f"LightGCN-temporal-{evaluation_year}.pth"
        )

        recommendations_file = (
            OUTPUT_DIR
            / (
                f"LightGCN-temporal-{evaluation_year}-"
                f"top10-recommendations.csv"
            )
        )

        if checkpoint_file.exists() and recommendations_file.exists():
            print()
            print(
                f"Skipping {evaluation_year}: "
                f"checkpoint and recommendations already exist."
            )
            continue

        result = run_one_year(
            full_df,
            evaluation_year,
        )

        all_results.append(result)

        # Reset random seeds between checkpoints.
        set_seed(SEED)

    # Consolidate outputs.
    combined_df = consolidate_outputs(
        all_results
    )

    # Final message.
    print()
    print("=" * 70)
    print(
        "ROLLING TEMPORAL LIGHTGCN EXPERIMENT COMPLETE"
    )
    print("=" * 70)

    print(
        f"Evaluation years completed: "
        f"{START_YEAR}-{END_YEAR}"
    )

    print(
        "Each evaluation year was kept outside "
        "its corresponding RecBole training dataset."
    )

    if len(combined_df) > 0:
        print(
            f"Total recommendations generated: "
            f"{len(combined_df):,}"
        )
