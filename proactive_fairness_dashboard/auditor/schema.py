from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RecommendationRecord:
    model: str
    timestamp: str
    user_id: str
    item_id: str
    rank: int
    score: Optional[float] = None


REQUIRED_COLUMNS = {
    "model",
    "timestamp",
    "user_id",
    "item_id",
    "rank",
}


def validate_recommendation_columns(columns) -> None:
    columns = set(columns)

    missing = REQUIRED_COLUMNS - columns

    if missing:
        raise ValueError(
            f"Recommendation output is missing required columns: "
            f"{sorted(missing)}"
        )
