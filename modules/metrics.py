import numpy as np
import pandas as pd
import torch


def compute_gini_coefficient(counts: np.ndarray) -> float:
    """
    Calculates the Gini coefficient of item recommendation exposure.
    0.0 = perfect equality (all items recommended equally)
    1.0 = absolute inequality (one item gets all recommendations)
    """
    if len(counts) == 0 or np.sum(counts) == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * sorted_counts)) / (n * np.sum(sorted_counts)))


def compute_catalog_coverage(unique_recommended_count: int, total_catalog_size: int) -> float:
    """Calculates the percentage of catalog items recommended at least once."""
    if total_catalog_size == 0:
        return 0.0
    return float((unique_recommended_count / total_catalog_size) * 100.0)


def evaluate_topk_predictions(topk_matrix: np.ndarray, total_items: int = 5000) -> dict:
    """
    Evaluates item exposure distribution metrics from a Top-K predictions array.
    
    :param topk_matrix: 2D NumPy array of shape (n_users, k) containing recommended item IDs.
    :param total_items: Total number of items in the system catalog.
    :return: Dictionary containing key fairness metrics and exposure counts.
    """
    flat_preds = topk_matrix.flatten()
    unique_items, counts = np.unique(flat_preds, return_counts=True)
    
    # Calculate Gini Disparity
    gini = compute_gini_coefficient(counts)
    
    # Calculate Catalog Coverage
    coverage_pct = compute_catalog_coverage(len(unique_items), total_items)
    
    # Calculate Tail vs Head Ratio (Items in bottom 80% of recommendations)
    sorted_counts = np.sort(counts)[::-1]
    head_cutoff = max(1, int(len(sorted_counts) * 0.2))
    head_exposure = np.sum(sorted_counts[:head_cutoff])
    total_exposure = np.sum(sorted_counts)
    head_concentration = float(head_exposure / total_exposure) if total_exposure > 0 else 0.0
    
    return {
        "gini_index": round(gini, 4),
        "catalog_coverage_pct": round(coverage_pct, 2),
        "active_recommended_items": int(len(unique_items)),
        "total_catalog_items": total_items,
        "head_concentration_ratio": round(head_concentration, 4),
        "item_counts": counts,
        "unique_item_ids": unique_items
    }


def load_recbole_checkpoint(checkpoint_path: str) -> dict:
    """
    Utility helper to inspect state dictionary key metadata from saved RecBole .pth files.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch = checkpoint.get("epoch", "Unknown")
        state_dict_keys = list(checkpoint.get("state_dict", {}).keys())
        return {
            "epoch": epoch,
            "num_layers_logged": len(state_dict_keys),
            "status": "Loaded successfully"
        }
    except Exception as e:
        return {"status": f"Error loading checkpoint: {str(e)}"}evaluate_checkpoint_fairness = evaluate_topk_predictions 
