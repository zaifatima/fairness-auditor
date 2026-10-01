import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class FairnessForecaster:
    """
    Predictive engine for longitudinal fairness monitoring and drift forecasting.
    Evaluates metric trajectories over time and predicts future exposure disparity.
    """

    def __init__(self, historical_logs: list[dict] = None):
        """
        :param historical_logs: List of dictionaries containing metric snapshots 
                                e.g., [{'epoch': 1, 'gini': 0.35, 'coverage': 42.1}, ...]
        """
        self.df_log = pd.DataFrame(historical_logs) if historical_logs else pd.DataFrame()

    def load_metric_history(self, log_data: list[dict]):
        """Populates or updates the historical metric log dataframe."""
        self.df_log = pd.DataFrame(log_data)

    def forecast_gini_drift(self, horizon: int = 7, deg: int = 1) -> pd.DataFrame:
        """
        Fits a time-series trend model over observed Gini disparity scores 
        and projects disparity metrics for future evaluation cycles.

        :param horizon: Number of recommendation cycles/epochs to project into the future.
        :param deg: Polynomial degree for trend fitting (1 = linear, 2 = quadratic curve).
        :return: DataFrame containing historical + forecasted Gini trajectories.
        """
        if self.df_log.empty or "epoch" not in self.df_log.columns or "gini" not in self.df_log.columns:
            # Return empty frame if logging data is insufficient
            return pd.DataFrame(columns=["epoch", "gini", "type"])

        observed_epochs = self.df_log["epoch"].values
        observed_gini = self.df_log["gini"].values

        # Fit trend model across available epochs
        poly_weights = np.polyfit(observed_epochs, observed_gini, deg=deg)

        # Generate future time steps
        last_epoch = int(observed_epochs[-1])
        future_epochs = np.arange(last_epoch + 1, last_epoch + 1 + horizon)
        projected_gini = np.polyval(poly_weights, future_epochs)

        # Combine observed and predicted sequences into unified structure
        historical_df = pd.DataFrame({
            "epoch": observed_epochs,
            "gini": observed_gini,
            "type": "Observed"
        })

        forecast_df = pd.DataFrame({
            "epoch": future_epochs,
            "gini": np.clip(projected_gini, 0.0, 1.0),  # Keep within valid Gini bounds [0, 1]
            "type": "Projected Drift"
        })

        return pd.concat([historical_df, forecast_df], ignore_index=True)

    def detect_distributional_drift(self, baseline_counts: np.ndarray, current_counts: np.ndarray) -> dict:
        """
        Performs a two-sample Kolmogorov-Smirnov (KS) test to detect statistical drift 
        in item exposure distribution between baseline and current evaluation batches.
        """
        if len(baseline_counts) == 0 or len(current_counts) == 0:
            return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False}

        ks_stat, p_val = ks_2samp(baseline_counts, current_counts)

        # Drift detected if p-value is below standard significance threshold (alpha = 0.05)
        return {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 5),
            "drift_detected": bool(p_val < 0.05)
        }

    def generate_intervention_advisory(self, projected_gini: float, threshold: float = 0.45) -> list[dict]:
        """
        Generates actionable mitigation steps for non-technical operators when 
        predicted Gini disparity breaches predefined tolerance bounds.
        """
        interventions = []

        if projected_gini > threshold:
            drift_delta = round(projected_gini - threshold, 3)
            interventions.append({
                "severity": "HIGH",
                "affected_component": "Item Popularity Bias (Head vs. Tail)",
                "predicted_issue": f"Gini exposure disparity exceeds threshold α ({threshold}) by +{drift_delta}",
                "action_recommended": "Inject provider exploration penalty; boost tail-item exposure weights in top-K ranking."
            })
            interventions.append({
                "severity": "MEDIUM",
                "affected_component": "Catalog Coverage Decay",
                "predicted_issue": "Long-tail recommendation suppression predicted in upcoming cycles.",
                "action_recommended": "Apply post-hoc re-ranking calibrated for equal item opportunity."
            })
        else:
            interventions.append({
                "severity": "NONE",
                "affected_component": "System Overall",
                "predicted_issue": "Disparity metrics remain within acceptable operational bounds.",
                "action_recommended": "No immediate intervention required. Continue background monitoring."
            })

        return interventions