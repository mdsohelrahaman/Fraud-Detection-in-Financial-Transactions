"""Convert fraud scores into actionable alerts."""
from typing import Iterable

import numpy as np
import pandas as pd


def score_to_severity(score: float) -> str:
    if score >= 0.90:
        return "CRITICAL"
    if score >= 0.75:
        return "HIGH"
    if score >= 0.50:
        return "MEDIUM"
    return "LOW"


def build_alerts(
    df: pd.DataFrame,
    scores: Iterable[float],
    score_column: str = "fraud_score",
) -> pd.DataFrame:
    """Attach normalized scores, severity, and review status to transactions."""
    alerts = df.copy()
    values = np.asarray(list(scores), dtype=float)
    if len(alerts) != len(values):
        raise ValueError("Number of scores must match number of transactions.")
    alerts[score_column] = np.clip(values, 0, 1)
    alerts["alert_level"] = alerts[score_column].map(score_to_severity)
    alerts["review_required"] = alerts["alert_level"].isin(["MEDIUM", "HIGH", "CRITICAL"])
    return alerts.sort_values(score_column, ascending=False)
