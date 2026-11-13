"""IQR-based PM2.5 anomaly detection."""

from typing import Dict, Tuple

import pandas as pd


def iqr_bounds(series: pd.Series) -> Tuple[float, float, float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    q1, q3 = values.quantile([0.25, 0.75]).tolist()
    iqr = q3 - q1
    return float(q1), float(q3), float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def detect_anomalies(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    q1, q3, lower, upper = iqr_bounds(data["PM2.5"])
    result = data[(data["PM2.5"] < lower) | (data["PM2.5"] > upper)].copy()
    return result.sort_values("datetime").reset_index(drop=True), {
        "q1": q1, "q3": q3, "lower_bound": lower, "upper_bound": upper,
    }


def detect_station_anomalies(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Classify readings using a separate IQR threshold for each station."""
    records = []
    bounds = []
    for station, group in data.groupby("station", sort=True):
        q1, q3, lower, upper = iqr_bounds(group["PM2.5"])
        flagged = group[(group["PM2.5"] < lower) | (group["PM2.5"] > upper)].copy()
        flagged["threshold_method"] = "station-specific IQR"
        records.append(flagged)
        bounds.append({
            "station": station, "q1": q1, "q3": q3,
            "lower_bound": lower, "upper_bound": upper,
            "anomaly_count": len(flagged), "records": len(group),
        })
    anomalies = pd.concat(records, ignore_index=True) if records else data.iloc[0:0].copy()
    return anomalies.sort_values(["datetime", "station"]).reset_index(drop=True), pd.DataFrame(bounds)
