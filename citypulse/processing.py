"""Shared serial/parallel aggregation operations."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from time import perf_counter
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class AnalysisResult:
    stats: Dict[str, float]
    station_summary: pd.DataFrame
    hourly_trend: pd.DataFrame
    daily_trend: pd.DataFrame
    monthly_trend: pd.DataFrame
    anomalies: pd.DataFrame
    records: int
    elapsed_seconds: float = 0.0


def analysis_results_equivalent(left: AnalysisResult, right: AnalysisResult) -> bool:
    """Compare all analytical outputs while ignoring wall-clock timing."""
    if left.records != right.records or set(left.stats) != set(right.stats):
        return False
    for key in left.stats:
        left_value, right_value = left.stats[key], right.stats[key]
        if pd.isna(left_value) or pd.isna(right_value):
            if not (pd.isna(left_value) and pd.isna(right_value)):
                return False
        elif isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            if abs(left_value - right_value) > 1e-9:
                return False
        elif left_value != right_value:
            return False
    for left_frame, right_frame in [
        (left.station_summary, right.station_summary),
        (left.hourly_trend, right.hourly_trend),
        (left.daily_trend, right.daily_trend),
        (left.monthly_trend, right.monthly_trend),
        (left.anomalies, right.anomalies),
    ]:
        try:
            pd.testing.assert_frame_equal(
                left_frame.reset_index(drop=True),
                right_frame.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-9,
                atol=1e-9,
            )
        except AssertionError:
            return False
    return True


def _ensure_datetime(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    if "datetime" not in frame:
        frame["datetime"] = pd.to_datetime(
            frame[["year", "month", "day", "hour"]], errors="coerce"
        )
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["date"] = frame["datetime"].dt.floor("D")
    frame["month_period"] = frame["datetime"].dt.to_period("M").astype(str)
    frame["hour_of_day"] = frame["datetime"].dt.hour
    return frame


def _group(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    result = frame.groupby(by, dropna=False)["PM2.5"].agg(["count", "sum", "mean"]).reset_index()
    return result


def _process_chunk(payload: Tuple[pd.DataFrame, float, float]) -> Dict[str, object]:
    data, lower, upper = payload
    frame = _ensure_datetime(data)
    pm = frame["PM2.5"]
    anomaly_mask = (pm < lower) | (pm > upper)
    station = frame.groupby("station")["PM2.5"].agg(["count", "sum", "mean", "max"]).reset_index()
    hourly = _group(frame, "hour_of_day")
    daily = _group(frame, "date")
    monthly = _group(frame, "month_period")
    anomaly_columns = [c for c in ["datetime", "station", "PM2.5", "PM10", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"] if c in frame]
    return {
        "count": int(pm.count()),
        "sum": float(pm.sum()),
        "sumsq": float((pm ** 2).sum()),
        "min": float(pm.min()),
        "max": float(pm.max()),
        "station": station,
        "hourly": hourly,
        "daily": daily,
        "monthly": monthly,
        "anomalies": frame.loc[anomaly_mask, anomaly_columns],
    }


def _combine_grouped(parts: List[Dict[str, object]], key: str, group_column: str) -> pd.DataFrame:
    combined = pd.concat([part[key] for part in parts], ignore_index=True)
    result = combined.groupby(group_column, dropna=False).agg(
        count=("count", "sum"), sum=("sum", "sum")
    ).reset_index()
    result["mean"] = result["sum"] / result["count"]
    return result.sort_values(group_column).reset_index(drop=True)


def _combine_station(parts: List[Dict[str, object]]) -> pd.DataFrame:
    combined = pd.concat([part["station"] for part in parts], ignore_index=True)
    result = combined.groupby("station", as_index=False).agg(
        count=("count", "sum"), sum=("sum", "sum"), max=("max", "max")
    )
    result["mean"] = result["sum"] / result["count"]
    return result.sort_values("station").reset_index(drop=True)


def _result_from_parts(parts: List[Dict[str, object]], elapsed: float, q1: float, q3: float) -> AnalysisResult:
    count = sum(int(part["count"]) for part in parts)
    total = sum(float(part["sum"]) for part in parts)
    sumsq = sum(float(part["sumsq"]) for part in parts)
    mean = total / count if count else 0.0
    variance = max(0.0, sumsq / count - mean ** 2) if count else 0.0
    anomalies = pd.concat([part["anomalies"] for part in parts], ignore_index=True)
    if not anomalies.empty:
        anomalies = anomalies.sort_values(["datetime", "station"]).reset_index(drop=True)
    return AnalysisResult(
        stats={
            "count": count, "mean": mean, "median": float("nan"),
            "std": variance ** 0.5, "min": min(float(part["min"]) for part in parts),
            "max": max(float(part["max"]) for part in parts), "q1": q1, "q3": q3,
            "lower_bound": q1 - 1.5 * (q3 - q1), "upper_bound": q3 + 1.5 * (q3 - q1),
            "anomaly_count": len(anomalies),
        },
        station_summary=_combine_station(parts),
        hourly_trend=_combine_grouped(parts, "hourly", "hour_of_day"),
        daily_trend=_combine_grouped(parts, "daily", "date"),
        monthly_trend=_combine_grouped(parts, "monthly", "month_period"),
        anomalies=anomalies,
        records=count,
        elapsed_seconds=elapsed,
    )


def _prepare(data: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    frame = _ensure_datetime(data)
    if "PM2.5_original_missing" in frame:
        frame = frame[~frame["PM2.5_original_missing"]]
    frame = frame.dropna(subset=["PM2.5"]).reset_index(drop=True)
    q1, q3 = frame["PM2.5"].quantile([0.25, 0.75]).tolist()
    return frame, float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1))


def _chunks(data: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    return [data.iloc[start:start + chunk_size].copy() for start in range(0, len(data), chunk_size)]


def chunk_count(records: int, chunk_size: int) -> int:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    return (records + chunk_size - 1) // chunk_size


def run_serial(data: pd.DataFrame, chunk_size: int = 50000) -> AnalysisResult:
    """Run the same chunk operation sequentially."""
    frame, lower, upper = _prepare(data)
    q1, q3 = frame["PM2.5"].quantile([0.25, 0.75]).tolist()
    started = perf_counter()
    parts = [_process_chunk((chunk, lower, upper)) for chunk in _chunks(frame, chunk_size)]
    result = _result_from_parts(parts, perf_counter() - started, float(q1), float(q3))
    result.stats["median"] = float(frame["PM2.5"].median())
    return result


def run_parallel(
    data: pd.DataFrame,
    workers: int = 2,
    chunk_size: int = 50000,
) -> AnalysisResult:
    """Run chunk operations in genuine worker processes using spawn on macOS."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    frame, lower, upper = _prepare(data)
    q1, q3 = frame["PM2.5"].quantile([0.25, 0.75]).tolist()
    started = perf_counter()
    payloads = [(chunk, lower, upper) for chunk in _chunks(frame, chunk_size)]
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        parts = list(executor.map(_process_chunk, payloads))
    result = _result_from_parts(parts, perf_counter() - started, float(q1), float(q3))
    result.stats["median"] = float(frame["PM2.5"].median())
    return result
