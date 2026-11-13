from pathlib import Path

import pandas as pd

from citypulse.anomaly import detect_station_anomalies, iqr_bounds
from citypulse.benchmark import benchmark
from citypulse.data_loader import discover_files, load_dataset
from citypulse.processing import run_parallel, run_serial


def test_loads_all_station_files_and_retains_station(tmp_path):
    source = Path("datasets")
    assert len(discover_files(source)) == 12
    data = load_dataset(source, tmp_path / "cleaned.csv", save_cleaned=True)
    assert len(data) == 420768
    assert data["station"].nunique() == 12
    assert pd.api.types.is_datetime64_any_dtype(data["datetime"])
    assert data["PM2.5"].isna().sum() == 0
    assert (tmp_path / "cleaned.csv").exists()


def test_serial_and_parallel_outputs_match():
    data = load_dataset(Path("datasets"), save_cleaned=False).head(2000)
    serial = run_serial(data, 500)
    parallel = run_parallel(data, workers=2, chunk_size=500)
    assert serial.stats["count"] == parallel.stats["count"]
    assert serial.stats["anomaly_count"] == parallel.stats["anomaly_count"]
    pd.testing.assert_frame_equal(serial.station_summary, parallel.station_summary)
    pd.testing.assert_frame_equal(serial.hourly_trend, parallel.hourly_trend)


def test_iqr_bounds_and_benchmark_formulas(tmp_path):
    series = pd.Series([1, 2, 2, 3, 100])
    q1, q3, lower, upper = iqr_bounds(series)
    assert q1 < q3 and upper > q3
    data = load_dataset(Path("datasets"), save_cleaned=False).head(1000)
    result = benchmark(data, sizes=(1000,), workers=2, chunk_size=250, output_path=tmp_path / "bench.csv")
    row = result.iloc[0]
    assert row["speedup"] == row["serial_seconds"] / row["parallel_seconds"]
    assert row["efficiency_percent"] == row["speedup"] / row["workers"] * 100
    assert (tmp_path / "bench.csv").exists()


def test_station_specific_anomalies_return_bounds():
    data = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=8, freq="h"),
        "station": ["A"] * 4 + ["B"] * 4,
        "PM2.5": [1, 2, 2, 30, 10, 11, 12, 13],
    })
    anomalies, bounds = detect_station_anomalies(data)
    assert set(bounds["station"]) == {"A", "B"}
    assert "anomaly_count" in bounds
    assert set(anomalies["threshold_method"]) == {"station-specific IQR"}
