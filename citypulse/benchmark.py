"""Measured serial versus multiprocessing benchmarks."""

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .processing import run_parallel, run_serial


def benchmark(
    data: pd.DataFrame,
    sizes: Iterable[object] = (10000, 50000, 100000, 200000, "full"),
    workers: int = 2,
    chunk_size: int = 50000,
    output_path: Path = Path("datasets/benchmark_results.csv"),
) -> pd.DataFrame:
    rows: List[dict] = []
    for requested in sizes:
        size = len(data) if requested == "full" else min(int(requested), len(data))
        sample = data.head(size).copy()
        serial = run_serial(sample, chunk_size)
        parallel = run_parallel(sample, workers, chunk_size)
        speedup = serial.elapsed_seconds / parallel.elapsed_seconds if parallel.elapsed_seconds else 0.0
        efficiency = speedup / workers * 100
        serial_mean = serial.stats["mean"]
        parallel_mean = parallel.stats["mean"]
        rows.append({
            "requested_size": requested, "records": size, "workers": workers,
            "chunk_size": chunk_size, "serial_seconds": serial.elapsed_seconds,
            "parallel_seconds": parallel.elapsed_seconds, "speedup": speedup,
            "efficiency_percent": efficiency,
            "serial_throughput": size / serial.elapsed_seconds if serial.elapsed_seconds else 0.0,
            "parallel_throughput": size / parallel.elapsed_seconds if parallel.elapsed_seconds else 0.0,
            "outputs_match": abs(serial_mean - parallel.stats["mean"]) < 1e-9
                and serial.stats["anomaly_count"] == parallel.stats["anomaly_count"],
        })
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result

