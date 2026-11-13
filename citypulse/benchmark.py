"""Measured serial versus multiprocessing benchmarks."""

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .processing import analysis_results_equivalent, run_parallel, run_serial


def benchmark(
    data: pd.DataFrame,
    sizes: Iterable[object] = (10000, 50000, 100000, 200000, "full"),
    workers: int = 2,
    chunk_size: int = 50000,
    output_path: Path = Path("datasets/benchmark_results.csv"),
    repeats: int = 1,
    warmup: bool = True,
) -> pd.DataFrame:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    rows: List[dict] = []
    for requested in sizes:
        size = len(data) if requested == "full" else min(int(requested), len(data))
        sample = data.head(size).copy()
        if warmup:
            run_serial(sample, chunk_size)
            run_parallel(sample, workers, chunk_size)
        serial_runs = []
        parallel_runs = []
        for repeat in range(repeats):
            if repeat % 2 == 0:
                serial_runs.append(run_serial(sample, chunk_size))
                parallel_runs.append(run_parallel(sample, workers, chunk_size))
            else:
                parallel_runs.append(run_parallel(sample, workers, chunk_size))
                serial_runs.append(run_serial(sample, chunk_size))
        serial = serial_runs[0]
        parallel = parallel_runs[0]
        serial_time = float(pd.Series([r.elapsed_seconds for r in serial_runs]).median())
        parallel_time = float(pd.Series([r.elapsed_seconds for r in parallel_runs]).median())
        speedup = serial_time / parallel_time if parallel_time else 0.0
        efficiency = speedup / workers * 100
        rows.append({
            "requested_size": requested, "records": size, "workers": workers,
            "chunk_size": chunk_size,             "serial_seconds": serial_time, "parallel_seconds": parallel_time,
            "repeats": repeats, "warmup": warmup, "speedup": speedup,
            "efficiency_percent": efficiency,
            "serial_throughput": size / serial_time if serial_time else 0.0,
            "parallel_throughput": size / parallel_time if parallel_time else 0.0,
            "outputs_match": analysis_results_equivalent(serial, parallel),
        })
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
