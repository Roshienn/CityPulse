# CITYPULSE

**Understanding Urban Air Quality Through Data Analysis.**

CITYPULSE is a Streamlit dashboard for the UCI Beijing Multi-Site Air Quality
dataset. It loads the 12 supplied station files without modifying them, retains
the source station, creates a validated `datetime`, interpolates missing
numeric observations, and writes `datasets/cleaned_citypulse.csv`.

## Features

- Overview of records, stations, PM2.5 distribution, monthly trends, and station comparison.
- Filterable data explorer with descriptive statistics and CSV export.
- Hourly, daily, monthly, station hotspot, and weather-correlation analysis.
- IQR anomaly detection with an exportable anomaly table, global or station-specific thresholds,
  anomaly percentages, and station counts.
- Identical serial and genuine `ProcessPoolExecutor` parallel aggregation.
- Configurable worker count and chunk size.
- Measured benchmark sizes (10,000, 50,000, 100,000, 200,000, and full data), speedup,
  efficiency, throughput, and output-correctness checks.

## Run

```bash
source venv/bin/activate
streamlit run app.py
```

The app uses Python 3.9-compatible code and the macOS-safe multiprocessing
`spawn` context. The first run creates the cleaned dataset; source CSVs remain
unchanged. For tests, install `pytest` in the existing environment if needed,
then run `PYTHONPATH=. python -m pytest -q`.

## Processing design

Both modes execute the same picklable chunk operation: PM2.5 statistics,
station aggregation, hourly/daily/monthly aggregation, and IQR anomaly
classification. Serial mode maps chunks in the current process. Parallel mode
maps those chunks to worker processes and combines partial sums/counts rather
than concatenating raw data in workers. This makes the comparison reproducible
and avoids simulated parallelism.

## Limitations

The dataset has no geographic coordinates, so “hotspots” means stations with
the highest observed average PM2.5, not a geographic heatmap. Interpolation is
limited to numeric observations within each station file. Wall-clock
benchmarks depend on machine load and multiprocessing startup overhead; the
dashboard reports measured values rather than fixed claims.

Overview, Explorer, Pollution Analysis, and Anomaly pages recalculate their
metrics and charts from the selected date and station filters. Benchmark
results are medians across the configured repeats after an optional warm-up;
the parallel wall-clock interval includes worker startup, serialization/data
transfer, computation, and result aggregation.
