"""CITYPULSE Streamlit dashboard."""

from pathlib import Path
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from citypulse.anomaly import detect_anomalies, detect_station_anomalies
from citypulse.benchmark import benchmark
from citypulse.data_loader import load_or_create_cleaned
from citypulse.processing import chunk_count, run_parallel, run_serial
from citypulse.visualizations import benchmark_chart, distribution_chart, station_chart, trend_chart

st.set_page_config(page_title="CITYPULSE", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: #0b1f33; }
[data-testid="stSidebar"] * { color: #e2f5f2 !important; }
.hero { background: linear-gradient(120deg,#0b1f33,#123e52); color:white; padding:2rem 2.2rem; border-radius:18px; margin-bottom:1.4rem; }
.hero h1 { margin:0; color:#5eead4; letter-spacing:.08em; font-weight:700; }
.hero p { margin:.4rem 0 0; color:#c7e9e5; }
[data-testid="stMetric"] { background:#ffffff; border:1px solid #dcebe9; padding:1rem; border-radius:14px; box-shadow:0 3px 14px #0b1f3310; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color:#41616d !important; font-weight:600 !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color:#0b1f33 !important; font-weight:700 !important; }
[data-testid="stMetricDelta"] { color:#0f766e !important; }
.section-title { color:#0f766e; font-weight:700; font-size:1.2rem; margin-top:.8rem; }
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading and cleaning station data...")
def get_data():
    return load_or_create_cleaned(Path(__file__).parent / "datasets")


@st.cache_data(show_spinner=False)
def cached_serial(data, chunk_size):
    return run_serial(data, chunk_size)


def fmt(value):
    return f"{value:,.1f}" if isinstance(value, float) else f"{value:,}"


def date_filter(frame, key):
    if frame.empty:
        return frame
    minimum = frame["datetime"].min().date()
    maximum = frame["datetime"].max().date()
    selected = st.date_input("Date range", (minimum, maximum), key=key)
    if not isinstance(selected, tuple) or len(selected) != 2:
        return frame.iloc[0:0]
    start, end = selected
    return frame[frame["datetime"].dt.date.between(start, end)]


def display_stats(result):
    stats = result.stats
    cards = st.columns(4)
    cards[0].metric("Records processed", f"{result.records:,}")
    cards[1].metric("Average PM2.5", f"{stats['mean']:.1f} µg/m³")
    cards[2].metric("Maximum PM2.5", f"{stats['max']:.1f} µg/m³")
    cards[3].metric("IQR anomalies", f"{stats['anomaly_count']:,}")


def valid_observations(frame, column):
    marker = f"{column}_original_missing"
    if marker in frame:
        frame = frame[~frame[marker]]
    return frame.dropna(subset=[column])


data = get_data()
POLLUTANTS = [column for column in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"] if column in data]
WEATHER = [column for column in ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"] if column in data]
IDENTIFIERS = {"No", "year", "month", "day", "hour", "datetime", "station"}

with st.sidebar:
    st.markdown("## 🌿 CITYPULSE")
    st.caption("Understanding Urban Air Quality Through Data Analysis.")
    page = st.radio("Navigate", [
        "Overview", "Data Explorer", "Pollution Analysis", "Anomaly Detection",
        "Serial Processing", "Parallel Processing", "Performance Comparison",
    ])
    st.divider()
    st.caption(f"{len(data):,} cleaned records · {data['station'].nunique()} stations")

st.markdown('<div class="hero"><h1>CITYPULSE</h1><p>Understanding Urban Air Quality Through Data Analysis.</p></div>', unsafe_allow_html=True)

if page == "Overview":
    filtered = date_filter(valid_observations(data, "PM2.5"), "overview_dates")
    if filtered.empty:
        st.warning("No records fall within this date range.")
        st.stop()
    result = cached_serial(filtered, 50000)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(filtered):,}")
    c2.metric("Average PM2.5", f"{filtered['PM2.5'].mean():.1f} µg/m³")
    c3.metric("Maximum PM2.5", f"{filtered['PM2.5'].max():.1f} µg/m³")
    highest = filtered.groupby("station")["PM2.5"].mean().idxmax()
    c4.metric("Highest-pollution station", highest)
    left, right = st.columns([1.5, 1])
    with left:
        st.caption("Monthly average PM2.5 reveals seasonal patterns and sustained changes.")
        st.plotly_chart(trend_chart(result.monthly_trend, "month_period", "Monthly pollution trend"), use_container_width=True)
    with right:
        st.caption("The distribution shows the frequency of observed PM2.5 concentrations.")
        st.plotly_chart(distribution_chart(filtered), use_container_width=True)
    st.plotly_chart(station_chart(result.station_summary), use_container_width=True)
    st.subheader("High-pollution readings")
    high = filtered[filtered["PM2.5"] >= 75].sort_values("PM2.5", ascending=False)
    st.caption("Readings at or above 75 µg/m³ are shown as a practical screening threshold, not a confirmed incident.")
    st.dataframe(high[["datetime", "station", "PM2.5"]].head(20), use_container_width=True)

elif page == "Data Explorer":
    st.markdown('<div class="section-title">Explore the cleaned master dataset</div>', unsafe_allow_html=True)
    stations = st.multiselect("Stations", sorted(data.station.unique()), default=sorted(data.station.unique()))
    filtered = date_filter(data[data.station.isin(stations)], "explorer_dates")
    preview_columns = st.multiselect(
        "Preview columns", list(data.columns), default=[c for c in data.columns if c not in {"No"}][:10]
    )
    if not preview_columns:
        st.warning("Select at least one preview column.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    c1.metric("Filtered records", f"{len(filtered):,}")
    missing_total = int(filtered.isna().sum().sum())
    c2.metric("Missing values", f"{missing_total:,}")
    c3.metric("Missing percentage", f"{missing_total / max(filtered.size, 1) * 100:.2f}%")
    st.dataframe(filtered[preview_columns].head(1000), use_container_width=True, height=400)
    st.download_button("Download filtered CSV", filtered.to_csv(index=False), "citypulse_filtered.csv", "text/csv")
    st.caption("The loader creates datetime, converts numeric fields, and interpolates missing numeric values within each station. Original CSV files are never changed.")
    missing = pd.DataFrame({
        "missing_count": filtered.isna().sum(),
        "missing_percent": filtered.isna().mean().mul(100),
    }).sort_values("missing_count", ascending=False)
    st.subheader("Missing-value breakdown")
    st.dataframe(missing, use_container_width=True)
    original_missing_columns = [column for column in filtered.columns if column.endswith("_original_missing")]
    if original_missing_columns:
        original_missing = filtered[original_missing_columns].sum().rename(
            lambda column: column.removesuffix("_original_missing")
        ).to_frame("original_missing_count")
        st.subheader("Original missing-value sentinel counts")
        st.caption("Original missing values include blank/NA values and the dataset's 999 sentinel. Imputed values remain visible for exploration but are excluded from PM2.5 processing and anomaly analysis.")
        st.dataframe(original_missing, use_container_width=True)
    summary_columns = [c for c in filtered.select_dtypes(include="number").columns if c not in IDENTIFIERS]
    st.subheader("Descriptive statistics")
    st.dataframe(filtered[summary_columns].describe().T, use_container_width=True)

elif page == "Pollution Analysis":
    selected_pollutant = st.selectbox("Pollutant", POLLUTANTS)
    stations = st.multiselect("Stations", sorted(data.station.unique()), default=sorted(data.station.unique()), key="analysis_stations")
    filtered = date_filter(valid_observations(data[data.station.isin(stations)], selected_pollutant), "analysis_dates")
    if filtered.empty:
        st.warning("No records match the selected filters.")
        st.stop()
    daily = filtered.set_index("datetime")[selected_pollutant].resample("D").mean().rename("mean").reset_index()
    monthly = filtered.set_index("datetime")[selected_pollutant].resample("MS").mean().rename("mean").reset_index()
    yearly = filtered.set_index("datetime")[selected_pollutant].resample("YS").mean().rename("mean").reset_index()
    daily["date"] = daily["datetime"].dt.date
    monthly["month"] = monthly["datetime"].dt.strftime("%Y-%m")
    yearly["year"] = yearly["datetime"].dt.year
    tabs = st.tabs(["Trends", "Stations and hotspots", "Correlation"])
    with tabs[0]:
        for frame, x, title in [(daily, "date", "Daily trend"), (monthly, "month", "Monthly trend"), (yearly, "year", "Yearly trend")]:
            st.plotly_chart(px.line(frame, x=x, y="mean", title=f"{title}: {selected_pollutant}", markers=True,
                                    labels={"mean": f"Average {selected_pollutant}", x: ""}), use_container_width=True)
        st.caption("Daily, monthly, and yearly averages are recalculated from the active station and date filters.")
    with tabs[1]:
        station = filtered.groupby("station")[selected_pollutant].agg(
            average="mean", maximum="max"
        ).reset_index().sort_values("average", ascending=False)
        high_threshold = filtered[selected_pollutant].quantile(0.95)
        high_counts = filtered.groupby("station")[selected_pollutant].apply(
            lambda values: (values >= high_threshold).sum()
        ).rename("high_readings").reset_index()
        station = station.merge(high_counts, on="station", how="left")
        st.plotly_chart(px.bar(station, x="average", y="station", orientation="h", color="average",
                               title=f"{selected_pollutant} comparison by station", color_continuous_scale="Teal"),
                        use_container_width=True)
        st.subheader("Top five potential hotspots")
        st.dataframe(station.head(5), use_container_width=True)
        st.caption(f"High readings are defined as the selected pollutant's 95th percentile ({high_threshold:.2f}); hotspots are station rankings, not geographic incident maps.")
    with tabs[2]:
        correlation_columns = [c for c in POLLUTANTS + WEATHER if c in filtered.columns]
        correlation = filtered[correlation_columns].corr(numeric_only=True)
        st.plotly_chart(px.imshow(correlation, text_auto=".2f", color_continuous_scale="Teal",
                                  zmin=-1, zmax=1, title="Pollutant and weather correlation heatmap"),
                        use_container_width=True)
        st.caption("Correlation indicates linear association, not causation.")

elif page == "Anomaly Detection":
    filtered = date_filter(valid_observations(data, "PM2.5"), "anomaly_dates")
    method = st.radio("Threshold method", ["Global IQR", "Station-specific IQR"], horizontal=True)
    if filtered.empty:
        st.warning("No records match the selected date range.")
        st.stop()
    if method == "Global IQR":
        anomalies, bounds = detect_anomalies(filtered)
        station_counts = anomalies.groupby("station").size().rename("anomaly_count").reset_index()
    else:
        anomalies, station_bounds = detect_station_anomalies(filtered)
        station_counts = anomalies.groupby("station").size().rename("anomaly_count").reset_index()
    st.subheader("IQR-based PM2.5 anomalies")
    c1, c2, c3 = st.columns(3)
    c1.metric("Anomalies", f"{len(anomalies):,}")
    c2.metric("Anomaly percentage", f"{len(anomalies) / len(filtered) * 100:.2f}%")
    if method == "Global IQR":
        c3.metric("Bounds", f"{bounds['lower_bound']:.1f} – {bounds['upper_bound']:.1f}")
    else:
        c3.metric("Eligible stations", f"{int(station_bounds['eligible'].sum())}/{len(station_bounds)}")
        st.dataframe(
            station_bounds[["station", "records", "eligible", "lower_bound", "upper_bound", "anomaly_count"]],
            use_container_width=True,
        )
    st.caption("IQR anomalies are statistically unusual readings relative to the selected population. They are not confirmed pollution incidents.")
    st.plotly_chart(px.bar(station_counts.sort_values("anomaly_count", ascending=False), x="station", y="anomaly_count",
                           title="Anomaly count by station", color="anomaly_count", color_continuous_scale="Teal"),
                    use_container_width=True)
    st.dataframe(anomalies.sort_values("PM2.5", ascending=False), use_container_width=True, height=500)
    st.download_button("Export anomalies CSV", anomalies.to_csv(index=False), "citypulse_anomalies.csv", "text/csv")

elif page == "Serial Processing":
    chunk = st.number_input("Chunk size", 1000, len(data), 50000, step=1000)
    st.caption("Sequential mode processes each chunk in the current Python process, one after another.")
    if st.button("Run serial analysis", type="primary"):
        result = run_serial(data, int(chunk))
        st.success(f"Completed in {result.elapsed_seconds:.3f} seconds")
        display_stats(result)
        st.caption(f"Configuration: chunk size {int(chunk):,} · {chunk_count(result.records, int(chunk))} chunks")
        st.dataframe(result.station_summary, use_container_width=True)

elif page == "Parallel Processing":
    supported_workers = [worker for worker in [1, 2, 4, 8] if worker <= (os.cpu_count() or 2)]
    workers = st.selectbox("Worker processes", supported_workers, index=min(1, len(supported_workers) - 1))
    chunk = st.number_input("Chunk size", 1000, len(data), 50000, step=1000)
    st.caption("Uses genuine ProcessPoolExecutor workers with macOS-safe spawn. Timing includes process startup, data transfer, computation, and result aggregation.")
    if st.button("Run parallel analysis", type="primary"):
        with st.spinner("Running worker processes..."):
            result = run_parallel(data, int(workers), int(chunk))
        st.success(f"Completed in {result.elapsed_seconds:.3f} seconds")
        display_stats(result)
        st.caption(f"Configuration: {workers} workers · chunk size {int(chunk):,} · {chunk_count(result.records, int(chunk))} chunks")
        st.info("Chunks are distributed through the process pool; smaller chunks can improve balancing but increase transfer and scheduling overhead.")
        st.dataframe(result.station_summary, use_container_width=True)

elif page == "Performance Comparison":
    supported_workers = [worker for worker in [1, 2, 4, 8] if worker <= (os.cpu_count() or 2)]
    workers = st.selectbox("Benchmark workers", supported_workers, index=min(1, len(supported_workers) - 1))
    chunk = st.number_input("Benchmark chunk size", 1000, len(data), 50000, step=1000)
    repeats = st.number_input("Measured repeats", 1, 5, 3)
    warmup = st.checkbox("Run warm-up before measurement", value=True)
    st.caption("Reported times are medians of repeated end-to-end runs. Warm-up reduces first-run effects; startup, transfer, computation, and aggregation remain inside parallel timing.")
    if st.button("Run benchmark suite", type="primary"):
        with st.spinner("Measuring actual serial and parallel execution..."):
            result = benchmark(data, workers=int(workers), chunk_size=int(chunk), repeats=int(repeats), warmup=warmup)
        run_number = int(st.session_state.get("benchmark_run_number", 0)) + 1
        st.session_state["benchmark_run_number"] = run_number
        result["run_label"] = f"Run {run_number}: {workers} workers / {int(chunk):,} chunk"
        history = st.session_state.get("benchmark_history")
        st.session_state["benchmark_history"] = result if history is None else pd.concat([history, result], ignore_index=True)
        st.session_state["benchmark"] = result
    result = st.session_state.get("benchmark")
    history = st.session_state.get("benchmark_history")
    if result is not None and history is not None and not history.empty:
        measured = history.copy()
        measured["time_difference_seconds"] = measured["serial_seconds"] - measured["parallel_seconds"]
        measured["time_difference_percent"] = measured["time_difference_seconds"] / measured["serial_seconds"].replace(0, pd.NA) * 100
        measured["parallel_faster"] = measured["speedup"] > 1
        measured["configuration"] = (
            measured["workers"].astype(str) + " workers / " +
            measured["chunk_size"].map(lambda value: f"{int(value):,}") + " chunk"
        )
        measured["dataset_label"] = measured.apply(
            lambda row: "Full" if row["requested_size"] == "full"
            else f"{int(row['records']) // 1000}k",
            axis=1,
        )
        dataset_order = ["10k", "50k", "100k", "200k", "Full"]
        fastest_parallel_row = measured.loc[measured["parallel_seconds"].idxmin()]
        highest_speedup_row = measured.loc[measured["speedup"].idxmax()]
        highest_efficiency_row = measured.loc[measured["efficiency_percent"].idxmax()]
        fastest_serial_row = measured.loc[measured["serial_seconds"].idxmin()]
        fastest_mode = "parallel" if fastest_parallel_row["parallel_seconds"] < fastest_serial_row["serial_seconds"] else "serial"
        summary = st.container()
        with summary:
            st.subheader("Measured performance summary")
            st.caption("Every comparison below uses recorded benchmark rows only. A speedup above 1 means parallel was faster; below 1 means serial was faster.")
            summary_cards = st.columns(4)
            summary_cards[0].metric("Fastest parallel execution", f"{fastest_parallel_row['parallel_seconds']:.3f}s",
                                    f"{int(fastest_parallel_row['records']):,} records · {fastest_parallel_row['configuration']}")
            summary_cards[1].metric("Highest measured speedup", f"{highest_speedup_row['speedup']:.2f}×",
                                    f"{int(highest_speedup_row['records']):,} records · {highest_speedup_row['configuration']}")
            summary_cards[2].metric("Highest parallel efficiency", f"{highest_efficiency_row['efficiency_percent']:.2f}%",
                                    f"{int(highest_efficiency_row['records']):,} records · {highest_efficiency_row['configuration']}")
            summary_cards[3].metric("Fastest mode", fastest_mode.title(),
                                    f"{fastest_serial_row['serial_seconds']:.3f}s serial vs {fastest_parallel_row['parallel_seconds']:.3f}s parallel")
            comparison_table = measured[[
                "requested_size", "dataset_label", "records", "configuration", "serial_seconds", "parallel_seconds",
                "time_difference_seconds", "time_difference_percent", "speedup",
                "efficiency_percent", "serial_throughput", "parallel_throughput",
                "outputs_match", "run_label",
            ]].sort_values(["records", "parallel_seconds"])
            st.dataframe(comparison_table, use_container_width=True)
            if highest_speedup_row["speedup"] > 1:
                st.success(f"Parallel processing achieved a measured speedup of {highest_speedup_row['speedup']:.2f}× for {int(highest_speedup_row['records']):,} records.")
            else:
                st.warning("Parallel processing did not achieve a measured speedup in the displayed results; serial is faster for every recorded comparison.")

        chart1, chart2, chart3, chart4 = st.tabs(["Execution time", "Speedup", "Efficiency", "Configuration comparisons"])
        with chart1:
            run_options = measured["run_label"].drop_duplicates().tolist()
            selected_run = st.selectbox("Dataset-size run", run_options, key="benchmark_size_run")
            size_result = measured[measured["run_label"] == selected_run].sort_values("records")
            st.plotly_chart(benchmark_chart(size_result), use_container_width=True)
            st.caption("This chart shows how serial and parallel wall-clock time scale with records for one measured worker/chunk configuration.")
            if len(size_result) > 1:
                grouped = size_result.melt(id_vars=["dataset_label"], value_vars=["serial_seconds", "parallel_seconds"],
                                           var_name="mode", value_name="seconds")
                grouped["dataset_label"] = pd.Categorical(grouped["dataset_label"], categories=dataset_order, ordered=True)
                st.plotly_chart(px.bar(grouped.sort_values("dataset_label"), x="dataset_label", y="seconds", color="mode", barmode="group",
                                       title="Grouped serial vs parallel execution time",
                                       category_orders={"dataset_label": dataset_order},
                                       labels={"dataset_label": "Dataset size", "seconds": "Execution time (seconds)", "mode": "Mode"}),
                                use_container_width=True)
                st.caption("Grouped bars make the absolute time difference visible at each equivalent dataset size.")
        with chart2:
            speedup = measured.sort_values("records")
            st.plotly_chart(px.line(speedup, x="records", y="speedup", color="configuration",
                                    markers=True, title="Measured speedup across dataset sizes and configurations",
                                    labels={"records": "Records", "speedup": "Speedup (serial / parallel)", "configuration": "Configuration"},
                                    hover_data=["workers", "chunk_size", "run_label"]),
                            use_container_width=True)
            st.caption("Values above 1.0 indicate parallel is faster; values below 1.0 indicate multiprocessing overhead outweighed its computation benefit.")
        with chart3:
            st.plotly_chart(px.line(measured.sort_values("records"), x="records", y="efficiency_percent",
                                    color="configuration", markers=True, title="Parallel efficiency",
                                    labels={"records": "Records", "efficiency_percent": "Efficiency (%)", "configuration": "Configuration"},
                                    hover_data=["workers", "chunk_size"]),
                            use_container_width=True)
            st.caption("Efficiency is speedup divided by worker count. Higher values indicate better worker utilization; values above 100% are not expected for this definition.")
        with chart4:
            st.markdown("**Configuration comparisons vary one setting at a time.** Dataset size and the other processing setting are held constant.")
            comparison_target = st.selectbox("Equivalent workload records", sorted(measured["records"].unique()), key="benchmark_comparison_records")
            equivalent = measured[measured["records"] == comparison_target].copy()
            worker_data = equivalent.groupby(["workers", "chunk_size"], as_index=False).agg(
                parallel_seconds=("parallel_seconds", "median"),
                run_count=("run_label", "nunique"),
            )
            worker_chunk_counts = equivalent.groupby("chunk_size")["workers"].nunique()
            comparable_worker_chunks = worker_chunk_counts[worker_chunk_counts > 1].index.tolist()
            if comparable_worker_chunks:
                selected_worker_chunk = st.selectbox(
                    "Chunk size for worker comparison",
                    sorted(comparable_worker_chunks),
                    format_func=lambda value: f"{int(value):,} records",
                    key="benchmark_worker_chunk",
                )
                worker_comparison = worker_data[worker_data["chunk_size"] == selected_worker_chunk].copy()
                st.plotly_chart(px.line(worker_comparison, x="workers", y="parallel_seconds", markers=True,
                                        title=f"Worker count vs parallel time ({int(comparison_target):,} records, {int(selected_worker_chunk):,} chunk)",
                                        labels={"workers": "Worker processes", "parallel_seconds": "Parallel time (seconds)"},
                                        hover_data=["chunk_size", "run_count"]), use_container_width=True)
                st.caption("Worker-count comparison holds dataset size and chunk size constant; only worker count varies.")
            else:
                tested_chunks = sorted(equivalent["chunk_size"].unique())
                st.info(
                    "Worker-count comparison unavailable for this dataset size. "
                    f"Benchmark at least two worker counts with the same chunk size "
                    f"(currently tested chunk sizes: {', '.join(f'{int(value):,}' for value in tested_chunks)})."
                )

            chunk_data = equivalent.groupby(["chunk_size", "workers"], as_index=False).agg(
                parallel_seconds=("parallel_seconds", "median"),
                run_count=("run_label", "nunique"),
            ).sort_values("chunk_size")
            comparable_chunk_workers = equivalent.groupby("workers")["chunk_size"].nunique()
            comparable_chunk_workers = comparable_chunk_workers[comparable_chunk_workers > 1].index.tolist()
            if comparable_chunk_workers:
                selected_chunk_workers = st.selectbox(
                    "Worker count for chunk comparison",
                    sorted(comparable_chunk_workers),
                    key="benchmark_chunk_workers",
                )
                chunk_comparison = chunk_data[chunk_data["workers"] == selected_chunk_workers].copy()
                st.plotly_chart(px.line(chunk_comparison, x="chunk_size", y="parallel_seconds", markers=True,
                                        title=f"Chunk size vs parallel time ({int(comparison_target):,} records, {int(selected_chunk_workers)} workers)",
                                        labels={"chunk_size": "Chunk size (records)", "parallel_seconds": "Parallel time (seconds)"},
                                        hover_data=["workers", "run_count"]), use_container_width=True)
                max_tested_chunk = int(chunk_comparison["chunk_size"].max())
                if comparison_target < max_tested_chunk:
                    st.warning(
                        f"This dataset has {int(comparison_target):,} records, smaller than at least one tested "
                        f"chunk size ({max_tested_chunk:,}). Those settings create only one chunk, so the comparison "
                        "does not measure multi-chunk scheduling behavior."
                    )
                else:
                    st.caption("Chunk-size comparison holds dataset size and worker count constant; only chunk size varies.")
            else:
                tested_workers = sorted(equivalent["workers"].unique())
                st.info(
                    "Chunk-size comparison unavailable for this dataset size. "
                    f"Benchmark at least two chunk sizes with the same worker count "
                    f"(currently tested worker counts: {', '.join(str(int(value)) for value in tested_workers)})."
                )
            if not equivalent.empty:
                best_config = equivalent.loc[equivalent["parallel_seconds"].idxmin()]
                st.metric("Fastest parallel configuration for selected workload",
                          f"{int(best_config['workers'])} workers / {int(best_config['chunk_size']):,} chunk",
                          f"{best_config['parallel_seconds']:.3f}s parallel")
        observations = []
        if (measured["parallel_faster"]).any():
            better = measured[measured["parallel_faster"]].sort_values("speedup", ascending=False).iloc[0]
            observations.append(f"Parallel performed best at {better['speedup']:.2f}× speedup ({int(better['records']):,} records, {better['configuration']}).")
        if (~measured["parallel_faster"]).any():
            worse = measured[~measured["parallel_faster"]].sort_values("speedup").iloc[0]
            observations.append(f"Serial was faster for {int(worse['records']):,} records at {worse['configuration']} ({worse['speedup']:.2f}× speedup).")
        comparable_size_runs = measured.groupby("run_label")["records"].nunique()
        if (comparable_size_runs > 1).any():
            observations.append("For a fixed worker/chunk configuration, the measured dataset-size series shows how fixed multiprocessing overhead changes as workload grows.")
        observations.append("Parallel timing includes process creation, serialization, data transfer, computation, and result aggregation; these costs can dominate small workloads.")
        st.subheader("Automatically generated observations")
        for observation in observations:
            st.markdown(f"- {observation}")
        st.download_button("Download benchmark CSV", measured.to_csv(index=False), "benchmark_results.csv", "text/csv")
    else:
        st.info("Run the benchmark suite to measure actual timings. Multiprocessing overhead is included.")
