"""CITYPULSE Streamlit dashboard."""

from pathlib import Path
import os

import pandas as pd
import streamlit as st

from citypulse.anomaly import detect_anomalies
from citypulse.benchmark import benchmark
from citypulse.data_loader import load_or_create_cleaned
from citypulse.processing import run_parallel, run_serial
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


data = get_data()
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
    result = cached_serial(data, 50000)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(data):,}")
    c2.metric("Average PM2.5", f"{data['PM2.5'].mean():.1f} µg/m³")
    c3.metric("Maximum PM2.5", f"{data['PM2.5'].max():.1f} µg/m³")
    c4.metric("Monitoring stations", str(data["station"].nunique()))
    left, right = st.columns([1.5, 1])
    with left:
        st.plotly_chart(trend_chart(result.monthly_trend, "month_period", "Monthly pollution trend"), use_container_width=True)
    with right:
        st.plotly_chart(distribution_chart(data), use_container_width=True)
    st.plotly_chart(station_chart(result.station_summary), use_container_width=True)

elif page == "Data Explorer":
    st.markdown('<div class="section-title">Explore the cleaned master dataset</div>', unsafe_allow_html=True)
    stations = st.multiselect("Stations", sorted(data.station.unique()), default=sorted(data.station.unique()))
    start, end = st.date_input("Date range", (data.datetime.min().date(), data.datetime.max().date()))
    filtered = data[data.station.isin(stations) & data.datetime.dt.date.between(start, end)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Filtered records", f"{len(filtered):,}")
    c2.metric("Missing values", f"{int(filtered.isna().sum().sum()):,}")
    c3.metric("Columns", str(len(filtered.columns)))
    st.dataframe(filtered.head(1000), use_container_width=True, height=400)
    st.download_button("Download filtered CSV", filtered.to_csv(index=False), "citypulse_filtered.csv", "text/csv")
    st.dataframe(filtered.describe(include="all").T, use_container_width=True)

elif page == "Pollution Analysis":
    result = cached_serial(data, 50000)
    st.plotly_chart(station_chart(result.station_summary), use_container_width=True)
    tab1, tab2, tab3 = st.tabs(["Hourly", "Daily", "Weather relationships"])
    with tab1:
        st.plotly_chart(trend_chart(result.hourly_trend, "hour_of_day", "Average PM2.5 by hour"), use_container_width=True)
    with tab2:
        st.plotly_chart(trend_chart(result.daily_trend.tail(365), "date", "Recent daily trend"), use_container_width=True)
    with tab3:
        weather = [c for c in ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "PM10", "SO2", "NO2", "CO", "O3"] if c in data]
        corr = data[["PM2.5"] + weather].corr(numeric_only=True)[["PM2.5"]].drop("PM2.5").sort_values("PM2.5")
        st.bar_chart(corr)
    st.subheader("Potential hotspots")
    hotspots = result.station_summary.sort_values("mean", ascending=False).head(5)
    st.dataframe(hotspots, use_container_width=True)

elif page == "Anomaly Detection":
    anomalies, bounds = detect_anomalies(data)
    st.subheader("IQR-based PM2.5 anomalies")
    c1, c2, c3 = st.columns(3)
    c1.metric("Anomalies", f"{len(anomalies):,}")
    c2.metric("Lower bound", f"{bounds['lower_bound']:.1f}")
    c3.metric("Upper bound", f"{bounds['upper_bound']:.1f}")
    st.dataframe(anomalies, use_container_width=True, height=500)
    st.download_button("Export anomalies CSV", anomalies.to_csv(index=False), "citypulse_anomalies.csv", "text/csv")

elif page == "Serial Processing":
    chunk = st.number_input("Chunk size", 1000, len(data), 50000, step=1000)
    if st.button("Run serial analysis", type="primary"):
        result = run_serial(data, int(chunk))
        st.success(f"Completed in {result.elapsed_seconds:.3f} seconds")
        st.json(result.stats)
        st.dataframe(result.station_summary, use_container_width=True)

elif page == "Parallel Processing":
    workers = st.number_input("Worker processes", 1, max(1, os.cpu_count() or 2), min(2, os.cpu_count() or 2))
    chunk = st.number_input("Chunk size", 1000, len(data), 50000, step=1000)
    st.caption("Uses ProcessPoolExecutor with macOS-safe spawn workers.")
    if st.button("Run parallel analysis", type="primary"):
        with st.spinner("Running worker processes..."):
            result = run_parallel(data, int(workers), int(chunk))
        st.success(f"Completed in {result.elapsed_seconds:.3f} seconds")
        st.json(result.stats)
        st.dataframe(result.station_summary, use_container_width=True)

elif page == "Performance Comparison":
    workers = st.number_input("Benchmark workers", 1, max(1, os.cpu_count() or 2), min(2, os.cpu_count() or 2))
    chunk = st.number_input("Benchmark chunk size", 1000, len(data), 50000, step=1000)
    if st.button("Run benchmark suite", type="primary"):
        with st.spinner("Measuring actual serial and parallel execution..."):
            result = benchmark(data, workers=int(workers), chunk_size=int(chunk))
        st.session_state["benchmark"] = result
    result = st.session_state.get("benchmark")
    if result is not None:
        st.dataframe(result, use_container_width=True)
        st.plotly_chart(benchmark_chart(result), use_container_width=True)
        st.download_button("Download benchmark CSV", result.to_csv(index=False), "benchmark_results.csv", "text/csv")
    else:
        st.info("Run the benchmark suite to measure actual timings. Multiprocessing overhead is included.")
