"""Plotly chart factories used by the dashboard."""

import plotly.express as px
import plotly.graph_objects as go

COLORS = ["#14b8a6", "#22c55e", "#0ea5e9", "#f59e0b", "#8b5cf6"]


def trend_chart(trend, x, title, color=None):
    fig = px.line(trend, x=x, y="mean", title=title, color=color, markers=True,
                  labels={"mean": "Average PM2.5 (µg/m³)", x: ""}, color_discrete_sequence=COLORS)
    fig.update_layout(template="plotly_white", hovermode="x unified", margin=dict(l=10, r=10, t=55, b=10))
    return fig


def station_chart(summary):
    ordered = summary.sort_values("mean", ascending=True)
    fig = px.bar(ordered, x="mean", y="station", orientation="h", title="Average PM2.5 by station",
                 color="mean", color_continuous_scale=["#99f6e4", "#0f766e"],
                 labels={"mean": "Average PM2.5 (µg/m³)", "station": ""})
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=55, b=10), coloraxis_showscale=False)
    return fig


def distribution_chart(data):
    fig = px.histogram(data, x="PM2.5", nbins=60, title="PM2.5 distribution",
                       labels={"PM2.5": "PM2.5 (µg/m³)"}, color_discrete_sequence=["#14b8a6"])
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=55, b=10))
    return fig


def benchmark_chart(result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result["records"], y=result["serial_seconds"], mode="lines+markers",
        name="Serial", line=dict(color="#0f766e", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=result["records"], y=result["parallel_seconds"], mode="lines+markers",
        name="Parallel", line=dict(color="#22c55e", width=3),
    ))
    fig.update_layout(
        template="plotly_white", title="Measured execution time by dataset size",
        xaxis_title="Records", yaxis_title="Seconds",
        hovermode="x unified", margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig
