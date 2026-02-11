import io

import pandas as pd
import plotly.express as px
import streamlit as st


def map_comparison_chart(data: pd.DataFrame):
    fig = px.bar(
        data,
        x="Model",
        y="Score",
        color="Metrics",
        barmode="group",
        color_discrete_map={
            "mAP50": "blue",
            "mAP50-95": "red",
        },
    )

    fig.update_layout(
        xaxis_title="Model YOLO",
        yaxis_title="Nilai Score (0.0 - 1.0)",
        font=dict(size=22),
        bargap=0.2,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=22),
        ),
        hoverlabel=dict(
            font_size=22,
            font_family="Arial",
        ),
        width=1800,
        height=900,
        margin=dict(l=120, r=80, t=80, b=120),
    )

    fig.update_xaxes(tickfont=dict(size=22), title_font=dict(size=22))

    fig.update_yaxes(tickfont=dict(size=22), title_font=dict(size=22), range=[0, 0.8])

    fig.update_traces(
        texttemplate="%{y:.4f}",
        textposition="auto",
        textfont=dict(size=22),
        hovertemplate=(
            "<b>Model:</b> %{x}<br>"
            "<b>Metrics:</b> %{fullData.name}<br>"
            "<b>Score:</b> %{y:.4f}<extra></extra>"
        ),
    )

    st.plotly_chart(fig, width="stretch")

    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=3)
    buf.seek(0)
    st.download_button(
        "📥 Download mAP Accuracy Charts",
        data=buf.getvalue(),
        file_name="map_accuracy.png",
        mime="image/png",
    )


def latency_stacked_chart(data: pd.DataFrame):
    metrics_order = ["Preprocessing (ms)", "Inference (ms)", "Postprocessing (ms)"]
    data["Metrics"] = pd.Categorical(
        data["Metrics"], categories=metrics_order, ordered=True
    )

    fig = px.bar(
        data,
        x="Model",
        y="Score",
        color="Metrics",
        barmode="stack",
        color_discrete_map={
            "Preprocessing (ms)": "#4CAF50",
            "Inference (ms)": "#FF5722",
            "Postprocessing (ms)": "#FFC107",
        },
        category_orders={"Metrics": metrics_order},
    )

    fig.update_layout(
        xaxis_title="Model YOLO",
        yaxis_title="Waktu Latency (ms)",
        font=dict(size=22),
        bargap=0.3,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=22),
        ),
        hoverlabel=dict(
            font_size=22,
            font_family="Arial",
        ),
        width=1800,
        height=900,
        margin=dict(l=120, r=80, t=80, b=120),
    )

    fig.update_xaxes(
        tickfont=dict(size=22),
        title_font=dict(size=22),
    )

    fig.update_yaxes(
        tickfont=dict(size=22),
        title_font=dict(size=22),
    )

    fig.update_traces(
        texttemplate="%{y:.2f}",
        textposition="auto",
        textfont=dict(size=22),
        hovertemplate=(
            "<b>Model:</b> %{x}<br>"
            "<b>Metrics:</b> %{fullData.name}<br>"
            "<b>Waktu:</b> %{y:.2f} ms<extra></extra>"
        ),
    )

    st.plotly_chart(fig, width="stretch")

    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=3)
    buf.seek(0)
    st.download_button(
        "📥 Download Model Eficiency Charts",
        data=buf.getvalue(),
        file_name="model_efficiency.png",
        mime="image/png",
    )


def tradeoff_scatter_chart(overall_data: pd.DataFrame):
    fig = px.scatter(
        overall_data,
        x="Total Time (ms)",
        y="mAP50-95",
        size="Parameters (M)",
        color="Model",
        color_discrete_sequence=px.colors.qualitative.Light24,
        hover_data={
            "Model": True,
            "Total Time (ms)": True,
            "mAP50-95": True,
            "Parameters (M)": True,
        },
        size_max=50,
    )

    fig.update_layout(
        xaxis=dict(title="Total Time (ms)"),
        yaxis=dict(title="mAP50-95"),
        font=dict(size=22),
        legend_title_text="Model",
        legend_title_font=dict(size=22),
        legend=dict(
            font=dict(size=22),
        ),
        hoverlabel=dict(
            font_size=22,
            font_family="Arial",
        ),
        width=1800,
        height=900,
        margin=dict(l=120, r=80, t=80, b=120),
    )

    fig.update_xaxes(
        tickfont=dict(size=22),
        title_font=dict(size=22),
    )

    fig.update_yaxes(
        tickfont=dict(size=22),
        title_font=dict(size=22),
    )

    fig.update_traces(
        textfont=dict(size=22),
        hovertemplate=(
            "<b>Model:</b> %{customdata[0]}<br>"
            "<b>Total Latency:</b> %{x:.2f} ms<br>"
            "<b>mAP50-95:</b> %{y:.3f}<br>"
            "<b>Parameters:</b> %{marker.size:.2f} M<extra></extra>"
        ),
    )

    st.plotly_chart(fig, width="stretch")

    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=3)
    buf.seek(0)
    st.download_button(
        "📥 Download Model Trade Off Scatter Plot",
        data=buf.getvalue(),
        file_name="scatter_trade-off.png",
        mime="image/png",
    )


def training_time_chart(overall_data: pd.DataFrame):
    fig = px.bar(
        overall_data,
        x="Model",
        y="Training Time (menit)",
        text="Training Time (menit)",
        color="Model",
        hover_data={"Training Time": True},
        color_discrete_sequence=px.colors.qualitative.Light24,
        height=600,
    )

    fig.update_layout(
        xaxis_title="Model YOLO",
        yaxis_title="Durasi Training (menit)",
        font=dict(size=13),
        bargap=0.3,
        showlegend=False,
    )

    fig.update_traces(
        texttemplate="%{text:.0f} m",
        textposition="outside",
        hovertemplate=(
            "<b>Model:</b> %{x}<br>"
            "<b>Durasi Training:</b> %{customdata[0]}<extra></extra>"
        ),
    )

    st.plotly_chart(fig, width="stretch")


def robustness_heatmap(class_data: pd.DataFrame, metric: str):
    model_order = {
        1: "YOLOv8n",
        2: "YOLOv8s",
        3: "YOLOv8m",
        4: "YOLOv8l",
        5: "YOLOv9t",
        6: "YOLOv9s",
        7: "YOLOv9m",
        8: "YOLOv9c",
        9: "YOLOv10n",
        10: "YOLOv10s",
        11: "YOLOv10m",
        12: "YOLOv10l",
        13: "YOLOv11n",
        14: "YOLOv11s",
        15: "YOLOv11m",
        16: "YOLOv11l",
        17: "YOLOv12n",
        18: "YOLOv12s",
        19: "YOLOv12m",
        20: "YOLOv12l",
    }

    model_list = list(model_order.values())

    if metric == "Default":
        metric = "mAP50"

    heatmap_data = class_data.pivot(index="Model", columns="Class", values=metric)

    existing_models_sorted = []
    for model in model_list:
        if model in heatmap_data.index:
            existing_models_sorted.append(model)

    heatmap_data = heatmap_data.reindex(existing_models_sorted)

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Kelas Emosi", y="Model", color=metric),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="RdYlGn",
        aspect="auto",
    ).update_traces(texttemplate="%{z:.4f}")

    fig.update_layout(
        width=2000,
        height=70 * len(existing_models_sorted),
        font=dict(size=22),
        margin=dict(l=180),
    )

    fig.update_xaxes(tickfont=dict(size=22), title_font=dict(size=22))
    fig.update_yaxes(tickfont=dict(size=22), title_font=dict(size=22))

    st.plotly_chart(fig, width="stretch")


def training_curve(data: pd.DataFrame, metric):
    model_order = {
        1: "YOLOv8n",
        2: "YOLOv8s",
        3: "YOLOv8m",
        4: "YOLOv8l",
        5: "YOLOv9t",
        6: "YOLOv9s",
        7: "YOLOv9m",
        8: "YOLOv9c",
        9: "YOLOv10n",
        10: "YOLOv10s",
        11: "YOLOv10m",
        12: "YOLOv10l",
        13: "YOLOv11n",
        14: "YOLOv11s",
        15: "YOLOv11m",
        16: "YOLOv11l",
        17: "YOLOv12n",
        18: "YOLOv12s",
        19: "YOLOv12m",
        20: "YOLOv12l",
    }
    
    visible_models = sorted(data["Model"].str.lower().unique())
    
    colors = [
        "#1f77b4",  
        "#d62728",  
        "#2ca02c",  
        "#ff7f0e",  
        "#9467bd",  
        "#17becf",  
    ]
    
    dynamic_color_map = {
        model: colors[i % len(colors)]
        for i, model in enumerate(visible_models)
    }

    ordered_models = []
    for m in model_order.values():
        ordered_models.append(m.lower())

    fig = px.line(
        data,
        x="epoch",
        y=metric,
        color="Model",
        markers=False,
        category_orders={"Model": ordered_models},
        color_discrete_map=dynamic_color_map,
    )

    max_val = data[metric].max()
    y_max = max_val * 1.10

    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title=metric.replace("_", " ").title(),
        font=dict(size=22),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=22),
        ),
        hoverlabel=dict(
            font_size=22,
            font_family="Arial",
        ),
        width=1800,
        height=900,
        margin=dict(l=120, r=80, t=80, b=120),
        hovermode="x unified",
    )

    fig.update_xaxes(tickfont=dict(size=22), title_font=dict(size=22))

    fig.update_yaxes(tickfont=dict(size=22), title_font=dict(size=22), range=[0, y_max])

    fig.update_traces(
        line=dict(width=2),
        textfont=dict(size=22),
        hovertemplate=(
            "<b>Model:</b> %{fullData.name}<br>"
            "<b>Epoch:</b> %{x}<br>"
            f"<b>{metric}:</b> " + "%{y:.4f}<extra></extra>"
        ),
    )

    st.plotly_chart(fig, width="stretch")

    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=3)
    buf.seek(0)
    st.download_button(
        "📥 Download Training Curve (High Resolution PNG)",
        data=buf.getvalue(),
        file_name="training_curve.png",
        mime="image/png",
    )
