import streamlit as st
import pandas as pd
import numpy
import os
import plotly.express as px
import plotly.graph_objects as go
from table.yolo_tabel import accuracy_table, efficiency_table, class_emotions_table
from utils.model_order import model_order

base_dir = os.path.dirname(os.path.abspath(__file__))
overall_path = os.path.join(base_dir, "data", "Testing Results-Final-Verdict.csv")
detailed_path = os.path.join(base_dir, "data", "yolo_metrics_detailed.csv")

st.set_page_config(page_title="Analisis Performa YOLO", layout="wide")

df = pd.read_csv(overall_path)

st.subheader("Tabel Raw Data Performa Arsitektur Model")
st.write(df)

st.subheader("Tabel Akurasi Model")

accuracy_data = df[
    [
        "Index",
        "Model",
        "Precision",
        "Recall",
        "F1-Score",
        "mAP50",
        "mAP50-95",
    ]
].copy()

a1, a2, a3 = st.columns(3)

with a1:
    selected_models = st.multiselect(
        "Filter Model:",
        accuracy_data["Model"].unique(),
        key="sl_accuracy_filter_model",
    )

    if selected_models:
        accuracy_data = accuracy_data[accuracy_data["Model"].isin(selected_models)]

with a2:
    selected_column = st.selectbox(
        "Pilih kolom untuk sorting:", accuracy_data.columns, key="sl_accuracy_column"
    )

with a3:
    sort_type = st.selectbox(
        "Mode Sort:",
        ["Terkecil → Terbesar", "Terbesar → Terkecil", "A → Z", "Z → A"],
        key="sl_accuracy_sort",
    )

match sort_type:
    case "Terkecil → Terbesar":
        accuracy_data = accuracy_data.sort_values(by=selected_column, ascending=True)

    case "Terbesar → Terkecil":
        accuracy_data = accuracy_data.sort_values(by=selected_column, ascending=False)

    case "A → Z":
        accuracy_data = accuracy_data.sort_values(
            by=selected_column, ascending=True, key=lambda x: x.astype(str)
        )

    case "Z → A":
        accuracy_data = accuracy_data.sort_values(
            by=selected_column, ascending=False, key=lambda x: x.astype(str)
        )

accuracy_table(accuracy_data)

st.subheader("Tabel Efficiency Model")

efficiency_data = df[
    [
        "Index",
        "Model",
        "Preprocessing (ms)",
        "Inference (ms)",
        "Postprocessing (ms)",
        "Total Time (ms)",
        "FPS",
        "Training Time",
    ]
].copy()

e1, e2, e3 = st.columns(3)

with e1:
    selected_models = st.multiselect(
        "Filter Model:",
        efficiency_data["Model"].unique(),
        key="sl_efficiency_filter_model",
    )

    if selected_models:
        efficiency_data = efficiency_data[
            efficiency_data["Model"].isin(selected_models)
        ]

with e2:
    selected_column = st.selectbox(
        "Pilih kolom untuk sorting:", efficiency_data.columns, key="sl_efficiency_col"
    )

with e3:
    sort_type = st.selectbox(
        "Sort:",
        ["Terkecil → Terbesar", "Terbesar → Terkecil", "A → Z", "Z → A"],
        key="sl_efficiency_sort",
    )

match sort_type:
    case "Terkecil → Terbesar":
        efficiency_data = efficiency_data.sort_values(
            by=selected_column, ascending=True
        )

    case "Terbesar → Terkecil":
        efficiency_data = efficiency_data.sort_values(
            by=selected_column, ascending=False
        )

    case "A → Z":
        efficiency_data = efficiency_data.sort_values(
            by=selected_column, ascending=True, key=lambda x: x.astype(str)
        )

    case "Z → A":
        efficiency_data = efficiency_data.sort_values(
            by=selected_column, ascending=False, key=lambda x: x.astype(str)
        )

efficiency_table(efficiency_data)

st.subheader("Tabel mAP50 per Kelas Emosi")

detailed_df = pd.read_csv(detailed_path)

pivot_detailed_df = detailed_df.pivot_table(
    index="Model", columns="Class", values="mAP50"
)

pivot_detailed_df = pivot_detailed_df.reset_index()
pivot_detailed_df = pivot_detailed_df.drop(columns=["index"], errors="ignore")

detailed_sorted = model_order(pivot_detailed_df)

c1, c2, c3 = st.columns(3)

with c1:
    selected_models = st.multiselect(
        "Filter Model:",
        detailed_sorted["Model"].unique(),
        key="sl_detailed_filter_model",
    )

    if selected_models:
        detailed_sorted = detailed_sorted[
            detailed_sorted["Model"].isin(selected_models)
        ]

with c2:
    selected_column = st.selectbox(
        "Pilih kolom untuk sorting:", detailed_sorted.columns, key="sl_detailed_col"
    )

with c3:
    sort_type = st.selectbox(
        "Sort:",
        ["Terkecil → Terbesar", "Terbesar → Terkecil", "A → Z", "Z → A"],
        key="sl_detailed_sort",
    )

match sort_type:
    case "Terkecil → Terbesar":
        detailed_sorted = detailed_sorted.sort_values(
            by=selected_column, ascending=True
        )

    case "Terbesar → Terkecil":
        detailed_sorted = detailed_sorted.sort_values(
            by=selected_column, ascending=False
        )

    case "A → Z":
        detailed_sorted = detailed_sorted.sort_values(
            by=selected_column, ascending=True, key=lambda x: x.astype(str)
        )

    case "Z → A":
        detailed_sorted = detailed_sorted.sort_values(
            by=selected_column, ascending=False, key=lambda x: x.astype(str)
        )

class_emotions_table(detailed_sorted)

st.subheader("Chart Perbandingan Akurasi Arsitekur Model: mAP50 & mAP50-95")

df_melted = df.melt(
    id_vars=["Model"],
    value_vars=["mAP50", "mAP50-95"],
    var_name="Metrik",
    value_name="Nilai Score",
)

map_comparison_chart = px.bar(
    df_melted,
    x="Model",
    y="Nilai Score",
    color="Metrik",
    barmode="group",
    color_discrete_map={"mAP50": "red", "mAP50-95": "blue"},
    height=700,
)

map_comparison_chart.update_layout(xaxis_title="Model", yaxis_title="Nilai mAP")
st.plotly_chart(map_comparison_chart, width="stretch")
