import pandas as pd
import streamlit as st


def table_filter(data: pd.DataFrame, prefix: str):
    a1, a2, a3 = st.columns(3)

    with a1:
        selected_models = st.multiselect(
            "Filter Model:",
            data["Model"].unique(),
            key=f"{prefix}_model",
        )

        if selected_models:
            data = data[data["Model"].isin(selected_models)]

    with a2:
        selected_column = st.selectbox(
            "Pilih kolom untuk sorting:",
            data.columns,
            key=f"{prefix}_column",
        )

    with a3:
        sort_type = st.selectbox(
            "Mode Sort:",
            ["Terkecil → Terbesar", "Terbesar → Terkecil", "A → Z", "Z → A"],
            key=f"{prefix}_sort",
        )

    match sort_type:
        case "Terkecil → Terbesar":
            if selected_column == "Training Time":
                data = data.sort_values(
                    by=selected_column, ascending=True, key=lambda x: pd.to_timedelta(x)
                )
            else:
                data = data.sort_values(by=selected_column, ascending=True)

        case "Terbesar → Terkecil":
            if selected_column == "Training Time":
                data = data.sort_values(
                    by=selected_column,
                    ascending=False,
                    key=lambda x: pd.to_timedelta(x),
                )
            else:
                data = data.sort_values(by=selected_column, ascending=False)

        case "A → Z":
            data = data.sort_values(
                by=selected_column, ascending=True, key=lambda x: x.astype(str)
            )

        case "Z → A":
            data = data.sort_values(
                by=selected_column, ascending=False, key=lambda x: x.astype(str)
            )

    return data


def chart_filter_options(data: pd.DataFrame, prefix: str):
    model_order_dict = {
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

    model_order = list(model_order_dict.values())

    c1, c2, c3 = st.columns(3)

    with c1:
        selected_models = st.multiselect(
            "Pilih Model yang Ditampilkan:",
            data["Model"].unique(),
            key=f"chart_model_{prefix}_filter",
        )

        if selected_models:
            data = data[data["Model"].isin(selected_models)]

    with c2:
        metrics_options = ["None"] + data["Metrics"].unique().tolist()

        if prefix.lower() == "latency":
            metrics_options.append("Total Time (ms)")

        sort_metric = st.selectbox(
            "Sorting berdasarkan:",
            metrics_options,
            key=f"chart_sort_{prefix}_metric",
        )

    with c3:
        sort_direction = st.selectbox(
            "Mode Sort:",
            ["Default", "Ascending", "Descending"],
            key=f"chart_sort_{prefix}_dir",
        )

    if sort_metric == "None" or sort_direction == "Default":
        filtered_order = [m for m in model_order if m in data["Model"].unique()]
        data["Model"] = pd.Categorical(
            data["Model"], categories=filtered_order, ordered=True
        )
        return data.sort_values("Model")

    wide = data.pivot_table(
        index="Model", columns="Metrics", values="Score"
    ).reset_index()

    if prefix.lower() == "latency" and sort_metric == "Total Time (ms)":
        wide["Total Time (ms)"] = (
            wide["Preprocessing (ms)"]
            + wide["Inference (ms)"]
            + wide["Postprocessing (ms)"]
        )

    sort = False
    if sort_direction == "Ascending":
        sort = True

    wide_sorted = wide.sort_values(by=sort_metric, ascending=sort)
    sorted_models = wide_sorted["Model"].tolist()
    data["Model"] = pd.Categorical(
        data["Model"], categories=sorted_models, ordered=True
    )

    return data.sort_values("Model")


def filter_and_sort_training_data(data: pd.DataFrame):
    model_order_dict = {
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
    
    default_rank = {v: k for k, v in model_order_dict.items()}

    t1, t2, t3 = st.columns(3)

    with t1:
        selected_models = st.multiselect(
            "Pilih Model yang Ditampilkan:",
            data["Model"].unique(),
            key="training_model_filter",
        )
        if selected_models:
            data = data[data["Model"].isin(selected_models)]

    with t2:
        sort_direction = st.selectbox(
            "Mode Sort:",
            ["Default", "Ascending", "Descending"],
            key="training_sort_dir",
        )

        if sort_direction == "Default":
            data["rank"] = data["Model"].map(default_rank)
            sorted_df = data.sort_values("rank")
        else:
            is_ascending = True if sort_direction == "Ascending" else False
            sorted_df = data.sort_values("Training Time (menit)", ascending=is_ascending)
            
        sorted_models = sorted_df["Model"].tolist()
        data["Model"] = pd.Categorical(
            data["Model"], categories=sorted_models, ordered=True
        )

    return data.sort_values("Model")


def filter_and_sort_heatmap_data(data: pd.DataFrame):
    h1, h2, h3 = st.columns(3)

    with h1:
        selected_models = st.multiselect(
            "Pilih Model yang Ditampilkan:",
            data["Model"].unique(),
            key="heatmap_model_filter",
        )
        if selected_models:
            data = data[data["Model"].isin(selected_models)]

    with h2:
        selected_metric = st.selectbox(
            "Pilih Metric:",
            options=["Default", "Precision", "Recall", "F1-Score", "mAP50", "mAP50-95"],
            index=0,
            key="heatmap_metric_filter",
        )

    return data, selected_metric
