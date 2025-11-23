import streamlit as st
import plotly.graph_objects as go


def accuracy_table(data):
    numerical_cols = ["Precision", "Recall", "F1-Score", "mAP50", "mAP50-95"]

    df_display = data.copy()
    df_display = df_display.drop(columns=["Index"], errors="ignore")

    for col in numerical_cols:
        max_val = df_display[col].max()

        df_display[col] = df_display[col].apply(
            lambda x: f"<b>{x}</b>" if x == max_val else x
        )

    row_height = 35
    header_height = 35
    dynamic_height = header_height + (len(df_display) * row_height)

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df_display.columns),
                    fill_color="lightgray",
                    align="center",
                    font=dict(size=15, color="black"),
                    height=header_height,
                ),
                cells=dict(
                    values=[df_display[col] for col in df_display.columns],
                    fill_color="white",
                    font=dict(size=14, color="black"),
                    height=row_height,
                    format=["html"] * len(df_display.columns),
                ),
            )
        ]
    ).update_layout(height=dynamic_height + 50, margin=dict(l=0, r=0, t=5, b=5))

    st.plotly_chart(fig_table, width="stretch")


def efficiency_table(data):
    numerical_cols = [
        "Preprocessing (ms)",
        "Inference (ms)",
        "Postprocessing (ms)",
        "Total Time (ms)",
        "FPS",
        "Training Time",
    ]

    df_display = data.copy()
    df_display = df_display.drop(columns=["Index"], errors="ignore")

    for col in numerical_cols:
        target_val = df_display[col].min() if col != "FPS" else df_display[col].max()

        df_display[col] = df_display[col].apply(
            lambda x: f"<b>{x}</b>" if x == target_val else x
        )

    row_height = 35
    header_height = 35
    dynamic_height = header_height + (len(df_display) * row_height)

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df_display.columns),
                    fill_color="lightgray",
                    align="center",
                    font=dict(size=15, color="black"),
                    height=header_height,
                ),
                cells=dict(
                    values=[df_display[col] for col in df_display.columns],
                    fill_color="white",
                    font=dict(size=14, color="black"),
                    height=row_height,
                    format=["html"] * len(df_display.columns),
                ),
            )
        ]
    ).update_layout(height=dynamic_height + 50, margin=dict(l=0, r=0, t=5, b=5))

    st.plotly_chart(fig_table, width="stretch")


def class_emotions_table(data):

    cols = [
        "anger",
        "content",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]

    df_display = data.copy()
    df_display = df_display.drop(columns=["Index"], errors="ignore")

    for col in cols:
        max_val = df_display[col].max()

        df_display[col] = df_display[col].apply(
            lambda x: f"<b>{x}</b>" if x == max_val else x
        )

    row_height = 35
    header_height = 35
    dynamic_height = header_height + (len(df_display) * row_height)

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df_display.columns),
                    fill_color="lightgray",
                    align="center",
                    font=dict(size=15, color="black"),
                    height=header_height,
                ),
                cells=dict(
                    values=[df_display[col] for col in df_display.columns],
                    fill_color="white",
                    font=dict(size=14, color="black"),
                    height=row_height,
                    format=["html"] * len(df_display.columns),
                ),
            )
        ]
    ).update_layout(height=dynamic_height + 50, margin=dict(l=0, r=0, t=5, b=5))

    st.plotly_chart(fig_table, width="stretch")
