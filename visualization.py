# visualization.py

import plotly.graph_objects as go
import plotly.express as px


# =========================================================
# FEATURE SET (must match audio_features + classifier)
# =========================================================

FEATURE_COLUMNS = ["tempo", "loudness", "brightness", "mfcc"]


# =========================================================
# 1. MOOD DISTRIBUTION BAR CHART
# =========================================================

def plot_mood_distribution(df):

    if "predicted_mood" not in df.columns:
        raise ValueError("Missing column: predicted_mood")

    mood_counts = (
        df["predicted_mood"]
        .value_counts()
        .reset_index()
    )

    mood_counts.columns = ["mood", "count"]

    fig = px.bar(
        mood_counts,
        x="mood",
        y="count",
        color="mood",
        text="count",
        title="Mood Distribution"
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Mood",
        yaxis_title="Track Count",
        showlegend=False
    )

    return fig


# =========================================================
# 2. PLAYLIST RADAR (AVERAGE FEATURES)
# =========================================================

import numpy as np
import plotly.graph_objects as go
def plot_playlist_radar(df):

    import plotly.graph_objects as go

    features = ["tempo", "loudness", "brightness", "mfcc"]

    # mean values
    values = df[features].mean()

    # normalize using dataset min/max
    normalized = []

    for col in features:

        min_val = df[col].min()
        max_val = df[col].max()
        value = values[col]

        if max_val - min_val == 0:
            norm = 0
        else:
            norm = (value - min_val) / (max_val - min_val)

        normalized.append(norm)

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=normalized,
            theta=features,
            fill="toself",
            mode="lines+markers+text",
            text=[f"{v:.2f}" for v in normalized],
            textposition="top center"
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Playlist Audio Feature Radar",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,1]
            )
        ),
        showlegend=False
    )

    return fig



# =========================================================
# 3. TRACK RADAR (SINGLE SONG)
# =========================================================

def plot_track_radar(track_row, df):

    features = ["tempo", "loudness", "brightness", "mfcc"]

    # normalize using dataset min/max
    normalized = []

    for col in features:
        min_val = df[col].min()
        max_val = df[col].max()

        value = track_row[col]

        if max_val - min_val == 0:
            norm = 0
        else:
            norm = (value - min_val) / (max_val - min_val)

        normalized.append(norm)

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=normalized,
            theta=features,
            fill="toself",
            name=track_row.get("title", "track")
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,1]
            )
        ),
        template="plotly_dark",
        showlegend=False
    )

    return fig


# =========================================================
# 4. FEATURE CORRELATION HEATMAP
# =========================================================

def plot_feature_heatmap(df):

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    corr = df[FEATURE_COLUMNS].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Audio Feature Correlation"
    )

    fig.update_layout(
        template="plotly_dark"
    )

    return fig