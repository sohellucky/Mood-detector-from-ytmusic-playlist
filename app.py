# app.py

import streamlit as st
import pandas as pd

from collect_playlist import fetch_playlist
from audio_signal_features import build_features
from mood_classifier import predict_mood, dominant_mood
from visualization import (
    plot_mood_distribution,
    plot_playlist_radar,
    plot_track_radar,
    plot_feature_heatmap
)

st.set_page_config(page_title="Playlist Mood Detector", layout="wide")


# --------------------------------------------------
# GLOBAL CSS
# --------------------------------------------------
st.markdown("""
<style>

body {
    font-family: Inter;
}

.sidebar-title{
    font-family: "Brush Script MT", cursive;
    font-size:48px;
    font-weight:700;
    margin-bottom:25px;
}

.metric-card {
    background:#1e263a;
    padding:20px;
    border-radius:20px;
    text-align:center;
}

.metric-card .value {
    font-size:36px;
    font-weight:700;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.markdown(
    """
    <div class="sidebar-title">
    Mood Detector
    </div>
    """,
    unsafe_allow_html=True
    )

    playlist_url = st.text_input(
        "Playlist URL",
        placeholder="Paste YouTube Music playlist URL"
    )

    max_videos = st.slider(
        "Max tracks",
        5,
        100,
        40
    )

    analyze = st.button("Analyze Playlist")


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🎧 Playlist Mood Analyzer")

st.write(
"""
Detect the emotional profile of a playlist using
machine learning and audio features.
"""
)


# --------------------------------------------------
# RUN PIPELINE
# --------------------------------------------------
df_tracks = None
df_features = None
predictions = None

if analyze and playlist_url:

    with st.spinner("Fetching playlist..."):

        df_tracks = fetch_playlist(
            playlist_url,
            limit=max_videos
        )

    with st.spinner("Extracting audio features..."):

        df_features = build_features(df_tracks)

    with st.spinner("Training / loading mood model..."):

        try:
            predictions = predict_mood(df_features)
        except:
            from mood_classifier import train_model
            train_model()
            predictions = predict_mood(df_features)


# --------------------------------------------------
# METRICS
# --------------------------------------------------
if predictions is not None:

    dom_mood, _ = dominant_mood(predictions)
    video_count = len(predictions)

    avg_tempo = round(df_features["tempo"].mean())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
            <label>Dominant Mood</label>
            <div class="value">{dom_mood.capitalize()}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
            <label>Tracks</label>
            <div class="value">{video_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
            <label>Avg Tempo</label>
            <div class="value">{avg_tempo}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()


# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Tracks",
    "Feature Analysis"
])


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
with tab1:

    if predictions is not None:

        col1, col2 = st.columns(2)

        with col1:
            fig = plot_mood_distribution(predictions)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = plot_playlist_radar(df_features)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Enter playlist URL and click analyze.")


# --------------------------------------------------
# TRACK TABLE
# --------------------------------------------------
with tab2:

    if predictions is not None:

        display = predictions.rename(columns={
            "title": "Title",
            "artist": "Artist",
            "predicted_mood": "Mood"
        })

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Track Emotion Radar")

        selected_track = st.selectbox(
            "Select Track",
            display["Title"]
        )

        track_row = df_features[
            df_features["title"] == selected_track
        ].iloc[0]

        radar = plot_track_radar(track_row, df_features)

        st.plotly_chart(radar, use_container_width=True)

    else:
        st.info("Run analysis to see tracks.")


# --------------------------------------------------
# FEATURE ANALYSIS
# --------------------------------------------------
with tab3:

    if df_features is not None:

        heatmap = plot_feature_heatmap(df_features)

        st.plotly_chart(heatmap, use_container_width=True)

    else:
        st.info("Run analysis first.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("Streamlit · Plotly · ML Mood Detection")