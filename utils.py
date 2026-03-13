# utils.py
# Shared helper functions compatible with:
# app.py, collect_playlist.py, audio_features.py,
# mood_classifier.py, visualization.py

import os
import re
import logging
import pandas as pd


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_title(text: str) -> str:
    """
    Remove common noise from YouTube titles.
    Example: (Official Video), [Lyrics], etc.
    """

    if not isinstance(text, str):
        return ""

    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)

    text = text.strip()

    return text


def normalize_artist(text: str) -> str:
    """
    Normalize artist names.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    text = re.sub(r"\s+", " ", text)

    return text


# =========================================================
# PLAYLIST ID EXTRACTION
# =========================================================

def extract_playlist_id(url_or_id: str) -> str:
    """
    Extract playlist ID from YouTube / YouTube Music URL.

    Compatible with collect_playlist.py
    """

    if not isinstance(url_or_id, str):
        return ""

    pattern = r"list=([a-zA-Z0-9_-]+)"
    match = re.search(pattern, url_or_id)

    if match:
        return match.group(1)

    return url_or_id.strip()


# =========================================================
# DIRECTORY UTILITIES
# =========================================================

def ensure_directory(path: str):
    """
    Ensure parent directory exists.

    Works for:
    data/
    models/
    assets/
    """

    directory = os.path.dirname(path)

    if directory == "":
        return

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# =========================================================
# CSV HELPERS
# =========================================================

def save_csv(df: pd.DataFrame, path: str):
    """
    Save dataframe safely.
    Used by multiple modules.
    """

    ensure_directory(path)

    df.to_csv(path, index=False)


def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV safely.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    return pd.read_csv(path)


# =========================================================
# LOGGER
# =========================================================

def setup_logger(name="playlist_mood_detector"):
    """
    Lightweight logger usable across modules.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


# =========================================================
# DATAFRAME HELPERS
# =========================================================

def remove_duplicates(df: pd.DataFrame, column="videoId") -> pd.DataFrame:
    """
    Remove duplicate tracks.
    """

    if column in df.columns:
        df = df.drop_duplicates(subset=[column])

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values safely.
    """

    df = df.fillna(0)

    return df


# =========================================================
# FEATURE NORMALIZATION
# =========================================================

def normalize_columns(df: pd.DataFrame, columns):
    """
    Normalize numeric columns between 0 and 1.
    """

    for col in columns:

        if col not in df.columns:
            continue

        min_val = df[col].min()
        max_val = df[col].max()

        if max_val - min_val == 0:
            df[col] = 0
        else:
            df[col] = (df[col] - min_val) / (max_val - min_val)

    return df