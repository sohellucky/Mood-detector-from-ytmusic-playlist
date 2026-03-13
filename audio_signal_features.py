import os
import librosa
import numpy as np
import pandas as pd
import yt_dlp
from tqdm import tqdm


AUDIO_DIR = "audio"
OUTPUT_CSV = "data/audio_features.csv"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


# --------------------------------------------------
# DOWNLOAD AUDIO FROM YOUTUBE
# --------------------------------------------------

def download_audio(video_id):

    url = f"https://www.youtube.com/watch?v={video_id}"

    output_path = f"{AUDIO_DIR}/{video_id}.mp3"

    if os.path.exists(output_path):
        return output_path

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
        "noplaylist": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except:
        return None

    return output_path


# --------------------------------------------------
# EXTRACT AUDIO FEATURES
# --------------------------------------------------

def extract_features(audio_path):

    try:

        y, sr = librosa.load(audio_path, sr=None)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        loudness = np.mean(np.abs(y))

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        brightness = np.mean(centroid)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_mean = np.mean(mfcc)

        return {
            "tempo": float(tempo),
            "loudness": float(loudness),
            "brightness": float(brightness),
            "mfcc": float(mfcc_mean)
        }

    except:
        return {
            "tempo": 0,
            "loudness": 0,
            "brightness": 0,
            "mfcc": 0
        }


# --------------------------------------------------
# BUILD FEATURE DATASET
# --------------------------------------------------

def build_features(tracks_df):

    results = []

    for _, row in tqdm(tracks_df.iterrows(), total=len(tracks_df)):

        video_id = row["videoId"]
        title = row["title"]
        artist = row["artist"]

        audio_file = download_audio(video_id)

        if audio_file is None:
            continue

        features = extract_features(audio_file)

        features["title"] = title
        features["artist"] = artist
        features["videoId"] = video_id

        results.append(features)

    df = pd.DataFrame(results)

    df.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved:", OUTPUT_CSV)

    return df


# --------------------------------------------------
# CLI TEST
# --------------------------------------------------

if __name__ == "__main__":

    playlist_csv = "data/playlist_tracks.csv"

    if not os.path.exists(playlist_csv):
        print("Missing playlist_tracks.csv")
        exit()

    tracks = pd.read_csv(playlist_csv)

    build_features(tracks)