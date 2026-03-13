# Playlist Mood Detector

Streamlit ML app that detects the mood of a YouTube Music playlist.
This project predicts the overall mood of a YouTube Music playlist using machine learning.
It collects songs from a playlist, extracts audio features, and classifies the playlist mood (e.g., calm, energetic, sad, happy).

The goal is to demonstrate an end-to-end ML pipeline including data collection, feature extraction, model training, and prediction.

Features

Fetch songs from a YouTube Music playlist

Extract audio features from each track

Aggregate features for the playlist

Predict playlist mood category

Export collected data to CSV for training

Tech Stack

Python

pandas

scikit-learn

ytmusicapi

transformers

tqdm

Project Pipeline

Collect Playlist Data

Retrieve tracks from a YouTube Music playlist using ytmusicapi.

Feature Extraction

Extract audio-related features such as:

tempo

energy

loudness

danceability

valence

Dataset Creation

Save track features and metadata to a CSV dataset.

Model Training

Train a machine learning model to classify mood.

Prediction

Input a playlist → output predicted mood.

Example Output

Playlist analysis:

Average energy: 0.82

Average tempo: 128 BPM

Average valence: 0.74

Predicted mood:

Energetic

Usage

Install dependencies:

pip install pandas ytmusicapi transformers torch tqdm scikit-learn

Run the data collection script:

python collect_data.py --playlist PLAYLIST_URL

Output dataset:

data/raw_tracks.csv

Train and run the model to predict playlist mood.

Applications

Music recommendation systems

Mood-based playlist generation

User listening behavior analysis

Music analytics tools

Future Improvements

Add lyrics sentiment analysis

Improve feature extraction using audio signal processing

Train deep learning models for better mood classification

Build a web interface for playlist mood analysis

## Run

pip install -r requirements.txt

streamlit run app.py
