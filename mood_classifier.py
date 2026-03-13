import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from utils import load_csv, ensure_directory

MODEL_PATH = "models/mood_model.pkl"


# ----------------------------------------
# AUDIO RULES TO GENERATE TRAINING LABELS
# ----------------------------------------

def audio_mood(row):

    tempo = row["tempo"]
    loudness = row["loudness"]
    brightness = row["brightness"]

    if tempo > 120 and loudness > 0.05:
        return "energetic"

    if tempo < 80 and loudness < 0.02:
        return "calm"

    if brightness < 1500:
        return "sad"

    return "happy"


# ----------------------------------------
# TRAIN MODEL
# ----------------------------------------

def train_model(input_csv="data/audio_features.csv"):

    df = load_csv(input_csv)

    # convert to numeric FIRST
    df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")
    df["loudness"] = pd.to_numeric(df["loudness"], errors="coerce")
    df["brightness"] = pd.to_numeric(df["brightness"], errors="coerce")
    df["mfcc"] = pd.to_numeric(df["mfcc"], errors="coerce")

    # replace NaN with column mean
    df = df.fillna(df.mean(numeric_only=True))

    # final safety (remove any remaining NaN rows)
    df = df.dropna()

    # create mood labels
    df["mood"] = df.apply(audio_mood, axis=1)

    print("\nOriginal mood distribution:")
    print(df["mood"].value_counts())

    # remove classes with <2 samples
    counts = df["mood"].value_counts()
    valid_classes = counts[counts >= 2].index

    df = df[df["mood"].isin(valid_classes)]

    print("\nFiltered mood distribution:")
    print(df["mood"].value_counts())

    feature_columns = [
        "tempo",
        "loudness",
        "brightness",
        "mfcc"
    ]

    X = df[feature_columns]
    y = df["mood"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report\n")
    print(classification_report(y_test, preds))

    ensure_directory("models")

    joblib.dump(
        {
            "model": model,
            "features": feature_columns
        },
        MODEL_PATH
    )

    print("\nModel saved →", MODEL_PATH)


# ----------------------------------------
# LOAD MODEL
# ----------------------------------------

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run train_model() first.")

    data = joblib.load(MODEL_PATH)

    if isinstance(data, dict):
        model = data["model"]
        features = data["features"]
    else:
        model = data
        features = ["tempo", "loudness", "brightness", "mfcc"]

    return model, features


# ----------------------------------------
# PREDICT MOOD
# ----------------------------------------

def predict_mood(features_df):

    model, feature_columns = load_model()

    # ensure numeric features
    for col in feature_columns:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

    features_df = features_df.fillna(features_df.mean(numeric_only=True))

    X = features_df[feature_columns]

    preds = model.predict(X)

    result = features_df.copy()
    result["predicted_mood"] = preds

    return result


# ----------------------------------------
# DOMINANT MOOD
# ----------------------------------------

def dominant_mood(predictions_df):

    if "predicted_mood" not in predictions_df.columns:
        raise ValueError("predicted_mood column not found")

    mood_counts = predictions_df["predicted_mood"].value_counts()

    dominant = mood_counts.idxmax()

    return dominant, mood_counts.to_dict()


# ----------------------------------------
# CLI
# ----------------------------------------

if __name__ == "__main__":
    train_model()