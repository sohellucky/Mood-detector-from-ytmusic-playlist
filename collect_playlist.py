import pandas as pd
from ytmusicapi import YTMusic
from tqdm import tqdm

from utils import extract_playlist_id, save_csv, ensure_directory


ytmusic = YTMusic()


# -----------------------------------------------------
# Fetch playlist tracks
# -----------------------------------------------------

def fetch_playlist(playlist_url, limit=None):

    playlist_id = extract_playlist_id(playlist_url)

    playlist = ytmusic.get_playlist(playlist_id, limit=limit)

    tracks = playlist.get("tracks", [])

    rows = []

    for track in tqdm(tracks, desc="Fetching tracks"):

        if not track:
            continue

        title = track.get("title", "")

        artists = track.get("artists", [])
        artist = artists[0]["name"] if artists else ""

        album = ""
        if track.get("album"):
            album = track["album"].get("name", "")

        duration = track.get("duration", "")

        video_id = track.get("videoId", "")

        if not video_id:
            continue

        rows.append({
            "title": title,
            "artist": artist,
            "album": album,
            "duration": duration,
            "videoId": video_id
        })

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(subset=["videoId"])
    df = df[df["title"].notna()]
    df = df.reset_index(drop=True)

    ensure_directory("data")

    # FIXED FILE NAME
    save_csv(df, "data/playlist_tracks.csv")

    print("\nSaved playlist tracks → data/playlist_tracks.csv")

    return df


# -----------------------------------------------------
# CLI
# -----------------------------------------------------

if __name__ == "__main__":

    playlist_url = input("Enter playlist URL: ")

    fetch_playlist(playlist_url)