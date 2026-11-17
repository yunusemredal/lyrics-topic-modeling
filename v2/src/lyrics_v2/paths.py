"""Repo-relative paths. All data lives under the git-ignored ``data/`` folder."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
TRACKS_DIR = DATA_DIR / "tracks"
LYRICS_DIR = DATA_DIR / "lyrics"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = REPO_ROOT / "v2" / "results"

TRACK_COLUMNS = [
    "year", "market", "artist", "track", "album",
    "popularity", "release_date", "spotify_id",
]

