import os
from pathlib import Path

HIST_DAYS = 35
DATA_PATH = Path(os.getenv(
    "DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
))
EXTERNAL_DATA_PATH = Path(os.getenv(
    "EXTERNAL_DIR",
    os.path.join(DATA_PATH, "external")
))
INTERIM_DATA_PATH = Path(os.getenv(
    "INTERIM_DIR",
    os.path.join(DATA_PATH, "interim")
))
RAW_DATA_PATH = Path(os.getenv("RAW_DIR", os.path.join(DATA_PATH, "raw")))
PROCESSED_DATA = Path(os.getenv("PROCESSED_DIR", os.path.join(DATA_PATH, "processed")))
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH", os.path.join(PROCESSED_DATA, "bike_thefts_lgbm.pkl"))
)
