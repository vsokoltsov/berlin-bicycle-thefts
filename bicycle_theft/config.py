import os

HIST_DAYS = 35
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
EXTERNAL_DATA_PATH = os.path.join(DATA_PATH, "external")
INTERIM_DATA_PATH = os.path.join(DATA_PATH, "interim")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_DATA = os.path.join(DATA_PATH, "processed")
MODEL_PATH = os.path.join(PROCESSED_DATA, "bike_thefts_lgbm.pkl")
