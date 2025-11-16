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

BEST_MODEL_PARAMS = {
  "objective": "tweedie",
  "learning_rate": 0.0902076388685726,
  "num_leaves": 157,
  "min_data_in_leaf": 161,
  "feature_fraction": 0.8750545328275411,
  "bagging_fraction": 0.8549838019082135,
  "bagging_freq": 3,
  "lambda_l1": 1.6040653714912276e-05,
  "lambda_l2": 1.71103772214094e-07,
  "max_depth": -1,
  "tweedie_variance_power": 1.8317103290147705,
  "metric": "rmse",
  "verbosity": -1,
  "seed": 42,
  "force_row_wise": True
}
