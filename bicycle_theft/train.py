import os
from typing import Any, Callable
from pathlib import Path
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from functools import partial
from sklearn.metrics import mean_absolute_error, mean_squared_error

import click

from .dataset import load_full_dataset
from .features import FeatureManager

from .config import PROCESSED_DATA

def make_lgb_objective(X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    rate_train: pd.Series,
    rate_test: pd.Series) -> Callable[optuna.Trial, Any | float]:

    def objective(
        trial: optuna.Trial
    ) -> Any | float:
        dtrain = lgb.Dataset(X_train, label=rate_train)
        dvalid = lgb.Dataset(X_test, label=rate_test, reference=dtrain)

        obj = trial.suggest_categorical("objective", ["tweedie", "poisson"])
        params = {
            "objective": obj,
            "metric": "rmse",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "feature_pre_filter": False,
            "verbosity": -1,
            "seed": 42,
            "force_row_wise": True
        }
        if obj == "tweedie":
            params["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=["train","val"],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=100),
            ]
        )

        trial.set_user_attr("best_iteration", int(model.best_iteration))
        val_rate_pred = model.predict(X_test, num_iteration=model.best_iteration)
        rmse_rate = mean_squared_error(rate_test, val_rate_pred) ** 0.5
        return rmse_rate

    return objective


@click.command()
@click.option(
    "--force",
    default=False,
    help="""
force to load dataset.
Default: false.
If dataset was already saved - returns file on disc
""",
)
def train(force: bool) -> None:
    best_model_params_file = Path(
        os.path.join(PROCESSED_DATA, "best_model_params.json")
    )
    preproc_file = Path(
        os.path.join(PROCESSED_DATA, "preproc.json")
    )
    df = load_full_dataset(force)
    feature_manager = FeatureManager()
    df_full = feature_manager.build_causal(df)
    print("Casual features were build")
    test_days = 60
    gap_days = 0
    date_col = "date" if "date" in df.columns else "start_date"

    d = pd.to_datetime(df[date_col]).dt.normalize()
    last = d.max()
    gap = pd.Timedelta(days=gap_days)

    test_start = last - pd.Timedelta(days=test_days) + pd.Timedelta(days=1)
    train_end = test_start - pd.Timedelta(days=1) - gap

    m_train = d <= train_end
    m_test = d >= test_start

    df_train = df_full.loc[m_train].copy()
    df_test = df_full.loc[m_test].copy()

    feature_manager.fit(df_train)
    print("Features were fitted")
    X_train = feature_manager.transform(df_train)
    X_test = feature_manager.transform(df_test)

    print("Dataframes were transformed")

    X_train = X_train.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    X_test = X_test.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    zero_like_tokens = [
        "y_lag",
        "y_roll",
        "y_diff",
        "zero_streak",
        "spatial_lag",
        "_roll7_prev",
        "_roll28_prev",
        "rate_prev_mean",
    ]
    zero_like_cols = [
        c for c in X_train.columns if any(tok in c for tok in zero_like_tokens)
    ]
    X_train[zero_like_cols] = X_train[zero_like_cols].fillna(0.0)
    X_test[zero_like_cols] = X_test[zero_like_cols].fillna(0.0)

    imp = X_train.median(numeric_only=True)
    X_train = X_train.fillna(imp)
    X_test = X_test.fillna(imp.reindex(X_train.columns))

    y_train = X_train[feature_manager.target_column].astype(float).values
    y_test = X_test[feature_manager.target_column].astype(float).values
    pop_train = X_train["population_total"].astype(float).clip(lower=1.0).values
    pop_test = X_test["population_total"].astype(float).clip(lower=1.0).values

    # target train metric - rate for 1000 of inhabitats
    rate_train = y_train / pop_train * 1000.0
    rate_test = y_test / pop_test * 1000.0

    print("Save train and test dataframes to parquet...")
    X_train.to_parquet(
        os.path.join(PROCESSED_DATA, "X_train.geoparquet.gzip"), compression="gzip"
    )
    X_test.to_parquet(
        os.path.join(PROCESSED_DATA, "X_test.geoparquet.gzip"), compression="gzip"
    )
    print("Dataframes are saved!")

    X_train = X_train[feature_manager.columns]
    X_test = X_test[feature_manager.columns]

    dtrain = lgb.Dataset(
        X_train, label=rate_train, feature_name=list(X_train.columns)
    )
    dtest = lgb.Dataset(
        X_test, label=rate_test, reference=dtrain, feature_name=list(X_train.columns)
    )
    if not best_model_params_file.is_file() or not preproc_file.is_file():
        print("Best model params files are not exist. Let us perfrom optuna optimizations...")
        obj = make_lgb_objective(X_train, X_test, rate_train, rate_test)
        study = optuna.create_study(direction="minimize", study_name="lgbm_biketheft")
        study.optimize(obj, n_trials=50, show_progress_bar=True)

        best_params = study.best_params.copy()
        best_params.update(dict(metric="rmse", verbosity=-1, seed=42, force_row_wise=True))
        with open(os.path.join(PROCESSED_DATA, "best_model_params.json"), "w") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)
        best_n = int(study.best_trial.user_attrs.get("best_iteration"))
        preproc_params = {
            "feature_columns": list(X_trval.columns),
            "median_imputer": imp.to_dict(),
            "zero_like_cols": zero_like_cols,
            "best_iteration": best_n,
        }
        with open(os.path.join(PROCESSED_DATA, "preproc.json"), "w") as f:
            json.dump(preproc_params, f, ensure_ascii=False, indent=2)
    else:
        with open(best_model_params_file, "r") as f:
            best_params = json.load(f)

        with open(preproc_file, "r") as f:
            preproc_params = json.load(f)

    best_n = preproc_params.get("best_iteration", 2827)
    print("PREPROC PARAMS ARE", preproc_params)
    gbm_final = lgb.train(
        best_params,
        dtrain,
        num_boost_round=best_n,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "test"],
        callbacks=[lgb.log_evaluation(period=500)],
    )

    rate_pred_test = gbm_final.predict(X_test, num_iteration=best_n)
    cnt_pred_test = np.clip(rate_pred_test, 0, None) * pop_test / 1000.0
    print(
        "Validation (COUNT)  MAE/RMSE:",
        mean_absolute_error(y_test, cnt_pred_test),
        mean_squared_error(y_test, cnt_pred_test) ** 0.5,
    )
    gbm_final.save_model(
        os.path.join(PROCESSED_DATA, "bike_thefts_lgbm.pkl"),
        num_iteration=gbm_final.best_iteration,
    )
    print("Model is saved!")


if __name__ == "__main__":
    train()
