import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

import click

from .dataset import load_full_dataset
from .features import FeatureManager

from .config import PROCESSED_DATA

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
    df = load_full_dataset(force)
    feature_manager = FeatureManager()
    df_full = feature_manager.build_causal(df)
    print("Casual features were build")
    test_days = 60
    gap_days  = 0
    date_col  = "date" if "date" in df.columns else "start_date"

    d    = pd.to_datetime(df[date_col]).dt.normalize()
    last = d.max()
    gap  = pd.Timedelta(days=gap_days)

    test_start = last - pd.Timedelta(days=test_days) + pd.Timedelta(days=1)
    train_end  = test_start - pd.Timedelta(days=1) - gap

    m_train = d <= train_end
    m_test  = d >= test_start

    df_train = df_full.loc[m_train].copy()
    df_test  = df_full.loc[m_test].copy()

    feature_manager.fit(df_train)
    print("Features were fitted")
    X_train = feature_manager.transform(df_train)
    X_test = feature_manager.transform(df_test)

    print("Dataframes were transformed")

    X_train = X_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    
    zero_like_tokens = ["y_lag","y_roll","y_diff","zero_streak",
                    "spatial_lag","_roll7_prev","_roll28_prev","rate_prev_mean"]
    zero_like_cols = [c for c in X_train.columns if any(tok in c for tok in zero_like_tokens)]
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
        os.path.join(PROCESSED_DATA, 'X_train.geoparquet.gzip'),
        compression='gzip'
    )
    X_test.to_parquet(
        os.path.join(PROCESSED_DATA, 'X_train.geoparquet.gzip'),
        compression='gzip'
    )
    print("Dataframes are saved!")

    X_train = X_train[feature_manager.columns]
    X_test = X_test[feature_manager.columns]

    dtrain = lgb.Dataset(X_train, label=rate_train, feature_name=list(X_train.columns))
    dtest = lgb.Dataset(X_test, label=rate_test, reference=dtrain, feature_name=list(X_train.columns))

    with open(os.path.join(PROCESSED_DATA, "best_model_params.json"), "r") as f:
        best_params = json.load(f)

    with open(os.path.join(PROCESSED_DATA, "preproc.json"), "r") as f:
        preproc_params = json.load(f)

    best_n = preproc_params.get('best_iteration', 2827)
    print("PREPROC PARAMS ARE", preproc_params)
    gbm_final = lgb.train(
        best_params,
        dtrain,
        num_boost_round=best_n,
        valid_sets=[dtrain, dtest], valid_names=["train", "test"],
        callbacks=[
                lgb.log_evaluation(period=500)
        ],
    )

    rate_pred_test = gbm_final.predict(X_test, num_iteration=best_n)
    cnt_pred_test  = np.clip(rate_pred_test, 0, None) * pop_test / 1000.0
    print("Validation (COUNT)  MAE/RMSE:",
      mean_absolute_error(y_test, cnt_pred_test),
      mean_squared_error(y_test, cnt_pred_test) ** 0.5)

    print("Saving model...")
    artifacts = {
        "feature_columns": list(X_train.columns),
        "median_imputer": imp.to_dict(),
        "zero_like_cols": zero_like_cols,
        "best_iteration": best_n,
    }
    gbm_final.save_model(
        os.path.join(PROCESSED_DATA, 'bike_thefts_lgbm.pkl'),
        num_iteration=gbm_final.best_iteration
    )
    with open(os.path.join(PROCESSED_DATA, "preproc.json"), "w") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)
    print("Model is saved!")

if __name__ == '__main__':
    train()
