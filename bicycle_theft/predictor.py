from doctest import DocFileCase
from typing import List, Dict, Any
from datetime import date
import os
import json
import pandas as pd
import numpy as np
import click
import lightgbm as lgb
from datetime import datetime, timedelta
from pydantic import TypeAdapter

from .dataset import load_full_dataset, Weather, LORMaps, Population, TrafficDensity
from .features import FeatureManager
from .models import PredictItem, PredictionResult
from dataclasses import dataclass, field

HIST_DAYS = 35

@dataclass
class Predictor:
    model: lgb.Booster
    artifacts: Dict[str, Any]
    weather_loader: Weather = field(init=False, default_factory=Weather)

    def build(self, items: List[PredictItem]) -> pd.DataFrame:
        df = load_full_dataset()
        df["date"] = pd.to_datetime(df["start_date"]).dt.normalize()
        df["lor"]  = (df["lor"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8))

        df_items = pd.DataFrame([it.model_dump() for it in items], columns=['lor', 'date'])
        df_items["lor"]  = (df_items["lor"].astype(str)
                        .str.replace(r"\D", "", regex=True).str.zfill(8))
        df_items["date"] = pd.to_datetime(df_items["date"], errors="coerce")\
                            .dt.tz_localize(None).dt.normalize()

        maps = LORMaps.load()
        maps_cols = ["lor", "PLR_NAME", "BEZ", "bez_name", "geometry"]
        df_items["start_date"] = pd.to_datetime(df_items["date"], errors="coerce")\
                            .dt.tz_localize(None).dt.normalize()

        panel = self._build_context_panel(df, df_items, hist_days=HIST_DAYS)
        panel = panel.merge(maps[maps_cols], on='lor', how='left')    

        median_imputer  = self.artifacts.get("median_imputer", {})
        zero_like_cols  = self.artifacts.get("zero_like_cols", [])

        fm = FeatureManager()
        df_full = fm.build_causal(panel)
        df_full = df_full.sort_values(["lor", "date"]).drop_duplicates(["lor","date"], keep="last")
        fm.load_state({
            "imp_median": self.artifacts['median_imputer'],
            "price_median_clip": self.artifacts['median_imputer']["price_median_clip"],
            "q_hot": self.artifacts['median_imputer']["is_hot"],
            "q_cold": self.artifacts['median_imputer']["is_cold"]
        })
        X_all = fm.transform(df_full)
        X_all["lor"]  = (X_all["lor"].astype(str)
                    .str.replace(r"\D", "", regex=True).str.zfill(8))
        X_all["date"] = pd.to_datetime(X_all["date"], errors="coerce")\
                        .dt.tz_localize(None).dt.normalize()
        missing_in_Xall = [c for c in zero_like_cols if c not in X_all.columns]
        for c in missing_in_Xall:
            X_all[c] = 0.0
        X_all[zero_like_cols] = X_all[zero_like_cols].fillna(0.0)
        X_all = X_all.fillna(pd.Series(median_imputer))

        pred_df = df_items.merge(X_all, on=["lor","date"], how="left", sort=False, validate="one_to_one")
        if zero_like_cols:
            pred_df[zero_like_cols] = pred_df[zero_like_cols].fillna(0.0)
        return pred_df

    def predict(self, df: pd.DataFrame) -> List[PredictionResult]:
        feature_columns = self.artifacts.get("feature_columns")
        median_imputer  = self.artifacts.get("median_imputer", {})
        best_n          = self.artifacts.get("best_iteration")
        X_req = (
            df[feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(pd.Series(median_imputer))
            .reindex(columns=feature_columns)
        )
        X_req = X_req.fillna(pd.Series(median_imputer)).reindex(columns=feature_columns)

        rate_hat = self.model.predict(X_req, num_iteration=best_n)
        pop_used = df["population_total"].clip(lower=1.0)
        count_hat = np.clip(rate_hat, 0, None) * pop_used / 1000.0

        return [
            PredictionResult(
                lor=l,
                date=d.strftime("%Y-%m-%d"),
                rate_hat_per_1000=float(r),
                count_hat=float(c),
                count_int=int(np.rint(c).astype(int))
            ) 
            for l, d, r, c in zip(df["lor"], df["date"], rate_hat, count_hat)
        ]


    def _build_context_panel(self, df_events: pd.DataFrame,
                        df_items: pd.DataFrame,
                        hist_days: int = 35,
                        force: bool = False) -> pd.DataFrame:
        population_df = Population.load(force)
        traffic_df = TrafficDensity.load(force)
        weather_df = self._get_weather_data(df_items)
        df_events["date"] = pd.to_datetime(df_events["start_date"]).dt.tz_localize(None).dt.normalize()
        df_events["lor"]  = df_events["lor"].astype(str).str.replace(r"\D","",regex=True).str.zfill(8)

        df_items["date"] = pd.to_datetime(df_items["date"]).dt.tz_localize(None).dt.normalize()
        df_items["lor"]  = df_items["lor"].astype(str).str.replace(r"\D","",regex=True).str.zfill(8)

        min_req = df_items["date"].min()
        max_req = df_items["date"].max()
        ctx_start = min_req - pd.Timedelta(days=hist_days)
        ctx_end   = max_req

        lors = sorted(df_items["lor"].unique().tolist())
        dates = pd.date_range(ctx_start, ctx_end, freq="D")
        panel = (pd.MultiIndex.from_product([lors, dates], names=["lor","date"])
                .to_frame(index=False))

        daily_ev = (df_events.groupby(["lor","date"]).size()
                    .rename("y_count").reset_index())
        panel = panel.merge(daily_ev, on=["lor","date"], how="left")
        panel["y_count"] = panel["y_count"].fillna(0.0)

        weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.tz_localize(None).dt.normalize()
        panel = panel.merge(weather_df, on="date", how="left")

        population_df["lor"] = population_df["lor"].astype(str).str.replace(r"\D","",regex=True).str.zfill(8)
        panel = panel.merge(population_df, on="lor", how="left")

        panel = panel.merge(traffic_df, on="lor", how="left")

        panel["start_date"] = panel["date"]

        return panel

    def _get_weather_data(self, req_df: pd.DataFrame) -> pd.DataFrame:
        cutoff = (pd.Timestamp.today().normalize() + pd.Timedelta(days=16))
        today_str = date.today().isoformat()
        forecast_df, archive_df = pd.DataFrame(), pd.DataFrame()

        forecast_items = req_df[req_df['date'] >= today_str]
        if not forecast_items.empty:
            df_over_limit = req_df[req_df["date"] > cutoff]
            if not df_over_limit.empty:
                raise ValueError("Prediction window is 16 days")
            forecast_df = self.weather_loader.forecast()

        archive_weather = req_df[req_df['date'] < today_str]
        if not archive_weather.empty:
            archive_df = self.weather_loader.before_date(today_str)

        return pd.concat([archive_df, forecast_df])



    