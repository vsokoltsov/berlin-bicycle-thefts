from typing import Dict, List, Self, Any
import re
import pandas as pd
import geopandas as gpd
import numpy as np
import libpysal as lps
import holidays
from shapely import wkb
from dataclasses import dataclass, field

@dataclass
class FeatureManager:
    state: Dict[str, Any] = field(default_factory=dict)

    @property
    def columns(self) -> List[str]:
        return [
            "dow",
            "is_weekend",
            "weekofyear",
            "sin_doy",
            "cos_doy",
            "is_holiday_BE",
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "sunshine_h",
            "precip_sum_lag1",
            "precip_sum_roll3",
            "t_mean_lag1",
            "sunshine_roll7",
            "is_rainy",
            "is_hot",
            "is_cold",
            "y_lag1",
            "y_lag7",
            "y_roll7",
            "y_diff1",
            "zero_streak",
            "rate_prev_mean_1k",
            "pop_density_km2",
            "poi_density_km2",
            "spatial_lag_y_roll7",
            "attempt_rate_roll7_prev",
            "price_median_clip_roll7_prev",
            "share_bt_diamond_frame_roll7_prev",
            "share_bt_step_through_roll7_prev",
            "share_bt_generic_roll7_prev",
            "share_bt_kids_roll7_prev",
            "share_bt_mtb_roll7_prev",
            "share_bt_other_roll7_prev"
        ]

    @property
    def target_column(self) -> str:
        return "y_count"

    @property
    def offset_column(self) -> str:
        return "offset_log_pop"


    def build_causal(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        df = self._base_features(df)
        df = self._calendar_features(df)
        df = self._dimension_features(df)
        df = self._history_features(df)
        df = self._events_content_features(df)
        return df

    def load_state(self, state_ext: Dict[str, Any]) -> None:
        self.state = {
            'price_median_clip': state_ext['price_median_clip'],
            'imp_median': state_ext['imp_median'],
            'q_hot': state_ext['q_hot'],
            'q_cold': state_ext['q_cold']
        }

    def fit(self, df_train: gpd.GeoDataFrame) -> Self:
        self.state["price_median_clip"] = float(
            df_train.loc[df_train["price"]>0,"price"].quantile(0.99)
        )
        self.state['imp_median'] = df_train.median(numeric_only=True).to_dict()
        daily_t = (
            df_train.assign(
                    date=pd.to_datetime(df_train["start_date"]).dt.normalize()
                ).groupby("date")["temperature_2m_mean"].mean()
            )
        self.state["q_hot"]  = float(daily_t.quantile(0.80)) if len(daily_t) else None
        self.state["q_cold"] = float(daily_t.quantile(0.20)) if len(daily_t) else None
        return self

    def transform(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        
        wx_cols = ["temperature_2m_mean","precipitation_sum","wind_speed_10m_max","sunshine_h"]
        w = (df[["date"] + wx_cols]
            .dropna(subset=["date"])
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .set_index("date"))

        # Precipitation: 1 day lag + 3 prev days sum (exclude current)
        w["precip_sum_lag1"]  = w["precipitation_sum"].shift(1)
        w["precip_sum_roll3"] = w["precipitation_sum"].shift(1).rolling(3, min_periods=1).sum()

        # Temperature: mean for the prev day
        w["t_mean_lag1"]      = w["temperature_2m_mean"].shift(1).mean()

        # Sum: average for last 7 days
        w["sunshine_roll7"]   = w["sunshine_h"].shift(1).rolling(7, min_periods=1).mean()

        # Rain day flag
        w["is_rainy"] = (w["precipitation_sum"] > 1.0).astype(int)

        w["is_hot"]  = int(self.state["q_hot"])
        w["is_cold"] = int(self.state["q_cold"])

        # Prepare daily weather df
        daily_weather = w.reset_index()[[
            "date",
            "precip_sum_lag1","precip_sum_roll3","t_mean_lag1","sunshine_roll7",
            "is_rainy","is_hot","is_cold"
        ]]

        df["date"] = pd.to_datetime(df["start_date"]).dt.normalize()

        # Delete previouse version of columns
        cols_to_drop = ["precip_sum_lag1","precip_sum_roll3","t_mean_lag1","sunshine_roll7",
                        "is_rainy","is_hot","is_cold"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

        # Merge two dataframes by date
        df = df.merge(daily_weather, on="date", how="left")

        # price_median_clip — median price for 99-percentile
        df["price_median_clip"] = self.state["price_median_clip"]

        _order = df.index
        df = df.sort_values(["lor","date"], kind="stable").copy()

        # First row in (lor, date) with daily feature value
        is_first_ld = df.groupby(["lor","date"]).cumcount().eq(0)

        for base_col in ["price_median_clip"]:
            tmp = np.where(is_first_ld, df[base_col], np.nan)
            # mean for last 7 days (no current date) within LOR
            rolled = (pd.Series(tmp)
                    .groupby(df["lor"])
                    .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()))
            out_col = base_col + "_roll7_prev"
            df[out_col] = rolled
            # Set this value for all rows for this day per LOR
            df[out_col] = df.groupby(["lor","date"])[out_col].transform("max")

        # Set initial order
        df = df.loc[_order]

        return df

    def _base_features(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        df["lor"] = df["lor"].astype(str).str.zfill(8)
        df["date"] = pd.to_datetime(df["start_date"]).dt.normalize()
        df['y_count'] = df.groupby(["date", "lor"])["lor"].transform("size")
        pop_day_lor = df.groupby(["lor","date"])["population_total"].transform("first")
        df["offset_log_pop"] = np.log(pop_day_lor.clip(lower=1.0))

        y_cnt_by_day = df.groupby(["lor","date"]).size().sort_index()
        grp = y_cnt_by_day.groupby(level=0)
        cum_sum = grp.cumsum()
        cum_cnt = grp.cumcount() + 1

        y_mean_prev_by_day = (cum_sum - y_cnt_by_day) / (cum_cnt - 1)
        y_mean_prev_by_day = y_mean_prev_by_day.where(cum_cnt > 1, np.nan).fillna(0.0)

        mi = pd.MultiIndex.from_frame(df[["lor","date"]])
        df["y_mean_prev"] = y_mean_prev_by_day.reindex(mi).to_numpy()
        df["rate_prev_mean"] = (df["y_mean_prev"] / pop_day_lor).fillna(0.0)
        df["rate_prev_mean_1k"] = 1000.0 * df["rate_prev_mean"]

        return df

    def _calendar_features(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        df["dow"]         = df["date"].dt.dayofweek
        df["is_weekend"]  = (df["dow"] >= 5).astype(int)
        df["month"]       = df["date"].dt.month
        df["weekofyear"]  = df["date"].dt.isocalendar().week.astype(int)
        years = sorted(df["date"].dt.year.unique())
        de_be = holidays.Germany(prov="BE", years=years) # type: ignore
        df["is_holiday_BE"] = df["date"].dt.date.map(lambda d: int(d in de_be))
        doy = df["date"].dt.dayofyear.astype(float)
        df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

        return df

    def _dimension_features(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        lor_polys = (df[["lor","geometry"]]
            .dropna(subset=["geometry"])
            .drop_duplicates(subset=["lor"])
            .copy())

        # Convert `geometry` field to Shapely
        geom_raw = lor_polys["geometry"]
        first = geom_raw.iloc[0]

        if isinstance(first, (bytes, bytearray, memoryview)):
            # WKB bytes
            geom = gpd.GeoSeries.from_wkb(geom_raw)
        elif isinstance(first, str):
            try:
                geom = gpd.GeoSeries.from_wkt(geom_raw)
            except Exception:
                geom = gpd.GeoSeries.from_wkb(geom_raw.apply(bytes.fromhex))
        else:
            # Shapely
            geom = gpd.GeoSeries(geom_raw)
        gpol = gpd.GeoDataFrame(lor_polys[["lor"]], geometry=geom)

        # Set CRS
        if gpol.crs is None:
            minx, miny, maxx, maxy = gpol.total_bounds
            # if coordinates like longitude / latitude
            if max(abs(minx),abs(maxx)) <= 180 and max(abs(miny),abs(maxy)) <= 90:
                gpol = gpol.set_crs(4326)
            else:
                gpol = gpol.set_crs(25833)

        gpol = gpol.to_crs(25833)
        gpol = (df[["lor","geometry"]]
        .dropna(subset=["geometry"])
        .drop_duplicates(subset=["lor"])
        .copy())

        gpol["lor"] = gpol["lor"].astype(str).str.zfill(8)

        first_geom = gpol["geometry"].iloc[0]
        if isinstance(first_geom, (bytes, bytearray, memoryview)):
            gpol["geometry"] = gpol["geometry"].apply(lambda b: wkb.loads(b))

        gpol = gpd.GeoDataFrame(gpol, geometry="geometry")

        if gpol.crs is None:
            minx, miny, maxx, maxy = gpol.total_bounds
            gpol = gpol.set_crs(4326 if max(abs(minx),abs(maxx))<=180 and max(abs(miny),abs(maxy))<=90 else 25833)
        gpol = gpol.to_crs(25833)

        # Set index to 'lor'
        if "lor" in gpol.columns:
            gpol = gpol.set_index("lor", drop=True)
        else:
            gpol.index.name = "lor"

        # Queen weights for the neighbours
        W = lps.weights.Queen.from_dataframe(gpol)
        W.transform = "r"
        id_order = W.id_order 

        df["lor"]  = df["lor"].astype(str).str.zfill(8)
        df["date"] = pd.to_datetime(df["date"] if "date" in df.columns else df["start_date"]).dt.normalize()

        y_cnt_by_day = df.groupby(["lor","date"]).size().sort_index()  # MultiIndex Series

        #  7-day past mean for each LOR
        y_roll7_by_day = (y_cnt_by_day
                            .groupby(level=0)
                            .apply(lambda s: self.__roll7_past(s.droplevel(0)))
                            .rename("y_roll7"))
        y_roll7_by_day.index = y_roll7_by_day.index.set_names(["lor","date"])

        # Spatial lag by days: W * y_roll7_vector
        parts = []
        for dt, s_day in y_roll7_by_day.groupby(level=1):
            # s_day: index = (lor, date)
            s_lor = s_day.droplevel(1)
            # Fill missing values with 0.0
            v = s_lor.reindex(id_order).fillna(0.0).to_numpy()
            neigh = W.sparse.dot(v).ravel()
            mi = pd.MultiIndex.from_product([id_order, [dt]], names=["lor","date"])
            parts.append(pd.Series(neigh, index=mi))

        spatial_lag_series = pd.concat(parts).rename("spatial_lag_y_roll7")

        # Add value back by (lor, date)
        mi_df = pd.MultiIndex.from_frame(df[["lor","date"]])
        df["spatial_lag_y_roll7"] = spatial_lag_series.reindex(mi_df).to_numpy()

        return df


    def _history_features(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        y_cnt_by_day = df.groupby(["lor","date"]).size().sort_index()

        mi_df = pd.MultiIndex.from_frame(df[["lor","date"]])
        acc: Dict[str, List] = {k: [] for k in ["y_lag1","y_lag7","y_roll7","y_diff1","zero_streak"]}

        # Loop through all LORs without external dataframes
        for lor, s in y_cnt_by_day.groupby(level=0):
            s_lor = s.droplevel(0)
            feats = self.__feat_one_lor(s_lor)
            for k, ser in feats.items():
                # MultiIndex (lor, date) for DataFrame mapping
                ser.index = pd.MultiIndex.from_product([[lor], ser.index], names=["lor","date"])
                acc[k].append(ser)

        # Concat for each feature and map to given df
        for k, parts in acc.items():
            if parts:
                feat_series = pd.concat(parts).sort_index()
                df[k] = feat_series.reindex(mi_df).to_numpy()
            else:
                df[k] = np.nan

        df["pop_density_km2"] = (
            df["population_total"] / df["area_km2"]
        ).replace([np.inf, -np.inf], np.nan)

        return df

    def _events_content_features(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gkey = [df["lor"], df["date"]]  # Group by lor and date

        if "attempt" not in df.columns:
            df["attempt"] = 0
        if "bicycle_type" not in df.columns:
            df["bicycle_type"] = pd.Series(index=df.index, dtype=object)

        # attempt_rate — number of attempts (0/1) per day in LOR
        attempt_bin = pd.to_numeric(df["attempt"], errors="coerce").fillna(0).clip(0, 1)
        df["attempt_rate"] = attempt_bin.groupby(gkey).transform("mean")

        # Share top-k bicycle types per day for LOR
        k = 5
        top_types = (df["bicycle_type"].astype(str).fillna("Unknown")
                    .value_counts().head(k).index.tolist())

        share_cols = []
        for t in top_types:
            slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(t)).strip("_").lower()
            col = f"share_bt_{slug}"
            share_cols.append(col)
            df[col] = (df["bicycle_type"].astype(str).eq(t).astype(int)).groupby(gkey).transform("mean")

        # Share "other" like `1 - sum(top-k)`
        df["share_bt_other"] = (1.0 - df[share_cols].sum(axis=1)).clip(lower=0.0)
        _order = df.index
        df = df.sort_values(["lor","date"], kind="stable").copy()

        # First row in (lor, date) with daily feature value
        is_first_ld = df.groupby(["lor","date"]).cumcount().eq(0)

        for base_col in ["attempt_rate"] + share_cols + ["share_bt_other"]:
            tmp = np.where(is_first_ld, df[base_col], np.nan)
            # mean for last 7 days (no current date) within LOR
            rolled = (pd.Series(tmp)
                    .groupby(df["lor"])
                    .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()))
            out_col = base_col + "_roll7_prev"
            df[out_col] = rolled
            # Set this value for all rows for this day per LOR
            df[out_col] = df.groupby(["lor","date"])[out_col].transform("max")

        # Set initial order
        df = df.loc[_order]
        return df


    def __feat_one_lor(self, s: pd.Series) -> dict[str, pd.Series]:
        s = s.sort_index()

        # Calender index for all values of this LOR
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        s_full = s.reindex(full_idx, fill_value=0)

        # Lags for last days
        y_lag1  = s_full.shift(1)
        y_lag7  = s_full.shift(7)

        # Rolling mean for previous days without today
        y_roll7  = s_full.shift(1).rolling(7,  min_periods=1).mean()

        # Difference between 'today' and 'yesterday'
        y_diff1 = s_full.diff(1)

        # zero_streak:  Length of consequent 0 (including today)
        arr = s_full.to_numpy()
        zs = np.zeros_like(arr, dtype=int)
        run = 0
        for i, v in enumerate(arr):
            if v == 0:
                run += 1
            else:
                run = 0
            zs[i] = run
        zero_streak = pd.Series(zs, index=s_full.index)

        # Values for observing dates only
        out = {
            "y_lag1":  y_lag1.reindex(s.index),
            "y_lag7":  y_lag7.reindex(s.index),
            "y_roll7":  y_roll7.reindex(s.index),
            "y_diff1":  y_diff1.reindex(s.index),
            "zero_streak": zero_streak.reindex(s.index),
        }
        return out

    def __roll7_past(self, s: pd.Series) -> pd.Series:
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        s_full = s.reindex(full_idx, fill_value=0)
        y_roll7 = s_full.shift(1).rolling(7, min_periods=1).mean()
        return y_roll7.reindex(s.index)

    
