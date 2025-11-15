import click
import os

import pandas as pd
import geopandas as gpd
import requests
from pathlib import Path
from shapely.geometry import Point

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
EXTERNAL_DATA_PATH = os.path.join(DATA_PATH, "external")
INTERIM_DATA_PATH = os.path.join(DATA_PATH, "interim")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")


def _file_exists(path: str) -> bool:
    return Path(path).is_file()


class BicycleThefts:
    INPUT_FILE_NAME = os.path.join(RAW_DATA_PATH, "Bicycle Theft Data.csv")
    OUTPUT_FILE_NAME = os.path.join(
        INTERIM_DATA_PATH, "bicycle_theft_utf8.parquet.gzip"
    )

    @classmethod
    def load(self, force: bool = False) -> pd.DataFrame:
        if not force and _file_exists(self.OUTPUT_FILE_NAME):
            print(f"File {self.OUTPUT_FILE_NAME} already exist")
            return pd.read_parquet(self.OUTPUT_FILE_NAME)

        df = pd.read_csv(self.INPUT_FILE_NAME, encoding="cp1252")
        df.columns = pd.Index([
            "created_at",
            "start_date",
            "start_hour",
            "end_date",
            "end_hour",
            "lor",
            "price",
            "attempt",
            "bicycle_type",
            "group",
            "type",
        ])
        for column in ["created_at", "start_date", "end_date"]:
            df[column] = pd.to_datetime(df[column], errors="coerce")

        bicycle_type_mapping = {
            "Damenfahrrad": "step_through",
            "Herrenfahrrad": "diamond_frame",
            "Mountainbike": "mtb",
            "Fahrrad": "generic",
            "diverse Fahrräder": "multiple",
            "Kinderfahrrad": "kids",
            "Lastenfahrrad": "cargo",
            "Rennrad": "road",
        }

        group_mapping = {
            "Fahrraddiebstahl": "bicycle_theft",
            "Keller- und Bodeneinbruch": "cellar_attic_burglary",
        }

        type_mapping = {
            "Sonstiger schwerer Diebstahl von Fahrrädern": "other_aggravated_bicycle_theft",
            "Sonstiger schwerer Diebstahl in/aus Keller/Boden von Fahrrädern": "other_aggravated_bicycle_theft_cellar_attic",
            "Einfacher Diebstahl von Fahrrädern": "simple_bicycle_theft",
            "Einfacher Diebstahl aus Keller/Boden von Fahrrädern": "simple_bicycle_theft_cellar_attic",
        }

        attempt_mapping = {"Unbekannt": 0, "Nein": 1, "Ja": 2}

        df["bicycle_type"] = df["bicycle_type"].map(bicycle_type_mapping)
        df["group"] = df["group"].map(group_mapping)
        df["type"] = df["type"].map(type_mapping)
        df["attempt"] = df["attempt"].map(attempt_mapping).astype(int)
        df["lor"] = (
            pd.to_numeric(df["lor"], errors="coerce")
            .astype("Int64")
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"\D", "", regex=True)
            .str.zfill(8)
        )
        df.to_parquet(self.OUTPUT_FILE_NAME, index=False, compression="gzip")
        return df


class LORMaps:
    SOURCE_URL = "https://tsb-opendata.s3.eu-central-1.amazonaws.com/lor_planungsgraeume_2021/lor_planungsraeume_2021.geojson"
    OUTPUT_FILE_PATH = os.path.join(EXTERNAL_DATA_PATH, "geo_data.geoparquet.gzip")

    @classmethod
    def load(self, force: bool = False) -> gpd.GeoDataFrame:
        if not force and _file_exists(self.OUTPUT_FILE_PATH):
            print(f"File {self.OUTPUT_FILE_PATH} already exist")
            return gpd.read_parquet(self.OUTPUT_FILE_PATH)

        gdf = gpd.read_file(self.SOURCE_URL)
        gdf["lor"] = (
            gdf["PLR_ID"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8)
        )
        bez_map = {
            "01": "Mitte",
            "02": "Friedrichshain-Kreuzberg",
            "03": "Pankow",
            "04": "Charlottenburg-Wilmersdorf",
            "05": "Spandau",
            "06": "Steglitz-Zehlendorf",
            "07": "Tempelhof-Schöneberg",
            "08": "Neukölln",
            "09": "Treptow-Köpenick",
            "10": "Marzahn-Hellersdorf",
            "11": "Lichtenberg",
            "12": "Reinickendorf",
        }
        gdf["bez_name"] = gdf["BEZ"].astype(str).str.zfill(2).map(bez_map)
        gdf.to_parquet(self.OUTPUT_FILE_PATH, compression="gzip")
        return gdf


class Weather:
    START_DATE = "2024-01-01"
    ARCHIVE_SOURCE = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_SOURCE = "https://api.open-meteo.com/v1/forecast"
    OUTPUT_FILE_PATH = os.path.join(EXTERNAL_DATA_PATH, "open_meteo.parquet.gzip")

    @classmethod
    def load(self, force: bool = False) -> pd.DataFrame:
        if not force and _file_exists(self.OUTPUT_FILE_PATH):
            print(f"File {self.OUTPUT_FILE_PATH} already exist")
            return pd.read_parquet(self.OUTPUT_FILE_PATH)

        params: dict[str, str | float] = {
            "latitude": 52.52,
            "longitude": 13.405,
            "start_date": "2024-01-01",
            "end_date": "2025-11-04",
            "daily": ",".join(
                [
                    "temperature_2m_mean",
                    "temperature_2m_min",
                    "temperature_2m_max",
                    "precipitation_sum",
                    "wind_speed_10m_max",
                    "sunshine_duration",
                ]
            ),
            "timezone": "Europe/Berlin",
        }
        j = requests.get(self.ARCHIVE_SOURCE, params=params, timeout=60).json()
        open_meteo_df = pd.DataFrame(j["daily"])
        if "sunshine_duration" in open_meteo_df:
            open_meteo_df["sunshine_h"] = open_meteo_df["sunshine_duration"] / 3600.0
        open_meteo_df["time"] = pd.to_datetime(open_meteo_df["time"])
        open_meteo_df.rename(columns={"time": "created_at"}, inplace=True)
        open_meteo_df.to_parquet(
            os.path.join(self.OUTPUT_FILE_PATH), compression="gzip"
        )
        return open_meteo_df

    def before_date(self, end_date: str) -> pd.DataFrame:
        params: dict[str, str | float] = {
            "latitude": 52.52,
            "longitude": 13.405,
            "start_date": self.START_DATE,
            "end_date": end_date,
            "daily": ",".join(
                [
                    "temperature_2m_mean",
                    "temperature_2m_min",
                    "temperature_2m_max",
                    "precipitation_sum",
                    "wind_speed_10m_max",
                    "sunshine_duration",
                ]
            ),
            "timezone": "Europe/Berlin",
        }
        j = requests.get(self.ARCHIVE_SOURCE, params=params, timeout=60).json()
        open_meteo_df = pd.DataFrame(j["daily"])
        if "sunshine_duration" in open_meteo_df:
            open_meteo_df["sunshine_h"] = open_meteo_df["sunshine_duration"] / 3600.0
        open_meteo_df["time"] = pd.to_datetime(open_meteo_df["time"])
        open_meteo_df.rename(columns={"time": "date"}, inplace=True)
        return open_meteo_df


    def forecast(self) -> pd.DataFrame:
        params: dict[str, str | float] = {
            "latitude": 52.52,
            "longitude": 13.405,
            "forecast_days": 16,
            "daily": ",".join(
                [
                    "temperature_2m_mean",
                    "temperature_2m_min",
                    "temperature_2m_max",
                    "precipitation_sum",
                    "wind_speed_10m_max",
                    "sunshine_duration",
                ]
            ),
            "timezone": "Europe/Berlin",
        }
        j = requests.get(self.FORECAST_SOURCE, params=params, timeout=60).json()
        open_meteo_df = pd.DataFrame(j["daily"])
        if "sunshine_duration" in open_meteo_df:
            open_meteo_df["sunshine_h"] = open_meteo_df["sunshine_duration"] / 3600.0
        open_meteo_df["time"] = pd.to_datetime(open_meteo_df["time"])
        open_meteo_df.rename(columns={"time": "date"}, inplace=True)
        return open_meteo_df
 

class Population:
    INPUT_FILE_NAME = os.path.join(RAW_DATA_PATH, "population.csv")
    OUTPUT_FILE_PATH = os.path.join(INTERIM_DATA_PATH, "population.parquet.gzip")

    @classmethod
    def load(self, force: bool = False) -> pd.DataFrame:
        if not force and _file_exists(self.OUTPUT_FILE_PATH):
            print(f"File {self.OUTPUT_FILE_PATH} already exist")
            return pd.read_parquet(self.OUTPUT_FILE_PATH)

        population_df = pd.read_csv(self.INPUT_FILE_NAME, sep=";")
        population_df.head(5)
        population_df = population_df.rename(columns={"RAUMID": "lor"})
        population_df["lor"] = (
            population_df["lor"]
            .astype(str)
            .str.replace(r"\D", "", regex=True)
            .str.zfill(8)
        )
        rename_keys = {
            "ZEIT": "population_snapshot_date",
            "BEZ": "bez_code",
            "PLR": "plr_code",
        }

        rename_E = {
            "E_E": "population_total",
            "E_EM": "population_male",
            "E_EW": "population_female",
            "E_E00_01": "age_0_1",
            "E_E14_15": "age_14_15",
            "E_E15_18": "age_15_18",
            "E_E18_21": "age_18_21",
            "E_E25_27": "age_25_27",
            "E_E55_60": "age_55_60",
            "E_E60_63": "age_60_64",
            "E_E80_85": "age_80_85",
        }
        columns_to_rename = {**rename_keys, **rename_E}
        columns = ["lor"] + list(rename_keys.values()) + list(rename_E.values())
        columns_to_rename
        population_df = population_df.rename(columns=columns_to_rename)[columns]
        population_df = population_df.fillna(0.0)
        population_df
        population_df.to_parquet(self.OUTPUT_FILE_PATH, compression="gzip")
        return population_df


class TrafficDensity:
    SOURCE = "https://overpass-api.de/api/interpreter"
    OUTPUT_FILE_PATH = os.path.join(EXTERNAL_DATA_PATH, "traffic_density.parquet.gzip")

    @classmethod
    def load(self, df_geo: gpd.GeoDataFrame, force: bool = False) -> pd.DataFrame:
        if not force and _file_exists(self.OUTPUT_FILE_PATH):
            print(f"File {self.OUTPUT_FILE_PATH} already exist")
            return pd.read_parquet(self.OUTPUT_FILE_PATH)

        query = """
        [out:json][timeout:60];
        area["name"="Berlin"]["boundary"="administrative"]->.a;
        (
        nwr(area.a)["amenity"="bicycle_parking"];
        nwr(area.a)["shop"="bicycle"];
        nwr(area.a)["railway"="station"];
        );
        out center tags;
        """
        r = requests.post(self.SOURCE, data={"data": query})
        r.raise_for_status()
        elements = r.json()["elements"]

        # Transform data to points (node -> lon/lat; way/rel -> center.lon/lat)
        rows = []
        for el in elements:
            tags = el.get("tags", {})
            if el["type"] == "node":
                lon, lat = el["lon"], el["lat"]
            else:
                c = el.get("center")
                if not c:
                    continue
                lon, lat = c["lon"], c["lat"]
            rows.append(
                {"id": f"{el['type']}/{el['id']}", "lon": lon, "lat": lat, **tags}
            )

        df_pois = pd.DataFrame(rows)

        # Classification of the given data
        def classify(rec: pd.Series) -> str:
            if rec.get("amenity") == "bicycle_parking":
                return "bike_parking"
            if rec.get("shop") == "bicycle":
                return "bike_shop"
            if rec.get("railway") == "station":
                return "rail_station"
            return "other"

        df_pois["kind"] = df_pois.apply(classify, axis=1)

        pois = gpd.GeoDataFrame(
            df_pois,
            geometry=[Point(xy) for xy in zip(df_pois["lon"], df_pois["lat"])],
            crs=4326,
        ).to_crs(25833)

        pois = pois.drop_duplicates(subset=["id"]).reset_index(drop=True)
        if pois.crs is None:
            pois = pois.set_crs(4326)
        pois = pois.to_crs(25833)

        # Get one polygon per LOR and build dataframe based on this
        lor_polys = (
            df_geo[["lor", "geometry"]]
            .dropna(subset=["geometry"])
            .drop_duplicates(subset=["lor"])
            .copy()
        )
        gdf_lor = gpd.GeoDataFrame(lor_polys, geometry="geometry")

        if gdf_lor.crs is None:
            minx, miny, maxx, maxy = gdf_lor.total_bounds
            if max(abs(minx), abs(maxx)) <= 180 and max(abs(miny), abs(maxy)) <= 90:
                gdf_lor = gdf_lor.set_crs(4326)
            else:
                gdf_lor = gdf_lor.set_crs(25833)

        gdf_lor_25833 = gdf_lor.to_crs(25833)

        joined = gpd.sjoin(
            pois, gdf_lor_25833[["lor", "geometry"]], how="inner", predicate="within"
        )

        area_km2 = (gdf_lor_25833.set_index("lor").area / 1e6).rename("area_km2")
        poi_cnt = joined.groupby("lor").size().rename("poi_cnt")

        poi_density: pd.DataFrame = (
            poi_cnt.to_frame().join(area_km2, how="right").fillna({"poi_cnt": 0})
        )
        poi_density["poi_density_km2"] = (
            poi_density["poi_cnt"] / poi_density["area_km2"]
        )
        poi_density.to_parquet(self.OUTPUT_FILE_PATH, compression="gzip")
        return poi_density


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
def load_data(force: bool) -> None:
    _ = load_full_dataset(force)


def load_full_dataset(force: bool = False) -> gpd.GeoDataFrame:
    output_file = os.path.join(INTERIM_DATA_PATH, "df_geo_etl.geoparquet.gzip")
    if not force and _file_exists(output_file):
        print(f"File {output_file} already exist")
        return gpd.read_parquet(output_file)
    
    df = BicycleThefts.load(force)

    geo_df = LORMaps.load(force)
    cols_geom = ["lor", "PLR_NAME", "BEZ", "bez_name", "geometry"]
    df_geo = df.merge(geo_df[cols_geom], on="lor", how="left")
    df_geo.columns = df_geo.columns.str.lower()

    meteo_df = Weather.load(force)
    df_geo = df_geo.join(meteo_df.set_index("created_at"), on="created_at")

    population_df = Population.load(force)
    df_geo = df_geo.join(population_df.set_index("lor"), on="lor")

    traffic_df = TrafficDensity.load(df_geo, force)
    df_geo = df_geo.join(traffic_df, on="lor")
    gdf = gpd.GeoDataFrame(df_geo, geometry="geometry", crs=geo_df.crs or 25833)
    gdf.to_parquet(
        os.path.join(INTERIM_DATA_PATH, "df_geo_etl.geoparquet.gzip"),
        compression="gzip",
    )
    return df_geo


if __name__ == "__main__":
    load_data()
