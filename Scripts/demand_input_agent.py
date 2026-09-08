"""Build model-ready static and hourly demand-surrogate inputs."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:  # Works both as a package import and as a directly imported script.
    from .surrounding_agent import get_surrounding_features
    from .weather_mapper import mapweather
except ImportError:
    from surrounding_agent import get_surrounding_features
    from weather_mapper import mapweather


WEATHER = [
    "AirTemperature", "AtmosphericPressure", "CloudCover", "DewPoint",
    "DiffuseHorizontalIrradiance", "DirectNormalIrradiance", "Rainfall",
    "RelativeHumidity", "Snowfall", "WindDirection", "WindSpeed",
]
TEMPORAL = ["h_sin", "h_cos", "d_sin", "d_cos", "s_heating", "s_cooling"]
USAGES = [
    "Bank", "Clinic", "CulturalFacility", "Domestic", "DrinkingEstablishment",
    "EatingEstablishment", "Hospital", "Hotel", "IndustrialFacility",
    "MultiResidential", "Non-Domestic", "Office", "Pharmacy",
    "ReligiousFacility", "RetailEstablishment", "School", "SingleResidential",
    "SportsFacility", "TransportFacility", "University",
]
USAGE_FEATURES = [f"usage_{name}" for name in USAGES]
SURROUNDING_FEATURES = [
    "surrounding_num_neighbours", "surrounding_density",
    "surrounding_max_height", "surrounding_mean_height",
    "surrounding_nearest_taller_distance",
    "surrounding_north_obstruction_angle", "surrounding_south_obstruction_angle",
    "surrounding_east_obstruction_angle", "surrounding_west_obstruction_angle",
    "surrounding_roof_obstruction_proxy",
]
GEOMETRY_FEATURES = [
    "height_m", "footprint_area_m2", "footprint_perimeter_m",
    "footprint_compactness", "estimated_volume_m3",
]
STATIC_FEATURES = GEOMETRY_FEATURES + USAGE_FEATURES + SURROUNDING_FEATURES
TIMESERIES_FEATURES = [f"w_{name}" for name in WEATHER] + TEMPORAL + USAGE_FEATURES + SURROUNDING_FEATURES + GEOMETRY_FEATURES

UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _uuid(value: str) -> str:
    match = UUID_RE.search(str(value))
    if not match:
        raise ValueError(f"No UUID found in building identifier: {value!r}")
    return match.group(0).lower()


def _geometry_values(wkt: str, height: float) -> dict[str, float]:
    dimension = 3 if re.match(r"^\s*(?:MULTI)?POLYGON\s+(?:Z|ZM)\b", str(wkt), re.I) else 2
    numbers = [float(value) for value in NUMBER_RE.findall(str(wkt))]
    points = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers) - dimension + 1, dimension)]
    if len(points) < 3:
        raise ValueError("Building footprint has fewer than three points")
    if points[0] == points[-1]:
        points.pop()
    lon0 = sum(point[0] for point in points) / len(points)
    lat0 = sum(point[1] for point in points) / len(points)
    xy = [
        ((lon - lon0) * 111_320.0 * math.cos(math.radians(lat0)), (lat - lat0) * 110_540.0)
        for lon, lat in points
    ]
    area2 = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1]))
    area = abs(area2) / 2.0
    perimeter = sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1]))
    compactness = 4.0 * math.pi * area / perimeter**2 if perimeter else 0.0
    return {
        "height_m": height,
        "footprint_area_m2": area,
        "footprint_perimeter_m": perimeter,
        "footprint_compactness": compactness,
        "estimated_volume_m3": area * height,
    }


def _load_geometry(geometry_loc: str | Path, wanted: set[str]) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(geometry_loc, dtype=str)
    id_col = next((name for name in ("iri", "building_iri", "uuid") if name in frame.columns), None)
    wkt_col = next((name for name in ("footprint_geometry", "geometry", "wkt") if name in frame.columns), None)
    height_col = next((name for name in ("measured_height", "building_height", "height_m") if name in frame.columns), None)
    if not id_col or not wkt_col or not height_col:
        raise ValueError("Geometry CSV needs an IRI/UUID, footprint WKT, and height column")
    frame["_uuid"] = frame[id_col].map(_uuid)
    rows = frame[frame["_uuid"].isin(wanted)].drop_duplicates("_uuid", keep="last")
    found = set(rows["_uuid"])
    if found != wanted:
        raise KeyError(f"Geometry missing for building UUIDs: {sorted(wanted - found)}")
    return {
        str(row["_uuid"]): _geometry_values(str(row[wkt_col]), float(row[height_col]))
        for _, row in rows.iterrows()
    }


def _load_usage(usage_loc: str | Path, wanted: set[str]) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(usage_loc)
    required = {"building_iri", "ontobuilt", "usageshare"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Usage CSV needs columns: {sorted(required)}")
    frame["_uuid"] = frame["building_iri"].map(_uuid)
    frame = frame[frame["_uuid"].isin(wanted)].copy()
    unknown = sorted(set(frame["ontobuilt"].astype(str)) - set(USAGES))
    if unknown:
        raise ValueError(f"Unknown usage types: {unknown}")
    result: dict[str, dict[str, float]] = {}
    for building_uuid in wanted:
        values = {name: 0.0 for name in USAGE_FEATURES}
        rows = frame[frame["_uuid"] == building_uuid]
        for _, row in rows.iterrows():
            values[f"usage_{row['ontobuilt']}"] += float(row["usageshare"])
        total = sum(values.values())
        if total <= 0:
            raise KeyError(f"No positive usage share for building {building_uuid}")
        result[building_uuid] = {name: value / total for name, value in values.items()}
    return result


def _load_weather(weather_iri: str) -> pd.DataFrame:
    station_uuid = _uuid(weather_iri)
    weather_path = Path(__file__).resolve().parents[1] / "data" / f"weather_{station_uuid}.csv"
    frame = pd.read_csv(weather_path)
    missing = [name for name in ["time", *WEATHER] if name not in frame.columns]
    if missing:
        raise ValueError(f"Weather CSV missing columns: {missing}")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    frame = frame.drop_duplicates("time").sort_values("time")

    # Select the best-covered calendar year and remove leap day, yielding 365*24.
    year = int(frame["time"].dt.year.value_counts().idxmax())
    frame = frame[frame["time"].dt.year == year]
    frame = frame[~((frame["time"].dt.month == 2) & (frame["time"].dt.day == 29))]
    if len(frame) != 8760:
        raise ValueError(f"Weather station {station_uuid} has {len(frame)} usable hours in {year}; expected 8760")
    if frame[WEATHER].isna().any().any():
        raise ValueError(f"Weather station {station_uuid} contains missing values")
    return frame.set_index("time")[WEATHER].astype(float)


def _temporal(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour.to_numpy()
    day = index.dayofyear.to_numpy()
    month = index.month.to_numpy()
    return pd.DataFrame(
        {
            "h_sin": np.sin(2 * math.pi * hour / 24),
            "h_cos": np.cos(2 * math.pi * hour / 24),
            "d_sin": np.sin(2 * math.pi * (day - 1) / 366),
            "d_cos": np.cos(2 * math.pi * (day - 1) / 366),
            "s_heating": np.isin(month, [1, 2, 3, 4, 10, 11, 12]).astype(float),
            "s_cooling": np.isin(month, [6, 7, 8]).astype(float),
        },
        index=index,
    )


def demand_input(
    IRI_lst: list[str],
    weather_IRI: str,
    geometry_loc: str,
    usage_loc: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return model-ready static and time-series inputs.

    For one building the shapes are exactly ``(35, 1)`` and ``(52, 8760)``.
    For multiple buildings, static columns are buildings and the time-series
    rows use a ``(building_IRI, feature)`` MultiIndex; select one 52-row block
    with ``timeseries_input.loc[building_IRI]``.
    """
    if not IRI_lst:
        raise ValueError("IRI_lst must contain at least one building")
    if len(set(map(_uuid, IRI_lst))) != len(IRI_lst):
        raise ValueError("IRI_lst contains duplicate building UUIDs")

    building_uuids = [_uuid(iri) for iri in IRI_lst]
    wanted = set(building_uuids)
    geometry = _load_geometry(geometry_loc, wanted)
    usage = _load_usage(usage_loc, wanted)
    weather_IRI = weather_IRI or mapweather(IRI_lst[0], geometry_loc)
    weather = _load_weather(weather_IRI)
    shared = pd.concat([weather.add_prefix("w_"), _temporal(weather.index)], axis=1)

    index_path = Path(__file__).with_name(f"{Path(geometry_loc).stem}_grid_index_50m.json")
    static_columns: dict[str, pd.Series] = {}
    timeseries_blocks = []
    for iri, building_uuid in zip(IRI_lst, building_uuids):
        surrounding_raw = get_surrounding_features(
            iri, geometry_csv=geometry_loc, index_path=index_path, revert_old=True
        )
        surrounding = {
            "surrounding_num_neighbours": float(surrounding_raw["num_neighbours_50m"]),
            "surrounding_density": float(surrounding_raw["density_50m"]),
            "surrounding_max_height": float(surrounding_raw["max_height_50m"]),
            "surrounding_mean_height": float(surrounding_raw["mean_height_50m"]),
            "surrounding_nearest_taller_distance": float(surrounding_raw["nearest_taller_distance"] or 50.0),
            "surrounding_north_obstruction_angle": float(surrounding_raw["north_obstruction_angle"]),
            "surrounding_south_obstruction_angle": float(surrounding_raw["south_obstruction_angle"]),
            "surrounding_east_obstruction_angle": float(surrounding_raw["east_obstruction_angle"]),
            "surrounding_west_obstruction_angle": float(surrounding_raw["west_obstruction_angle"]),
            "surrounding_roof_obstruction_proxy": float(surrounding_raw["roof_obstruction_proxy"]),
        }
        values = {**geometry[building_uuid], **usage[building_uuid], **surrounding}
        static_columns[iri] = pd.Series(values).reindex(STATIC_FEATURES).astype(float)

        hourly = shared.copy()
        for feature in USAGE_FEATURES + SURROUNDING_FEATURES + GEOMETRY_FEATURES:
            hourly[feature] = values[feature]
        block = hourly[TIMESERIES_FEATURES].T
        block.index = pd.MultiIndex.from_product([[iri], block.index], names=["building_IRI", "feature"])
        timeseries_blocks.append(block)

    static_input = pd.DataFrame(static_columns).reindex(STATIC_FEATURES)
    timeseries_input = pd.concat(timeseries_blocks)
    if static_input.shape != (35, len(IRI_lst)) or timeseries_input.shape != (52 * len(IRI_lst), 8760):
        raise RuntimeError("Internal feature-shape validation failed")
    return static_input, timeseries_input


__all__ = ["demand_input", "STATIC_FEATURES", "TIMESERIES_FEATURES"]
