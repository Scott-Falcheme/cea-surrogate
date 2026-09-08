"""End-to-end solar surrogate workflow for the Final_bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from . import radiation_ROM as radiation_rom
    from .solartech_ROM import convert, pv_absorption_ratio
    from ..surrounding_agent import parse_wkt_points, uuid_from_iri_or_uuid
    from ..weather_mapper import mapweather
except ImportError:
    import radiation_ROM as radiation_rom
    from solartech_ROM import convert, pv_absorption_ratio
    from surrounding_agent import parse_wkt_points, uuid_from_iri_or_uuid
    from weather_mapper import mapweather


ORIENTATIONS = ("R", "N", "S", "E", "W")
TECHNOLOGIES = ("PV", "ET_Q", "ET_E", "FP_Q", "FP_E", "Th_ET", "Th_FP")
RADIATION_SURFACES = ("roof", "north", "south", "east", "west")
DEFAULT_WEATHER_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class BuildingGeometry:
    footprint: list[tuple[float, float]]
    height_m: float


@dataclass
class SolarInput:
    building_iri: str
    building_uuid: str
    timestamps: pd.DatetimeIndex
    weather: pd.DataFrame
    target_geometry: BuildingGeometry
    neighbour_geometries: list[BuildingGeometry]


@dataclass
class RadiationResult:
    raw_Whm2: np.ndarray
    filtered_Whm2: np.ndarray
    filtered_kW: np.ndarray
    gross_area_m2: np.ndarray
    opaque_area_m2: np.ndarray
    eligible_area_m2: np.ndarray
    metadata: dict[str, Any]


@dataclass
class SolarResult:
    values: np.ndarray
    suitable_areas_m2: dict[str, float]


def _station_uuid(weather_iri: str) -> str:
    return uuid_from_iri_or_uuid(weather_iri)


def _load_weather(weather_iri: str, weather_dir: Path) -> pd.DataFrame:
    path = weather_dir / f"weather_{_station_uuid(weather_iri)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Weather CSV not found for {weather_iri}: {path}")
    frame = pd.read_csv(path)
    required = {"time", "AirTemperature"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Weather CSV is missing columns {sorted(missing)}: {path}")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    frame = frame.drop_duplicates("time").sort_values("time")
    year = int(frame["time"].dt.year.value_counts().idxmax())
    frame = frame[frame["time"].dt.year == year]
    frame = frame[~((frame["time"].dt.month == 2) & (frame["time"].dt.day == 29))]
    if len(frame) != 8760:
        raise ValueError(f"Weather station {_station_uuid(weather_iri)} has {len(frame)} usable hours in {year}; expected 8760")
    if frame["AirTemperature"].isna().any():
        raise ValueError(f"Weather station {_station_uuid(weather_iri)} contains missing air temperatures")
    return frame.set_index("time")


def _radiation_result(frame: pd.DataFrame, metadata: dict[str, Any]) -> RadiationResult:
    def hourly(suffix: str) -> np.ndarray:
        return np.column_stack([frame[f"{surface}_{suffix}"].to_numpy(float) for surface in RADIATION_SURFACES])

    def static(suffix: str) -> np.ndarray:
        return np.asarray([float(frame[f"{surface}_{suffix}"].iloc[0]) for surface in RADIATION_SURFACES])

    opaque = static("area_m2")
    eligible_fraction = static("eligible_area_fraction")
    return RadiationResult(
        raw_Whm2=hourly("raw_Whm2"),
        filtered_Whm2=hourly("Whm2"),
        filtered_kW=hourly("kW"),
        gross_area_m2=static("gross_area_m2"),
        opaque_area_m2=opaque,
        eligible_area_m2=opaque * eligible_fraction,
        metadata=metadata,
    )


def solar_surrogate(
    building_iris: list[str],
    geometry_loc: str | Path,
    weather_dir: str | Path | None = None,
    index_path: str | Path | None = None,
    radius_m: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return solar technology time series (35 x 8760 per building) and suitable areas (5 x building)."""
    if not building_iris:
        raise ValueError("building_iris must contain at least one building IRI")

    geometry_path = Path(geometry_loc)
    weather_path = Path(weather_dir) if weather_dir is not None else DEFAULT_WEATHER_DIR
    if not geometry_path.exists():
        raise FileNotFoundError(f"Geometry CSV not found: {geometry_path}")
    if not weather_path.is_dir():
        raise FileNotFoundError(f"Weather directory not found: {weather_path}")

    time_blocks: list[pd.DataFrame] = []
    area_columns: dict[str, np.ndarray] = {}
    absorption_cache: dict[str, np.ndarray] = {}
    original_weather_dir = radiation_rom.DATA_DIR
    radiation_rom.DATA_DIR = weather_path
    try:
        for building_iri in building_iris:
            weather_iri = mapweather(building_iri, geometry_path)
            weather = _load_weather(weather_iri, weather_path)
            if weather_iri not in absorption_cache:
                absorption_cache[weather_iri] = pv_absorption_ratio(weather_iri, weather_path)
            absorption = absorption_cache[weather_iri]
            if absorption.shape != (8760, 5):
                raise ValueError(f"pv_absorption_ratio returned {absorption.shape}; expected (8760, 5)")
            radiation_frame, metadata = radiation_rom.calculate_radiation(
                building_IRI=building_iri,
                geometry_csv=str(geometry_path),
                weather_IRI=weather_iri,
                radius_m=radius_m,
                index_path=Path(index_path) if index_path is not None else None,
            )
            radiation = _radiation_result(radiation_frame, metadata)
            if not pd.DatetimeIndex(radiation_frame["time"]).equals(weather.index.tz_localize(None)):
                raise ValueError(f"Radiation and weather timestamps do not align for {building_iri}")

            flat = convert(
                radiation_kw=radiation.filtered_kW[None, :, :],
                area_m2=radiation.opaque_area_m2[None, :],
                ambient_c=weather["AirTemperature"].to_numpy(float),
                absorption_ratio=absorption,
                apply_aggregate_annual_filter=False,
            )
            if flat.shape != (1, 8760, 35):
                raise ValueError(f"solartech_ROM.convert returned {flat.shape}; expected (1, 8760, 35)")

            # convert() is orientation-major; expose README's [hour, technology, orientation].
            values = flat[0].reshape(8760, 5, 7).transpose(0, 2, 1)
            row_index = pd.MultiIndex.from_product(
                [[building_iri], TECHNOLOGIES, ORIENTATIONS],
                names=["building_IRI", "technology", "orientation"],
            )
            time_blocks.append(pd.DataFrame(values.transpose(1, 2, 0).reshape(35, 8760), index=row_index, columns=weather.index))
            area_columns[building_iri] = radiation.eligible_area_m2
    finally:
        radiation_rom.DATA_DIR = original_weather_dir

    timeseries_solar_tech = pd.concat(time_blocks, axis=0)
    if len(building_iris) == 1:
        timeseries_solar_tech.index = timeseries_solar_tech.index.droplevel("building_IRI")
    solar_suitable_areas = pd.DataFrame(area_columns, index=pd.Index(ORIENTATIONS, name="orientation"))
    return timeseries_solar_tech, solar_suitable_areas


__all__ = [
    "ORIENTATIONS", "TECHNOLOGIES", "BuildingGeometry", "SolarInput",
    "RadiationResult", "SolarResult", "solar_surrogate",
]
