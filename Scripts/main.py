"""Public orchestration interface for the Final_bundle surrogate models."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

try:
    from .demand_input_agent import demand_input
    from .demand_pipline import DEMAND_COLUMNS, demand_surrogate
    from .solar.solar_pipline import ORIENTATIONS, TECHNOLOGIES, solar_surrogate
    from .weather_mapper import mapweather
except ImportError:  # Support ``from Scripts.main import ...`` and direct execution.
    from demand_input_agent import demand_input
    from demand_pipline import DEMAND_COLUMNS, demand_surrogate
    from solar.solar_pipline import ORIENTATIONS, TECHNOLOGIES, solar_surrogate
    from weather_mapper import mapweather


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BUNDLE_ROOT / "Outputs"
TARGET_MAPPING = BUNDLE_ROOT / "data" / "target_mapping.csv"
UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
SOLAR_COLUMNS = [
    f"{technology}_{orientation}"
    for technology in TECHNOLOGIES
    for orientation in ORIENTATIONS
]
STATIC_COLUMNS = [*DEMAND_COLUMNS, *[f"solar_suitable_area_{o}_m2" for o in ORIENTATIONS]]


@dataclass
class surrogate_result:
    """Results returned by :func:`surrogate`; unavailable modes remain ``None``."""

    flag: int
    IRIs: list[str]
    aggregate_demand: pd.DataFrame | None = None
    timeseries_demand: pd.DataFrame | None = None
    timeseries_solar_tech: pd.DataFrame | None = None
    solar_suitable_areas: pd.DataFrame | None = None
    error_metrics: dict = field(default_factory=dict)


def _uuid(identifier: str) -> str:
    match = UUID_RE.search(str(identifier))
    if not match:
        raise ValueError(f"No UUID found in building IRI: {identifier!r}")
    return match.group(0).lower()


def _demand_for_buildings(
    iris: list[str], geometry_loc: str, usage_loc: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run buildings separately so each one uses its nearest weather station."""
    aggregate_blocks: list[pd.DataFrame] = []
    hourly_blocks: dict[str, pd.DataFrame] = {}
    for iri in iris:
        weather_iri = mapweather(iri, geometry_loc)
        inputs = demand_input([iri], weather_iri, geometry_loc, usage_loc)
        aggregate, hourly = demand_surrogate(inputs)
        aggregate_blocks.append(aggregate)
        hourly_blocks[iri] = hourly.loc[:, DEMAND_COLUMNS]
    aggregate_all = pd.concat(aggregate_blocks).reindex(iris)
    hourly_all = pd.concat(hourly_blocks, axis=1)
    hourly_all.columns.names = ["building_IRI", "variable"]
    return aggregate_all, hourly_all


def _normalise_solar_timeseries(frame: pd.DataFrame, iris: list[str]) -> pd.DataFrame:
    blocks: dict[str, pd.DataFrame] = {}
    for iri in iris:
        rows = frame.loc[iri] if len(iris) > 1 else frame
        expected = pd.MultiIndex.from_product(
            [TECHNOLOGIES, ORIENTATIONS], names=["technology", "orientation"]
        )
        rows = rows.reindex(expected)
        if rows.isna().all(axis=1).any():
            raise ValueError(f"Solar output is missing technology/orientation rows for {iri}")
        block = rows.T
        block.columns = SOLAR_COLUMNS
        blocks[iri] = block
    result = pd.concat(blocks, axis=1)
    result.columns.names = ["building_IRI", "variable"]
    return result


def surrogate(
    IRI_lst: list[str],
    geometry_loc: str,
    usage_loc: str,
    flag: int = 4,
) -> surrogate_result:
    """Run aggregate demand, hourly demand, solar, or the comprehensive workflow.

    Flags: 1 aggregate demand; 2 hourly demand; 3 hourly solar; 4 all outputs.
    """
    iris = list(IRI_lst)
    if flag not in (1, 2, 3, 4):
        raise ValueError("flag must be one of 1, 2, 3, or 4")
    if not iris:
        raise ValueError("IRI_lst must contain at least one building IRI")
    if len({_uuid(iri) for iri in iris}) != len(iris):
        raise ValueError("IRI_lst contains duplicate building UUIDs")

    result = surrogate_result(flag=flag, IRIs=iris)
    if flag in (1, 2, 4):
        aggregate, hourly = _demand_for_buildings(iris, geometry_loc, usage_loc)
        if flag in (1, 4):
            result.aggregate_demand = aggregate
        if flag in (2, 4):
            result.timeseries_demand = hourly

    if flag in (3, 4):
        solar_raw, areas = solar_surrogate(iris, geometry_loc)
        result.timeseries_solar_tech = _normalise_solar_timeseries(solar_raw, iris)
        result.solar_suitable_areas = areas.reindex(index=ORIENTATIONS, columns=iris)
    return result


def _timeseries_uuid_map(mapping_path: str | Path = TARGET_MAPPING) -> dict[str, str]:
    mapping = pd.read_csv(mapping_path, dtype=str)
    required = {"timeseries_table", "building_uuid"}
    if not required.issubset(mapping.columns):
        raise ValueError(f"target_mapping.csv needs columns {sorted(required)}")
    subset = mapping[["building_uuid", "timeseries_table"]].dropna().copy()
    subset["building_uuid"] = subset["building_uuid"].str.lower()
    if subset["building_uuid"].duplicated().any():
        raise ValueError("target_mapping.csv contains duplicate building_uuid values")
    return dict(zip(subset["building_uuid"], subset["timeseries_table"]))


def save_timeseries_results(
    surrogate_results: surrogate_result,
    output: bool = True,
) -> pd.DataFrame:
    """Return an 8760-row frame and optionally save one 39-column CSV per building."""
    iris = surrogate_results.IRIs
    blocks: dict[str, pd.DataFrame] = {}
    for iri in iris:
        pieces: list[pd.DataFrame] = []
        if surrogate_results.timeseries_demand is not None:
            pieces.append(surrogate_results.timeseries_demand[iri].reindex(columns=DEMAND_COLUMNS))
        if surrogate_results.timeseries_solar_tech is not None:
            pieces.append(surrogate_results.timeseries_solar_tech[iri].reindex(columns=SOLAR_COLUMNS))
        if not pieces:
            raise ValueError("surrogate_results contains no time-series predictions")
        block = pd.concat(pieces, axis=1)
        if surrogate_results.flag == 4 and block.shape[1] != 39:
            raise RuntimeError(f"Comprehensive output for {iri} has {block.shape[1]} columns, expected 39")
        blocks[iri] = block

    combined = pd.concat(blocks, axis=1)
    combined.columns.names = ["building_IRI", "variable"]
    if output:
        uuid_map = _timeseries_uuid_map()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for iri, block in blocks.items():
            building_uuid = _uuid(iri)
            if building_uuid not in uuid_map:
                raise KeyError(f"No timeseries table UUID mapped for building {building_uuid}")
            block.to_csv(OUTPUT_DIR / f"building_{uuid_map[building_uuid]}.csv", index_label="time")
    return combined


def save_static_results(
    surrogate_results: surrogate_result,
    output: bool = True,
) -> pd.DataFrame:
    """Return one row per building and optionally save ``Outputs/building_scalar.csv``."""
    iris = surrogate_results.IRIs
    pieces: list[pd.DataFrame] = []
    if surrogate_results.aggregate_demand is not None:
        pieces.append(surrogate_results.aggregate_demand.reindex(index=iris, columns=DEMAND_COLUMNS))
    if surrogate_results.solar_suitable_areas is not None:
        areas = surrogate_results.solar_suitable_areas.reindex(index=ORIENTATIONS, columns=iris).T
        areas.columns = [f"solar_suitable_area_{o}_m2" for o in ORIENTATIONS]
        pieces.append(areas)
    if not pieces:
        raise ValueError("surrogate_results contains no static predictions")
    combined = pd.concat(pieces, axis=1).reindex(iris)
    combined.index.name = "building_IRI"
    if surrogate_results.flag == 4:
        combined = combined.reindex(columns=STATIC_COLUMNS)
        if combined.shape[1] != 9:
            raise RuntimeError(f"Comprehensive static output has {combined.shape[1]} columns, expected 9")
    if output:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(OUTPUT_DIR / "building_scalar.csv")
    return combined


__all__ = [
    "surrogate_result", "surrogate", "save_timeseries_results", "save_static_results"
]
