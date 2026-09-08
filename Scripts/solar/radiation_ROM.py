"""Lightweight, non-ML surrogate for CEA's annual radiation calculation.

The model deliberately reduces a building to five receiving surfaces (roof,
north, south, east and west).  pvlib transposes hourly DNI/DHI to each plane,
Embree tests direct-sun visibility against extruded neighbouring footprints,
and a small cosine-weighted hemisphere sample estimates diffuse sky view.

The default paths target the local Kaiserslautern CEA capture so that this file
can also benchmark one building against CEA's ``*_radiation.csv`` output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pvlib
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DATA_DIR = HERE.parents[1] / "data"
try:
    from .. import surrounding_agent as sf
except ImportError:
    sys.path.insert(0, str(HERE.parent))
    import surrounding_agent as sf


ORIENTATIONS = ("roof", "north", "south", "east", "west")
PVLIB_AZIMUTH = {"roof": 180.0, "north": 0.0, "east": 90.0, "south": 180.0, "west": 270.0}
TILT = {"roof": 0.0, "north": 90.0, "east": 90.0, "south": 90.0, "west": 90.0}
CEA_PREFIX = {"roof": "roofs_top", "north": "walls_north", "south": "walls_south", "east": "walls_east", "west": "walls_west"}
DEFAULT_WWR = {"north": 0.16, "south": 0.16, "east": 0.16, "west": 0.16}


@dataclass(frozen=True)
class Surface:
    name: str
    origin: np.ndarray
    normal: np.ndarray
    area_m2: float


@lru_cache(maxsize=4)
def _geometry_rows(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path)
    iri_column = next((name for name in ("iri", "building_iri", "uuid") if name in frame.columns), None)
    height_column = next((name for name in ("measured_height", "building_height", "height_m") if name in frame.columns), None)
    geometry_column = next((name for name in ("footprint_geometry", "geometry", "wkt") if name in frame.columns), None)
    if iri_column is None or height_column is None or geometry_column is None:
        raise ValueError("Geometry CSV needs an IRI/UUID, height, and footprint geometry column")
    rows: dict[str, dict[str, Any]] = {}
    for item in frame.to_dict("records"):
        uuid = sf.uuid_from_iri_or_uuid(str(item[iri_column]))
        rows[uuid] = {
            "iri": str(item[iri_column]),
            "height_m": float(item[height_column]),
            "lonlat": sf.parse_wkt_points(str(item[geometry_column])),
        }
    return rows


def _open_ring(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return points[:-1] if len(points) > 1 and points[0] == points[-1] else points


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized even-odd point-in-polygon test without optional geometry packages."""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        crosses = (y1 > y) != (y2 > y)
        x_cross = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1
        inside ^= crosses & (x < x_cross)
        x1, y1 = x2, y2
    return inside


def _roof_surfaces(points: list[tuple[float, float]], height: float, grid: float, maximum: int) -> tuple[list[Surface], float]:
    polygon = np.asarray(points, dtype=float)
    area = sf.polygon_area_m2(points)
    if area <= 0:
        return [], grid
    minimum = polygon.min(axis=0)
    maximum_xy = polygon.max(axis=0)
    effective = max(grid, math.sqrt(area / maximum)) if maximum > 0 else grid
    for _ in range(8):
        xs = np.arange(minimum[0] + effective / 2.0, maximum_xy[0], effective)
        ys = np.arange(minimum[1] + effective / 2.0, maximum_xy[1], effective)
        if len(xs) and len(ys):
            xx, yy = np.meshgrid(xs, ys)
            candidates = np.column_stack((xx.ravel(), yy.ravel()))
            centres = candidates[_points_in_polygon(candidates, polygon)]
        else:
            centres = np.empty((0, 2))
        if 0 < len(centres) <= maximum:
            break
        if len(centres) == 0:
            centres = np.asarray([np.mean(polygon, axis=0)])
            break
        effective *= math.sqrt(len(centres) / maximum) * 1.001
    patch_area = area / len(centres)
    return [
        Surface("roof", np.array([xy[0], xy[1], height + 0.03]), np.array([0.0, 0.0, 1.0]), patch_area)
        for xy in centres
    ], effective


def _target_surfaces(
    points_xy: list[tuple[float, float]], height: float,
    roof_grid: float = 10.0, walls_grid: float = 200.0,
    max_roof_sensors: int = 256, max_wall_sensors: int = 128,
) -> tuple[list[Surface], dict[str, Any]]:
    points = _open_ring(points_xy)
    if len(points) < 3 or height <= 0 or roof_grid <= 0 or walls_grid <= 0:
        raise ValueError("Target footprint needs at least three vertices and positive height")
    roofs, effective_roof_grid = _roof_surfaces(points, height, roof_grid, max_roof_sensors)
    is_ccw = sf.signed_polygon_area_m2(points) >= 0.0
    edges = []
    for start, end in zip(points, points[1:] + points[:1]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        nx, ny = ((dy / length, -dx / length) if is_ccw else (-dy / length, dx / length))
        direction = sf.wall_normal_direction(nx, ny)
        if direction is None:
            continue
        edges.append((np.asarray(start), np.asarray(end), length, np.asarray([nx, ny, 0.0]), direction))
    requested = sum(max(1, math.ceil(length / walls_grid)) * max(1, math.ceil(height / walls_grid)) for _, _, length, _, _ in edges)
    effective_walls_grid = walls_grid
    if max_wall_sensors > 0 and requested > max_wall_sensors:
        effective_walls_grid *= math.sqrt(requested / max_wall_sensors) * 1.001
    walls: list[Surface] = []
    for start, end, length, normal, direction in edges:
        nx = max(1, math.ceil(length / effective_walls_grid))
        nz = max(1, math.ceil(height / effective_walls_grid))
        patch_area = length * height / (nx * nz)
        for ix in range(nx):
            fraction = (ix + 0.5) / nx
            xy = start + fraction * (end - start)
            for iz in range(nz):
                origin = np.array([xy[0], xy[1], (iz + 0.5) * height / nz]) + 0.03 * normal
                walls.append(Surface(direction, origin, normal, patch_area))
    metadata = {
        "requested_roof_grid_m": roof_grid,
        "requested_walls_grid_m": walls_grid,
        "effective_roof_grid_m": effective_roof_grid,
        "effective_walls_grid_m": effective_walls_grid,
        "max_roof_sensors": max_roof_sensors,
        "max_wall_sensors": max_wall_sensors,
    }
    return roofs + walls, metadata


def _prism_mesh(points_xy: list[tuple[float, float]], height: float) -> trimesh.Trimesh:
    """Triangulate an extruded footprint without optional shapely/triangle dependencies."""
    ring = _open_ring(points_xy)
    n = len(ring)
    centre = np.mean(np.asarray(ring, dtype=float), axis=0)
    vertices = [[x, y, 0.0] for x, y in ring] + [[x, y, height] for x, y in ring]
    bottom_c, top_c = len(vertices), len(vertices) + 1
    vertices.extend([[centre[0], centre[1], 0.0], [centre[0], centre[1], height]])
    faces: list[list[int]] = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend([[i, j, n + j], [i, n + j, n + i]])
        faces.extend([[bottom_c, j, i], [top_c, n + i, n + j]])
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def _scene_and_surfaces(
    building_uuid: str, geometry_csv: Path, radius_m: float, index_path: Path | None,
    roof_grid: float, walls_grid: float, max_roof_sensors: int, max_wall_sensors: int,
) -> tuple[list[Surface], RayMeshIntersector | None, dict[str, Any]]:
    rows = _geometry_rows(geometry_csv)
    target = rows[building_uuid]
    neighbour_info = sf.get_surrounding_features(
        building_uuid, geometry_csv=geometry_csv, radius_m=radius_m,
        index_path=index_path, include_neighbour_uuids=True,
    )
    key = f"neighbour_uuids_{int(radius_m)}m" if float(radius_m).is_integer() else f"neighbour_uuids_{radius_m}m"
    neighbour_ids = [str(value) for value in neighbour_info.get(key, [])]
    lon0, lat0 = sf.polygon_centroid(target["lonlat"])
    target_xy = sf.project_points(target["lonlat"], lon0, lat0)
    surfaces, grid_metadata = _target_surfaces(
        target_xy, target["height_m"], roof_grid, walls_grid, max_roof_sensors, max_wall_sensors
    )

    meshes = []
    for uuid in neighbour_ids:
        row = rows.get(uuid)
        if row is None or row["height_m"] <= 0:
            continue
        meshes.append(_prism_mesh(sf.project_points(row["lonlat"], lon0, lat0), row["height_m"]))
    if meshes:
        mesh = trimesh.util.concatenate(meshes)
        intersector: RayMeshIntersector | None = RayMeshIntersector(mesh)
        triangles = int(len(mesh.faces))
    else:
        intersector = None
        triangles = 0
    context = {
        "longitude": lon0, "latitude": lat0, "height_m": target["height_m"],
        "neighbour_count": len(neighbour_ids), "mesh_triangles": triangles,
        "neighbour_uuids": neighbour_ids,
        "sensor_count": len(surfaces),
        "sensor_count_by_orientation": {name: sum(surface.name == name for surface in surfaces) for name in ORIENTATIONS},
        **grid_metadata,
    }
    return surfaces, intersector, context


def _sun_vectors(apparent_zenith: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    zen = np.radians(apparent_zenith)
    azi = np.radians(azimuth)
    return np.column_stack((np.sin(zen) * np.sin(azi), np.sin(zen) * np.cos(azi), np.cos(zen)))


def _cosine_hemisphere(normal: np.ndarray, count: int) -> np.ndarray:
    # Deterministic low-discrepancy cosine-weighted hemisphere sample.
    i = np.arange(count, dtype=float) + 0.5
    r = np.sqrt(i / count)
    phi = i * (math.pi * (3.0 - math.sqrt(5.0)))
    local = np.column_stack((r * np.cos(phi), r * np.sin(phi), np.sqrt(1.0 - r * r)))
    n = normal / np.linalg.norm(normal)
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    tangent = np.cross(helper, n); tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(n, tangent)
    return local[:, 0, None] * tangent + local[:, 1, None] * bitangent + local[:, 2, None] * n


def _visibility(intersector: RayMeshIntersector | None, surface: Surface, sun_vectors: np.ndarray, daylight: np.ndarray, sky_rays: int) -> tuple[np.ndarray, float]:
    visible = np.ones(len(sun_vectors), dtype=float)
    if intersector is None:
        return visible, 1.0
    active = daylight & ((sun_vectors @ surface.normal) > 0)
    if np.any(active):
        origins = np.repeat(surface.origin[None, :], int(active.sum()), axis=0)
        visible[active] = (~intersector.intersects_any(origins, sun_vectors[active])).astype(float)
    sky_dirs = _cosine_hemisphere(surface.normal, sky_rays)
    sky_origins = np.repeat(surface.origin[None, :], sky_rays, axis=0)
    sky_view = float(np.mean(~intersector.intersects_any(sky_origins, sky_dirs)))
    return visible, sky_view


@lru_cache(maxsize=16)
def _read_weather(weather_IRI: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build an EPW-like pvlib weather object from a bundled weather CSV."""
    station_uuid = sf.uuid_from_iri_or_uuid(weather_IRI)
    weather_path = DATA_DIR / f"weather_{station_uuid}.csv"
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather CSV not found for {weather_IRI}: {weather_path}")

    source = pd.read_csv(weather_path)
    required = {"time", "AirTemperature", "DiffuseHorizontalIrradiance", "DirectNormalIrradiance"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Weather CSV is missing columns {sorted(missing)}: {weather_path}")

    source["time"] = pd.to_datetime(source["time"], utc=True, errors="raise")
    source = source.drop_duplicates("time").sort_values("time")
    year = int(source["time"].dt.year.value_counts().idxmax())
    source = source[source["time"].dt.year == year]
    source = source[~((source["time"].dt.month == 2) & (source["time"].dt.day == 29))]

    weather = pd.DataFrame(
        {
            "temp_air": pd.to_numeric(source["AirTemperature"], errors="raise").to_numpy(float),
            "dhi": pd.to_numeric(source["DiffuseHorizontalIrradiance"], errors="raise").to_numpy(float),
            "dni": pd.to_numeric(source["DirectNormalIrradiance"], errors="raise").to_numpy(float),
        },
        index=pd.DatetimeIndex(source["time"]),
    )
    if len(weather) != 8760:
        raise ValueError(f"Weather station {station_uuid} has {len(weather)} usable hours in {year}; expected 8760")
    if weather.isna().any().any():
        raise ValueError(f"Weather station {station_uuid} contains missing irradiance values")
    return weather, {
        "weather_IRI": weather_IRI,
        "weather_path": str(weather_path.resolve()),
        "weather_year": year,
    }


def calculate_radiation(
    building_IRI: str, geometry_csv: str, weather_IRI: str, radius_m: float = 50.0,
    sky_rays: int = 256, albedo: float = 0.2, index_path: Path | None = None,
    solar_time_offset_minutes: float = 30.0,
    hourly_threshold_Whm2: float = 50.0,
    annual_threshold_kWhm2: float = 800.0,
    wwr: float = 0.16,
    roof_grid: float = 10.0,
    walls_grid: float = 200.0,
    max_roof_sensors: int = 256,
    max_wall_sensors: int = 128,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not 0.0 <= wwr < 1.0:
        raise ValueError(f"WWR must be in [0, 1), got {wwr}")
    start = time.perf_counter()
    building_uuid = sf.uuid_from_iri_or_uuid(building_IRI)
    geometry_path = Path(geometry_csv)
    index_path = Path(index_path) if index_path is not None else None
    weather, meta = _read_weather(weather_IRI)
    surfaces, intersector, context = _scene_and_surfaces(
        building_uuid, geometry_path, radius_m, index_path,
        roof_grid, walls_grid, max_roof_sensors, max_wall_sensors,
    )
    solar_times = weather.index + pd.Timedelta(minutes=solar_time_offset_minutes)
    position = pvlib.solarposition.get_solarposition(solar_times, context["latitude"], context["longitude"], altitude=float(meta.get("altitude", 0.0)))
    zenith = position["apparent_zenith"].to_numpy()
    azimuth = position["azimuth"].to_numpy()
    daylight = zenith < 90.0
    sun_vectors = _sun_vectors(zenith, azimuth)
    dni = np.clip(weather["dni"].to_numpy(dtype=float), 0, None)
    dhi = np.clip(weather["dhi"].to_numpy(dtype=float), 0, None)
    ghi = np.clip(dhi + dni * np.clip(np.cos(np.radians(zenith)), 0.0, None), 0, None)

    output: dict[str, Any] = {"time": weather.index.tz_localize(None)}
    sky_view: dict[str, float] = {}
    areas: dict[str, float] = {}
    annual_raw_kWhm2: dict[str, float] = {}
    eligible: dict[str, bool] = {}
    hourly_values_removed: dict[str, int] = {}
    poa_by_orientation = {
        name: pvlib.irradiance.get_total_irradiance(
            surface_tilt=TILT[name], surface_azimuth=PVLIB_AZIMUTH[name],
            solar_zenith=zenith, solar_azimuth=azimuth, dni=dni, ghi=ghi, dhi=dhi,
            albedo=albedo, model="isotropic",
        )
        for name in ORIENTATIONS
    }
    for name in ORIENTATIONS:
        group = [surface for surface in surfaces if surface.name == name]
        gross_total = sum(surface.area_m2 for surface in group)
        window_total = gross_total * wwr if name != "roof" else 0.0
        opaque_factor = 1.0 - wwr if name != "roof" else 1.0
        opaque_total = gross_total * opaque_factor
        raw_power = np.zeros(len(weather), dtype=float)
        filtered_power = np.zeros(len(weather), dtype=float)
        visible_weighted = np.zeros(len(weather), dtype=float)
        sky_weighted = 0.0
        eligible_area = 0.0
        removed = 0
        poa = poa_by_orientation[name]
        for surface in group:
            visible, svf = _visibility(intersector, surface, sun_vectors, daylight, sky_rays)
            direct = np.asarray(poa["poa_direct"], dtype=float) * visible
            sky = np.asarray(poa["poa_sky_diffuse"], dtype=float) * svf
            ground = np.asarray(poa["poa_ground_diffuse"], dtype=float)
            irradiance_raw = np.clip(np.nan_to_num(direct + sky + ground), 0.0, None)
            annual_raw = float(irradiance_raw.sum() / 1000.0)
            is_eligible = annual_raw >= annual_threshold_kWhm2
            irradiance = np.where(irradiance_raw > hourly_threshold_Whm2, irradiance_raw, 0.0) if is_eligible else np.zeros_like(irradiance_raw)
            opaque_area = surface.area_m2 * opaque_factor
            raw_power += irradiance_raw * opaque_area / 1000.0
            filtered_power += irradiance * opaque_area / 1000.0
            visible_weighted += visible * surface.area_m2
            sky_weighted += svf * surface.area_m2
            eligible_area += opaque_area if is_eligible else 0.0
            removed += int(np.count_nonzero((irradiance_raw > 0.0) & (irradiance_raw <= hourly_threshold_Whm2))) if is_eligible else int(np.count_nonzero(irradiance_raw > 0.0))
        raw_average = raw_power * 1000.0 / opaque_total if opaque_total > 0 else np.zeros(len(weather))
        filtered_average = filtered_power * 1000.0 / opaque_total if opaque_total > 0 else np.zeros(len(weather))
        output[f"{name}_raw_Whm2"] = raw_average
        output[f"{name}_Whm2"] = filtered_average
        output[f"{name}_direct_visible"] = visible_weighted / gross_total if gross_total > 0 else np.zeros(len(weather))
        output[f"{name}_gross_area_m2"] = gross_total
        output[f"{name}_window_area_m2"] = window_total
        output[f"{name}_area_m2"] = opaque_total
        output[f"{name}_eligible"] = eligible_area > 0
        output[f"{name}_eligible_area_fraction"] = eligible_area / opaque_total if opaque_total > 0 else 0.0
        output[f"{name}_sensor_count"] = len(group)
        output[f"{name}_kW"] = filtered_power
        sky_view[name] = sky_weighted / gross_total if gross_total > 0 else 0.0
        areas[name] = opaque_total
        annual_raw_kWhm2[name] = float(raw_average.sum() / 1000.0)
        eligible[name] = eligible_area > 0
        hourly_values_removed[name] = removed
    frame = pd.DataFrame(output)
    frame["total_kW"] = frame[[f"{name}_kW" for name in ORIENTATIONS]].sum(axis=1)
    context.update({
        "engine": "trimesh.ray.ray_pyembree.RayMeshIntersector (Embree/embreex)",
        "sky_rays_per_surface": sky_rays, "surface_area_m2": areas,
        "sky_view_factor": sky_view, "hours": len(frame),
        "solar_time_offset_minutes": solar_time_offset_minutes,
        "hourly_threshold_Whm2": hourly_threshold_Whm2,
        "annual_threshold_kWhm2": annual_threshold_kWhm2,
        "wwr": {name: (0.0 if name == "roof" else wwr) for name in ORIENTATIONS},
        "area_model": "opaque wall area = gross facade area * (1 - WWR); roof unchanged",
        "annual_raw_kWhm2": annual_raw_kWhm2,
        "surface_eligible": eligible,
        "hourly_values_removed": hourly_values_removed,
        "runtime_seconds": time.perf_counter() - start,
    })
    return frame, context


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building", required=True, help="Building UUID or building IRI")
    parser.add_argument("--geometry-csv", type=Path, default=WORKSPACE / "data" / "building_geometry_full.csv")
    parser.add_argument("--weather-iri", required=True, help="Weather-station IRI mapped to data/weather_<UUID>.csv")
    parser.add_argument("--cea-radiation", type=Path, help="Optional CEA *_radiation.csv comparison")
    parser.add_argument("--index-path", type=Path, default=HERE / "building_geometry_grid_index_50m.json")
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--sky-rays", type=int, default=256)
    parser.add_argument("--solar-time-offset-minutes", type=float, default=30.0,
                        help="Offset applied to bundled weather timestamps for pvlib solar position")
    parser.add_argument("--hourly-threshold-Whm2", type=float, default=50.0,
                        help="CEA solar-stage cutoff: values <= threshold become zero")
    parser.add_argument("--annual-threshold-kWhm2", type=float, default=800.0,
                        help="CEA solar-stage surface eligibility threshold")
    parser.add_argument("--wwr", type=float, default=0.16,
                        help="Fixed window-to-wall ratio for all four orientations")
    parser.add_argument("--roof-grid", type=float, default=10.0,
                        help="Roof receiver spacing in metres")
    parser.add_argument("--walls-grid", type=float, default=200.0,
                        help="Wall receiver spacing in metres along each footprint edge and height")
    parser.add_argument("--max-roof-sensors", type=int, default=256)
    parser.add_argument("--max-wall-sensors", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=HERE / "radiation_proxy_output")
    return parser


def main() -> None:
    end_to_end_start = time.perf_counter()
    args = _parser().parse_args()
    uuid = sf.uuid_from_iri_or_uuid(args.building)
    cea_name = uuid
    cea_path = args.cea_radiation
    proxy, report = calculate_radiation(
        args.building, str(args.geometry_csv), args.weather_iri, args.radius_m, args.sky_rays,
        index_path=args.index_path, solar_time_offset_minutes=args.solar_time_offset_minutes,
        hourly_threshold_Whm2=args.hourly_threshold_Whm2,
        annual_threshold_kWhm2=args.annual_threshold_kWhm2,
        wwr=args.wwr,
        roof_grid=args.roof_grid,
        walls_grid=args.walls_grid,
        max_roof_sensors=args.max_roof_sensors,
        max_wall_sensors=args.max_wall_sensors,
    )
    report.update({"building_uuid": uuid, "weather_IRI": args.weather_iri})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = args.output_dir / f"{cea_name}_proxy_hourly.csv"
    report_path = args.output_dir / f"{cea_name}_report.json"
    proxy.to_csv(proxy_path, index=False)
    if cea_path is not None and cea_path.exists():
        comparison, summary = compare_with_cea(
            proxy, cea_path, args.hourly_threshold_Whm2, args.annual_threshold_kWhm2,
        )
        comparison_path = args.output_dir / f"{cea_name}_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        report["comparison"] = summary
        report["comparison_csv"] = str(comparison_path.resolve())
    report["hourly_csv"] = str(proxy_path.resolve())
    report["end_to_end_runtime_seconds"] = time.perf_counter() - end_to_end_start
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
