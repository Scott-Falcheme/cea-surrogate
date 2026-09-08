"""Neighbourhood geometry features for CEA surrogate inputs.

This module builds a cached square-grid index for building geometries and
returns surrounding-building features for one building UUID/IRI.

The geometry source may contain either of these column sets:
    iri, measured_height, footprint_geometry
    uuid, building_height, geometry

Default source:
    kaiserslautern/adminer/Minorities/building_geometry.csv

Example:
    from codex.surrounding_features import get_surrounding_features

    features = get_surrounding_features(
        "777861fa-19e7-4280-9ed1-59ad8c18181f",
        plot_geometries=True,
    )
    print(features)
"""

from __future__ import annotations

import csv
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

DEFAULT_GEOMETRY_CSV = (
    Path(__file__).resolve().parents[1]
    / "kaiserslautern"
    / "adminer"
    / "Minorities"
    / "building_geometry.csv"
)
BUILDING_IRI_PREFIX = "https://theworldavatar.io/kg/Building/"


def uuid_from_iri_or_uuid(value: str) -> str:
    """Extract a UUID from either a UUID string or full building IRI."""
    match = UUID_RE.search(str(value))
    if not match:
        raise ValueError(f"No UUID found in building identifier: {value!r}")
    return match.group(0).lower()


def parse_wkt_points(wkt: str) -> list[tuple[float, float]]:
    """Parse lon/lat points from POLYGON / POLYGON Z WKT text."""
    text = str(wkt)
    points: list[tuple[float, float]] = []
    numbers = [float(n) for n in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]
    stride = 3 if "POLYGON Z" in text.upper() else 2
    for i in range(0, len(numbers) - stride + 1, stride):
        lon, lat = numbers[i], numbers[i + 1]
        points.append((lon, lat))
    if len(points) < 3:
        raise ValueError(f"Could not parse polygon WKT: {wkt[:80]!r}")
    return points


def polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Return a robust lon/lat centroid approximation."""
    # The WKT closes the ring by repeating the first point; dropping it avoids
    # overweighting that vertex.
    if points[0] == points[-1]:
        points = points[:-1]
    lon = sum(p[0] for p in points) / len(points)
    lat = sum(p[1] for p in points) / len(points)
    return lon, lat


def project_points(
    points: list[tuple[float, float]], lon0: float, lat0: float
) -> list[tuple[float, float]]:
    """Project lon/lat to local metres with an equirectangular approximation."""
    cos_lat = math.cos(math.radians(lat0))
    return [((lon - lon0) * 111_320.0 * cos_lat, (lat - lat0) * 110_574.0) for lon, lat in points]


def polygon_area_m2(points_xy: list[tuple[float, float]]) -> float:
    """Shoelace area in square metres."""
    if len(points_xy) < 3:
        return 0.0
    area2 = 0.0
    for (x1, y1), (x2, y2) in zip(points_xy, points_xy[1:] + points_xy[:1]):
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5


def signed_polygon_area_m2(points_xy: list[tuple[float, float]]) -> float:
    """Signed shoelace area in square metres; positive means counter-clockwise."""
    if len(points_xy) < 3:
        return 0.0
    if points_xy[0] == points_xy[-1]:
        points_xy = points_xy[:-1]
    area2 = 0.0
    for (x1, y1), (x2, y2) in zip(points_xy, points_xy[1:] + points_xy[:1]):
        area2 += x1 * y2 - x2 * y1
    return area2 * 0.5


def bearing_direction(dx: float, dy: float) -> str | None:
    """Map a target-to-neighbour vector to N/S/E/W cardinal sector."""
    if dx == 0.0 and dy == 0.0:
        return None
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "north" if dy > 0 else "south"


def wall_normal_direction(nx: float, ny: float) -> str | None:
    """Map a wall outward normal vector to the nearest cardinal direction."""
    if abs(nx) < 1e-12 and abs(ny) < 1e-12:
        return None
    if abs(nx) >= abs(ny):
        return "east" if nx > 0 else "west"
    return "north" if ny > 0 else "south"


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    eps: float = 1e-9,
) -> bool:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    dot = (px - x1) * (px - x2) + (py - y1) * (py - y2)
    return dot <= eps


def _point_in_polygon_inclusive(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Return True if point lies inside or on the boundary of a polygon."""
    if polygon[0] == polygon[-1]:
        polygon = polygon[:-1]

    x, y = point
    inside = False
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        if _point_on_segment(point, start, end):
            return True
        x1, y1 = start
        x2, y2 = end
        if (y1 > y) != (y2 > y):
            x_intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersection:
                inside = not inside
    return inside


def _segment_edge_intersection_t_values(
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
    eps: float = 1e-9,
) -> list[float]:
    """Return segment parameters where a segment intersects a polygon edge."""
    px, py = seg_start
    rx = seg_end[0] - seg_start[0]
    ry = seg_end[1] - seg_start[1]
    qx, qy = edge_start
    sx = edge_end[0] - edge_start[0]
    sy = edge_end[1] - edge_start[1]

    denom = rx * sy - ry * sx
    q_minus_p_x = qx - px
    q_minus_p_y = qy - py

    if abs(denom) <= eps:
        cross = q_minus_p_x * ry - q_minus_p_y * rx
        if abs(cross) > eps:
            return []

        seg_len2 = rx * rx + ry * ry
        if seg_len2 <= eps:
            return []
        t0 = ((edge_start[0] - px) * rx + (edge_start[1] - py) * ry) / seg_len2
        t1 = ((edge_end[0] - px) * rx + (edge_end[1] - py) * ry) / seg_len2
        lo = max(0.0, min(t0, t1))
        hi = min(1.0, max(t0, t1))
        if hi < lo - eps:
            return []
        return [max(0.0, min(1.0, lo)), max(0.0, min(1.0, hi))]

    t = (q_minus_p_x * sy - q_minus_p_y * sx) / denom
    u = (q_minus_p_x * ry - q_minus_p_y * rx) / denom
    if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
        return [max(0.0, min(1.0, t))]
    return []


def _unique_sorted(values: list[float], eps: float = 1e-9) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        clipped = max(0.0, min(1.0, value))
        if not result or abs(clipped - result[-1]) > eps:
            result.append(clipped)
    return result


def _segment_polygon_overlap_intervals(
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
    polygon: list[tuple[float, float]],
    eps: float = 1e-9,
) -> list[tuple[float, float]]:
    """Return segment-parameter intervals covered by a polygon footprint."""
    if polygon[0] == polygon[-1]:
        polygon = polygon[:-1]

    t_values = [0.0, 1.0]
    for edge_start, edge_end in zip(polygon, polygon[1:] + polygon[:1]):
        t_values.extend(_segment_edge_intersection_t_values(seg_start, seg_end, edge_start, edge_end, eps))

    t_values = _unique_sorted(t_values, eps)
    intervals: list[tuple[float, float]] = []
    dx = seg_end[0] - seg_start[0]
    dy = seg_end[1] - seg_start[1]
    for t0, t1 in zip(t_values, t_values[1:]):
        if t1 - t0 <= eps:
            continue
        mid = (t0 + t1) / 2.0
        midpoint = (seg_start[0] + dx * mid, seg_start[1] + dy * mid)
        if _point_in_polygon_inclusive(midpoint, polygon):
            intervals.append((t0, t1))
    return intervals


def _covered_interval_area(
    intervals: list[tuple[float, float, float]],
    segment_length_m: float,
    eps: float = 1e-9,
) -> float:
    """Calculate covered wall area from interval-height overlaps on one wall segment."""
    if not intervals or segment_length_m <= 0.0:
        return 0.0
    t_values = _unique_sorted([value for interval in intervals for value in interval[:2]], eps)
    area_m2 = 0.0
    for t0, t1 in zip(t_values, t_values[1:]):
        if t1 - t0 <= eps:
            continue
        mid = (t0 + t1) / 2.0
        covered_height = max((height for start, end, height in intervals if start - eps <= mid <= end + eps), default=0.0)
        area_m2 += (t1 - t0) * segment_length_m * covered_height
    return area_m2


def _read_geometry_rows(geometry_csv: Path) -> list[dict[str, str]]:
    with geometry_csv.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _geometry_fields(row: dict[str, str]) -> tuple[str, float]:
    """Return footprint WKT and height for either supported Adminer CSV schema."""
    geometry = row.get("footprint_geometry") or row.get("geometry")
    height = row.get("measured_height") or row.get("building_height")
    if geometry is None or height is None:
        raise KeyError(
            "Geometry CSV must contain either footprint_geometry/measured_height "
            "or geometry/building_height columns."
        )
    return geometry, float(height)


def _building_iri(row: dict[str, str]) -> str:
    """Return the supplied IRI, or construct the canonical IRI from UUID."""
    iri = (row.get("iri") or "").strip()
    if iri:
        return iri
    uuid = (row.get("uuid") or "").strip()
    if not uuid:
        raise KeyError("Geometry CSV must contain either an iri or uuid column.")
    return BUILDING_IRI_PREFIX + uuid_from_iri_or_uuid(uuid)


def _read_footprints_for_iris(
    geometry_csv: Path, wanted_iris: set[str]
) -> dict[str, list[tuple[float, float]]]:
    """Read footprint polygons for the requested building IRIs only."""
    footprints_by_iri = _load_all_footprints_cached(str(geometry_csv.resolve()))
    footprints = {iri: footprints_by_iri[iri] for iri in wanted_iris if iri in footprints_by_iri}
    missing = wanted_iris - footprints.keys()
    if missing:
        raise KeyError(f"Footprint geometry not found for {len(missing)} building(s).")
    return footprints


@lru_cache(maxsize=4)
def _load_all_footprints_cached(geometry_csv: str) -> dict[str, list[tuple[float, float]]]:
    """Cache parsed footprints for efficient batch feature generation."""
    footprints: dict[str, list[tuple[float, float]]] = {}
    with Path(geometry_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            geometry, _ = _geometry_fields(row)
            footprints[_building_iri(row)] = parse_wkt_points(geometry)
    return footprints


def _index_file_name(cell_size_m: float, geometry_csv: Path) -> str:
    stem = geometry_csv.stem
    size = int(cell_size_m) if float(cell_size_m).is_integer() else str(cell_size_m).replace(".", "p")
    return f"{stem}_grid_index_{size}m.json"


def build_grid_index(
    geometry_csv: str | Path = DEFAULT_GEOMETRY_CSV,
    cell_size_m: float = 50.0,
    index_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Build the grid-to-building-IRI index JSON if it does not exist.

    The JSON is stored in the current working directory by default.
    It contains both the grid mapping and compact per-building geometry data,
    so feature calculation does not need to re-parse the CSV every time.
    """
    geometry_csv = Path(geometry_csv)
    if not geometry_csv.exists():
        raise FileNotFoundError(f"Geometry CSV not found: {geometry_csv}")

    index_path = Path(index_path) if index_path else Path.cwd() / _index_file_name(cell_size_m, geometry_csv)
    if index_path.exists() and not force:
        return index_path

    rows = _read_geometry_rows(geometry_csv)
    parsed = []
    lon_sum = 0.0
    lat_sum = 0.0
    count = 0

    # The source can contain multiple polygon rows for one building UUID.
    # The index model stores one footprint per IRI, so retain the final CSV row,
    # matching the module's cached-footprint dictionary semantics.
    rows_by_iri = {_building_iri(row): row for row in rows}
    for iri, row in rows_by_iri.items():
        uuid = uuid_from_iri_or_uuid(iri)
        geometry, height_m = _geometry_fields(row)
        points = parse_wkt_points(geometry)
        lon_c, lat_c = polygon_centroid(points)
        parsed.append(
            {
                "iri": iri,
                "uuid": uuid,
                "height_m": height_m,
                "points": points,
                "centroid_lon": lon_c,
                "centroid_lat": lat_c,
            }
        )
        lon_sum += lon_c
        lat_sum += lat_c
        count += 1

    lon0 = lon_sum / count
    lat0 = lat_sum / count

    buildings: dict[str, dict[str, Any]] = {}
    uuid_to_iri: dict[str, str] = {}
    grid: dict[str, list[str]] = {}

    for item in parsed:
        centroid_x, centroid_y = project_points([(item["centroid_lon"], item["centroid_lat"])], lon0, lat0)[0]
        area_m2 = polygon_area_m2(project_points(item["points"], lon0, lat0))
        cell_x = math.floor(centroid_x / cell_size_m)
        cell_y = math.floor(centroid_y / cell_size_m)
        cell_key = f"{cell_x},{cell_y}"

        iri = item["iri"]
        buildings[iri] = {
            "uuid": item["uuid"],
            "height_m": item["height_m"],
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "footprint_area_m2": area_m2,
            "cell": [cell_x, cell_y],
        }
        uuid_to_iri[item["uuid"]] = iri
        grid.setdefault(cell_key, []).append(iri)

    payload = {
        "metadata": {
            "geometry_csv": str(geometry_csv.resolve()),
            "cell_size_m": cell_size_m,
            "projection": "local equirectangular",
            "origin_lon": lon0,
            "origin_lat": lat0,
            "building_count": len(buildings),
        },
        "grid": grid,
        "uuid_to_iri": uuid_to_iri,
        "buildings": buildings,
    }

    index_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return index_path


@lru_cache(maxsize=8)
def _load_index_cached(index_path: str) -> dict[str, Any]:
    return json.loads(Path(index_path).read_text(encoding="utf-8"))


def _candidate_neighbour_iris(index: dict[str, Any], target: dict[str, Any], radius_m: float) -> list[str]:
    cell_size = float(index["metadata"]["cell_size_m"])
    cx, cy = target["cell"]
    reach = math.ceil(radius_m / cell_size) + 1
    iris: list[str] = []
    grid = index["grid"]
    for gx in range(cx - reach, cx + reach + 1):
        for gy in range(cy - reach, cy + reach + 1):
            iris.extend(grid.get(f"{gx},{gy}", []))
    return iris


def _polygon_prism_faces(
    points_xy: list[tuple[float, float]], height_m: float
) -> list[list[tuple[float, float, float]]]:
    """Create bottom, top, and wall faces for a footprint extruded by height."""
    if points_xy[0] == points_xy[-1]:
        points_xy = points_xy[:-1]

    bottom = [(x, y, 0.0) for x, y in points_xy]
    top = [(x, y, height_m) for x, y in points_xy]
    faces = [bottom, top]

    for i, current in enumerate(points_xy):
        next_i = (i + 1) % len(points_xy)
        faces.append(
            [
                (current[0], current[1], 0.0),
                (points_xy[next_i][0], points_xy[next_i][1], 0.0),
                (points_xy[next_i][0], points_xy[next_i][1], height_m),
                (current[0], current[1], height_m),
            ]
        )
    return faces


def _set_3d_axes_equal(ax: Any, xs: list[float], ys: list[float], zs: list[float]) -> None:
    """Keep metres visually comparable across x, y, and z axes."""
    x_mid = (max(xs) + min(xs)) / 2.0
    y_mid = (max(ys) + min(ys)) / 2.0
    z_mid = (max(zs) + min(zs)) / 2.0
    half_range = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0) / 2.0

    ax.set_xlim(x_mid - half_range, x_mid + half_range)
    ax.set_ylim(y_mid - half_range, y_mid + half_range)
    ax.set_zlim(max(0.0, z_mid - half_range), z_mid + half_range)
    try:
        ax.set_box_aspect((1, 1, 0.55))
    except AttributeError:
        pass


def _plot_surrounding_geometries(
    *,
    index: dict[str, Any],
    geometry_csv: Path,
    target_iri: str,
    neighbours: list[tuple[str, dict[str, Any], float]],
    radius_m: float,
) -> None:
    """Plot target and surrounding buildings as interactive 3D prisms."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as exc:
        raise ImportError(
            "plot_geometries=True requires matplotlib. Install it in the active Python "
            "environment or run this script from an environment that already has it."
        ) from exc

    target = index["buildings"][target_iri]
    selected_iris = {target_iri, *(iri for iri, _, _ in neighbours)}
    footprints = _read_footprints_for_iris(geometry_csv, selected_iris)
    lon0 = float(index["metadata"]["origin_lon"])
    lat0 = float(index["metadata"]["origin_lat"])
    tx = float(target["centroid_x"])
    ty = float(target["centroid_y"])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    all_x: list[float] = []
    all_y: list[float] = []
    all_z: list[float] = [0.0]

    def add_building(iri: str, face_color: str, edge_color: str, alpha: float) -> None:
        building = index["buildings"][iri]
        points_xy = project_points(footprints[iri], lon0, lat0)
        centred = [(x - tx, y - ty) for x, y in points_xy]
        height = float(building["height_m"])
        faces = _polygon_prism_faces(centred, height)
        collection = Poly3DCollection(
            faces,
            facecolors=face_color,
            edgecolors=edge_color,
            linewidths=0.7,
            alpha=alpha,
        )
        ax.add_collection3d(collection)
        all_x.extend(x for x, _ in centred)
        all_y.extend(y for _, y in centred)
        all_z.append(height)

    for iri, _, _ in neighbours:
        add_building(iri, face_color="#7B2CBF", edge_color="#4B136E", alpha=0.55)
    add_building(target_iri, face_color="#FFD60A", edge_color="#9A6A00", alpha=0.9)

    if all_x and all_y:
        padding = max(radius_m * 0.1, 5.0)
        all_x.extend([-radius_m - padding, radius_m + padding])
        all_y.extend([-radius_m - padding, radius_m + padding])
        _set_3d_axes_equal(ax, all_x, all_y, all_z)

    ax.set_xlabel("East/West from target centroid (m)")
    ax.set_ylabel("North/South from target centroid (m)")
    ax.set_zlabel("Height (m)")
    ax.set_title(f"Surrounding buildings within {radius_m:g} m")
    ax.view_init(elev=28, azim=-45)
    ax.legend(
        handles=[
            Patch(facecolor="#FFD60A", edgecolor="#9A6A00", label="Target building"),
            Patch(facecolor="#7B2CBF", edgecolor="#4B136E", label="Surroundings"),
        ],
        loc="upper right",
    )
    plt.tight_layout()
    plt.show()


def _calculate_wall_coverage_ratios(
    *,
    index: dict[str, Any],
    geometry_csv: Path,
    target_iri: str,
    neighbours: list[tuple[str, dict[str, Any], float]],
) -> dict[str, float]:
    """Calculate directional target-wall area covered by neighbouring building volumes."""
    ratios = {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0}
    if not neighbours:
        return ratios

    target = index["buildings"][target_iri]
    selected_iris = {target_iri, *(iri for iri, _, _ in neighbours)}
    footprints = _read_footprints_for_iris(geometry_csv, selected_iris)
    lon0 = float(index["metadata"]["origin_lon"])
    lat0 = float(index["metadata"]["origin_lat"])

    target_xy = project_points(footprints[target_iri], lon0, lat0)
    if target_xy[0] == target_xy[-1]:
        target_xy = target_xy[:-1]

    target_height = max(0.0, float(target["height_m"]))
    if target_height == 0.0 or len(target_xy) < 3:
        return ratios

    signed_area = signed_polygon_area_m2(target_xy)
    is_ccw = signed_area >= 0.0
    direction_total_area = {direction: 0.0 for direction in ratios}
    direction_covered_area = {direction: 0.0 for direction in ratios}

    neighbour_polygons = []
    for iri, building, _ in neighbours:
        points_xy = project_points(footprints[iri], lon0, lat0)
        if points_xy[0] == points_xy[-1]:
            points_xy = points_xy[:-1]
        if len(points_xy) >= 3:
            neighbour_polygons.append((points_xy, max(0.0, float(building["height_m"]))))

    for start, end in zip(target_xy, target_xy[1:] + target_xy[:1]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length = math.hypot(dx, dy)
        if segment_length <= 0.0:
            continue

        # For a counter-clockwise exterior ring the polygon interior is on the
        # left side of each edge, so the outward normal is the right normal.
        if is_ccw:
            nx, ny = dy / segment_length, -dx / segment_length
        else:
            nx, ny = -dy / segment_length, dx / segment_length

        direction = wall_normal_direction(nx, ny)
        if direction is None:
            continue

        wall_area = segment_length * target_height
        direction_total_area[direction] += wall_area

        covered_intervals: list[tuple[float, float, float]] = []
        for neighbour_polygon, neighbour_height in neighbour_polygons:
            if neighbour_height <= 0.0:
                continue
            covered_height = min(target_height, neighbour_height)
            for t0, t1 in _segment_polygon_overlap_intervals(start, end, neighbour_polygon):
                covered_intervals.append((t0, t1, covered_height))

        direction_covered_area[direction] += _covered_interval_area(covered_intervals, segment_length)

    for direction in ratios:
        total_area = direction_total_area[direction]
        if total_area > 0.0:
            ratios[direction] = min(1.0, max(0.0, direction_covered_area[direction] / total_area))

    return ratios


def get_directional_surface_areas(
    building_uuid_or_iri: str,
    geometry_csv: str | Path = DEFAULT_GEOMETRY_CSV,
    cell_size_m: float = 50.0,
    index_path: str | Path | None = None,
) -> dict[str, float]:
    """Return roof and cardinal facade areas in square metres.

    Wall edges use the same outward-normal-to-cardinal assignment as the wall
    coverage features, so these areas align exactly with the R/N/S/E/W solar
    output convention used by the surrogate dataset.
    """
    geometry_csv = Path(geometry_csv)
    index_path = build_grid_index(geometry_csv, cell_size_m=cell_size_m, index_path=index_path)
    index = _load_index_cached(str(index_path.resolve()))
    query_uuid = uuid_from_iri_or_uuid(building_uuid_or_iri)
    target_iri = index["uuid_to_iri"].get(query_uuid)
    if target_iri is None:
        raise KeyError(f"Building UUID not found in grid index: {query_uuid}")

    target = index["buildings"][target_iri]
    footprint = _read_footprints_for_iris(geometry_csv, {target_iri})[target_iri]
    lon0 = float(index["metadata"]["origin_lon"])
    lat0 = float(index["metadata"]["origin_lat"])
    target_xy = project_points(footprint, lon0, lat0)
    if target_xy[0] == target_xy[-1]:
        target_xy = target_xy[:-1]

    areas = {
        "roof_area_m2": polygon_area_m2(target_xy),
        "north_facade_area_m2": 0.0,
        "south_facade_area_m2": 0.0,
        "east_facade_area_m2": 0.0,
        "west_facade_area_m2": 0.0,
    }
    height = max(0.0, float(target["height_m"]))
    if height == 0.0 or len(target_xy) < 3:
        return areas

    is_ccw = signed_polygon_area_m2(target_xy) >= 0.0
    for start, end in zip(target_xy, target_xy[1:] + target_xy[:1]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length = math.hypot(dx, dy)
        if segment_length <= 0.0:
            continue
        if is_ccw:
            nx, ny = dy / segment_length, -dx / segment_length
        else:
            nx, ny = -dy / segment_length, dx / segment_length
        direction = wall_normal_direction(nx, ny)
        if direction is not None:
            areas[f"{direction}_facade_area_m2"] += segment_length * height
    return areas


def get_surrounding_features(
    building_uuid_or_iri: str,
    plot_geometries: bool = False,
    geometry_csv: str | Path = DEFAULT_GEOMETRY_CSV,
    cell_size_m: float = 50.0,
    radius_m: float = 50.0,
    index_path: str | Path | None = None,
    print_neighbours: bool = False,
    include_neighbour_uuids: bool = False,
    revert_old: bool = False,
) -> dict[str, float | int | None | list[str]]:
    """Return surrounding-building features for one building UUID or IRI.

    Features returned by default:
        num_neighbours_50m
        density_50m
        max_height_50m
        mean_height_50m
        nearest_taller_distance
        north_obstruction_angle
        south_obstruction_angle
        east_obstruction_angle
        west_obstruction_angle
        roof_obstruction_proxy
        north_wall_coverage_ratio
        south_wall_coverage_ratio
        east_wall_coverage_ratio
        west_wall_coverage_ratio

    Optional:
        - print_neighbours=True prints UUIDs of all neighbours within radius_m.
        - include_neighbour_uuids=True includes them in the returned dict under
          neighbour_uuids_50m.
        - plot_geometries=True opens an interactive 3D matplotlib window with
          the target building in yellow and surrounding buildings in purple.
        - revert_old=True returns only the original 10 surrounding features.

    Notes:
        - Neighbours are selected by centroid distance <= radius_m.
        - density_50m is total neighbour footprint area divided by the
          circular 50 m search area.
        - obstruction angles are degrees of atan((h_neighbour-h_target)/d),
          clipped at zero and grouped by target-to-neighbour direction.
        - roof_obstruction_proxy is the maximum obstruction angle over all
          neighbouring buildings, useful as a compact shading-context feature.
        - wall coverage ratios are in [0, 1], with walls assigned to the nearest
          cardinal direction using their projected outward normal.
    """
    geometry_csv = Path(geometry_csv)
    index_path = build_grid_index(geometry_csv, cell_size_m=cell_size_m, index_path=index_path)
    index = _load_index_cached(str(index_path.resolve()))

    query_uuid = uuid_from_iri_or_uuid(building_uuid_or_iri)
    target_iri = index["uuid_to_iri"].get(query_uuid)
    if target_iri is None:
        raise KeyError(
            f"Building UUID not found in grid index: {query_uuid}. "
            f"The grid index is built from {index['metadata']['geometry_csv']}; "
            "the queried building must appear there with footprint_geometry and measured_height."
        )

    buildings = index["buildings"]
    target = buildings[target_iri]
    tx = float(target["centroid_x"])
    ty = float(target["centroid_y"])
    th = float(target["height_m"])

    neighbours: list[tuple[str, dict[str, Any], float]] = []
    for iri in _candidate_neighbour_iris(index, target, radius_m):
        if iri == target_iri:
            continue
        b = buildings[iri]
        dx = float(b["centroid_x"]) - tx
        dy = float(b["centroid_y"]) - ty
        distance = math.hypot(dx, dy)
        if 0.0 < distance <= radius_m:
            neighbours.append((iri, b, distance))

    num_neighbours = len(neighbours)
    search_area_m2 = math.pi * radius_m * radius_m
    total_neighbour_area = sum(float(b["footprint_area_m2"]) for _, b, _ in neighbours)
    density = total_neighbour_area / search_area_m2 if search_area_m2 > 0 else 0.0

    if neighbours:
        heights = [float(b["height_m"]) for _, b, _ in neighbours]
        max_height = max(heights)
        mean_height = sum(heights) / len(heights)
    else:
        max_height = 0.0
        mean_height = 0.0

    taller_distances = [dist for _, b, dist in neighbours if float(b["height_m"]) > th]
    nearest_taller = min(taller_distances) if taller_distances else None

    direction_angles = {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0}
    roof_proxy = 0.0

    for _, b, distance in neighbours:
        dx = float(b["centroid_x"]) - tx
        dy = float(b["centroid_y"]) - ty
        dh = max(0.0, float(b["height_m"]) - th)
        angle = math.degrees(math.atan2(dh, distance)) if distance > 0 else 0.0
        direction = bearing_direction(dx, dy)
        if direction is not None:
            direction_angles[direction] = max(direction_angles[direction], angle)
        roof_proxy = max(roof_proxy, angle)

    suffix = f"{int(radius_m)}m" if float(radius_m).is_integer() else f"{radius_m}m"
    neighbour_uuids = [str(b["uuid"]) for _, b, _ in sorted(neighbours, key=lambda item: item[2])]

    if print_neighbours:
        print(f"Neighbours within {radius_m:g} m for {query_uuid}:")
        if neighbour_uuids:
            for uuid in neighbour_uuids:
                print(uuid)
        else:
            print("(none)")

    if plot_geometries:
        _plot_surrounding_geometries(
            index=index,
            geometry_csv=geometry_csv,
            target_iri=target_iri,
            neighbours=neighbours,
            radius_m=radius_m,
        )

    features: dict[str, float | int | None | list[str]] = {
        f"num_neighbours_{suffix}": num_neighbours,
        f"density_{suffix}": density,
        f"max_height_{suffix}": max_height,
        f"mean_height_{suffix}": mean_height,
        "nearest_taller_distance": nearest_taller,
        "north_obstruction_angle": direction_angles["north"],
        "south_obstruction_angle": direction_angles["south"],
        "east_obstruction_angle": direction_angles["east"],
        "west_obstruction_angle": direction_angles["west"],
        "roof_obstruction_proxy": roof_proxy,
    }

    if not revert_old:
        wall_ratios = _calculate_wall_coverage_ratios(
            index=index,
            geometry_csv=geometry_csv,
            target_iri=target_iri,
            neighbours=neighbours,
        )
        features.update(
            {
                "north_wall_coverage_ratio": wall_ratios["north"],
                "south_wall_coverage_ratio": wall_ratios["south"],
                "east_wall_coverage_ratio": wall_ratios["east"],
                "west_wall_coverage_ratio": wall_ratios["west"],
            }
        )

    if include_neighbour_uuids:
        features[f"neighbour_uuids_{suffix}"] = neighbour_uuids

    return features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate 50 m surrounding features for a building.")
    parser.add_argument("building", help="Building UUID or full IRI")
    parser.add_argument("--geometry-csv", type=Path, default=DEFAULT_GEOMETRY_CSV)
    parser.add_argument("--cell-size-m", type=float, default=50.0)
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--print-neighbours", action="store_true", help="Print UUIDs of buildings within radius.")
    parser.add_argument("--include-neighbours", action="store_true", help="Include neighbour UUIDs in JSON output.")
    parser.add_argument("--revert-old", action="store_true", help="Return the original 10-feature output.")
    parser.add_argument(
        "--plot-geometries",
        action="store_true",
        help="Open an interactive 3D window of target and surrounding building geometries.",
    )
    args = parser.parse_args()

    result = get_surrounding_features(
        args.building,
        geometry_csv=args.geometry_csv,
        cell_size_m=args.cell_size_m,
        radius_m=args.radius_m,
        index_path=args.index_path,
        print_neighbours=args.print_neighbours,
        include_neighbour_uuids=args.include_neighbours,
        plot_geometries=args.plot_geometries,
        revert_old=args.revert_old,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
