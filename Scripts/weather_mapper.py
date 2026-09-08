"""Map a building to its nearest Kaiserslautern weather-station IRI.

Only the Python standard library is used.  ``geometry_csv`` must contain an
``iri`` (or ``building_iri``) column and a WKT geometry column named
``footprint_geometry`` (or ``geometry`` / ``wkt``).
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path


# (station IRI, latitude, longitude), sourced from Kaiserslautern all_stations.csv.
WEATHER_STATIONS = (
    ("https://www.theworldavatar.com/kg/ontoems/ReportingStation_ffb16e9d-825f-48ae-bc83-05ff587ef5fb", 49.42237, 7.73918),
    ("https://www.theworldavatar.com/kg/ontoems/ReportingStation_b5792e91-6d44-4612-9f5e-755f6aa9fc88", 49.39394, 7.67277),
    ("https://www.theworldavatar.com/kg/ontoems/ReportingStation_493f06ee-80ca-4fca-ac43-81ecf1c85c48", 49.39380, 7.75211),
    ("https://www.theworldavatar.com/kg/ontoems/ReportingStation_deb98ddb-cbb2-44bb-8c02-5e463582e084", 49.39393, 7.88223),
    ("https://www.theworldavatar.com/kg/ontoems/ReportingStation_27334e2b-8bbe-4949-b556-05dcdd74dcfb", 49.39458, 7.71708),
    ("https://www.theworldavatar.com/kg/ontoems/ReportingStation_fd360e95-5155-4b07-a9d0-8242557562b7", 49.40225, 7.78262),
    ("https://www.theworldavatar.com/kg/ontoems/ReportingStation_b9e3aa38-d6ea-47e4-81dc-c3d0dffdd4d2", 49.43431, 7.76503),
    ("https://www.theworldavatar.com/kg/ontoems/ReportingStation_d55c150f-9d60-4e3c-9fe6-c2cf486750d7", 49.42233, 7.68617),
    ("https://www.theworldavatar.com/kg/ontoems/ReportingStation_0ef2cab7-daef-4ffd-96b8-eb3b8b366fed", 49.42793, 7.71278),
    ("https://www.theworldavatar.com/kg/ontoems/ReportingStation_f2d348c5-99f2-4b01-87e4-f760d8ca8123", 52.52000, 14.41000),
    ("https://www.theworldavatar.com/kg/ontoems/ReportingStation_7376c51e-e511-4684-afa2-6579ce6e0de6", 49.44044, 7.73313),
    ("https://www.theworldavatar.com/kg/ontoems/ReportingStation_69b2f245-a0c9-481c-ba30-1a634fe692b0", 49.44128, 7.79138),
)

UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _identifier_key(value: str) -> str:
    """Use the UUID when present, otherwise compare the complete IRI."""
    match = UUID_RE.search(str(value))
    return match.group(0).lower() if match else str(value).strip().rstrip("/")


def _wkt_centroid(wkt: str) -> tuple[float, float]:
    """Return (latitude, longitude) from a POLYGON/POLYGON Z footprint."""
    upper = wkt.upper()
    dimension = 3 if re.match(r"^\s*(?:MULTI)?POLYGON\s+(?:Z|ZM)\b", upper) else 2
    values = [float(value) for value in NUMBER_RE.findall(wkt)]
    if len(values) < dimension * 3:
        raise ValueError("Building footprint contains fewer than three points")
    points = [(values[i], values[i + 1]) for i in range(0, len(values) - dimension + 1, dimension)]
    if points[0] == points[-1]:
        points.pop()

    # Polygon centroid in lon/lat coordinates; fall back to the vertex mean
    # for degenerate footprints.
    cross_sum = cx_sum = cy_sum = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        cross = x1 * y2 - x2 * y1
        cross_sum += cross
        cx_sum += (x1 + x2) * cross
        cy_sum += (y1 + y2) * cross
    if abs(cross_sum) < 1e-15:
        longitude = sum(point[0] for point in points) / len(points)
        latitude = sum(point[1] for point in points) / len(points)
    else:
        longitude = cx_sum / (3.0 * cross_sum)
        latitude = cy_sum / (3.0 * cross_sum)
    return latitude, longitude


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance using the haversine formula."""
    radius_m = 6_371_008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_m * math.asin(math.sqrt(a))


def mapweather(building_IRI: str, geometry_csv: str | Path) -> str:
    """Return the IRI of the weather station nearest to ``building_IRI``.

    ``building_IRI`` may also be a bare building UUID.  Distance is measured
    from the building-footprint centroid to each weather station.
    """
    wanted = _identifier_key(building_IRI)
    geometry_path = Path(geometry_csv)
    with geometry_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Geometry CSV has no header: {geometry_path}")
        iri_column = next((name for name in ("iri", "building_iri", "uuid") if name in reader.fieldnames), None)
        geometry_column = next(
            (name for name in ("footprint_geometry", "geometry", "wkt") if name in reader.fieldnames), None
        )
        if iri_column is None or geometry_column is None:
            raise ValueError(
                "Geometry CSV needs iri/building_iri/uuid and footprint_geometry/geometry/wkt columns"
            )
        footprint = next(
            (row[geometry_column] for row in reader if _identifier_key(row[iri_column]) == wanted),
            None,
        )

    if footprint is None:
        raise KeyError(f"Building not found in geometry CSV: {building_IRI}")
    building_lat, building_lon = _wkt_centroid(footprint)
    return min(
        WEATHER_STATIONS,
        key=lambda station: _distance_m(building_lat, building_lon, station[1], station[2]),
    )[0]


__all__ = ["mapweather"]
