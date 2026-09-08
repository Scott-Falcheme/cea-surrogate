"""Benchmark Final_bundle demand and solar inference on random Kaiserslautern buildings.

The benchmark samples building UUIDs whose mapped time-series CSV exists in the
authoritative 2714 directory and whose geometry and usage records are available.
It times the deployed three-level demand cascade and the full solar ROM chain
separately. Predictions are processed in memory-safe batches and are not saved.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parent
DATA_DIR = BUNDLE_ROOT / "data"
SOLAR_SCRIPT_DIR = SCRIPT_DIR / "solar"
DEFAULT_TARGET_DIR = Path(
    r"C:\Users\ROG\OneDrive - University of Cambridge\Study\THESIS"
    r"\City data\Kaiserslautern\2714"
)
BUILDING_IRI_PREFIX = "https://theworldavatar.io/kg/Building/"
TABLE_FILE_RE = re.compile(
    r"^building_([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.csv$"
)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SOLAR_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SOLAR_SCRIPT_DIR))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from demand_input_agent import demand_input  # noqa: E402
from demand_pipline import DEMAND_COLUMNS, demand_surrogate  # noqa: E402
from solar.solar_pipline import solar_surrogate  # noqa: E402
from weather_mapper import mapweather  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--geometry", type=Path, default=DATA_DIR / "building_geometry_full.csv")
    parser.add_argument("--usage", type=Path, default=DATA_DIR / "building_usage.csv")
    parser.add_argument("--mapping", type=Path, default=DATA_DIR / "target_mapping.csv")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--side", choices=("both", "demand", "solar"), default="both")
    parser.add_argument(
        "--output-dir", type=Path, default=BUNDLE_ROOT / "runtime_results"
    )
    parser.add_argument(
        "--warmup", action=argparse.BooleanOptionalAction, default=True,
        help="Run one extra, untimed building before each selected benchmark side.",
    )
    return parser.parse_args()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _target_table_uuids(target_dir: Path) -> set[str]:
    if not target_dir.is_dir():
        raise FileNotFoundError(f"2714 target directory not found: {target_dir}")
    uuids: set[str] = set()
    for path in target_dir.iterdir():
        match = TABLE_FILE_RE.match(path.name)
        if match:
            uuids.add(match.group(1).lower())
    if not uuids:
        raise ValueError(f"No building_<UUID>.csv files found in {target_dir}")
    return uuids


def eligible_building_uuids(
    target_dir: Path, mapping_path: Path, geometry_path: Path, usage_path: Path
) -> list[str]:
    """Return sorted building UUIDs available to both deployed chains."""
    for path, label in (
        (mapping_path, "target mapping"),
        (geometry_path, "geometry CSV"),
        (usage_path, "usage CSV"),
    ):
        _require_file(path, label)

    target_tables = _target_table_uuids(target_dir)
    mapping = pd.read_csv(
        mapping_path, usecols=["timeseries_table", "building_uuid"], dtype=str
    ).dropna()
    mapping["timeseries_table"] = mapping["timeseries_table"].str.lower()
    mapping["building_uuid"] = mapping["building_uuid"].str.lower()
    mapping = mapping[mapping["timeseries_table"].isin(target_tables)]

    geometry_ids = set(
        pd.read_csv(geometry_path, usecols=["uuid"], dtype=str)["uuid"]
        .dropna().str.lower()
    )
    usage_ids = set(
        pd.read_csv(usage_path, usecols=["building_iri"], dtype=str)["building_iri"]
        .dropna().str.lower()
    )
    eligible = sorted(set(mapping["building_uuid"]) & geometry_ids & usage_ids)
    if not eligible:
        raise ValueError("No common buildings across targets, mapping, geometry, and usage")
    return eligible


def _sync_accelerator() -> None:
    """Synchronise CUDA, if used, so wall-clock timing includes queued kernels."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _prepare_demand_inputs(
    iris: list[str], geometry_path: Path, usage_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build inputs using each building's nearest weather station."""
    by_weather: dict[str, list[str]] = {}
    for iri in iris:
        by_weather.setdefault(mapweather(iri, str(geometry_path)), []).append(iri)
    static_blocks: list[pd.DataFrame] = []
    hourly_blocks: list[pd.DataFrame] = []
    for weather_iri, group in by_weather.items():
        static, hourly = demand_input(
            group, weather_iri, str(geometry_path), str(usage_path)
        )
        static_blocks.append(static)
        hourly_blocks.append(hourly)
    static_all = pd.concat(static_blocks, axis=1).reindex(columns=iris)
    hourly_all = pd.concat(hourly_blocks, axis=0).loc[iris]
    return static_all, hourly_all


def _run_warmup(
    iri: str, side: str, geometry_path: Path, usage_path: Path
) -> None:
    if side == "demand":
        demand_surrogate(_prepare_demand_inputs([iri], geometry_path, usage_path))
    else:
        solar_surrogate([iri], str(geometry_path))


def benchmark_side(
    side: str,
    iris: list[str],
    batch_size: int,
    geometry_path: Path,
    usage_path: Path,
) -> tuple[dict, list[dict]]:
    batch_rows: list[dict] = []
    _sync_accelerator()
    wall_start = time.perf_counter()
    total_preparation = 0.0
    total_prediction = 0.0
    completed = 0
    for batch_number, start in enumerate(range(0, len(iris), batch_size), start=1):
        batch = iris[start : start + batch_size]
        preparation_start = time.perf_counter()
        demand_inputs = (
            _prepare_demand_inputs(batch, geometry_path, usage_path)
            if side == "demand" else None
        )
        preparation_elapsed = time.perf_counter() - preparation_start
        _sync_accelerator()
        prediction_start = time.perf_counter()
        if side == "demand":
            _, output = demand_surrogate(demand_inputs)
            expected = len(batch) * len(DEMAND_COLUMNS)
            actual = output.shape[1]
        else:
            output, _ = solar_surrogate(batch, str(geometry_path))
            expected = len(batch) * 35
            actual = len(output) if len(batch) > 1 else output.shape[0]
        _sync_accelerator()
        prediction_elapsed = time.perf_counter() - prediction_start
        if actual != expected:
            raise RuntimeError(
                f"{side} output has {actual} channel rows/columns; expected {expected}"
            )
        total_preparation += preparation_elapsed
        total_prediction += prediction_elapsed
        completed += len(batch)
        row = {
            "side": side,
            "batch": batch_number,
            "n_buildings": len(batch),
            "preparation_seconds": preparation_elapsed,
            "prediction_seconds": prediction_elapsed,
            "prediction_seconds_per_building": prediction_elapsed / len(batch),
            "prediction_buildings_per_second": len(batch) / prediction_elapsed,
        }
        batch_rows.append(row)
        print(
            f"[{side}] {completed}/{len(iris)} buildings; "
            f"prep {preparation_elapsed:.3f} s; prediction {prediction_elapsed:.3f} s; "
            f"wall {time.perf_counter() - wall_start:.3f} s",
            flush=True,
        )
        gc.collect()

    wall_elapsed = time.perf_counter() - wall_start
    channels = 4 if side == "demand" else 35
    summary = {
        "side": side,
        "n_buildings": len(iris),
        "batch_size": batch_size,
        "preparation_seconds": total_preparation,
        "prediction_seconds": total_prediction,
        "wall_seconds": wall_elapsed,
        "prediction_seconds_per_building": total_prediction / len(iris),
        "prediction_buildings_per_second": len(iris) / total_prediction,
        "hourly_output_values": len(iris) * 8760 * channels,
        "hourly_output_values_per_prediction_second": (
            len(iris) * 8760 * channels / total_prediction
        ),
    }
    return summary, batch_rows


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    eligible = eligible_building_uuids(
        args.target_dir, args.mapping, args.geometry, args.usage
    )
    extra = 1 if args.warmup else 0
    if len(eligible) < args.sample_size + extra:
        raise ValueError(
            f"Requested {args.sample_size} timed buildings plus {extra} warm-up, "
            f"but only {len(eligible)} buildings are eligible"
        )

    rng = random.Random(args.seed)
    selected = rng.sample(eligible, args.sample_size + extra)
    warmup_iri = BUILDING_IRI_PREFIX + selected[0] if args.warmup else None
    timed_uuids = selected[extra:]
    timed_iris = [BUILDING_IRI_PREFIX + value for value in timed_uuids]
    sides = ("demand", "solar") if args.side == "both" else (args.side,)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_frame = pd.DataFrame(
        {"sample_order": range(1, len(timed_uuids) + 1), "building_uuid": timed_uuids}
    )
    sample_frame.to_csv(
        args.output_dir / f"runtime_sample_{args.sample_size}.csv", index=False
    )

    summaries: list[dict] = []
    batches: list[dict] = []
    for side in sides:
        if warmup_iri is not None:
            print(f"[{side}] warming up with one untimed building", flush=True)
            _run_warmup(warmup_iri, side, args.geometry, args.usage)
            gc.collect()
        summary, batch_rows = benchmark_side(
            side, timed_iris, args.batch_size, args.geometry, args.usage
        )
        summaries.append(summary)
        batches.extend(batch_rows)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_dir / "runtime_summary.csv", index=False)
    pd.DataFrame(batches).to_csv(args.output_dir / "runtime_batches.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "sample_size": args.sample_size,
        "eligible_buildings": len(eligible),
        "warmup": args.warmup,
        "warmup_building_iri": warmup_iri,
        "sides": list(sides),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pandas": pd.__version__,
        },
        "paths": {
            "target_dir": str(args.target_dir.resolve()),
            "mapping": str(args.mapping.resolve()),
            "geometry": str(args.geometry.resolve()),
            "usage": str(args.usage.resolve()),
        },
        "results": summaries,
    }
    (args.output_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nRuntime summary", flush=True)
    print(summary_frame.to_string(index=False), flush=True)
    print(f"Results written to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
