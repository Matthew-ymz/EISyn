#!/usr/bin/env python3
"""Download the minimal ORAS5 monthly fields required by UniCM Modeformer."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import tarfile
import time
import urllib.request
from pathlib import Path

from tqdm import tqdm


BASE_URL = (
    "https://icdc.cen.uni-hamburg.de/thredds/fileServer/"
    "ftpthredds/EASYInit/oras5/r1x1"
)
VARIABLES = ("sosstsst", "so20chgt")


def write_status(
    path: Path,
    *,
    phase: str,
    current: int,
    total: int,
    started: float,
    message: str = "",
) -> None:
    elapsed = time.monotonic() - started
    rate = current / elapsed if elapsed > 0 else 0.0
    payload = {
        "phase": phase,
        "current": current,
        "total": total,
        "unit": "annual archives",
        "elapsed_seconds": elapsed,
        "eta_seconds": (total - current) / rate if rate > 0 else None,
        "message": message,
        "updated_at": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def expected_monthly_files(target: Path, variable: str, year: int) -> list[Path]:
    return [
        target / variable / f"{variable}_ORAS5_1m_{year}{month:02d}_r1x1.nc"
        for month in range(1, 13)
    ]


def safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = [member for member in handle.getmembers() if member.isfile()]
        for member in members:
            output = (destination / member.name).resolve()
            if destination_resolved not in output.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        handle.extractall(destination, members=members)


def download_one(
    target: Path,
    variable: str,
    year: int,
    timeout: int,
) -> tuple[str, int, str]:
    expected = expected_monthly_files(target, variable, year)
    if all(path.is_file() and path.stat().st_size > 0 for path in expected):
        return variable, year, "cached"

    destination = target / variable
    destination.mkdir(parents=True, exist_ok=True)
    archive_name = f"{variable}_ORAS5_1m_{year}_r1x1.tar.gz"
    archive = target / "_archives" / variable / archive_name
    archive.parent.mkdir(parents=True, exist_ok=True)
    part = archive.with_suffix(archive.suffix + ".part")
    url = f"{BASE_URL}/{variable}/opa0/{archive_name}"

    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            part.unlink(missing_ok=True)
            archive.unlink(missing_ok=True)
            request = urllib.request.Request(url, headers={"User-Agent": "EISyn-ORAS5/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response, part.open(
                "wb"
            ) as out:
                expected_bytes = int(response.headers.get("Content-Length", "0"))
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if expected_bytes and part.stat().st_size != expected_bytes:
                raise RuntimeError(
                    f"truncated response: {part.stat().st_size}/{expected_bytes} bytes"
                )
            os.replace(part, archive)
            safe_extract(archive, destination)
            break
        except Exception as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            archive.unlink(missing_ok=True)
            if attempt == 4:
                raise RuntimeError(
                    f"{variable} {year}: download failed after {attempt} attempts"
                ) from last_error
            time.sleep(attempt)

    missing = [path.name for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"{variable} {year}: missing extracted files: {missing}")
    archive.unlink()
    return variable, year, "downloaded"


def validate(target: Path, start_year: int, end_year: int) -> None:
    import xarray as xr

    expected_count = (end_year - start_year + 1) * 12
    for variable in VARIABLES:
        files = sorted((target / variable).glob(f"{variable}_ORAS5_1m_*_r1x1.nc"))
        selected = [
            path
            for path in files
            if start_year <= int(path.name.split("_1m_")[1][:4]) <= end_year
        ]
        if len(selected) != expected_count:
            raise RuntimeError(
                f"{variable}: expected {expected_count} monthly files, found {len(selected)}"
            )
        for sample in (selected[0], selected[-1]):
            with xr.open_dataset(sample, engine="scipy") as dataset:
                if variable not in dataset.variables:
                    raise RuntimeError(f"{sample}: variable {variable!r} not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=Path("data/ORAS5/icdc_r1x1_opa0"))
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2014)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("docs/log/oras5_download_progress.json"),
    )
    args = parser.parse_args()

    jobs = [
        (variable, year)
        for variable in VARIABLES
        for year in range(args.start_year, args.end_year + 1)
    ]
    started = time.monotonic()
    write_status(
        args.status,
        phase="download",
        current=0,
        total=len(jobs),
        started=started,
    )

    completed = 0
    downloaded = 0
    cached = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(download_one, args.target, variable, year, args.timeout): (
                    variable,
                    year,
                )
                for variable, year in jobs
            }
            with tqdm(total=len(jobs), desc="ORAS5", unit="archive", mininterval=1.0) as bar:
                for future in concurrent.futures.as_completed(futures):
                    variable, year, state = future.result()
                    completed += 1
                    downloaded += state == "downloaded"
                    cached += state == "cached"
                    bar.update(1)
                    bar.set_postfix(variable=variable, year=year, cached=cached)
                    write_status(
                        args.status,
                        phase="download",
                        current=completed,
                        total=len(jobs),
                        started=started,
                        message=f"{variable} {year}: {state}",
                    )

        write_status(
            args.status,
            phase="validate",
            current=completed,
            total=len(jobs),
            started=started,
            message="checking monthly file counts and NetCDF variables",
        )
        validate(args.target, args.start_year, args.end_year)
        write_status(
            args.status,
            phase="complete",
            current=completed,
            total=len(jobs),
            started=started,
            message=f"downloaded={downloaded}, cached={cached}",
        )
    except Exception as exc:
        write_status(
            args.status,
            phase="failed",
            current=completed,
            total=len(jobs),
            started=started,
            message=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
