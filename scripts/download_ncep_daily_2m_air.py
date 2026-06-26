#!/usr/bin/env python3
"""Download NCEP/NCAR Reanalysis daily 2m air temperature files."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis.dailyavgs/surface_gauss"
DEFAULT_OUTPUT_DIR = Path("data/ncep_reanalysis_air/daily_2m")


def remote_info(url: str, timeout: int) -> dict[str, object]:
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=timeout) as response:
        return {
            "status": int(response.status),
            "content_length": int(response.headers.get("Content-Length", "0") or 0),
            "last_modified": response.headers.get("Last-Modified"),
        }


def download_file(url: str, output_path: Path, expected_size: int, timeout: int, chunk_size: int = 1024 * 1024) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    if expected_size <= 0:
        request = Request(url)
        with urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
    else:
        if tmp_path.exists() and tmp_path.stat().st_size > expected_size:
            tmp_path.unlink()
        position = tmp_path.stat().st_size if tmp_path.exists() else 0
        with tmp_path.open("ab") as handle:
            while position < expected_size:
                end = min(position + int(chunk_size) - 1, expected_size - 1)
                headers = {"Range": f"bytes={position}-{end}"}
                last_error: Exception | None = None
                for _ in range(12):
                    try:
                        request = Request(url, headers=headers)
                        with urlopen(request, timeout=timeout) as response:
                            data = response.read()
                        expected_chunk = end - position + 1
                        if len(data) != expected_chunk:
                            raise IOError(f"range {position}-{end} returned {len(data)} bytes, expected {expected_chunk}")
                        handle.write(data)
                        handle.flush()
                        position += len(data)
                        last_error = None
                        break
                    except (HTTPError, URLError, TimeoutError, OSError) as exc:
                        last_error = exc
                if last_error is not None:
                    raise last_error
    actual_size = tmp_path.stat().st_size
    if expected_size > 0 and actual_size != expected_size:
        raise IOError(f"Downloaded size mismatch for {output_path.name}: got {actual_size}, expected {expected_size}")
    tmp_path.replace(output_path)


def download_year(year: int, output_dir: Path, timeout: int, force: bool) -> dict[str, object]:
    filename = f"air.2m.gauss.{year}.nc"
    url = f"{BASE_URL}/{filename}"
    output_path = output_dir / filename
    info = remote_info(url, timeout=timeout)
    expected_size = int(info["content_length"])
    if output_path.exists() and not force:
        local_size = output_path.stat().st_size
        if expected_size <= 0 or local_size == expected_size:
            return {"year": year, "status": "exists", "path": str(output_path), **info}
        output_path.unlink()
    download_file(url, output_path, expected_size=expected_size, timeout=timeout)
    return {"year": year, "status": "downloaded", "path": str(output_path), **info}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1948)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "download_manifest.jsonl"
    failures: list[dict[str, object]] = []
    with manifest_path.open("a", encoding="utf-8") as manifest:
        years = list(range(int(args.start_year), int(args.end_year) + 1))
        workers = max(1, int(args.workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_year = {
                executor.submit(download_year, year, args.output_dir, int(args.timeout), bool(args.force)): year
                for year in years
            }
            for future in as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    record = future.result()
                    print(f"{year}: {record['status']} {record.get('content_length', '')}")
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    record = {"year": year, "status": "failed", "error": repr(exc)}
                    failures.append(record)
                    print(f"{year}: failed {exc}", file=sys.stderr)
                manifest.write(json.dumps(record, sort_keys=True) + "\n")
                manifest.flush()
    if failures:
        raise SystemExit(f"{len(failures)} downloads failed; see {manifest_path}")


if __name__ == "__main__":
    main()
