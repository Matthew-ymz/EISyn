#!/usr/bin/env python3
"""Publish live null-permutation progress from an existing JSONL cache."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--total", type=int, default=200)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    started = time.monotonic()
    initial = -1
    while True:
        try:
            current = len(args.cache.read_text(encoding="utf-8").splitlines())
        except FileNotFoundError:
            current = 0
        if initial < 0:
            initial = current
        elapsed = time.monotonic() - started
        rate = (current - initial) / elapsed if elapsed > 0 else 0.0
        alive = True
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            alive = False
        phase = "null_permutations" if alive else ("complete" if current >= args.total else "failed")
        payload: dict[str, object] = {
            "phase": phase,
            "current": current,
            "total": args.total,
            "unit": "null replicate",
            "elapsed_seconds": elapsed,
            "eta_seconds": (args.total - current) / rate if rate > 0 else None,
            "metrics": {"rate_replicates_per_second": rate} if rate > 0 else {},
            "updated_at": time.time(),
        }
        write_status(args.status, payload)
        if not alive:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
