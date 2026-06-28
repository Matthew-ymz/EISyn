#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
"${PYTHON:-python}" reproduce_from_cache.py
