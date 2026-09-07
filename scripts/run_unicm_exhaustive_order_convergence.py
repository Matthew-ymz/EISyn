"""Compatibility entry point for the corrected independent-source exhaustive run.

The original empirical-covariance sample sweep is retained in its result logs.
Recompute from the original 16,384 cached predictions using the product prior;
never load the legacy per-lead Syn caches or launch more predictions here.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_unicm_order_syn import main

if __name__ == '__main__':
    main()
