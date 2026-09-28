#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_bsc_rain_examples.py — Quality-filtered BSC rain event example gallery.

Reads rf_dataset.csv and regenerates src/BSCRainExamples/ with 5 examples
per BSC code.  Every plot is fully annotated:

  - Horizontal dotted baseline at pre-event VWC level
  - Circle marker at peak VWC (timestamp re-computed from raw data)
  - Dash-dot vertical line at drying endpoint (timestamp re-computed)
  - Monospace metrics table (ΔVWC, Δdyn%, drying h, temp sum)

Selection criteria per BSC code (heaviest events first):
  (1) All three VWC depths have complete data  (delta_vwc-10/-30/-45 non-NaN)
  (2) At least one depth shows >= MIN_IMPACT_PCT % of local dynamic range increase

Usage:
    python src/plot_bsc_rain_examples.py
    python src/plot_bsc_rain_examples.py --n 3 --pct 30
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

import argparse
import matplotlib
matplotlib.use("Agg")

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Import shared logic from _poc_irrigation_v3 (same src/ directory)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _poc_irrigation_v3 import (
    OUT_RF,
    N_EXAMPLES_GLOBAL,
    plot_bsc_code_examples,
)

OUT_DIR = _HERE / "BSCRainExamples"


def main(n_per_code: int = N_EXAMPLES_GLOBAL, min_impact_pct: float = 20.0) -> None:
    print("=" * 60)
    print("  BSC Rain Example Gallery  (quality-filtered)")
    print("=" * 60)

    if not OUT_RF.exists():
        print(f"\n  ERROR: {OUT_RF} not found.")
        print("  Run src/_poc_irrigation_v3.py first to generate rf_dataset.csv.")
        sys.exit(1)

    df = pd.read_csv(OUT_RF, dtype={"eui": str}, low_memory=False)
    print(f"  Loaded {len(df):,} events from {df['eui'].nunique()} sensors")

    plot_bsc_code_examples(
        df,
        out_dir=OUT_DIR,
        n_per_code=n_per_code,
        min_impact_pct=min_impact_pct,
    )

    print("\n  Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n",   type=int,   default=N_EXAMPLES_GLOBAL,
                    help="Examples per BSC code (default %(default)s)")
    ap.add_argument("--pct", type=float, default=20.0,
                    help="Min VWC increase as %% of dynamic range (default %(default)s)")
    args = ap.parse_args()
    main(n_per_code=args.n, min_impact_pct=args.pct)
