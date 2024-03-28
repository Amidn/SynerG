#!/usr/bin/env python3
"""
GW170817 / GRB 170817A Multi-Messenger Analysis Pipeline
=========================================================

A complete example of multi-messenger astrophysics data analysis, combining
gravitational wave data from LIGO/Virgo with gamma-ray data from Fermi GBM.

This pipeline demonstrates:
  1. Downloading and processing GW strain data from GWOSC
  2. Downloading and processing Fermi GBM light curve data
  3. Signal processing (whitening, bandpass, Q-transform)
  4. Constructing gamma-ray light curves from time-tagged events
  5. Multi-messenger comparison (timing, localization, energetics)
  6. Publication-quality visualizations

Usage
-----
    python main.py              # Run the full pipeline
    python main.py --gw-only    # Run only GW analysis
    python main.py --grb-only   # Run only GRB analysis
    python main.py --compare    # Run only comparison (needs data cached)

Output
------
    plots/                      # All generated figures
    Console                     # Event summaries and comparison report

Dependencies
------------
    See requirements.txt

Author
------
    Multi-messenger astrophysics analysis sample
    Created as a template for GW+EM counterpart studies
"""

import sys
import logging
import argparse
import time
import matplotlib.pyplot as plt

from config import PLOT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="GW170817 / GRB 170817A Multi-Messenger Analysis",
    )
    parser.add_argument(
        "--gw-only", action="store_true",
        help="Run only the gravitational wave analysis.",
    )
    parser.add_argument(
        "--grb-only", action="store_true",
        help="Run only the Fermi GBM analysis.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run only the comparison module (requires cached data).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not call plt.show() (useful for batch runs).",
    )
    return parser.parse_args()


def run_full_pipeline():
    """Run the entire multi-messenger analysis."""
    t_start = time.time()

    print("\n" + "#" * 70)
    print("#  GW170817 / GRB 170817A  Multi-Messenger Analysis Pipeline")
    print("#" * 70 + "\n")

    # ---- Step 1: Gravitational Wave Analysis ----
    print("=" * 60)
    print("  STEP 1: Gravitational Wave Analysis (LIGO/Virgo)")
    print("=" * 60)
    from gw_analysis import run_gw_analysis
    gw_results = run_gw_analysis()

    # ---- Step 2: Gamma-Ray Burst Analysis ----
    print("\n" + "=" * 60)
    print("  STEP 2: Gamma-Ray Burst Analysis (Fermi GBM)")
    print("=" * 60)
    from grb_analysis import run_grb_analysis
    grb_results = run_grb_analysis()

    # ---- Step 3: Multi-Messenger Comparison ----
    print("\n" + "=" * 60)
    print("  STEP 3: Multi-Messenger Comparison")
    print("=" * 60)
    from comparison import run_comparison
    cmp_results = run_comparison(
        gw_results["strain_data"],
        grb_results["gbm_data"],
    )

    elapsed = time.time() - t_start
    print(f"\nPipeline complete in {elapsed:.1f} seconds.")
    print(f"All plots saved to: {PLOT_DIR}/")
    print(f"Generated figures:")
    import os
    for f in sorted(os.listdir(PLOT_DIR)):
        if f.endswith(".png"):
            print(f"  {f}")

    return {
        "gw": gw_results,
        "grb": grb_results,
        "comparison": cmp_results,
    }


def run_gw_only():
    from gw_analysis import run_gw_analysis
    return run_gw_analysis()


def run_grb_only():
    from grb_analysis import run_grb_analysis
    return run_grb_analysis()


def run_compare_only():
    from comparison import (
        print_comparison_report,
        plot_detection_delay_diagram,
        plot_speed_of_gravity_constraint,
    )
    print_comparison_report()
    plot_detection_delay_diagram()
    plot_speed_of_gravity_constraint()


def main():
    args = parse_args()

    if args.gw_only:
        run_gw_only()
    elif args.grb_only:
        run_grb_only()
    elif args.compare:
        run_compare_only()
    else:
        run_full_pipeline()

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
