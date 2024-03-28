"""
Multi-messenger comparison module for GW170817 and GRB 170817A.

Produces joint visualizations and quantitative comparisons between the
gravitational wave and electromagnetic counterpart observations.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch

from config import (
    gw_event, grb_event, analysis, PLOT_DIR, PLOT_STYLE,
    DETECTOR_COLORS, DETECTOR_LABELS,
)
from gw_analysis import (
    whiten_strain, bandpass_strain, get_event_properties,
)
from grb_analysis import (
    make_lightcurve_from_tte, get_grb_properties,
    RELEVANT_NAI_DETECTORS,
)

logger = logging.getLogger(__name__)
rcParams.update(PLOT_STYLE)

# Speed of light
C_M_S = 299792458.0  # m/s


# ============================================================================
# Timing comparison
# ============================================================================

def compute_timing_comparison():
    """
    Compute the detection time delay between GW and GRB signals.

    Returns
    -------
    dict
        Timing metrics including delay, fractional speed difference bound,
        and light travel time.
    """
    gw_gps = gw_event.gps_time
    grb_gps = grb_event.trigger_time_gps
    delay_sec = grb_gps - gw_gps

    # Distance in meters
    distance_m = gw_event.distance_mpc * 3.0857e22  # Mpc to meters

    # Light travel time
    light_travel_sec = distance_m / C_M_S

    # Fractional speed difference bound:
    # |v_gw - v_em| / c < delay / travel_time
    # (this is a simplified upper bound)
    frac_speed_diff = abs(delay_sec) / light_travel_sec

    return {
        "gw_gps_time": gw_gps,
        "grb_gps_time": grb_gps,
        "delay_seconds": delay_sec,
        "distance_mpc": gw_event.distance_mpc,
        "distance_meters": distance_m,
        "light_travel_time_sec": light_travel_sec,
        "light_travel_time_years": light_travel_sec / (365.25 * 86400),
        "fractional_speed_difference": frac_speed_diff,
        "speed_ratio_bound": f"|v_GW/c - 1| < {frac_speed_diff:.1e}",
    }


def compute_energy_comparison():
    """
    Compare the energy scales of the GW and EM signals.

    Returns
    -------
    dict
        Energy and luminosity comparisons.
    """
    # GW energy (approximate): ~0.04 M_sun c^2 radiated in GW for BNS
    gw_energy_erg = 0.04 * 1.989e33 * C_M_S**2 * 1e4  # rough estimate

    grb_iso_energy = grb_event.isotropic_energy_erg

    return {
        "gw_radiated_energy_erg": gw_energy_erg,
        "grb_isotropic_energy_erg": grb_iso_energy,
        "energy_ratio_gw_to_grb": gw_energy_erg / grb_iso_energy,
        "grb_peak_energy_kev": grb_event.peak_energy_kev,
        "grb_T90_sec": grb_event.duration_sec,
        "note": (
            "GRB 170817A was unusually faint compared to typical sGRBs, "
            "likely because the jet was observed off-axis."
        ),
    }


def compute_localization_comparison():
    """
    Compare sky localization from GW and EM observations.

    Returns
    -------
    dict
        Localization areas and the confirmed host galaxy position.
    """
    return {
        "gw_90pct_area_sq_deg": 28.0,  # LIGO/Virgo 90% credible region
        "fermi_localization_sq_deg": "~1000 (very broad, all-sky monitor)",
        "optical_counterpart_ra": gw_event.ra_deg,
        "optical_counterpart_dec": gw_event.dec_deg,
        "host_galaxy": gw_event.host_galaxy,
        "note": (
            "GW localization (~28 sq deg) was precise enough for optical "
            "follow-up. Fermi GBM alone had very poor localization. "
            "The combination confirmed the association."
        ),
    }


# ============================================================================
# Combined timeline visualization
# ============================================================================

def plot_combined_timeline(gw_strain_data: dict, gbm_data: dict, save=True):
    """
    Create a combined multi-panel figure showing the GW chirp signal and
    the GRB light curve on the same time axis around the merger.

    This is the key multi-messenger plot.

    Parameters
    ----------
    gw_strain_data : dict
        detector -> TimeSeries mapping for GW strain.
    gbm_data : dict
        detector -> {tte: ..., ctime: ...} mapping for GBM.
    save : bool
        Save figure to disk.
    """
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 0.3, 1.2], hspace=0.08)

    time_window = (-6, 4)  # seconds around GW merger
    merger_gps = gw_event.gps_time

    # ---- Panel 1: LIGO Hanford whitened strain ----
    ax_h1 = fig.add_subplot(gs[0])
    if "H1" in gw_strain_data:
        strain_h1 = gw_strain_data["H1"]
        wh = bandpass_strain(whiten_strain(strain_h1))
        t = wh.times.value - merger_gps
        ax_h1.plot(t, wh.value, color=DETECTOR_COLORS["H1"], linewidth=0.5)

    ax_h1.set_xlim(*time_window)
    ax_h1.set_ylabel("Whitened strain")
    ax_h1.set_title("LIGO Hanford (H1)", fontsize=13, loc="left")
    ax_h1.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax_h1.tick_params(labelbottom=False)

    # ---- Panel 2: LIGO Livingston whitened strain ----
    ax_l1 = fig.add_subplot(gs[1], sharex=ax_h1)
    if "L1" in gw_strain_data:
        strain_l1 = gw_strain_data["L1"]
        wl = bandpass_strain(whiten_strain(strain_l1))
        t = wl.times.value - merger_gps
        ax_l1.plot(t, wl.value, color=DETECTOR_COLORS["L1"], linewidth=0.5)

    ax_l1.set_ylabel("Whitened strain")
    ax_l1.set_title("LIGO Livingston (L1)", fontsize=13, loc="left")
    ax_l1.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax_l1.tick_params(labelbottom=False)

    # ---- Panel 3: Timing annotation strip ----
    ax_ann = fig.add_subplot(gs[2], sharex=ax_h1)
    ax_ann.set_ylim(0, 1)
    ax_ann.axvline(0, color="blue", linewidth=2, label="GW Merger")
    ax_ann.axvline(
        grb_event.delay_from_gw_sec, color="red", linewidth=2,
        label=f"GBM Trigger (+{grb_event.delay_from_gw_sec:.3f} s)",
    )
    ax_ann.annotate(
        "",
        xy=(grb_event.delay_from_gw_sec, 0.5),
        xytext=(0, 0.5),
        arrowprops=dict(arrowstyle="<->", color="purple", lw=2),
    )
    ax_ann.text(
        grb_event.delay_from_gw_sec / 2, 0.7,
        f"\u0394t = {grb_event.delay_from_gw_sec:.3f} s",
        ha="center", va="center", fontsize=13, color="purple",
        fontweight="bold",
    )
    ax_ann.set_yticks([])
    ax_ann.legend(loc="upper right", fontsize=10, ncol=2)
    ax_ann.set_title("Detection Delay", fontsize=13, loc="left")
    ax_ann.tick_params(labelbottom=False)

    # ---- Panel 4: Fermi GBM combined NaI light curve ----
    ax_gbm = fig.add_subplot(gs[3], sharex=ax_h1)

    combined_rates = None
    combined_errors_sq = None
    bin_centers = None

    for det in RELEVANT_NAI_DETECTORS:
        if det not in gbm_data or "tte" not in gbm_data[det]:
            continue
        # NB: GBM times are relative to GBM trigger. We shift to GW merger frame.
        lc = make_lightcurve_from_tte(
            gbm_data[det]["tte"],
            time_range=(time_window[0] - grb_event.delay_from_gw_sec,
                        time_window[1] - grb_event.delay_from_gw_sec),
        )
        shifted_times = lc["bin_centers"] + grb_event.delay_from_gw_sec

        if combined_rates is None:
            combined_rates = lc["rates"].copy()
            combined_errors_sq = (lc["rate_errors"] ** 2).copy()
            bin_centers = shifted_times
        else:
            combined_rates += lc["rates"]
            combined_errors_sq += lc["rate_errors"] ** 2

    if combined_rates is not None:
        combined_errors = np.sqrt(combined_errors_sq)
        ax_gbm.step(
            bin_centers, combined_rates,
            where="mid", color=DETECTOR_COLORS["Fermi"], linewidth=0.8,
        )
        ax_gbm.fill_between(
            bin_centers,
            combined_rates - combined_errors,
            combined_rates + combined_errors,
            alpha=0.2, color=DETECTOR_COLORS["Fermi"], step="mid",
        )

    ax_gbm.axvline(0, color="blue", linestyle=":", alpha=0.5, label="GW Merger")
    ax_gbm.axvline(
        grb_event.delay_from_gw_sec, color="red", linestyle=":",
        alpha=0.5, label="GBM Trigger",
    )
    ax_gbm.set_ylabel("Combined NaI rate\n(counts/s)")
    ax_gbm.set_xlabel(f"Time relative to GW merger (s)  [GPS {merger_gps}]")
    ax_gbm.set_title(
        "Fermi GBM (NaI n1+n2+n5, 10 to 1000 keV)",
        fontsize=13, loc="left",
    )
    ax_gbm.legend(loc="upper right", fontsize=10)

    fig.suptitle(
        "GW170817 / GRB 170817A  Multi-Messenger Timeline",
        fontsize=18, y=0.98, fontweight="bold",
    )

    if save:
        path = os.path.join(PLOT_DIR, "multimessenger_timeline.png")
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved: %s", path)

    return fig


def plot_detection_delay_diagram(save=True):
    """
    Create a schematic diagram showing the detection sequence and delay
    between GW and EM signals, with annotations.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    timing = compute_timing_comparison()

    # Time axis
    t_range = (-3, 5)
    ax.set_xlim(*t_range)
    ax.set_ylim(-0.5, 3.5)

    # GW merger bar
    ax.barh(3, 0.05, left=-0.025, height=0.4, color="blue", alpha=0.8)
    ax.text(-0.5, 3, "GW170817\nMerger", ha="right", va="center",
            fontsize=12, color="blue", fontweight="bold")

    # GW signal duration (last ~100 seconds of inspiral visible, show last few)
    ax.barh(3, 3, left=-3, height=0.2, color="blue", alpha=0.15)
    ax.text(-1.5, 3.25, "GW inspiral signal", ha="center", fontsize=9,
            color="blue", style="italic")

    # Fermi trigger
    grb_t = timing["delay_seconds"]
    ax.barh(2, grb_event.duration_sec, left=grb_t, height=0.4,
            color="red", alpha=0.4)
    ax.barh(2, 0.05, left=grb_t - 0.025, height=0.4, color="red", alpha=0.8)
    ax.text(grb_t + grb_event.duration_sec + 0.2, 2,
            f"GRB 170817A\n(T90 ~ {grb_event.duration_sec}s)",
            ha="left", va="center", fontsize=12, color="red", fontweight="bold")

    # INTEGRAL
    ax.barh(1, 0.05, left=grb_t - 0.025, height=0.4,
            color=DETECTOR_COLORS["INTEGRAL"], alpha=0.8)
    ax.text(grb_t + 0.3, 1, "INTEGRAL SPI-ACS\n(independent detection)",
            ha="left", va="center", fontsize=11,
            color=DETECTOR_COLORS["INTEGRAL"])

    # Chandra X-ray (9 days later, just annotate)
    ax.annotate(
        "Chandra X-ray\n(+9 days)",
        xy=(4.5, 0), fontsize=10, color="purple", ha="center",
        fontstyle="italic",
    )

    # Delay arrow
    ax.annotate(
        "", xy=(grb_t, 2.7), xytext=(0, 2.7),
        arrowprops=dict(arrowstyle="<->", color="purple", lw=2.5),
    )
    ax.text(
        grb_t / 2, 2.85,
        f"\u0394t = {grb_t:.3f} s",
        ha="center", fontsize=14, color="purple", fontweight="bold",
    )

    ax.set_xlabel("Time relative to GW merger (s)", fontsize=13)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["X-ray", "INTEGRAL", "Fermi GBM", "LIGO/Virgo"])
    ax.set_title(
        "Multi-Messenger Detection Sequence for GW170817 / GRB 170817A",
        fontsize=15,
    )
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "detection_delay_diagram.png")
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved: %s", path)

    return fig


def plot_speed_of_gravity_constraint(save=True):
    """
    Visualize the constraint on the speed of gravitational waves vs light.

    The 1.7s delay over ~130 Mly constrains |v_GW/c - 1| to ~10^-15.
    """
    timing = compute_timing_comparison()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: log scale comparison of constraints
    constraints = {
        "GW170817\n(this event)": timing["fractional_speed_difference"],
        "Binary pulsar\n(Hulse-Taylor)": 1e-2,
        "Solar system\n(Laplace bound)": 1e-8,
        "Pre-2017\ntheory priors": 1e-10,
    }

    names = list(constraints.keys())
    values = list(constraints.values())
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    ax1.barh(names, values, color=colors, alpha=0.7, edgecolor="black")
    ax1.set_xscale("log")
    ax1.set_xlabel("|v_GW/c - 1| (upper bound)", fontsize=12)
    ax1.set_title("Constraints on Speed of Gravity", fontsize=14)
    ax1.invert_yaxis()

    for i, v in enumerate(values):
        ax1.text(v * 2, i, f"{v:.0e}", va="center", fontsize=11)

    # Right panel: schematic of the measurement
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect("equal")

    # Source
    ax2.plot(1, 5, "o", markersize=20, color="orange", zorder=5)
    ax2.text(1, 5.8, "NGC 4993\n(40 Mpc)", ha="center", fontsize=10)

    # Earth
    ax2.plot(9, 5, "o", markersize=15, color="blue", zorder=5)
    ax2.text(9, 5.8, "Earth", ha="center", fontsize=10)

    # GW arrow (arrives first)
    ax2.annotate(
        "", xy=(8.5, 5.3), xytext=(1.5, 5.3),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    )
    ax2.text(5, 5.6, "Gravitational Waves", ha="center", fontsize=11,
             color="blue")

    # EM arrow (arrives 1.7s later)
    ax2.annotate(
        "", xy=(8.5, 4.7), xytext=(1.5, 4.7),
        arrowprops=dict(arrowstyle="->", color="red", lw=2, linestyle="--"),
    )
    ax2.text(5, 4.2, "Gamma Rays (+1.7 s)", ha="center", fontsize=11,
             color="red")

    ax2.text(
        5, 2.5,
        f"Travel time: ~{timing['light_travel_time_years']:.0e} years\n"
        f"Delay: {timing['delay_seconds']:.3f} s\n"
        f"|v_GW/c - 1| < {timing['fractional_speed_difference']:.1e}",
        ha="center", fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                  edgecolor="gray"),
    )

    ax2.set_title("Speed of Gravity Measurement", fontsize=14)
    ax2.axis("off")

    fig.suptitle(
        "GW170817: Constraining the Speed of Gravitational Waves",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "speed_of_gravity.png")
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved: %s", path)

    return fig


# ============================================================================
# Summary report
# ============================================================================

def print_comparison_report():
    """Print a comprehensive comparison report to the console."""
    timing = compute_timing_comparison()
    energy = compute_energy_comparison()
    loc = compute_localization_comparison()

    print("\n" + "=" * 70)
    print("  MULTI-MESSENGER COMPARISON REPORT: GW170817 / GRB 170817A")
    print("=" * 70)

    print("\n  TIMING")
    print(f"    GW merger (GPS)        : {timing['gw_gps_time']}")
    print(f"    GBM trigger (GPS)      : {timing['grb_gps_time']:.3f}")
    print(f"    Detection delay        : {timing['delay_seconds']:.3f} s")
    print(f"    Distance               : {timing['distance_mpc']} Mpc")
    print(f"    Light travel time      : {timing['light_travel_time_years']:.2e} yr")
    print(f"    Speed constraint       : {timing['speed_ratio_bound']}")

    print("\n  ENERGY")
    print(f"    GW radiated energy     : ~{energy['gw_radiated_energy_erg']:.1e} erg")
    print(f"    GRB iso. energy        : {energy['grb_isotropic_energy_erg']:.1e} erg")
    print(f"    Ratio (GW/GRB)         : ~{energy['energy_ratio_gw_to_grb']:.0e}")
    print(f"    GRB E_peak             : {energy['grb_peak_energy_kev']} keV")
    print(f"    Note                   : {energy['note']}")

    print("\n  SKY LOCALIZATION")
    print(f"    GW 90% area            : {loc['gw_90pct_area_sq_deg']} sq deg")
    print(f"    Fermi GBM area         : {loc['fermi_localization_sq_deg']}")
    print(f"    Optical counterpart    : RA={loc['optical_counterpart_ra']}, "
          f"Dec={loc['optical_counterpart_dec']}")
    print(f"    Host galaxy            : {loc['host_galaxy']}")
    print(f"    Note                   : {loc['note']}")

    print("\n" + "=" * 70 + "\n")


# ============================================================================
# Pipeline runner
# ============================================================================

def run_comparison(gw_strain_data: dict, gbm_data: dict):
    """
    Execute the full comparison pipeline:
      1. Print comparison report
      2. Plot combined timeline
      3. Plot detection delay diagram
      4. Plot speed of gravity constraint

    Parameters
    ----------
    gw_strain_data : dict
        From gw_analysis.
    gbm_data : dict
        From grb_analysis.

    Returns
    -------
    dict
        Contains timing, energy, localization comparison results.
    """
    print_comparison_report()

    print("[CMP] Plotting combined multi-messenger timeline...")
    plot_combined_timeline(gw_strain_data, gbm_data)

    print("[CMP] Plotting detection delay diagram...")
    plot_detection_delay_diagram()

    print("[CMP] Plotting speed of gravity constraint...")
    plot_speed_of_gravity_constraint()

    return {
        "timing": compute_timing_comparison(),
        "energy": compute_energy_comparison(),
        "localization": compute_localization_comparison(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_comparison_report()
    plot_detection_delay_diagram()
    plot_speed_of_gravity_constraint()
    plt.show()
