"""
Fermi GBM analysis module for GRB 170817A.

Handles downloading Fermi GBM trigger data (TTE and CTIME), extracting light
curves, computing spectral properties, and generating GRB-specific plots.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.io import fits
from astropy.utils.data import download_file

from config import (
    grb_event, gw_event, analysis, DATA_DIR, PLOT_DIR, PLOT_STYLE,
    DETECTOR_COLORS,
)

logger = logging.getLogger(__name__)
rcParams.update(PLOT_STYLE)


# ============================================================================
# Fermi GBM detector info
# ============================================================================

# NaI detectors (12 total: n0 to n9, na, nb) and BGO detectors (b0, b1)
# For GRB 170817A, the most illuminated NaI detectors were n1, n2, n5
# and BGO detector b0.

RELEVANT_NAI_DETECTORS = ["n1", "n2", "n5"]
RELEVANT_BGO_DETECTORS = ["b0"]

NAI_ENERGY_RANGE_KEV = (8.0, 900.0)
BGO_ENERGY_RANGE_KEV = (200.0, 40000.0)

# Colors for individual GBM detectors
GBM_DET_COLORS = {
    "n1": "#2ecc71",
    "n2": "#27ae60",
    "n5": "#1abc9c",
    "b0": "#e67e22",
}


# ============================================================================
# Data fetching
# ============================================================================

def build_gbm_url(detector: str, data_type: str = "ctime") -> str:
    """
    Construct the HEASARC URL for a Fermi GBM data file.

    Parameters
    ----------
    detector : str
        Detector name, e.g. 'n1', 'b0'.
    data_type : str
        One of 'ctime' (binned), 'tte' (time-tagged events), 'cspec'.

    Returns
    -------
    str
        URL to the FITS file.
    """
    trigger = grb_event.fermi_trigger_name
    base = (
        f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/"
        f"2017/{trigger}/current/"
    )

    filename = f"glg_{data_type}_{detector}_{trigger}_v00.fit"
    return base + filename


def fetch_gbm_ctime(detector: str):
    """
    Download and read a Fermi GBM CTIME (binned count rate) file.

    Returns
    -------
    dict with keys:
        times : ndarray, bin center times (MET seconds)
        counts : ndarray, shape (n_times, n_channels)
        channel_edges : ndarray, energy channel edges in keV
        trigger_time : float, trigger time in MET
    """
    url = build_gbm_url(detector, "ctime")
    logger.info("Downloading GBM CTIME for %s: %s", detector, url)

    local = download_file(url, cache=True)

    with fits.open(local) as hdul:
        # Extension 0: Primary header with trigger time
        trigger_time = hdul[0].header.get("TRIGTIME", 0.0)

        # Extension 1: SPECTRUM table (time-binned spectra)
        data = hdul[2].data  # SPECTRUM extension
        times = data["TIME"]  # MET seconds
        counts = data["COUNTS"]  # shape (n_times, n_channels)

        # Extension 2: EBOUNDS (energy channel boundaries)
        ebounds = hdul[1].data
        e_min = ebounds["E_MIN"]
        e_max = ebounds["E_MAX"]

    return {
        "times": times,
        "counts": counts,
        "e_min": e_min,
        "e_max": e_max,
        "trigger_time": trigger_time,
        "detector": detector,
    }


def fetch_gbm_tte(detector: str):
    """
    Download and read a Fermi GBM TTE (Time-Tagged Event) file.

    TTE data gives individual photon arrival times and energies,
    providing the highest time resolution.

    Returns
    -------
    dict with keys:
        photon_times : ndarray, arrival times (MET seconds)
        pha : ndarray, pulse height channel for each photon
        e_min, e_max : ndarray, channel energy boundaries (keV)
        trigger_time : float
    """
    url = build_gbm_url(detector, "tte")
    logger.info("Downloading GBM TTE for %s: %s", detector, url)

    local = download_file(url, cache=True)

    with fits.open(local) as hdul:
        trigger_time = hdul[0].header.get("TRIGTIME", 0.0)

        events = hdul[2].data
        photon_times = events["TIME"]
        pha = events["PHA"]

        ebounds = hdul[1].data
        e_min = ebounds["E_MIN"]
        e_max = ebounds["E_MAX"]

    return {
        "photon_times": photon_times,
        "pha": pha,
        "e_min": e_min,
        "e_max": e_max,
        "trigger_time": trigger_time,
        "detector": detector,
    }


def fetch_all_gbm_data():
    """
    Fetch CTIME and TTE data for all relevant GBM detectors.

    Returns
    -------
    dict
        Nested dict: result[detector] = {"ctime": ..., "tte": ...}
    """
    all_data = {}

    for det in RELEVANT_NAI_DETECTORS + RELEVANT_BGO_DETECTORS:
        det_data = {}
        try:
            det_data["ctime"] = fetch_gbm_ctime(det)
            logger.info("CTIME loaded for %s", det)
        except Exception as exc:
            logger.warning("CTIME fetch failed for %s: %s", det, exc)

        try:
            det_data["tte"] = fetch_gbm_tte(det)
            logger.info("TTE loaded for %s", det)
        except Exception as exc:
            logger.warning("TTE fetch failed for %s: %s", det, exc)

        if det_data:
            all_data[det] = det_data

    return all_data


# ============================================================================
# Light curve construction
# ============================================================================

def make_lightcurve_from_tte(tte_data, time_range=None, binsize=None,
                              energy_range=None):
    """
    Bin TTE photon events into a light curve.

    Parameters
    ----------
    tte_data : dict
        Output of fetch_gbm_tte().
    time_range : tuple of float, optional
        (t_start, t_end) in seconds relative to trigger. Defaults to config.
    binsize : float, optional
        Bin width in seconds. Defaults to config.
    energy_range : tuple of float, optional
        (E_low, E_high) in keV. Defaults to config.

    Returns
    -------
    dict with keys:
        bin_centers : ndarray (seconds relative to trigger)
        rates : ndarray (counts/s)
        rate_errors : ndarray
        trigger_time : float
    """
    if time_range is None:
        time_range = analysis.gbm_time_range
    if binsize is None:
        binsize = analysis.gbm_time_binsize
    if energy_range is None:
        energy_range = analysis.gbm_energy_range_kev

    trigger = tte_data["trigger_time"]
    rel_times = tte_data["photon_times"] - trigger

    # Energy selection
    channels = tte_data["pha"]
    e_lo = tte_data["e_min"]
    e_hi = tte_data["e_max"]

    # Find channels within the desired energy range
    valid_channels = np.where(
        (e_lo >= energy_range[0]) & (e_hi <= energy_range[1])
    )[0]

    mask = (
        (rel_times >= time_range[0])
        & (rel_times <= time_range[1])
        & np.isin(channels, valid_channels)
    )
    selected_times = rel_times[mask]

    # Histogram
    bin_edges = np.arange(time_range[0], time_range[1] + binsize, binsize)
    counts, _ = np.histogram(selected_times, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rates = counts / binsize
    rate_errors = np.sqrt(counts) / binsize

    return {
        "bin_centers": bin_centers,
        "rates": rates,
        "rate_errors": rate_errors,
        "trigger_time": trigger,
        "detector": tte_data["detector"],
        "energy_range": energy_range,
    }


def make_lightcurve_from_ctime(ctime_data, time_range=None):
    """
    Extract a light curve from CTIME data by summing over energy channels.

    Parameters
    ----------
    ctime_data : dict
        Output of fetch_gbm_ctime().
    time_range : tuple, optional
        (t_start, t_end) relative to trigger.

    Returns
    -------
    dict with keys: rel_times, total_rate, trigger_time, detector
    """
    if time_range is None:
        time_range = analysis.gbm_time_range

    trigger = ctime_data["trigger_time"]
    rel_times = ctime_data["times"] - trigger
    total_counts = ctime_data["counts"].sum(axis=1)

    # Estimate bin widths from time differences
    dt = np.diff(rel_times)
    dt = np.append(dt, dt[-1])
    rates = total_counts / dt

    mask = (rel_times >= time_range[0]) & (rel_times <= time_range[1])

    return {
        "rel_times": rel_times[mask],
        "total_rate": rates[mask],
        "trigger_time": trigger,
        "detector": ctime_data["detector"],
    }


# ============================================================================
# Property extraction
# ============================================================================

def get_grb_properties():
    """
    Return a summary dict of GRB 170817A properties.

    Returns
    -------
    dict
    """
    return {
        "event_name": grb_event.name,
        "trigger_time_gps": grb_event.trigger_time_gps,
        "trigger_time_utc": grb_event.trigger_time_utc,
        "delay_from_gw_sec": grb_event.delay_from_gw_sec,
        "duration_T90_sec": grb_event.duration_sec,
        "fermi_trigger_name": grb_event.fermi_trigger_name,
        "isotropic_energy_erg": grb_event.isotropic_energy_erg,
        "peak_energy_kev": grb_event.peak_energy_kev,
        "integral_detected": grb_event.integral_detected,
        "detectors_used": RELEVANT_NAI_DETECTORS + RELEVANT_BGO_DETECTORS,
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_lightcurve_tte(gbm_data: dict, save=True):
    """
    Plot TTE light curves for all relevant detectors in a multi-panel figure.

    Parameters
    ----------
    gbm_data : dict
        Output of fetch_all_gbm_data().
    save : bool
        Save plot to disk.
    """
    detectors_with_tte = [
        d for d in gbm_data if "tte" in gbm_data[d]
    ]
    n = len(detectors_with_tte)
    if n == 0:
        logger.warning("No TTE data available for plotting.")
        return None

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, det in zip(axes, detectors_with_tte):
        lc = make_lightcurve_from_tte(gbm_data[det]["tte"])
        color = GBM_DET_COLORS.get(det, "#333333")

        ax.step(
            lc["bin_centers"], lc["rates"],
            where="mid", color=color, linewidth=0.8,
        )
        ax.fill_between(
            lc["bin_centers"],
            lc["rates"] - lc["rate_errors"],
            lc["rates"] + lc["rate_errors"],
            alpha=0.2, color=color, step="mid",
        )

        ax.axvline(0, color="red", linestyle=":", alpha=0.7, label="GBM Trigger")
        ax.axvline(
            -grb_event.delay_from_gw_sec,
            color="blue", linestyle="--", alpha=0.7,
            label="GW Merger",
        )
        ax.set_ylabel("Rate (counts/s)")
        det_type = "NaI" if det.startswith("n") else "BGO"
        ax.set_title(
            f"Fermi GBM {det_type} detector {det}  "
            f"({lc['energy_range'][0]:.0f} to {lc['energy_range'][1]:.0f} keV)",
            fontsize=13,
        )
        ax.legend(loc="upper right", fontsize=10)

    axes[-1].set_xlabel("Time relative to GBM trigger (s)")
    fig.suptitle(
        f"GRB 170817A Fermi GBM Light Curves (TTE, {analysis.gbm_time_binsize}s bins)",
        fontsize=16, y=1.01,
    )
    fig.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "grb_lightcurve_tte.png")
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved: %s", path)

    return fig


def plot_combined_nai_lightcurve(gbm_data: dict, save=True):
    """
    Plot a summed NaI light curve combining the brightest detectors.
    This gives better SNR for the faint GRB 170817A signal.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    combined_rates = None
    combined_errors_sq = None
    bin_centers = None

    for det in RELEVANT_NAI_DETECTORS:
        if det not in gbm_data or "tte" not in gbm_data[det]:
            continue
        lc = make_lightcurve_from_tte(gbm_data[det]["tte"])
        if combined_rates is None:
            combined_rates = lc["rates"].copy()
            combined_errors_sq = (lc["rate_errors"] ** 2).copy()
            bin_centers = lc["bin_centers"]
        else:
            combined_rates += lc["rates"]
            combined_errors_sq += lc["rate_errors"] ** 2

    if combined_rates is None:
        logger.warning("No NaI TTE data available for combined light curve.")
        return None

    combined_errors = np.sqrt(combined_errors_sq)

    ax.step(
        bin_centers, combined_rates,
        where="mid", color=DETECTOR_COLORS["Fermi"], linewidth=1.0,
    )
    ax.fill_between(
        bin_centers,
        combined_rates - combined_errors,
        combined_rates + combined_errors,
        alpha=0.2, color=DETECTOR_COLORS["Fermi"], step="mid",
    )

    ax.axvline(0, color="red", linestyle=":", linewidth=1.5, label="GBM Trigger")
    ax.axvline(
        -grb_event.delay_from_gw_sec,
        color="blue", linestyle="--", linewidth=1.5,
        label=f"GW Merger (t = {-grb_event.delay_from_gw_sec:+.3f} s)",
    )

    # Shade the burst interval
    ax.axvspan(0, grb_event.duration_sec, alpha=0.08, color="red",
               label=f"~T90 ({grb_event.duration_sec} s)")

    ax.set_xlabel("Time relative to GBM trigger (s)")
    ax.set_ylabel("Combined NaI rate (counts/s)")
    ax.set_title(
        f"GRB 170817A Combined NaI ({', '.join(RELEVANT_NAI_DETECTORS)}) Light Curve",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    fig.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "grb_combined_nai.png")
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved: %s", path)

    return fig


def plot_gbm_count_spectrum(gbm_data: dict, save=True):
    """
    Plot the energy spectrum (count rate vs energy) around the trigger.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for det in RELEVANT_NAI_DETECTORS:
        if det not in gbm_data or "tte" not in gbm_data[det]:
            continue

        tte = gbm_data[det]["tte"]
        trigger = tte["trigger_time"]
        rel_t = tte["photon_times"] - trigger

        # Select photons during the burst (0 to T90)
        burst_mask = (rel_t >= 0) & (rel_t <= grb_event.duration_sec)
        burst_channels = tte["pha"][burst_mask]

        e_centers = 0.5 * (tte["e_min"] + tte["e_max"])
        e_widths = tte["e_max"] - tte["e_min"]

        channel_counts = np.bincount(
            burst_channels, minlength=len(e_centers)
        )[:len(e_centers)]
        rate_per_kev = channel_counts / (grb_event.duration_sec * e_widths)

        color = GBM_DET_COLORS.get(det, "#333")
        ax.step(e_centers, rate_per_kev, where="mid", color=color, label=det)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Rate (counts / s / keV)")
    ax.set_title("GRB 170817A Count Spectrum (during T90)", fontsize=14)
    ax.set_xlim(8, 1000)
    ax.legend()
    fig.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "grb_count_spectrum.png")
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved: %s", path)

    return fig


def print_grb_summary():
    """Print a formatted summary of GRB 170817A properties."""
    props = get_grb_properties()
    print("\n" + "=" * 60)
    print("  GRB 170817A Event Summary")
    print("=" * 60)
    print(f"  Event name       : {props['event_name']}")
    print(f"  Trigger GPS      : {props['trigger_time_gps']:.3f}")
    print(f"  Trigger UTC      : {props['trigger_time_utc']}")
    print(f"  Delay from GW    : {props['delay_from_gw_sec']:.3f} s")
    print(f"  T90 duration     : {props['duration_T90_sec']:.1f} s")
    print(f"  Fermi trigger    : {props['fermi_trigger_name']}")
    print(f"  Iso. energy      : {props['isotropic_energy_erg']:.1e} erg")
    print(f"  E_peak           : {props['peak_energy_kev']:.0f} keV")
    print(f"  INTEGRAL detect  : {props['integral_detected']}")
    print(f"  GBM detectors    : {props['detectors_used']}")
    print("=" * 60 + "\n")


# ============================================================================
# Pipeline runner
# ============================================================================

def run_grb_analysis():
    """
    Execute the full GRB analysis pipeline:
      1. Print GRB summary
      2. Fetch GBM data
      3. Plot individual TTE light curves
      4. Plot combined NaI light curve
      5. Plot count spectrum

    Returns
    -------
    dict
        Contains gbm_data and grb_properties.
    """
    print_grb_summary()

    print("[GRB] Fetching Fermi GBM data from HEASARC...")
    gbm_data = fetch_all_gbm_data()

    if not gbm_data:
        raise RuntimeError(
            "No GBM data could be fetched. Check internet and astropy install."
        )

    print(f"[GRB] Loaded data for detectors: {list(gbm_data.keys())}")

    print("[GRB] Plotting TTE light curves...")
    plot_lightcurve_tte(gbm_data)

    print("[GRB] Plotting combined NaI light curve...")
    plot_combined_nai_lightcurve(gbm_data)

    print("[GRB] Plotting count spectrum...")
    plot_gbm_count_spectrum(gbm_data)

    return {
        "gbm_data": gbm_data,
        "grb_properties": get_grb_properties(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_grb_analysis()
    plt.show()
