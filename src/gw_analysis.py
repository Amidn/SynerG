"""
Gravitational Wave analysis module for GW170817.

Handles downloading strain data from GWOSC, signal processing (bandpass,
whitening, Q-transform), and generating GW-specific visualizations.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from config import (
    gw_event, analysis, DATA_DIR, PLOT_DIR, PLOT_STYLE,
    DETECTOR_COLORS, DETECTOR_LABELS,
)

logger = logging.getLogger(__name__)
rcParams.update(PLOT_STYLE)


# ============================================================================
# Data fetching
# ============================================================================

def fetch_strain_data(detector: str, duration: int = None):
    """
    Download strain data for a given detector around GW170817.

    Parameters
    ----------
    detector : str
        Detector key, e.g. 'H1', 'L1', 'V1'.
    duration : int, optional
        Segment length in seconds. Defaults to config value.

    Returns
    -------
    gwpy.timeseries.TimeSeries
        Strain time series.
    """
    from gwpy.timeseries import TimeSeries

    if duration is None:
        duration = analysis.strain_segment_duration

    gps_start = int(gw_event.gps_time) - duration // 2
    gps_end = int(gw_event.gps_time) + duration // 2

    logger.info(
        "Fetching %s strain: GPS [%d, %d] (%d s)",
        detector, gps_start, gps_end, duration,
    )

    strain = TimeSeries.fetch_open_data(
        detector,
        gps_start,
        gps_end,
        sample_rate=analysis.strain_sample_rate,
        cache=True,
    )
    return strain


def fetch_all_detectors():
    """
    Fetch strain data for every detector that observed GW170817.

    Returns
    -------
    dict
        Mapping detector key to TimeSeries.
    """
    data = {}
    for det in gw_event.detectors:
        try:
            data[det] = fetch_strain_data(det)
            logger.info("Successfully fetched %s data.", det)
        except Exception as exc:
            logger.warning("Could not fetch %s: %s", det, exc)
    return data


# ============================================================================
# Signal processing
# ============================================================================

def bandpass_strain(strain, flow=None, fhigh=None):
    """Apply a band-pass filter to the strain data."""
    if flow is None:
        flow = analysis.bandpass_low
    if fhigh is None:
        fhigh = analysis.bandpass_high

    filtered = strain.bandpass(flow, fhigh)
    return filtered


def whiten_strain(strain, segment_duration=4, overlap=2):
    """
    Whiten the strain using an estimated PSD.

    Parameters
    ----------
    strain : TimeSeries
        Raw strain.
    segment_duration : float
        FFT segment length for PSD estimation (seconds).
    overlap : float
        Overlap between FFT segments (seconds).

    Returns
    -------
    TimeSeries
        Whitened strain.
    """
    whitened = strain.whiten(segment_duration, overlap)
    return whitened


def compute_qtransform(strain, detector: str):
    """
    Compute the Q-transform spectrogram around the merger.

    Parameters
    ----------
    strain : TimeSeries
        Input strain data.
    detector : str
        Detector key (used for centering the output segment).

    Returns
    -------
    Spectrogram
        Q-transform output.
    """
    outseg_start = gw_event.gps_time + analysis.qtransform_outseg[0]
    outseg_end = gw_event.gps_time + analysis.qtransform_outseg[1]

    qt = strain.q_transform(
        frange=analysis.qtransform_frange,
        qrange=analysis.qtransform_qrange,
        outseg=(outseg_start, outseg_end),
    )
    return qt


def compute_spectrogram(strain):
    """Compute a standard spectrogram of the strain data."""
    fft_samples = int(analysis.spectrogram_fftlength * analysis.strain_sample_rate)
    overlap_samples = int(analysis.spectrogram_overlap * fft_samples)

    spec = strain.spectrogram2(
        fftlength=analysis.spectrogram_fftlength,
        overlap=analysis.spectrogram_overlap * analysis.spectrogram_fftlength,
    )
    return spec


# ============================================================================
# Property extraction
# ============================================================================

def get_event_properties():
    """
    Return a summary dict of GW170817 properties.

    Returns
    -------
    dict
        Event metadata for display or comparison.
    """
    return {
        "event_name": gw_event.name,
        "gps_time": gw_event.gps_time,
        "utc_time": gw_event.utc_time,
        "ra_deg": gw_event.ra_deg,
        "dec_deg": gw_event.dec_deg,
        "distance_mpc": gw_event.distance_mpc,
        "host_galaxy": gw_event.host_galaxy,
        "network_snr": gw_event.network_snr,
        "chirp_mass_msun": gw_event.chirp_mass_msun,
        "m1_range": gw_event.m1_range_msun,
        "m2_range": gw_event.m2_range_msun,
        "detectors": gw_event.detectors,
        "observing_run": gw_event.observing_run,
    }


def fetch_skymap():
    """
    Download the official LIGO/Virgo sky localization map.

    Returns
    -------
    numpy.ndarray
        HEALPix probability map.
    dict
        Header metadata.
    """
    from astropy.utils.data import download_file
    import healpy as hp

    skymap_url = (
        "https://dcc.ligo.org/public/0146/G1701985/001/"
        "LALInference_v2.fits.gz"
    )

    logger.info("Downloading GW170817 skymap...")
    local_path = download_file(skymap_url, cache=True)

    prob, header = hp.read_map(local_path, h=True)
    header = dict(header)
    return prob, header


# ============================================================================
# Visualization
# ============================================================================

def plot_raw_strain(strain_data: dict, save=True):
    """
    Plot raw strain for all detectors in a multi-panel figure.

    Parameters
    ----------
    strain_data : dict
        Mapping detector key to TimeSeries.
    save : bool
        If True, save to PLOT_DIR.
    """
    n_det = len(strain_data)
    fig, axes = plt.subplots(n_det, 1, figsize=(14, 4 * n_det), sharex=True)
    if n_det == 1:
        axes = [axes]

    for ax, (det, strain) in zip(axes, strain_data.items()):
        t = strain.times.value - gw_event.gps_time
        ax.plot(t, strain.value, color=DETECTOR_COLORS[det], linewidth=0.4)
        ax.set_ylabel("Strain")
        ax.set_title(DETECTOR_LABELS[det], fontsize=13)
        ax.axvline(0, color="k", linestyle=":", alpha=0.6, label="Merger")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel(f"Time relative to merger (s)  [GPS {gw_event.gps_time}]")
    fig.suptitle("GW170817 Raw Strain Data", fontsize=16, y=1.01)
    fig.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "gw_raw_strain.png")
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved: %s", path)

    return fig


def plot_whitened_bandpassed(strain_data: dict, save=True):
    """
    Plot whitened and bandpassed strain in the seconds around merger.

    Shows the chirp signal clearly visible in H1 and L1.
    """
    n_det = len(strain_data)
    fig, axes = plt.subplots(n_det, 1, figsize=(14, 4 * n_det), sharex=True)
    if n_det == 1:
        axes = [axes]

    for ax, (det, strain) in zip(axes, strain_data.items()):
        whitened = whiten_strain(strain)
        filtered = bandpass_strain(whitened)

        t = filtered.times.value - gw_event.gps_time
        ax.plot(t, filtered.value, color=DETECTOR_COLORS[det], linewidth=0.5)
        ax.set_ylabel("Whitened strain")
        ax.set_title(
            f"{DETECTOR_LABELS[det]} (bandpass {analysis.bandpass_low}"
            f" to {analysis.bandpass_high} Hz)",
            fontsize=13,
        )
        ax.set_xlim(-10, 2)
        ax.axvline(0, color="k", linestyle=":", alpha=0.6, label="Merger")
        ax.legend(loc="upper left")

    axes[-1].set_xlabel(f"Time relative to merger (s)  [GPS {gw_event.gps_time}]")
    fig.suptitle(
        "GW170817 Whitened & Bandpassed Strain", fontsize=16, y=1.01
    )
    fig.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "gw_whitened_strain.png")
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved: %s", path)

    return fig


def plot_qtransform(strain_data: dict, save=True):
    """
    Plot Q-transform spectrograms for all detectors.

    The Q-transform highlights the characteristic binary inspiral chirp in
    the time-frequency plane.
    """
    n_det = len(strain_data)
    fig, axes = plt.subplots(
        1, n_det, figsize=(6 * n_det, 5), sharey=True
    )
    if n_det == 1:
        axes = [axes]

    for ax, (det, strain) in zip(axes, strain_data.items()):
        try:
            qt = compute_qtransform(strain, det)
            ax.imshow(qt, aspect="auto", origin="lower")
            ax.set_title(DETECTOR_LABELS[det], fontsize=13)
            ax.set_xlabel(f"Time relative to merger (s)")
            if ax == axes[0]:
                ax.set_ylabel("Frequency (Hz)")
            ax.colorbar(label="Normalized energy")
        except Exception as exc:
            ax.text(
                0.5, 0.5, f"Q-transform failed:\n{exc}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="red",
            )
            ax.set_title(DETECTOR_LABELS[det], fontsize=13)

    fig.suptitle(
        "GW170817 Q-transform Spectrograms", fontsize=16, y=1.02
    )
    fig.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "gw_qtransform.png")
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved: %s", path)

    return fig


def plot_skymap(prob=None, save=True):
    """
    Plot the GW sky localization map with the source marked.

    Parameters
    ----------
    prob : ndarray, optional
        HEALPix probability. If None, will be downloaded.
    save : bool
        If True, save to PLOT_DIR.
    """
    import healpy as hp

    if prob is None:
        prob, _ = fetch_skymap()

    fig = plt.figure(figsize=(12, 7))
    hp.mollview(
        prob,
        title="GW170817 Sky Localization (LIGO/Virgo)",
        unit="Probability",
        cmap="OrRd",
        fig=fig.number,
    )
    hp.graticule()

    # Mark the actual source position
    theta = np.radians(90.0 - gw_event.dec_deg)
    phi = np.radians(gw_event.ra_deg)
    hp.projscatter(
        theta, phi,
        marker="*", color="blue", s=200, edgecolors="black",
        linewidths=0.8, zorder=10,
    )
    hp.projtext(
        theta, phi + 0.1,
        f"  {gw_event.host_galaxy}",
        color="blue", fontsize=11, fontweight="bold",
    )

    if save:
        path = os.path.join(PLOT_DIR, "gw_skymap.png")
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved: %s", path)

    return fig


def print_event_summary():
    """Print a formatted summary of GW170817 properties."""
    props = get_event_properties()
    print("\n" + "=" * 60)
    print("  GW170817 Event Summary")
    print("=" * 60)
    print(f"  Event name       : {props['event_name']}")
    print(f"  GPS time         : {props['gps_time']}")
    print(f"  UTC time         : {props['utc_time']}")
    print(f"  RA, Dec (deg)    : {props['ra_deg']}, {props['dec_deg']}")
    print(f"  Distance         : {props['distance_mpc']} Mpc")
    print(f"  Host galaxy      : {props['host_galaxy']}")
    print(f"  Network SNR      : {props['network_snr']}")
    print(f"  Chirp mass       : {props['chirp_mass_msun']} M_sun")
    print(f"  m1 range         : {props['m1_range']} M_sun")
    print(f"  m2 range         : {props['m2_range']} M_sun")
    print(f"  Detectors        : {', '.join(props['detectors'])}")
    print(f"  Observing run    : {props['observing_run']}")
    print("=" * 60 + "\n")


# ============================================================================
# Pipeline runner
# ============================================================================

def run_gw_analysis():
    """
    Execute the full GW analysis pipeline:
      1. Print event summary
      2. Fetch strain data
      3. Plot raw strain
      4. Plot whitened/bandpassed strain
      5. Plot Q-transform
      6. Plot skymap

    Returns
    -------
    dict
        Contains strain_data, event_properties, skymap_prob.
    """
    print_event_summary()

    print("[GW] Fetching strain data from GWOSC...")
    strain_data = fetch_all_detectors()

    if not strain_data:
        raise RuntimeError(
            "No strain data could be fetched. Check your internet connection "
            "and that gwpy/gwosc are installed."
        )

    print(f"[GW] Loaded data for: {list(strain_data.keys())}")

    print("[GW] Plotting raw strain...")
    plot_raw_strain(strain_data)

    print("[GW] Plotting whitened and bandpassed strain...")
    plot_whitened_bandpassed(strain_data)

    print("[GW] Computing and plotting Q-transforms...")
    plot_qtransform(strain_data)

    print("[GW] Fetching and plotting skymap...")
    try:
        prob, header = fetch_skymap()
        plot_skymap(prob)
    except Exception as exc:
        logger.warning("Skymap failed (healpy may not be installed): %s", exc)
        prob = None

    return {
        "strain_data": strain_data,
        "event_properties": get_event_properties(),
        "skymap_prob": prob,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_gw_analysis()
    plt.show()
