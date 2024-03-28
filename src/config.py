"""
Configuration and constants for GW170817 / GRB 170817A multi-messenger analysis.

This module centralizes all event parameters, detector metadata, data paths,
and plotting defaults used throughout the analysis pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ============================================================================
# Directory setup
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

for d in [DATA_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================================
# GW170817 event parameters
# ============================================================================
@dataclass
class GWEventParams:
    """Parameters for the gravitational wave event GW170817."""

    name: str = "GW170817"
    gps_time: float = 1187008882.43  # merger GPS time
    utc_time: str = "2017-08-17 12:41:04.4 UTC"
    ra_deg: float = 197.45  # right ascension (degrees), host galaxy NGC 4993
    dec_deg: float = -23.38  # declination (degrees)
    distance_mpc: float = 40.0  # luminosity distance in Mpc
    distance_mpc_err: Tuple[float, float] = (8.0, 14.0)  # (lower, upper) uncertainty
    host_galaxy: str = "NGC 4993"
    redshift: float = 0.0099
    network_snr: float = 32.4
    chirp_mass_msun: float = 1.186  # detector frame chirp mass
    m1_range_msun: Tuple[float, float] = (1.36, 1.60)
    m2_range_msun: Tuple[float, float] = (1.17, 1.36)
    catalog: str = "GWTC-1-confident"
    observing_run: str = "O2"

    # Detectors that observed the event
    detectors: List[str] = field(default_factory=lambda: ["H1", "L1", "V1"])
    detector_names: Dict[str, str] = field(default_factory=lambda: {
        "H1": "LIGO Hanford",
        "L1": "LIGO Livingston",
        "V1": "Virgo",
    })


# ============================================================================
# GRB 170817A parameters
# ============================================================================
@dataclass
class GRBEventParams:
    """Parameters for the gamma-ray burst GRB 170817A."""

    name: str = "GRB 170817A"
    trigger_time_gps: float = 1187008882.43 + 1.734  # ~1.7s after GW merger
    trigger_time_utc: str = "2017-08-17 12:41:06.5 UTC"
    delay_from_gw_sec: float = 1.734  # delay relative to GW merger
    duration_sec: float = 2.0  # approximate T90

    # Fermi GBM trigger info
    fermi_trigger_name: str = "bn170817529"
    fermi_trigger_number: str = "524666471"

    # INTEGRAL detection
    integral_detected: bool = True
    integral_instrument: str = "SPI-ACS"

    # Observed properties
    isotropic_energy_erg: float = 3.1e46  # much lower than typical sGRBs
    peak_energy_kev: float = 185.0

    # Fermi GBM data URL base
    fermi_data_url: str = (
        "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/2017/"
        "bn170817529/current/"
    )


# ============================================================================
# Detector configuration for plotting and analysis
# ============================================================================
DETECTOR_COLORS = {
    "H1": "#ee0000",       # red
    "L1": "#4ba6ff",       # blue
    "V1": "#9b59b6",       # purple
    "Fermi": "#2ecc71",    # green
    "INTEGRAL": "#e67e22",  # orange
}

DETECTOR_LABELS = {
    "H1": "LIGO Hanford (H1)",
    "L1": "LIGO Livingston (L1)",
    "V1": "Virgo (V1)",
    "Fermi": "Fermi GBM",
    "INTEGRAL": "INTEGRAL SPI-ACS",
}


# ============================================================================
# Analysis parameters
# ============================================================================
@dataclass
class AnalysisParams:
    """Parameters controlling the analysis pipeline."""

    # Strain data
    strain_sample_rate: int = 4096  # Hz
    strain_segment_duration: int = 32  # seconds around event
    bandpass_low: float = 30.0  # Hz
    bandpass_high: float = 1700.0  # Hz (BNS signal range)

    # Q-transform
    qtransform_qrange: Tuple[float, float] = (4, 64)
    qtransform_frange: Tuple[float, float] = (30.0, 500.0)
    qtransform_outseg: Tuple[float, float] = (-4.0, 1.0)  # relative to merger

    # Spectrogram
    spectrogram_fftlength: float = 1.0  # seconds
    spectrogram_overlap: float = 0.9  # fraction

    # GBM light curve
    gbm_time_range: Tuple[float, float] = (-2.0, 10.0)  # seconds around trigger
    gbm_time_binsize: float = 0.064  # seconds
    gbm_energy_range_kev: Tuple[float, float] = (10.0, 1000.0)

    # Sky map
    skymap_nside: int = 512


# ============================================================================
# Plot styling
# ============================================================================
PLOT_STYLE = {
    "figure.figsize": (12, 7),
    "figure.dpi": 150,
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}


# ============================================================================
# Instantiate default parameter objects
# ============================================================================
gw_event = GWEventParams()
grb_event = GRBEventParams()
analysis = AnalysisParams()
