#!/usr/bin/env python3
"""
Generate realistic mock preview plots for the GW170817 multi-messenger
analysis pipeline. These are illustrative only and use simulated data
to show what the real analysis output looks like.

Every plot includes a clear "SIMULATED DATA" watermark.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os

# ============================================================================
# Setup
# ============================================================================
PLOT_DIR = "/home/claude/preview_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

STYLE = {
    "figure.dpi": 150,
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
}
plt.rcParams.update(STYLE)

COLOR_H1 = "#ee0000"
COLOR_L1 = "#4ba6ff"
COLOR_V1 = "#9b59b6"
COLOR_FERMI = "#2ecc71"
COLOR_INTEGRAL = "#e67e22"

WATERMARK = "SIMULATED PREVIEW"

np.random.seed(170817)


def add_watermark(fig, text=WATERMARK):
    """Add a diagonal watermark across the figure."""
    fig.text(
        0.5, 0.5, text,
        fontsize=38, color="red", alpha=0.12,
        ha="center", va="center", rotation=30,
        fontweight="bold",
        transform=fig.transFigure, zorder=999,
    )


# ============================================================================
# 1. BNS chirp waveform simulator
# ============================================================================

def simulate_bns_chirp(t, t_merger=0.0, chirp_mass_msun=1.186, phi0=0.0):
    """
    Generate an approximate Newtonian inspiral chirp waveform for a BNS.
    This is the leading-order post-Newtonian formula.
    """
    G = 6.674e-11
    c = 3e8
    Msun = 1.989e30
    Mc = chirp_mass_msun * Msun
    Mc_geo = G * Mc / c**3  # chirp mass in seconds

    tau = t_merger - t
    tau = np.clip(tau, 1e-4, None)

    # Newtonian chirp phase and frequency
    phase = phi0 - 2.0 * (5.0 * Mc_geo)**(-5.0/8.0) * tau**(5.0/8.0)
    freq = (1.0 / (8.0 * np.pi * Mc_geo)) * (5.0 * Mc_geo / tau)**(3.0/8.0)

    # Amplitude grows as tau^(-1/4)
    amp = tau**(-1.0/4.0)
    amp = amp / np.max(amp) * 4.0  # normalize

    # Cut off after merger
    mask_post = t > t_merger
    h = amp * np.cos(2 * np.pi * phase)
    h[mask_post] = 0.0

    # Taper near merger (ringdown approximation)
    ringdown_t = t[mask_post] - t_merger
    if len(ringdown_t) > 0:
        h[mask_post] = 2.0 * np.exp(-ringdown_t / 0.003) * np.cos(
            2 * np.pi * 1600 * ringdown_t
        )

    return h, freq


# ============================================================================
# Plot 1: Whitened Strain (H1 + L1 + V1)
# ============================================================================

def plot_whitened_strain():
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    dt = 1.0 / 4096
    t = np.arange(-5, 0.15, dt)

    configs = [
        ("H1", COLOR_H1, "LIGO Hanford (H1)", 2.5, 0.0),
        ("L1", COLOR_L1, "LIGO Livingston (L1)", 2.3, 0.003),
        ("V1", COLOR_V1, "Virgo (V1)", 0.3, 0.012),
    ]

    for ax, (det, color, label, snr_scale, t_delay) in zip(axes, configs):
        h, freq = simulate_bns_chirp(t, t_merger=0.0 + t_delay)
        noise = np.random.normal(0, 0.8, len(t))
        signal = h * snr_scale + noise

        from scipy.ndimage import uniform_filter1d
        signal = uniform_filter1d(signal, size=3)

        ax.plot(t, signal, color=color, linewidth=0.35, alpha=0.85)
        ax.set_ylabel("Whitened strain")
        ax.set_title(f"{label}  (bandpass 30 to 1700 Hz)", fontsize=13, loc="left")
        ax.axvline(0, color="k", linestyle=":", alpha=0.5, label="Merger")
        ax.set_xlim(-5, 0.12)
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Time relative to merger (s)  [GPS 1187008882.43]")
    fig.suptitle("GW170817  Whitened & Bandpassed Strain", fontsize=16, y=0.98)
    add_watermark(fig)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "preview_gw_whitened_strain.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================================
# Plot 2: Q-transform spectrograms
# ============================================================================

def plot_qtransform():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    t_bins = np.linspace(-4, 0.5, 600)
    f_bins = np.linspace(30, 500, 400)
    T, F = np.meshgrid(t_bins, f_bins)

    configs = [
        ("LIGO Hanford (H1)", COLOR_H1, 1.0),
        ("LIGO Livingston (L1)", COLOR_L1, 0.9),
        ("Virgo (V1)", COLOR_V1, 0.12),
    ]

    for ax, (label, color, snr_scale) in zip(axes, configs):
        # Background noise
        Z = np.random.exponential(0.5, T.shape)

        # Chirp track: f(t) ~ (-t)^(-3/8) for Newtonian inspiral
        tau = np.clip(-t_bins, 0.001, None)
        chirp_freq = 40.0 * (tau[0] / tau)**(3.0/8.0)
        chirp_freq = np.clip(chirp_freq, 30, 500)

        # Paint the chirp track onto the spectrogram
        for i, (tc, fc) in enumerate(zip(t_bins, chirp_freq)):
            if tc > 0:
                continue
            width = 8.0 + 15.0 * (1.0 / (1.0 + abs(tc)))
            amplitude = snr_scale * (3.0 + 20.0 / (1.0 + abs(tc)**0.5))
            track = amplitude * np.exp(-0.5 * ((f_bins - fc) / width)**2)
            Z[:, i] += track

        im = ax.pcolormesh(
            T, F, Z,
            cmap="viridis", shading="auto",
            vmin=0, vmax=np.percentile(Z, 99.5),
        )
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Time relative to merger (s)")
        if ax == axes[0]:
            ax.set_ylabel("Frequency (Hz)")

    fig.colorbar(im, ax=axes, label="Normalized energy", shrink=0.8)
    fig.suptitle("GW170817  Q-transform Spectrograms", fontsize=16, y=1.02)
    add_watermark(fig)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "preview_gw_qtransform.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================================
# Plot 3: Fermi GBM Light Curves
# ============================================================================

def plot_gbm_lightcurves():
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    dt = 0.064
    t = np.arange(-2, 8, dt)
    trigger_t = 0.0

    # Background rates for each detector (counts/s)
    det_configs = [
        ("NaI n1", "#2ecc71", 800, 120),
        ("NaI n2", "#27ae60", 750, 100),
        ("NaI n5", "#1abc9c", 900, 90),
        ("BGO b0", "#e67e22", 350, 40),
    ]

    for ax, (name, color, bg_rate, burst_amp) in zip(axes, det_configs):
        # Background with Poisson noise
        bg = bg_rate + np.random.normal(0, np.sqrt(bg_rate), len(t))

        # Burst signal: short pulse starting at trigger, lasting ~2s
        burst = np.zeros_like(t)
        burst_mask = (t >= 0) & (t <= 2.0)
        # Shape: fast rise, exponential decay
        t_burst = t[burst_mask]
        burst_profile = burst_amp * np.exp(-t_burst / 0.5)
        # Add a second, weaker peak
        burst_profile += burst_amp * 0.4 * np.exp(-((t_burst - 0.4) / 0.2)**2)
        burst[burst_mask] = burst_profile

        signal = bg + burst
        errors = np.sqrt(np.abs(signal) * dt) / dt

        ax.step(t, signal, where="mid", color=color, linewidth=0.8)
        ax.fill_between(
            t, signal - errors, signal + errors,
            alpha=0.15, color=color, step="mid",
        )
        ax.axvline(0, color="red", linestyle=":", alpha=0.7, label="GBM Trigger")
        ax.axvline(
            -1.734, color="blue", linestyle="--", alpha=0.7, label="GW Merger",
        )
        ax.set_ylabel("Rate (cts/s)")
        e_range = "10 to 900 keV" if name.startswith("NaI") else "200 to 40000 keV"
        ax.set_title(f"Fermi GBM {name}  ({e_range})", fontsize=13, loc="left")
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time relative to GBM trigger (s)")
    fig.suptitle(
        "GRB 170817A  Fermi GBM Light Curves (TTE, 64ms bins)",
        fontsize=16, y=0.99,
    )
    add_watermark(fig)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "preview_grb_lightcurves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================================
# Plot 4: Combined NaI Light Curve
# ============================================================================

def plot_combined_nai():
    fig, ax = plt.subplots(figsize=(14, 5))

    dt = 0.064
    t = np.arange(-2, 8, dt)

    combined = np.zeros_like(t)
    combined_var = np.zeros_like(t)

    for bg_rate, amp in [(800, 120), (750, 100), (900, 90)]:
        bg = bg_rate + np.random.normal(0, np.sqrt(bg_rate), len(t))
        burst = np.zeros_like(t)
        mask = (t >= 0) & (t <= 2.0)
        tb = t[mask]
        burst[mask] = amp * np.exp(-tb / 0.5) + amp * 0.4 * np.exp(-((tb - 0.4)/0.2)**2)
        combined += bg + burst
        combined_var += np.abs(bg + burst) * dt

    errors = np.sqrt(combined_var) / dt

    ax.step(t, combined, where="mid", color=COLOR_FERMI, linewidth=1.0)
    ax.fill_between(
        t, combined - errors, combined + errors,
        alpha=0.2, color=COLOR_FERMI, step="mid",
    )
    ax.axvline(0, color="red", linestyle=":", linewidth=1.5, label="GBM Trigger")
    ax.axvline(
        -1.734, color="blue", linestyle="--", linewidth=1.5,
        label="GW Merger (t = -1.734 s)",
    )
    ax.axvspan(0, 2.0, alpha=0.06, color="red", label="~T90 (2 s)")

    ax.set_xlabel("Time relative to GBM trigger (s)")
    ax.set_ylabel("Combined NaI rate (counts/s)")
    ax.set_title("GRB 170817A  Combined NaI (n1+n2+n5) Light Curve", fontsize=14)
    ax.legend(fontsize=11)
    add_watermark(fig)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "preview_grb_combined_nai.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================================
# Plot 5: Multi-Messenger Timeline (THE key figure)
# ============================================================================

def plot_multimessenger_timeline():
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 0.3, 1.2], hspace=0.08)

    dt = 1.0 / 4096
    t_gw = np.arange(-6, 3, dt)

    # ---- Panel 1: H1 ----
    ax1 = fig.add_subplot(gs[0])
    h1, _ = simulate_bns_chirp(t_gw)
    noise1 = np.random.normal(0, 0.8, len(t_gw))
    from scipy.ndimage import uniform_filter1d
    sig1 = uniform_filter1d(h1 * 2.5 + noise1, size=3)
    ax1.plot(t_gw, sig1, color=COLOR_H1, linewidth=0.3)
    ax1.set_ylabel("Whitened strain")
    ax1.set_title("LIGO Hanford (H1)", fontsize=13, loc="left")
    ax1.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax1.set_xlim(-6, 3)
    ax1.tick_params(labelbottom=False)

    # ---- Panel 2: L1 ----
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    h2, _ = simulate_bns_chirp(t_gw, phi0=0.5)
    noise2 = np.random.normal(0, 0.8, len(t_gw))
    sig2 = uniform_filter1d(h2 * 2.3 + noise2, size=3)
    ax2.plot(t_gw, sig2, color=COLOR_L1, linewidth=0.3)
    ax2.set_ylabel("Whitened strain")
    ax2.set_title("LIGO Livingston (L1)", fontsize=13, loc="left")
    ax2.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax2.tick_params(labelbottom=False)

    # ---- Panel 3: Timing annotation ----
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_ylim(0, 1)
    ax3.axvline(0, color="blue", linewidth=2.5, label="GW Merger")
    ax3.axvline(1.734, color="red", linewidth=2.5, label="GBM Trigger (+1.734 s)")
    ax3.annotate(
        "", xy=(1.734, 0.5), xytext=(0, 0.5),
        arrowprops=dict(arrowstyle="<->", color="purple", lw=2.5),
    )
    ax3.text(0.867, 0.72, "\u0394t = 1.734 s", ha="center", fontsize=14,
             color="purple", fontweight="bold")
    ax3.set_yticks([])
    ax3.legend(loc="upper right", fontsize=10, ncol=2)
    ax3.set_title("Detection Delay", fontsize=13, loc="left")
    ax3.tick_params(labelbottom=False)

    # ---- Panel 4: Fermi GBM ----
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    dt_gbm = 0.064
    t_gbm = np.arange(-6, 3, dt_gbm)  # relative to GW merger
    # GBM trigger is at t=1.734 relative to merger
    bg = 2400 + np.random.normal(0, 40, len(t_gbm))
    burst = np.zeros_like(t_gbm)
    mask = (t_gbm >= 1.734) & (t_gbm <= 3.734)
    tb = t_gbm[mask] - 1.734
    burst[mask] = 310 * np.exp(-tb/0.5) + 130 * np.exp(-((tb-0.4)/0.2)**2)
    signal = bg + burst
    errors = np.sqrt(np.abs(signal) * dt_gbm) / dt_gbm

    ax4.step(t_gbm, signal, where="mid", color=COLOR_FERMI, linewidth=0.8)
    ax4.fill_between(
        t_gbm, signal - errors, signal + errors,
        alpha=0.2, color=COLOR_FERMI, step="mid",
    )
    ax4.axvline(0, color="blue", linestyle=":", alpha=0.5, label="GW Merger")
    ax4.axvline(1.734, color="red", linestyle=":", alpha=0.5, label="GBM Trigger")
    ax4.set_ylabel("Combined NaI rate\n(counts/s)")
    ax4.set_xlabel("Time relative to GW merger (s)  [GPS 1187008882.43]")
    ax4.set_title("Fermi GBM (NaI n1+n2+n5, 10 to 1000 keV)", fontsize=13, loc="left")
    ax4.legend(loc="upper right", fontsize=10)

    fig.suptitle(
        "GW170817 / GRB 170817A  Multi-Messenger Timeline",
        fontsize=18, y=0.99, fontweight="bold",
    )
    add_watermark(fig)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "preview_multimessenger_timeline.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================================
# Plot 6: Detection Delay Diagram
# ============================================================================

def plot_detection_delay():
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.set_xlim(-3.5, 6)
    ax.set_ylim(-0.5, 3.8)

    # GW merger
    ax.barh(3, 0.08, left=-0.04, height=0.4, color="blue", alpha=0.9, zorder=5)
    ax.text(-0.7, 3, "GW170817\nMerger", ha="right", va="center",
            fontsize=12, color="blue", fontweight="bold")

    # GW inspiral duration
    ax.barh(3, 3.0, left=-3.0, height=0.18, color="blue", alpha=0.12)
    ax.text(-1.5, 3.25, "GW inspiral signal (~100 s visible, last 3 s shown)",
            ha="center", fontsize=9, color="blue", style="italic")

    # Fermi trigger
    grb_t = 1.734
    ax.barh(2, 2.0, left=grb_t, height=0.35, color="red", alpha=0.3, zorder=4)
    ax.barh(2, 0.08, left=grb_t-0.04, height=0.4, color="red", alpha=0.9, zorder=5)
    ax.text(grb_t + 2.3, 2, "GRB 170817A\n(T90 ~ 2 s)", ha="left", va="center",
            fontsize=12, color="red", fontweight="bold")

    # INTEGRAL
    ax.barh(1, 0.08, left=grb_t-0.04, height=0.4, color=COLOR_INTEGRAL, alpha=0.9, zorder=5)
    ax.text(grb_t + 0.4, 1, "INTEGRAL SPI-ACS\n(independent detection)",
            ha="left", va="center", fontsize=11, color=COLOR_INTEGRAL)

    # Chandra X-ray
    ax.annotate(
        "Chandra X-ray\n(+9 days)", xy=(5.3, 0), fontsize=10, color="purple",
        ha="center", fontstyle="italic",
    )
    ax.plot(5.3, 0, ">", markersize=12, color="purple")

    # Delay arrow
    ax.annotate(
        "", xy=(grb_t, 2.75), xytext=(0, 2.75),
        arrowprops=dict(arrowstyle="<->", color="purple", lw=2.5),
    )
    ax.text(grb_t/2, 2.92, f"\u0394t = {grb_t:.3f} s",
            ha="center", fontsize=14, color="purple", fontweight="bold")

    ax.set_xlabel("Time relative to GW merger (s)", fontsize=13)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["X-ray", "INTEGRAL", "Fermi GBM", "LIGO/Virgo"])
    ax.set_title(
        "Multi-Messenger Detection Sequence", fontsize=15, pad=10,
    )
    ax.grid(axis="x", alpha=0.3)
    add_watermark(fig)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "preview_detection_delay.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================================
# Plot 7: Speed of Gravity Constraint
# ============================================================================

def plot_speed_of_gravity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: bar chart of constraints
    constraints = {
        "GW170817\n(this measurement)": 4.3e-16,
        "Binary pulsar\n(indirect, Hulse-Taylor)": 1e-2,
        "Solar system\n(Laplace bound)": 2e-8,
    }
    names = list(constraints.keys())
    values = list(constraints.values())
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    bars = ax1.barh(names, values, color=colors, alpha=0.75, edgecolor="black",
                    linewidth=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("|v$_{GW}$/c - 1|  (upper bound)", fontsize=12)
    ax1.set_title("Constraints on Speed of Gravity", fontsize=14)
    ax1.invert_yaxis()
    ax1.set_xlim(1e-17, 1)

    for i, v in enumerate(values):
        ax1.text(v * 3, i, f"{v:.0e}", va="center", fontsize=11, fontweight="bold")

    # Right: schematic
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect("equal")

    # Source
    ax2.plot(1.2, 5, "o", markersize=22, color="#ff8c00", zorder=5)
    ax2.plot(1.2, 5, "o", markersize=26, color="#ff8c00", alpha=0.3, zorder=4)
    ax2.text(1.2, 6.2, "NGC 4993\n(BNS Merger)\n40 Mpc", ha="center", fontsize=10,
             fontweight="bold")

    # Earth
    ax2.plot(8.8, 5, "o", markersize=16, color="#4169e1", zorder=5)
    ax2.text(8.8, 6.2, "Earth\nDetectors", ha="center", fontsize=10,
             fontweight="bold")

    # GW path
    ax2.annotate(
        "", xy=(8.3, 5.4), xytext=(1.8, 5.4),
        arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.5),
    )
    ax2.text(5, 5.8, "Gravitational Waves (LIGO/Virgo)", ha="center",
             fontsize=11, color="blue", fontweight="bold")

    # EM path
    ax2.annotate(
        "", xy=(8.3, 4.6), xytext=(1.8, 4.6),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5, linestyle="--"),
    )
    ax2.text(5, 4.0, "Gamma Rays (Fermi/INTEGRAL)  +1.7 s", ha="center",
             fontsize=11, color="red", fontweight="bold")

    # Result box
    result_text = (
        "Travel time: ~1.3 \u00d7 10$^8$ years\n"
        "Detection delay: 1.734 s\n"
        "\n"
        "|v$_{GW}$/c - 1| < 4.3 \u00d7 10$^{-16}$"
    )
    ax2.text(
        5, 2.0, result_text,
        ha="center", fontsize=12, fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.6", facecolor="#fffde7",
            edgecolor="#666", linewidth=1.5,
        ),
    )

    ax2.set_title("Speed of Gravity Measurement", fontsize=14)
    ax2.axis("off")

    fig.suptitle(
        "GW170817: Constraining the Speed of Gravitational Waves",
        fontsize=16, fontweight="bold",
    )
    add_watermark(fig)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "preview_speed_of_gravity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================================
# Plot 8: Count Spectrum
# ============================================================================

def plot_count_spectrum():
    fig, ax = plt.subplots(figsize=(10, 6))

    energies = np.logspace(np.log10(8), np.log10(900), 80)
    e_centers = 0.5 * (energies[:-1] + energies[1:])

    det_configs = [
        ("n1", "#2ecc71"),
        ("n2", "#27ae60"),
        ("n5", "#1abc9c"),
    ]

    for name, color in det_configs:
        # Simulated broken power law spectrum
        rate = 50.0 * (e_centers / 100)**(-0.8)
        # Add a cutoff around peak energy
        rate *= np.exp(-(e_centers / 400)**2)
        # Add noise
        rate *= (1 + 0.2 * np.random.randn(len(e_centers)))
        rate = np.clip(rate, 0.01, None)

        ax.step(e_centers, rate, where="mid", color=color, linewidth=1.2, label=name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Rate (counts / s / keV)")
    ax.set_title("GRB 170817A  Count Spectrum (during T90)", fontsize=14)
    ax.set_xlim(8, 1000)
    ax.set_ylim(0.1, 200)
    ax.legend(title="NaI detectors", fontsize=11)
    add_watermark(fig)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "preview_grb_count_spectrum.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Generating preview plots...\n")

    paths = []
    paths.append(plot_whitened_strain())
    paths.append(plot_qtransform())
    paths.append(plot_gbm_lightcurves())
    paths.append(plot_combined_nai())
    paths.append(plot_multimessenger_timeline())
    paths.append(plot_detection_delay())
    paths.append(plot_speed_of_gravity())
    paths.append(plot_count_spectrum())

    print(f"\nDone! {len(paths)} preview plots generated in {PLOT_DIR}/")
