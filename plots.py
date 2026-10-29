"""plots.py — the two-panel demo figure: ray-traced propagation model
(left) and STFT spectrogram with DEMON-detected tonal lines overlaid
(right)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dsp import Spectrogram
from ray_tracing import Ray, ShadowGrid
from sound_speed import SoundSpeedProfile


def plot_propagation(
    ax,
    profile: SoundSpeedProfile,
    rays: list[Ray],
    source_depth_m: float,
    bottom_depth_m: float,
    shadow_grid: ShadowGrid | None = None,
):
    if shadow_grid is not None:
        # Shade the un-insonified cells so the shadow zone is visually obvious,
        # not just implied by the absence of ray lines.
        shadow_mask = np.where(shadow_grid.insonified, np.nan, 1.0)
        ax.pcolormesh(
            shadow_grid.range_edges_m / 1000.0, shadow_grid.depth_edges_m,
            shadow_mask, cmap="Reds", vmin=0, vmax=1.6, alpha=0.35, shading="flat",
        )

    for ray in rays:
        ax.plot(ray.r / 1000.0, ray.z, linewidth=0.5, color="steelblue", alpha=0.7)

    axis_d, _ = profile.sofar_axis()
    ax.axhline(axis_d, color="darkorange", linestyle="--", linewidth=1.2, label=f"SOFAR axis ({axis_d:.0f} m)")
    ax.scatter([0], [source_depth_m], color="black", zorder=5, s=30, marker="v", label=f"Source ({source_depth_m:.0f} m)")

    ax.set_ylim(bottom_depth_m, 0)
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Ray-Traced Propagation Model\n(shaded = acoustic shadow zone, no ray coverage)")
    ax.legend(loc="lower right", fontsize=8)


def plot_spectrogram(ax, spectrogram: Spectrogram, tonal_detections, demon_result=None, freq_max_hz: float = 100.0):
    mask = spectrogram.freqs_hz <= freq_max_hz
    mesh = ax.pcolormesh(
        spectrogram.times_s, spectrogram.freqs_hz[mask], spectrogram.magnitude_db[mask, :],
        cmap="viridis", shading="auto",
    )
    for d in tonal_detections:
        ax.axhline(d.freq_hz, color="red", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.text(spectrogram.times_s[-1] * 1.01, d.freq_hz, f"{d.freq_hz:.1f} Hz", fontsize=7, va="center", color="red")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    title = "STFT Spectrogram (red = detected tonal lines)"
    if demon_result is not None and demon_result.detections:
        top = demon_result.detections[0]
        title += f"\nDEMON candidate blade rate: {top.freq_hz:.2f} Hz ({top.margin_db:.1f} dB)"
    ax.set_title(title)
    return mesh


def plot_two_panel(
    profile: SoundSpeedProfile,
    rays: list[Ray],
    source_depth_m: float,
    bottom_depth_m: float,
    spectrogram: Spectrogram,
    tonal_detections,
    demon_result,
    shadow_grid: ShadowGrid | None,
    save_path: str,
) -> str:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_propagation(ax_left, profile, rays, source_depth_m, bottom_depth_m, shadow_grid=shadow_grid)
    mesh = plot_spectrogram(ax_right, spectrogram, tonal_detections, demon_result=demon_result)
    fig.colorbar(mesh, ax=ax_right, label="dB")
    fig.suptitle("Undersea Acoustics (educational demo) -- Environment + Signal Picture", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
