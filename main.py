"""NUWC Undersea Acoustic Intelligence Suite CLI.

EDUCATIONAL / UNCLASSIFIED DEMONSTRATION ONLY. See config.SCOPE_DISCLAIMER,
printed on every command. Sound-speed profile is synthetic (WOA23-grounded).
The hydrophone background is a real public OOI recording; the detected
tonal/DEMON contact is a disclosed SYNTHETIC test signal, never a real
vessel.
"""
from __future__ import annotations

import os
import sys

import click

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import dashboard
import pipeline
from dashboard import console


@click.group()
def cli() -> None:
    """NUWC Undersea Acoustic Intelligence Suite: sound-speed profiling,
    Snell's-law ray tracing / shadow-zone mapping, and a DEMON + tonal
    signal-processing pipeline on a real public hydrophone recording with
    a disclosed synthetic contact overlay.

    \b
    EDUCATIONAL / UNCLASSIFIED DEMONSTRATION ONLY. No real submarine
    acoustic data exists in the public domain. The detected contact is a
    disclosed SYNTHETIC test signal, never a real vessel.
    """


@cli.command()
def profile() -> None:
    """Show the sound-speed profile table and the SOFAR channel axis."""
    dashboard.print_banner()
    p = pipeline.run_propagation()
    dashboard.print_profile_table(p.profile)


@cli.command()
@click.option("--source-depth", type=float, default=pipeline.DEFAULT_SOURCE_DEPTH_M, help="Source depth in meters.")
@click.option("--out", default=None, help="PNG output path (default: output/propagation.png)")
def raytrace(source_depth: float, out: str | None) -> None:
    """Run the 2D Snell's law ray tracer and report shadow zones."""
    dashboard.print_banner()
    console.print(f"Tracing {len(pipeline.DEFAULT_LAUNCH_ANGLES_DEG)} rays from {source_depth:.0f} m source depth...")
    p = pipeline.run_propagation(source_depth_m=source_depth)
    dashboard.print_shadow_summary(p.shadow_km)

    from plots import plot_propagation
    import matplotlib.pyplot as plt
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out = out or os.path.join(config.OUTPUT_DIR, "propagation.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_propagation(ax, p.profile, p.rays, p.source_depth_m, p.bottom_depth_m, shadow_grid=p.shadow_grid)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    console.print(f"Saved propagation plot to {out}")


@cli.command()
@click.option("--out", default=None, help="PNG output path (default: output/spectrogram.png)")
@click.option("--live", is_flag=True, default=False, help="Fetch a fresh hydrophone segment live via obspy/FDSN instead of the bundled one.")
def spectrogram(out: str | None, live: bool) -> None:
    """Build the hydrophone recording and run the STFT spectrogram +
    tonal peak-picker."""
    dashboard.print_banner()
    console.print("Assembling hydrophone recording (real OOI background + synthetic contact)...")
    sig = pipeline.run_signal_analysis(use_live=live)
    console.print(f"[dim]{sig.recording.background_source}[/dim]")
    dashboard.print_detection_table("Tonal Lines (LOFAR peak-picking)", sig.tonal_detections)

    from plots import plot_spectrogram
    import matplotlib.pyplot as plt
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out = out or os.path.join(config.OUTPUT_DIR, "spectrogram.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    mesh = plot_spectrogram(ax, sig.spectrogram, sig.tonal_detections, demon_result=sig.demon)
    fig.colorbar(mesh, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    console.print(f"Saved spectrogram plot to {out}")


@cli.command()
@click.option("--live", is_flag=True, default=False, help="Fetch a fresh hydrophone segment live instead of the bundled one.")
def demon(live: bool) -> None:
    """Run DEMON envelope-modulation analysis and report candidate
    blade-rate lines."""
    dashboard.print_banner()
    sig = pipeline.run_signal_analysis(use_live=live)
    console.print(f"Analysis band: {sig.demon.analysis_band_hz} Hz, envelope spectrum out to {sig.demon.max_demon_freq_hz} Hz")
    dashboard.print_detection_table("DEMON Envelope-Modulation Lines", sig.demon.detections)
    console.print(f"[yellow]{config.SYNTHETIC_CONTACT['label']}[/yellow] -- shaft rate "
                  f"{config.SYNTHETIC_CONTACT['shaft_rate_hz']} Hz x {config.SYNTHETIC_CONTACT['blade_count']} blades "
                  f"= {config.SYNTHETIC_CONTACT['shaft_rate_hz']*config.SYNTHETIC_CONTACT['blade_count']:.2f} Hz")


@cli.command()
@click.option("--live", is_flag=True, default=False, help="Fetch a fresh hydrophone segment live instead of the bundled one.")
def brief(live: bool) -> None:
    """Run the full pipeline (ray trace + DEMON/tonal signal analysis)
    and generate the two-panel figure + ASW Environment Brief PDF."""
    dashboard.print_banner()
    console.print("Running full pipeline (ray trace + signal analysis + brief)...")
    prop, sig, brief_text, figure_path, pdf_path = pipeline.run_full_pipeline(use_live=live)
    dashboard.print_shadow_summary(prop.shadow_km)
    dashboard.print_detection_table("Tonal Lines", sig.tonal_detections)
    dashboard.print_detection_table("DEMON Lines", sig.demon.detections)
    dashboard.print_brief(brief_text)
    console.print(f"Saved figure: {figure_path}")
    console.print(f"Saved PDF brief: {pdf_path}")
    if config.DEMO_MODE:
        console.print("[dim]DEMO_MODE=True -- brief uses a deterministic template, not a live Claude call.[/dim]")


@cli.command()
@click.pass_context
def demo(ctx: click.Context) -> None:
    """Alias for `brief` -- full pipeline, bundled real hydrophone data,
    no API key or network needed."""
    ctx.invoke(brief, live=False)


@cli.command(name="fetch-hydrophone")
@click.option("--start", required=True, help="ISO8601 UTC start time, e.g. 2024-06-01T00:00:00")
@click.option("--duration", type=float, default=180.0, help="Duration in seconds.")
@click.option("--out", required=True, help="Output .npy path.")
def fetch_hydrophone(start: str, duration: float, out: str) -> None:
    """Pull a fresh real hydrophone segment from OOI/EarthScope (network
    OO, station AXBA1, channel HDH) and cache it as a .npy file. Not used
    by demo/brief; this is for pulling current data independently."""
    from hydrophone import fetch_live_background
    console.print(f"Fetching {duration}s from {start} via EarthScope FDSN (OO.AXBA1..HDH)...")
    data, sr = fetch_live_background(start, duration)
    import numpy as np
    np.save(out, data.astype(np.int32))
    console.print(f"Wrote {len(data)} samples at {sr} Hz to {out}")


if __name__ == "__main__":
    cli()
