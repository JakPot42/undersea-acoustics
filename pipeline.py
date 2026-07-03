"""pipeline.py — orchestrates the full P18 pipeline so main.py's CLI
commands (and tests) share one implementation instead of duplicating it."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

import config
from brief import AssessmentInputs, generate_brief
from dsp import DemonResult, Spectrogram, compute_demon, compute_spectrogram, tonal_lines
from hydrophone import Recording, build_demo_recording
from pdf_report import build_asw_brief_pdf
from plots import plot_two_panel
from ray_tracing import Ray, ShadowGrid, compute_shadow_grid, shadow_intervals_at_depth, trace_fan
from sound_speed import SoundSpeedProfile, build_profile

DEFAULT_SOURCE_DEPTH_M = 50.0
DEFAULT_MAX_RANGE_M = 80_000.0
DEFAULT_LAUNCH_ANGLES_DEG = np.linspace(-20, 20, 161)
DEFAULT_SHADOW_TARGET_DEPTHS_M = [100, 300, 600, 1000, 2000]


@dataclass
class PropagationResult:
    profile: SoundSpeedProfile
    rays: list[Ray]
    shadow_grid: ShadowGrid
    shadow_km: dict
    source_depth_m: float
    bottom_depth_m: float


def run_propagation(
    source_depth_m: float = DEFAULT_SOURCE_DEPTH_M,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    bottom_depth_m: float | None = None,
    target_depths_m: list[float] | None = None,
) -> PropagationResult:
    bottom_depth_m = bottom_depth_m or config.REGION_BOTTOM_DEPTH_M
    target_depths_m = target_depths_m or DEFAULT_SHADOW_TARGET_DEPTHS_M

    profile = build_profile()
    rays = trace_fan(profile, source_depth_m, DEFAULT_LAUNCH_ANGLES_DEG, max_range_m, bottom_depth_m, ds=15.0)
    grid = compute_shadow_grid(rays, max_range_m, bottom_depth_m, range_bins=320, depth_bins=200)
    shadow_km = {
        d: [(a / 1000.0, b / 1000.0) for a, b in shadow_intervals_at_depth(grid, d)]
        for d in target_depths_m
    }
    return PropagationResult(profile, rays, grid, shadow_km, source_depth_m, bottom_depth_m)


@dataclass
class SignalResult:
    recording: Recording
    spectrogram: Spectrogram
    tonal_detections: list
    demon: DemonResult


def run_signal_analysis(duration_s: float | None = None, use_live: bool = False) -> SignalResult:
    rec = build_demo_recording(duration_s=duration_s, use_live=use_live)
    spec = compute_spectrogram(rec.samples, rec.sample_rate_hz)
    tonals = tonal_lines(spec)
    demon = compute_demon(rec.samples, rec.sample_rate_hz)
    return SignalResult(rec, spec, tonals, demon)


def run_full_pipeline(output_dir: str = config.OUTPUT_DIR, use_live: bool = False) -> tuple[PropagationResult, SignalResult, str, str, str]:
    """Runs the whole pipeline end to end: ray trace, signal analysis,
    two-panel figure, Claude/template brief, PDF. Returns
    (propagation, signal, brief_text, figure_path, pdf_path)."""
    os.makedirs(output_dir, exist_ok=True)
    prop = run_propagation()
    sig = run_signal_analysis(use_live=use_live)

    figure_path = os.path.join(output_dir, "propagation_and_spectrogram.png")
    plot_two_panel(
        prop.profile, prop.rays, prop.source_depth_m, prop.bottom_depth_m,
        sig.spectrogram, sig.tonal_detections, sig.demon, prop.shadow_grid, figure_path,
    )

    axis_d, axis_c = prop.profile.sofar_axis()
    inputs = AssessmentInputs(
        region_name=config.REGION_NAME,
        region_citation=config.REGION_CITATION,
        source_depth_m=prop.source_depth_m,
        sofar_axis_depth_m=axis_d,
        sofar_axis_speed_ms=axis_c,
        surface_speed_ms=float(prop.profile.speed_ms[0]),
        bottom_speed_ms=float(prop.profile.speed_ms[-1]),
        shadow_zone_km=prop.shadow_km,
        tonal_detections=sig.tonal_detections,
        demon_detections=sig.demon.detections,
        background_source=sig.recording.background_source,
    )
    brief_text = generate_brief(inputs)

    pdf_path = os.path.join(output_dir, "asw_environment_brief.pdf")
    build_asw_brief_pdf(pdf_path, figure_path, brief_text, config.REGION_NAME)

    return prop, sig, brief_text, figure_path, pdf_path
