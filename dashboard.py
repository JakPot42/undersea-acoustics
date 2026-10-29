"""Rich terminal dashboard — ASCII-safe (box.ASCII2) for Windows cp1252
console compatibility, same convention as every other CLI in this
portfolio."""
from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config

console = Console(width=115)

_BANNER = "[bold cyan]undersea-acoustics[/bold cyan]  [dim]v1.0, educational demo[/dim]"


def print_banner() -> None:
    console.print()
    console.print(_BANNER)
    console.print(Panel(config.SCOPE_DISCLAIMER, box=box.ASCII2, border_style="yellow", title="[bold yellow]Scope[/bold yellow]"))


def print_profile_table(profile) -> None:
    console.rule("[bold]Sound Speed Profile (Mackenzie 1981, WOA23-grounded)[/bold]")
    axis_d, axis_c = profile.sofar_axis()
    t = Table(box=box.ASCII2)
    t.add_column("Depth (m)", justify="right")
    t.add_column("Temp (C)", justify="right")
    t.add_column("Salinity (PSU)", justify="right")
    t.add_column("Sound speed (m/s)", justify="right")
    t.add_column("")
    for d, temp, s, c in zip(config.WOA_DEPTH_M, config.WOA_TEMP_C, config.WOA_SALINITY_PSU,
                              profile.speed_at(config.WOA_DEPTH_M)):
        mark = "<- SOFAR axis (min)" if abs(d - axis_d) < 1e-6 else ""
        t.add_row(f"{d:.0f}", f"{temp:.1f}", f"{s:.2f}", f"{c:.1f}", mark)
    console.print(t)
    console.print(f"[dim]{config.REGION_CITATION}[/dim]")


def print_shadow_summary(shadow_km: dict) -> None:
    console.rule("[bold]Shadow Zone Summary (no direct/refracted ray coverage)[/bold]")
    t = Table(box=box.ASCII2)
    t.add_column("Target depth (m)", justify="right")
    t.add_column("Shadow ranges (km)", overflow="fold")
    for depth_m, intervals in sorted(shadow_km.items()):
        span = "; ".join(f"{a:.1f}-{b:.1f}" for a, b in intervals) or "fully covered"
        t.add_row(f"{depth_m:.0f}", span)
    console.print(t)


def print_detection_table(title: str, detections, top_n: int = 10) -> None:
    console.rule(f"[bold]{title}[/bold]")
    if not detections:
        console.print("[dim]No detections cleared the adaptive threshold.[/dim]")
        return
    shown = detections[:top_n]
    t = Table(box=box.ASCII2)
    t.add_column("Frequency (Hz)", justify="right")
    t.add_column("Magnitude (dB)", justify="right")
    t.add_column("Noise floor (dB)", justify="right")
    t.add_column("Margin (dB)", justify="right")
    for d in shown:
        t.add_row(f"{d.freq_hz:.2f}", f"{d.magnitude_db:.1f}", f"{d.threshold_db - 8.0:.1f}", f"{d.margin_db:.1f}")
    console.print(t)
    if len(detections) > top_n:
        console.print(f"[dim]... {len(detections) - top_n} more marginal detections not shown (ranked by margin above noise floor).[/dim]")


def print_brief(text: str) -> None:
    console.rule("[bold]Acoustic Environment Brief[/bold]")
    console.print(text)
    console.print()
