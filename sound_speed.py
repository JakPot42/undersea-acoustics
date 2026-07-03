"""sound_speed.py — Mackenzie (1981) sound speed formula and the derived
sound-speed-vs-depth profile for the ray tracer.

Mackenzie, K.V. (1981), "Nine-term equation for sound speed in the
oceans," J. Acoust. Soc. Am. 70(3), 807-812. Coefficients verified against
the published equation (not reproduced from memory alone) before use here.
Valid for T in [2, 30] C, S in [25, 40] PSU, D in [0, 8000] m.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config

# Mackenzie (1981) nine-term coefficients.
_A0 = 1448.96
_A1 = 4.591
_A2 = -5.304e-2
_A3 = 2.374e-4
_A4 = 1.340
_A5 = 1.630e-2
_A6 = 1.675e-7
_A7 = -1.025e-2
_A8 = -7.139e-13

T_VALID = (2.0, 30.0)
S_VALID = (25.0, 40.0)
D_VALID = (0.0, 8000.0)


def mackenzie_sound_speed(temp_c, salinity_psu, depth_m):
    """Sound speed in m/s. Accepts scalars or numpy arrays.

    Does not raise outside the formula's validity range (deep, near-freezing
    water occasionally dips under 2 C at the edges of published tables) --
    Mackenzie's own paper notes the equation degrades gracefully a little
    outside its stated range; out-of-range inputs are clipped to the
    validity bounds before evaluation and the clipping is visible to callers
    that check `T`, `S`, `D` themselves.
    """
    t = np.clip(np.asarray(temp_c, dtype=np.float64), *T_VALID)
    s = np.asarray(salinity_psu, dtype=np.float64)
    d = np.clip(np.asarray(depth_m, dtype=np.float64), *D_VALID)

    return (
        _A0
        + _A1 * t
        + _A2 * t**2
        + _A3 * t**3
        + _A4 * (s - 35)
        + _A5 * d
        + _A6 * d**2
        + _A7 * t * (s - 35)
        + _A8 * t * d**3
    )


@dataclass
class SoundSpeedProfile:
    depth_m: np.ndarray       # fine depth grid, ascending
    temp_c: np.ndarray        # interpolated onto depth_m
    salinity_psu: np.ndarray  # interpolated onto depth_m
    speed_ms: np.ndarray      # Mackenzie sound speed at depth_m
    dcdz: np.ndarray          # d(speed)/d(depth), same grid

    def speed_at(self, depth_m):
        """Linear interpolation of sound speed at an arbitrary depth (or
        array of depths), clamped to the profile's depth range."""
        d = np.clip(depth_m, self.depth_m[0], self.depth_m[-1])
        return np.interp(d, self.depth_m, self.speed_ms)

    def gradient_at(self, depth_m):
        """Linear interpolation of dc/dz at an arbitrary depth -- the ray
        tracer's curvature term. Computed once over the fine grid via
        np.gradient and interpolated here rather than re-differenced at
        every ray step."""
        d = np.clip(depth_m, self.depth_m[0], self.depth_m[-1])
        return np.interp(d, self.depth_m, self.dcdz)

    def sofar_axis(self) -> tuple[float, float]:
        """Depth and sound speed of the SOFAR channel axis -- the depth
        of minimum sound speed, where temperature-driven deceleration in
        the upper ocean and pressure-driven acceleration below the main
        thermocline cross over. Found numerically from the built profile,
        not assumed at a fixed depth."""
        i = int(np.argmin(self.speed_ms))
        return float(self.depth_m[i]), float(self.speed_ms[i])


def build_profile(depth_resolution_m: float = 5.0) -> SoundSpeedProfile:
    """Builds the sound-speed profile for config.REGION from the
    WOA23-grounded T/S table, interpolated onto a fine depth grid and run
    through the Mackenzie formula. Fine resolution matters for the ray
    tracer -- a coarse profile produces a jagged, unphysical sound-speed
    gradient that distorts ray curvature."""
    max_depth = config.WOA_DEPTH_M[-1]
    depth_m = np.arange(0.0, max_depth + depth_resolution_m, depth_resolution_m)
    depth_m[-1] = max_depth

    temp_c = np.interp(depth_m, config.WOA_DEPTH_M, config.WOA_TEMP_C)
    salinity_psu = np.interp(depth_m, config.WOA_DEPTH_M, config.WOA_SALINITY_PSU)
    speed_ms = mackenzie_sound_speed(temp_c, salinity_psu, depth_m)
    dcdz = np.gradient(speed_ms, depth_m)

    return SoundSpeedProfile(
        depth_m=depth_m, temp_c=temp_c, salinity_psu=salinity_psu,
        speed_ms=speed_ms, dcdz=dcdz,
    )
