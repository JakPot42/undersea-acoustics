"""ray_tracing.py — 2D range-independent Snell's law ray tracer.

Physics: in a horizontally-stratified ocean (sound speed c depends only on
depth z, not range r), Snell's law generalizes to a conserved ray
invariant xi = cos(theta)/c(z), where theta is the grazing angle measured
from the horizontal. Rather than track theta directly (which needs
branch-handling / arccos near turning points), this integrates the
standard range-independent ray equations in (r, z, eta) with
eta = sin(theta)/c(z):

    dr/ds = c(z) * xi
    dz/ds = c(z) * eta
    d(eta)/ds = -(1/c(z)^2) * dc/dz

xi is fixed by the launch angle and is invariant along the whole ray
(that invariance IS Snell's law here); eta evolves smoothly through zero
at a turning point, so no special-casing is needed there -- this
formulation (and the eta variable name) follows the standard treatment in
Jensen, Kuperman, Porter & Schmidt, "Computational Ocean Acoustics."

Surface (z=0) and bottom (z=bottom_depth_m) are modeled as simple
specular reflectors (mirror the ray, flip eta's sign) -- no reflection
loss or bottom-interaction attenuation is modeled. That's a real
simplification: a full model (e.g. BELLHOP) would attenuate
bottom-bounce rays. Here it only matters for the ray-density shadow-zone
picture, not for absolute transmission loss, which this tool does not
compute.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation

from sound_speed import SoundSpeedProfile


@dataclass
class Ray:
    launch_angle_deg: float
    r: np.ndarray   # range, m
    z: np.ndarray   # depth, m
    turning_points: list = field(default_factory=list)   # depths where eta crossed zero
    surface_bounces: int = 0
    bottom_bounces: int = 0


def _derivatives(z: float, eta: float, xi: float, profile: SoundSpeedProfile):
    c = float(profile.speed_at(z))
    dcdz = float(profile.gradient_at(z))
    dz = c * eta
    deta = -(1.0 / c**2) * dcdz
    dr = c * xi
    return dr, dz, deta


def trace_ray(
    profile: SoundSpeedProfile,
    source_depth_m: float,
    launch_angle_deg: float,
    max_range_m: float,
    bottom_depth_m: float,
    ds: float = 20.0,
) -> Ray:
    """Traces one ray via 4th-order Runge-Kutta in arc length s.

    launch_angle_deg > 0 means launched downward (depth increasing),
    < 0 means launched upward (toward the surface).
    """
    theta0 = np.radians(launch_angle_deg)
    c0 = float(profile.speed_at(source_depth_m))
    xi = np.cos(theta0) / c0        # ray invariant -- constant for the whole ray
    eta = np.sin(theta0) / c0

    r = 0.0
    z = float(source_depth_m)
    rs = [r]
    zs = [z]
    turning_points = []
    surface_bounces = 0
    bottom_bounces = 0
    prev_eta_sign = np.sign(eta) if eta != 0 else 0.0

    n_steps = int(max_range_m / max(ds * abs(xi) * c0, 1e-6)) + 5000
    # Cap steps generously by arc length rather than range, since a ray
    # can spend a long arc length covering little range near a turning
    # point (eta -> 0, dz/ds -> 0 there too, dr/ds ~ c*xi stays roughly
    # steady) -- guard against runaway loops on pathological inputs.
    max_steps = 200_000

    step = 0
    while r < max_range_m and step < max_steps:
        # RK4 on (z, eta); r accumulates via its own derivative (dr/ds = c*xi)
        k1r, k1z, k1e = _derivatives(z, eta, xi, profile)
        k2r, k2z, k2e = _derivatives(z + 0.5 * ds * k1z, eta + 0.5 * ds * k1e, xi, profile)
        k3r, k3z, k3e = _derivatives(z + 0.5 * ds * k2z, eta + 0.5 * ds * k2e, xi, profile)
        k4r, k4z, k4e = _derivatives(z + ds * k3z, eta + ds * k3e, xi, profile)

        r += (ds / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
        z_new = z + (ds / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
        eta_new = eta + (ds / 6.0) * (k1e + 2 * k2e + 2 * k3e + k4e)

        new_sign = np.sign(eta_new) if eta_new != 0 else prev_eta_sign
        if new_sign != 0 and prev_eta_sign != 0 and new_sign != prev_eta_sign:
            turning_points.append(z_new)
        if new_sign != 0:
            prev_eta_sign = new_sign

        # Surface reflection.
        if z_new < 0.0:
            z_new = -z_new
            eta_new = -eta_new
            surface_bounces += 1
        # Bottom reflection.
        if z_new > bottom_depth_m:
            z_new = 2 * bottom_depth_m - z_new
            eta_new = -eta_new
            bottom_bounces += 1

        z = z_new
        eta = eta_new
        rs.append(r)
        zs.append(z)
        step += 1

    return Ray(
        launch_angle_deg=launch_angle_deg,
        r=np.array(rs),
        z=np.array(zs),
        turning_points=turning_points,
        surface_bounces=surface_bounces,
        bottom_bounces=bottom_bounces,
    )


def trace_fan(
    profile: SoundSpeedProfile,
    source_depth_m: float,
    launch_angles_deg,
    max_range_m: float,
    bottom_depth_m: float,
    ds: float = 20.0,
) -> list[Ray]:
    """Traces a whole fan of rays at once, integrated together as arrays
    (one RK4 step advances every ray in the fan in a single vectorized
    call) instead of looping `trace_ray` per angle. Same physics and same
    per-ray RK4 update as `trace_ray` -- this exists purely for speed: a
    161-ray fan looping `trace_ray` takes ~80s on this machine (millions
    of scalar np.interp calls dominate runtime); batched across rays it's
    a couple of seconds. Verified to match `trace_ray` step-for-step
    (see tests/test_ray_tracing.py) before switching the CLI over to it.
    """
    angles = np.asarray(launch_angles_deg, dtype=np.float64)
    k = len(angles)
    theta0 = np.radians(angles)
    c0 = float(profile.speed_at(source_depth_m))
    xi = np.cos(theta0) / c0     # fixed ray invariant per ray, shape (k,)

    r = np.zeros(k)
    z = np.full(k, float(source_depth_m))
    eta = np.sin(theta0) / c0

    # Conservative upper bound on steps needed: the slowest ray to cover
    # max_range is the one with smallest xi (steepest angle), using the
    # profile's slowest sound speed as a floor on dr/ds.
    min_rate = ds * max(np.min(np.abs(xi)) * float(profile.speed_ms.min()), 1e-9)
    max_steps = int(max_range_m / min_rate * 1.3) + 200

    r_hist = np.zeros((max_steps + 1, k))
    z_hist = np.full((max_steps + 1, k), float(source_depth_m))
    surface_bounces = np.zeros(k, dtype=int)
    bottom_bounces = np.zeros(k, dtype=int)
    turning_points: list[list[float]] = [[] for _ in range(k)]
    prev_eta_sign = np.sign(eta)

    def derivs(z_arr, eta_arr):
        c = profile.speed_at(z_arr)
        dcdz = profile.gradient_at(z_arr)
        return c * xi, c * eta_arr, -(1.0 / c**2) * dcdz

    n_finished_at = np.full(k, max_steps, dtype=int)  # step index at which each ray first exceeds max_range
    done_flag = np.zeros(k, dtype=bool)

    for step in range(1, max_steps + 1):
        k1r, k1z, k1e = derivs(z, eta)
        k2r, k2z, k2e = derivs(z + 0.5 * ds * k1z, eta + 0.5 * ds * k1e)
        k3r, k3z, k3e = derivs(z + 0.5 * ds * k2z, eta + 0.5 * ds * k2e)
        k4r, k4z, k4e = derivs(z + ds * k3z, eta + ds * k3e)

        r = r + (ds / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
        z_new = z + (ds / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
        eta_new = eta + (ds / 6.0) * (k1e + 2 * k2e + 2 * k3e + k4e)

        new_sign = np.where(eta_new != 0, np.sign(eta_new), prev_eta_sign)
        turned = (new_sign != 0) & (prev_eta_sign != 0) & (new_sign != prev_eta_sign)
        for idx in np.nonzero(turned)[0]:
            turning_points[idx].append(float(z_new[idx]))
        nonzero = new_sign != 0
        prev_eta_sign = np.where(nonzero, new_sign, prev_eta_sign)

        below_surface = z_new < 0.0
        z_new = np.where(below_surface, -z_new, z_new)
        eta_new = np.where(below_surface, -eta_new, eta_new)
        surface_bounces += below_surface.astype(int)

        above_bottom = z_new > bottom_depth_m
        z_new = np.where(above_bottom, 2 * bottom_depth_m - z_new, z_new)
        eta_new = np.where(above_bottom, -eta_new, eta_new)
        bottom_bounces += above_bottom.astype(int)

        z = z_new
        eta = eta_new
        r_hist[step] = r
        z_hist[step] = z

        newly_done = (~done_flag) & (r >= max_range_m)
        n_finished_at[newly_done] = step
        done_flag |= newly_done
        if done_flag.all():
            break

    rays = []
    for idx, angle in enumerate(angles):
        n = n_finished_at[idx] + 1
        rays.append(
            Ray(
                launch_angle_deg=float(angle),
                r=r_hist[:n, idx].copy(),
                z=z_hist[:n, idx].copy(),
                turning_points=turning_points[idx],
                surface_bounces=int(surface_bounces[idx]),
                bottom_bounces=int(bottom_bounces[idx]),
            )
        )
    return rays


@dataclass
class ShadowGrid:
    range_edges_m: np.ndarray     # length N+1
    depth_edges_m: np.ndarray     # length M+1
    insonified: np.ndarray        # boolean, shape (M, N) -- depth rows, range cols


def compute_shadow_grid(
    rays: list[Ray],
    max_range_m: float,
    bottom_depth_m: float,
    range_bins: int = 400,
    depth_bins: int = 200,
    close_gaps: bool = True,
) -> ShadowGrid:
    """Rasterizes every ray's path onto a range/depth grid to mark which
    cells are reached by at least one ray ("insonified" in the simple
    ray-density sense used here -- this is NOT a transmission-loss
    calculation, just ray coverage). Cells no ray ever passes through are
    the shadow zone.

    A finite ray fan naturally leaves thin rasterization gaps between
    adjacent rays as they diverge with range, even where the ocean is
    genuinely insonified -- an artifact of fan density vs. grid
    resolution, not a real acoustic effect. `close_gaps` runs a light
    1-cell binary dilation along the range axis only (never across depth,
    so it can't paper over a real depth-direction shadow zone) to remove
    that discretization noise. Disable it to see the raw ray hits."""
    range_edges = np.linspace(0.0, max_range_m, range_bins + 1)
    depth_edges = np.linspace(0.0, bottom_depth_m, depth_bins + 1)
    insonified = np.zeros((depth_bins, range_bins), dtype=bool)

    for ray in rays:
        r_idx = np.clip(np.digitize(ray.r, range_edges) - 1, 0, range_bins - 1)
        z_idx = np.clip(np.digitize(ray.z, depth_edges) - 1, 0, depth_bins - 1)
        insonified[z_idx, r_idx] = True

    if close_gaps:
        insonified = binary_dilation(insonified, structure=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]))

    return ShadowGrid(range_edges_m=range_edges, depth_edges_m=depth_edges, insonified=insonified)


def shadow_intervals_at_depth(grid: ShadowGrid, target_depth_m: float) -> list[tuple[float, float]]:
    """Range intervals (in m) at a given depth that are NOT covered by any
    ray -- i.e. the shadow zone at that depth. Used to make the ray-trace
    result concrete for the acoustic brief ("a contact at this depth is outside
    direct/refracted coverage between range X and Y")."""
    row = np.clip(np.digitize([target_depth_m], grid.depth_edges_m)[0] - 1, 0, len(grid.depth_edges_m) - 2)
    covered = grid.insonified[row]
    intervals = []
    in_shadow = False
    start = None
    for i, hit in enumerate(covered):
        if not hit and not in_shadow:
            in_shadow = True
            start = grid.range_edges_m[i]
        elif hit and in_shadow:
            in_shadow = False
            intervals.append((start, grid.range_edges_m[i]))
    if in_shadow:
        intervals.append((start, grid.range_edges_m[-1]))
    return intervals
