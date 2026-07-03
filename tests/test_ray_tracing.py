import numpy as np
import pytest

from ray_tracing import (
    compute_shadow_grid,
    shadow_intervals_at_depth,
    trace_fan,
    trace_ray,
)
from sound_speed import build_profile

PROFILE = build_profile()


def test_downward_launch_increases_depth_initially():
    ray = trace_ray(PROFILE, source_depth_m=50, launch_angle_deg=10.0, max_range_m=2000, bottom_depth_m=5200, ds=15)
    assert ray.z[5] > ray.z[0]


def test_upward_launch_decreases_depth_initially():
    ray = trace_ray(PROFILE, source_depth_m=500, launch_angle_deg=-10.0, max_range_m=2000, bottom_depth_m=5200, ds=15)
    assert ray.z[5] < ray.z[0]


def test_surface_reflection_never_goes_negative():
    ray = trace_ray(PROFILE, source_depth_m=20, launch_angle_deg=-15.0, max_range_m=30000, bottom_depth_m=5200, ds=15)
    assert ray.z.min() >= 0.0
    assert ray.surface_bounces >= 1


def test_bottom_reflection_never_exceeds_bottom_depth():
    bottom = 5200
    ray = trace_ray(PROFILE, source_depth_m=50, launch_angle_deg=25.0, max_range_m=60000, bottom_depth_m=bottom, ds=15)
    assert ray.z.max() <= bottom + 1e-6
    assert ray.bottom_bounces >= 1


def test_range_is_monotonically_nondecreasing():
    ray = trace_ray(PROFILE, source_depth_m=50, launch_angle_deg=12.0, max_range_m=40000, bottom_depth_m=5200, ds=15)
    assert np.all(np.diff(ray.r) >= -1e-9)


def test_axial_ray_stays_close_to_the_sofar_axis():
    axis_depth, _ = PROFILE.sofar_axis()
    ray = trace_ray(PROFILE, source_depth_m=axis_depth, launch_angle_deg=0.0, max_range_m=50000, bottom_depth_m=5200, ds=20)
    # The axial ray is the degenerate/critical ray of the channel -- it
    # should stay within a modest excursion of the axis, not run off to
    # the surface or seafloor.
    assert ray.z.max() < axis_depth + 500
    assert ray.z.min() > axis_depth - 500


def test_vectorized_fan_matches_scalar_trace_ray_exactly():
    scalar = trace_ray(PROFILE, source_depth_m=50, launch_angle_deg=7.0, max_range_m=20000, bottom_depth_m=5200, ds=15)
    fanned = trace_fan(PROFILE, source_depth_m=50, launch_angles_deg=[7.0], max_range_m=20000, bottom_depth_m=5200, ds=15)[0]
    n = min(len(scalar.r), len(fanned.r))
    assert np.allclose(scalar.r[:n], fanned.r[:n])
    assert np.allclose(scalar.z[:n], fanned.z[:n])
    assert scalar.surface_bounces == fanned.surface_bounces
    assert scalar.bottom_bounces == fanned.bottom_bounces


def test_fan_produces_one_ray_per_launch_angle():
    angles = [-10.0, -5.0, 0.0, 5.0, 10.0]
    rays = trace_fan(PROFILE, source_depth_m=50, launch_angles_deg=angles, max_range_m=10000, bottom_depth_m=5200, ds=20)
    assert len(rays) == len(angles)
    assert [r.launch_angle_deg for r in rays] == angles


def test_shadow_zone_exists_below_a_near_surface_source():
    # Classic textbook case: a near-surface source above a subsurface sound
    # channel leaves a genuine shadow zone at intermediate range/depth
    # before refracted rays reconverge. This is a physics regression guard,
    # not just a shape check -- if the ray tracer breaks, this goes empty.
    angles = np.linspace(-20, 20, 81)
    rays = trace_fan(PROFILE, source_depth_m=50, launch_angles_deg=angles, max_range_m=60000, bottom_depth_m=5200, ds=20)
    grid = compute_shadow_grid(rays, max_range_m=60000, bottom_depth_m=5200, range_bins=240, depth_bins=150)
    intervals = shadow_intervals_at_depth(grid, target_depth_m=300)
    assert len(intervals) > 0
    total_shadow_km = sum((b - a) for a, b in intervals) / 1000.0
    assert total_shadow_km > 5.0
