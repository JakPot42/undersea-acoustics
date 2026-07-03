import numpy as np
import pytest

from sound_speed import build_profile, mackenzie_sound_speed


def test_mackenzie_known_value_t10_s35_d1000():
    # Hand-computed from the published Mackenzie (1981) nine-term equation:
    # 1448.96 + 4.591*10 - 5.304e-2*100 + 2.374e-4*1000 + 1.340*0
    #   + 1.630e-2*1000 + 1.675e-7*1e6 - 1.025e-2*10*0 - 7.139e-13*10*1e9
    assert mackenzie_sound_speed(10.0, 35.0, 1000.0) == pytest.approx(1506.263761, abs=1e-6)


def test_mackenzie_known_value_freezing_surface():
    assert mackenzie_sound_speed(0.0, 35.0, 0.0) == pytest.approx(1457.9317392, abs=1e-6)


def test_mackenzie_salinity_above_35_increases_speed():
    base = mackenzie_sound_speed(10.0, 35.0, 100.0)
    saltier = mackenzie_sound_speed(10.0, 36.0, 100.0)
    assert saltier > base


def test_mackenzie_accepts_arrays():
    out = mackenzie_sound_speed(np.array([5.0, 10.0]), np.array([35.0, 35.0]), np.array([0.0, 100.0]))
    assert out.shape == (2,)


def test_mackenzie_clips_out_of_range_temperature_rather_than_raising():
    # Deep near-freezing water occasionally dips just under Mackenzie's
    # stated 2 C floor; must not raise.
    result = mackenzie_sound_speed(-1.0, 35.0, 4000.0)
    clipped = mackenzie_sound_speed(2.0, 35.0, 4000.0)
    assert result == pytest.approx(clipped)


def test_build_profile_sofar_axis_is_a_true_local_minimum():
    profile = build_profile()
    axis_depth, axis_speed = profile.sofar_axis()
    assert 400 <= axis_depth <= 1500  # physically expected range for this water-mass structure
    assert axis_speed == pytest.approx(profile.speed_ms.min())
    assert axis_speed < profile.speed_ms[0]     # slower than the surface
    assert axis_speed < profile.speed_ms[-1]    # slower than the seafloor


def test_speed_at_matches_grid_and_interpolates_between():
    profile = build_profile()
    assert profile.speed_at(profile.depth_m[10]) == pytest.approx(profile.speed_ms[10])
    mid = (profile.depth_m[10] + profile.depth_m[11]) / 2
    interp = profile.speed_at(mid)
    lo, hi = sorted([profile.speed_ms[10], profile.speed_ms[11]])
    assert lo <= interp <= hi


def test_speed_at_clamps_outside_profile_range():
    profile = build_profile()
    assert profile.speed_at(-500) == pytest.approx(profile.speed_ms[0])
    assert profile.speed_at(999999) == pytest.approx(profile.speed_ms[-1])


def test_gradient_negative_near_surface_positive_near_bottom():
    # Temperature-driven deceleration dominates near the surface (negative
    # dc/dz); pressure-driven acceleration dominates near the seafloor
    # (positive dc/dz) -- this asymmetry is what creates the SOFAR axis.
    profile = build_profile()
    assert profile.gradient_at(50) < 0
    assert profile.gradient_at(4500) > 0
