import numpy as np
import pytest

import config
from dsp import compute_demon, compute_spectrogram, peak_pick, tonal_lines
from hydrophone import build_demo_recording


def test_peak_pick_finds_an_injected_sinusoid_above_noise():
    sr = 200.0
    n = 20000
    t = np.arange(n) / sr
    rng = np.random.default_rng(1)
    signal = rng.normal(size=n) + 8.0 * np.sin(2 * np.pi * 30.0 * t)
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    mag_db = 20 * np.log10(np.abs(np.fft.rfft(signal)) + 1e-12)
    detections = peak_pick(freqs, mag_db, freq_range_hz=(1, 99), threshold_db=8.0)
    assert any(abs(d.freq_hz - 30.0) < 0.5 for d in detections)


def test_peak_pick_empty_input_does_not_crash():
    assert peak_pick(np.array([]), np.array([]), threshold_db=8.0) == []


def test_compute_spectrogram_shape():
    sr = 200.0
    samples = np.random.default_rng(0).normal(size=4000)
    spec = compute_spectrogram(samples, sr, nperseg=256)
    assert spec.magnitude_db.shape[0] == len(spec.freqs_hz)
    assert spec.magnitude_db.shape[1] == len(spec.times_s)
    assert spec.freqs_hz.max() <= sr / 2 + 1e-6


def test_tonal_lines_recovers_synthetic_contact_frequencies():
    rec = build_demo_recording(duration_s=60.0)
    spec = compute_spectrogram(rec.samples, rec.sample_rate_hz)
    detections = tonal_lines(spec, freq_range_hz=(1, 100), threshold_db=6.0)
    found = {round(d.freq_hz) for d in detections}
    for f_true in config.SYNTHETIC_CONTACT["tonal_frequencies_hz"]:
        assert any(abs(f_found - f_true) < 1.0 for f_found in found)


def test_compute_demon_recovers_synthetic_blade_rate():
    rec = build_demo_recording(duration_s=90.0)
    result = compute_demon(rec.samples, rec.sample_rate_hz)
    true_blade_rate = config.SYNTHETIC_CONTACT["shaft_rate_hz"] * config.SYNTHETIC_CONTACT["blade_count"]
    assert len(result.detections) > 0
    top = result.detections[0]
    assert abs(top.freq_hz - true_blade_rate) < 0.3
    assert top.margin_db > 10.0   # should be a very confident detection, not a marginal one
