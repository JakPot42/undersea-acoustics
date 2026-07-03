import numpy as np
import pytest

import config
from hydrophone import (
    build_demo_recording,
    generate_synthetic_contact,
    load_bundled_background,
)


def test_bundled_background_loads_and_matches_documented_sample_rate():
    raw, sr = load_bundled_background()
    assert sr == config.OOI_SAMPLE_RATE_HZ
    assert len(raw) > 0
    assert raw.std() > 0   # not silence/all-zero


def test_synthetic_contact_is_deterministic_given_same_seed():
    a = generate_synthetic_contact(5000, 200.0, config.SYNTHETIC_CONTACT)
    b = generate_synthetic_contact(5000, 200.0, config.SYNTHETIC_CONTACT)
    assert np.allclose(a, b)


def test_synthetic_contact_tonals_are_spectrally_visible():
    sr = 200.0
    n = 20000
    contact = generate_synthetic_contact(n, sr, config.SYNTHETIC_CONTACT)
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    mag = np.abs(np.fft.rfft(contact))
    for f_target in config.SYNTHETIC_CONTACT["tonal_frequencies_hz"]:
        i = int(np.argmin(np.abs(freqs - f_target)))
        local_median = np.median(mag[max(0, i - 30):i + 30])
        assert mag[i] > 5 * local_median


def test_build_demo_recording_combines_real_background_and_synthetic_contact():
    rec = build_demo_recording(duration_s=30.0)
    assert rec.contact_present is True
    assert "OOI" in rec.background_source
    assert len(rec.samples) == pytest.approx(30.0 * rec.sample_rate_hz, abs=rec.sample_rate_hz)
    assert rec.contact_params["label"].startswith("SYNTHETIC")


def test_build_demo_recording_trims_to_requested_duration():
    rec = build_demo_recording(duration_s=10.0)
    assert len(rec.samples) == int(10.0 * rec.sample_rate_hz)
