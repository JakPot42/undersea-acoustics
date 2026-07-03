"""hydrophone.py — hydrophone recording assembly.

Two pieces, kept strictly separate and always labeled:

1. REAL background: an actual unclassified hydrophone recording from
   NSF's Ocean Observatories Initiative (network OO, station AXBA1,
   channel HDH -- see config.OOI_*), fetched live via obspy's FDSN client
   or loaded from the bundled cached segment in data/. This is real
   ambient ocean noise (surf, currents, marine life, distant shipping) --
   not a submarine, not classified, not calibrated to physical units.

2. SYNTHETIC contact: a fabricated tonal + DEMON-modulated cavitation
   signal built from config.SYNTHETIC_CONTACT, added on top of the real
   background. Every function and output that touches this signal labels
   it synthetic -- see config.py's module docstring for why real
   submarine acoustic data cannot exist in a public dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

import config


@dataclass
class Recording:
    samples: np.ndarray       # combined signal, normalized (background: zero-mean/unit-std)
    sample_rate_hz: float
    background_source: str    # human-readable provenance string
    contact_present: bool
    contact_params: dict | None   # config.SYNTHETIC_CONTACT if contact_present, else None


def load_bundled_background() -> tuple[np.ndarray, float]:
    """Loads the real, cached OOI hydrophone segment bundled in data/."""
    raw = np.load(config.OOI_BUNDLED_FILE).astype(np.float64)
    return raw, config.OOI_SAMPLE_RATE_HZ


def fetch_live_background(start_iso: str, duration_s: float) -> tuple[np.ndarray, float]:
    """Pulls a fresh real segment from EarthScope/IRIS via obspy's FDSN
    client -- same station/channel as the bundled data, different time
    window. Network access required; not used by the default demo path.
    Raises on any client/network failure -- callers fall back to the
    bundled segment, same doctrine as every live-data path elsewhere in
    this portfolio (catch broadly, degrade to bundled/demo data, never
    crash the CLI over a network hiccup)."""
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client

    client = Client("EARTHSCOPE")
    t0 = UTCDateTime(start_iso)
    stream = client.get_waveforms(
        config.OOI_NETWORK, config.OOI_STATION, config.OOI_LOCATION,
        config.OOI_CHANNEL, t0, t0 + duration_s,
    )
    stream.merge(fill_value="interpolate")
    trace = stream[0]
    return trace.data.astype(np.float64), float(trace.stats.sampling_rate)


def _normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


def _bandlimited_noise(n: int, sample_rate_hz: float, band_hz: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(size=n)
    nyq = sample_rate_hz / 2.0
    low, high = band_hz[0] / nyq, min(band_hz[1] / nyq, 0.999)
    sos = butter(4, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, noise)
    return filtered / (filtered.std() + 1e-12)


def generate_synthetic_contact(n_samples: int, sample_rate_hz: float, params: dict = None) -> np.ndarray:
    """Builds the synthetic 'contact of interest': narrowband tonal lines
    (stable machinery tones, config.SYNTHETIC_CONTACT['tonal_frequencies_hz'])
    plus broadband cavitation-band noise amplitude-modulated at blade rate
    (shaft_rate_hz * blade_count) -- the physical mechanism DEMON analysis
    is built to recover. All amplitudes are expressed in units of the
    (unit-variance) background noise this gets added to, so the SNR is
    directly the amplitude value."""
    params = params or config.SYNTHETIC_CONTACT
    rng = np.random.default_rng(params["seed"])
    t = np.arange(n_samples) / sample_rate_hz

    tonals = np.zeros(n_samples)
    for f in params["tonal_frequencies_hz"]:
        phase = rng.uniform(0, 2 * np.pi)
        tonals += np.sin(2 * np.pi * f * t + phase)
    if params["tonal_frequencies_hz"]:
        tonals = tonals / len(params["tonal_frequencies_hz"]) * params["tonal_amplitude"]

    blade_rate_hz = params["shaft_rate_hz"] * params["blade_count"]
    cavitation = _bandlimited_noise(n_samples, sample_rate_hz, params["cavitation_band_hz"], rng)
    modulation_depth = 0.7
    envelope = 1.0 + modulation_depth * np.cos(2 * np.pi * blade_rate_hz * t)
    cavitation = cavitation * envelope * params["cavitation_amplitude"]

    return tonals + cavitation


def build_demo_recording(duration_s: float | None = None, use_live: bool = False) -> Recording:
    """Assembles the full demo recording: real OOI background (bundled by
    default, or freshly fetched if use_live=True and the network
    cooperates) + the synthetic contact overlay. This is the single
    function the CLI/DSP pipeline calls."""
    if use_live:
        try:
            raw, sr = fetch_live_background(config.OOI_BUNDLED_START, duration_s or config.OOI_BUNDLED_DURATION_S)
            source = f"LIVE fetch: {config.OOI_CITATION}"
        except Exception:
            raw, sr = load_bundled_background()
            source = f"[live fetch failed, fell back to bundled] {config.OOI_CITATION}"
    else:
        raw, sr = load_bundled_background()
        source = f"Bundled: {config.OOI_CITATION}"

    background = _normalize(raw)
    if duration_s is not None:
        n = min(len(background), int(duration_s * sr))
        background = background[:n]

    contact = generate_synthetic_contact(len(background), sr, config.SYNTHETIC_CONTACT)
    combined = background + contact

    return Recording(
        samples=combined,
        sample_rate_hz=sr,
        background_source=source,
        contact_present=True,
        contact_params=dict(config.SYNTHETIC_CONTACT),
    )
