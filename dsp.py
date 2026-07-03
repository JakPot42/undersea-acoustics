"""dsp.py — FFT/STFT spectrogram, DEMON envelope analysis, and tonal
peak-picking.

Peak-picking (both the LOFAR tonal-line picker and the DEMON blade-rate
picker) uses a cell-averaging adaptive noise-floor threshold: for each
frequency bin, estimate the local noise floor from a ring of "training"
cells around it (skipping a "guard" band immediately adjacent so the
peak's own skirt doesn't pollute the floor estimate), then flag bins that
are a local maximum AND clear the floor by a fixed dB margin. This is the
same cell-averaging idea as CA-CFAR (P59's dsp-lib, Rohling 1983) applied
per-frequency-bin instead of per-range-bin -- conceptually the same
detection discipline, different axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, stft


@dataclass
class Spectrogram:
    freqs_hz: np.ndarray
    times_s: np.ndarray
    magnitude_db: np.ndarray   # shape (len(freqs_hz), len(times_s))


def compute_spectrogram(samples: np.ndarray, sample_rate_hz: float, nperseg: int = 512, noverlap: int | None = None) -> Spectrogram:
    noverlap = noverlap if noverlap is not None else nperseg // 2
    freqs, times, Zxx = stft(samples, fs=sample_rate_hz, nperseg=nperseg, noverlap=noverlap)
    magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    return Spectrogram(freqs_hz=freqs, times_s=times, magnitude_db=magnitude_db)


@dataclass
class Detection:
    freq_hz: float
    magnitude_db: float
    threshold_db: float
    margin_db: float


def _rolling_noise_floor_db(magnitude_db: np.ndarray, guard_bins: int, training_bins: int) -> np.ndarray:
    """Cell-averaging (median) noise-floor estimate per bin -- see module
    docstring. A plain Python loop over bins; the arrays this is called on
    (spectrogram-averaged spectrum, DEMON spectrum) are at most a few
    thousand bins, so this is not a performance concern here."""
    n = len(magnitude_db)
    floor = np.empty(n)
    for i in range(n):
        lo = magnitude_db[max(0, i - guard_bins - training_bins):max(0, i - guard_bins)]
        hi = magnitude_db[min(n, i + guard_bins + 1):min(n, i + guard_bins + 1 + training_bins)]
        cells = np.concatenate([lo, hi])
        floor[i] = np.median(cells) if len(cells) else magnitude_db[i]
    return floor


def peak_pick(
    freqs_hz: np.ndarray,
    magnitude_db: np.ndarray,
    freq_range_hz: tuple[float, float] | None = None,
    guard_bins: int = 3,
    training_bins: int = 20,
    threshold_db: float = 8.0,
) -> list[Detection]:
    """Adaptive-threshold local-maximum peak picker -- see module docstring
    for the CA-CFAR-style method. Returns detections sorted by descending
    margin above the noise floor (strongest, most confident lines first)."""
    if freq_range_hz is not None:
        mask = (freqs_hz >= freq_range_hz[0]) & (freqs_hz <= freq_range_hz[1])
        freqs_hz = freqs_hz[mask]
        magnitude_db = magnitude_db[mask]

    if len(magnitude_db) < (2 * guard_bins + 2):
        return []

    floor = _rolling_noise_floor_db(magnitude_db, guard_bins, training_bins)
    threshold = floor + threshold_db
    peak_idx, _ = find_peaks(magnitude_db, height=threshold)

    detections = [
        Detection(
            freq_hz=float(freqs_hz[i]),
            magnitude_db=float(magnitude_db[i]),
            threshold_db=float(threshold[i]),
            margin_db=float(magnitude_db[i] - threshold[i]),
        )
        for i in peak_idx
    ]
    return sorted(detections, key=lambda d: d.margin_db, reverse=True)


def tonal_lines(spectrogram: Spectrogram, freq_range_hz: tuple[float, float] = (1.0, 100.0), threshold_db: float = 8.0) -> list[Detection]:
    """Peak-picks tonal lines from the TIME-AVERAGED spectrum (persistent
    tones stay elevated when averaged across the whole recording; noise
    and transients wash out) -- this is what makes a tonal a tonal."""
    mean_spectrum_db = 20 * np.log10(np.mean(10 ** (spectrogram.magnitude_db / 20), axis=1) + 1e-12)
    return peak_pick(spectrogram.freqs_hz, mean_spectrum_db, freq_range_hz=freq_range_hz, threshold_db=threshold_db)


@dataclass
class DemonResult:
    demon_freqs_hz: np.ndarray
    demon_magnitude_db: np.ndarray
    detections: list[Detection]
    analysis_band_hz: tuple[float, float]
    max_demon_freq_hz: float


def compute_demon(
    samples: np.ndarray,
    sample_rate_hz: float,
    analysis_band_hz: tuple[float, float] = (10.0, 95.0),
    max_demon_freq_hz: float = 20.0,
    threshold_db: float = 8.0,
) -> DemonResult:
    """DEMON pipeline: bandpass to the broadband cavitation-noise band ->
    envelope detector (Hilbert-transform magnitude, the analytic-signal
    equivalent of rectify+lowpass) -> FFT of the envelope, which lives at
    the much lower blade-rate/shaft-rate frequencies (a few Hz) -> peak
    pick for candidate blade-rate lines."""
    nyq = sample_rate_hz / 2.0
    low, high = analysis_band_hz[0] / nyq, min(analysis_band_hz[1] / nyq, 0.999)
    sos = butter(4, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, samples)

    envelope = np.abs(hilbert(filtered))
    envelope = envelope - envelope.mean()

    n = len(envelope)
    demon_freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    demon_mag = np.abs(np.fft.rfft(envelope))
    demon_mag_db = 20 * np.log10(demon_mag + 1e-12)

    keep = demon_freqs <= max_demon_freq_hz
    demon_freqs = demon_freqs[keep]
    demon_mag_db = demon_mag_db[keep]

    detections = peak_pick(demon_freqs, demon_mag_db, freq_range_hz=(0.5, max_demon_freq_hz), threshold_db=threshold_db)

    return DemonResult(
        demon_freqs_hz=demon_freqs,
        demon_magnitude_db=demon_mag_db,
        detections=detections,
        analysis_band_hz=analysis_band_hz,
        max_demon_freq_hz=max_demon_freq_hz,
    )
