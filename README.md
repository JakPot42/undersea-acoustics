# undersea-acoustics: sound propagation and passive acoustic signal processing (educational demo)

Educational demo on synthetic ocean data and a public hydrophone recording.
Not affiliated with the US Navy or NUWC.

## Scope — EDUCATIONAL / UNCLASSIFIED DEMONSTRATION ONLY

Printed on every CLI command (`config.SCOPE_DISCLAIMER`):

- Ocean environment data is a **synthetic** temperature/salinity profile,
  grounded in published WOA23 climatological values for the western North
  Atlantic — not a live pull of the actual gridded dataset (see "Data
  sourcing decisions" below).
- The hydrophone background is a **real, public, unclassified** recording
  (NSF Ocean Observatories Initiative). The tonal/DEMON contact detected
  in it is a **disclosed synthetic test signal** with published parameters
  (shaft rate, blade count, tonal frequencies) — it does not represent,
  and must never be presented as, a real vessel.
- **No dataset of actual submarine acoustic signatures is publicly
  available anywhere** — that data is classified by definition. This tool
  was never exposed to, and does not process, any classified or
  controlled data.

---

## What it is

The two halves of an underwater acoustics pipeline, talking to each other:

1. **Ocean environment model**: Mackenzie (1981) sound-speed formula → a
   2D range-independent Snell's-law ray tracer → a range/depth map of
   where sound does and doesn't reach (the acoustic shadow zone).
2. **Passive sonar signal processing**: STFT spectrogram → LOFAR-style
   tonal peak-picking → DEMON envelope-modulation analysis on a real
   hydrophone recording with a synthetic contact overlaid.

The demo overlays the DEMON/tonal detections onto the propagation model
and has Claude (or a deterministic template in `DEMO_MODE`) write a
one-page acoustic environment brief from the already-computed numbers — Claude
never computes a physics or DSP result itself, same "extracts/narrates,
never decides" doctrine as every other project in this portfolio.

```
config.py (WOA23-grounded profile, region, citations, synthetic-contact params)
  |
  +-- sound_speed.py  --> Mackenzie (1981) formula, SOFAR axis
  |         |
  |         v
  +-- ray_tracing.py  --> Snell's law ray fan, shadow-zone grid
  |
  +-- hydrophone.py   --> real OOI background + synthetic contact overlay
  |         |
  |         v
  +-- dsp.py          --> STFT spectrogram, DEMON, CA-CFAR-style peak-picking
  |
  +-- plots.py / pdf_report.py --> two-panel figure + PDF
  |
  +-- brief.py         --> Claude synthesis / deterministic template
  |
  +-- pipeline.py / main.py --> CLI orchestration
```

---

## The physics, verified before writing code

**Mackenzie (1981) sound speed formula** — the nine-term equation was
checked against the published JASA paper's coefficients before use
(`sound_speed.py`), not reproduced from memory alone:

```
c = 1448.96 + 4.591T - 5.304e-2 T^2 + 2.374e-4 T^3 + 1.340(S-35)
  + 1.630e-2 D + 1.675e-7 D^2 - 1.025e-2 T(S-35) - 7.139e-13 T D^3
```
T = temperature (C), S = salinity (PSU), D = depth (m). Valid for
T in [2,30], S in [25,40], D in [0,8000].

**SOFAR channel**: temperature-driven deceleration dominates in the upper
ocean; pressure-driven acceleration dominates below the main thermocline.
The crossover produces a sound-speed *minimum* at depth — the built
profile puts it at **1000 m, 1487.9 m/s** for this region, well inside
the textbook 600-1300 m mid-latitude range. Sound near that axis keeps
refracting back toward it instead of hitting the lossy surface/seafloor,
so transmission loss becomes cylindrical instead of spherical — this is
the mechanism behind SOSUS-scale long-range detection.

**Ray tracing** (`ray_tracing.py`) integrates the standard
range-independent ray equations (Jensen, Kuperman, Porter & Schmidt,
*Computational Ocean Acoustics*) in `(r, z, eta)` with the ray invariant
`xi = cos(theta)/c(z)` held fixed per ray — this formulation makes Snell's
law an automatic conservation law and lets rays pass smoothly through
turning points with no branch/arccos handling. RK4 integration, ds=15m,
vectorized across the whole ray fan (see "Performance" below).

**Shadow zones**: with a near-surface source above a subsurface sound
channel (the setup used here — source at 50 m, channel axis at 1000 m),
rays launched at even modest downward angles get bent deep into the
channel and don't return to shallow depth for tens of km, leaving a
genuine void in ray coverage at intermediate range/depth — the classic
"double wedge" shadow-zone pattern visible in `output/propagation_and_spectrogram.png`.
This is a **ray-density coverage map, not a transmission-loss
calculation** — no absolute dB loss is computed, and surface/bottom
reflections are modeled as simple lossless specular bounces (a real model
like BELLHOP would attenuate bottom-interacting rays).

**DEMON analysis** (`dsp.py`): bandpass to the broadband cavitation-noise
band → Hilbert-transform envelope → FFT of the envelope, which lives at
the much lower blade-rate frequency. **Tonal peak-picking**: local maxima
of the time-averaged spectrum that clear a cell-averaging adaptive
noise-floor threshold — the same idea as the CA-CFAR detector (`dsp-lib`,
Rohling 1983), applied per-frequency-bin instead of per-range-bin.

---

## Data sourcing decisions (made explicitly, not defaulted into)

**WOA23 temperature/salinity**: the real NOAA World Ocean Atlas 2023
NetCDF grids exist and need no authentication, but they're large
(multi-hundred-MB global files) behind a query interface, not a clean
single-call API — impractical to pull on every CLI run. `config.py`
instead ships a **synthetic profile grounded in published WOA23-documented
ranges** for the western North Atlantic subtropical gyre (Subtropical
Mode Water / "18°C Water" in the upper thermocline, North Atlantic Deep
Water ~2-3°C/34.9-35.0 PSU below ~1500 m) — same `DEMO_MODE`-by-default
discipline as every other project in this portfolio. The region chosen is
deep open ocean (~30-35°N, 60-70°W, 5200 m bottom depth) rather than the
shallow coastal water of Narragansett Bay, because a SOFAR channel and
real shadow zones require deep water.

**Hydrophone recording**: no unclassified public dataset of real
submarine acoustic signatures exists anywhere. What *is* real and public:
NSF's Ocean Observatories Initiative Regional Cabled Array runs broadband
hydrophones at Axial Seamount, archived at EarthScope (formerly IRIS DMC)
and retrievable with **no authentication** via `obspy`'s FDSN client —
confirmed live during this build (network `OO`, station `AXBA1`, channel
`HDH`, 200 Hz). A real 3-minute segment (2023-01-15, no gaps) is bundled
in `data/ooi_axba1_hdh_20230115_raw.npy` for a reproducible offline demo;
`fetch-hydrophone` pulls a fresh segment live. The disclosed synthetic
contact (`config.SYNTHETIC_CONTACT`) is added on top — parameters chosen
to be physically plausible for large-vessel machinery (shaft rate 2.3 Hz,
5 blades → 11.5 Hz blade rate; tonals at 41.5/63.0/78.25/96.0 Hz), not
drawn from or representative of any real vessel.

---

## Verification, not just "it runs"

Every physics/DSP claim above was checked numerically against known
ground truth before trusting the pipeline, not just eyeballed:

- **Mackenzie formula**: two hand-computed test cases in
  `tests/test_sound_speed.py` match to 1e-6.
- **SOFAR axis**: lands at 1000 m / 1487.9 m/s — inside the textbook
  600-1300 m range, and numerically confirmed as the true minimum of the
  built profile (not assumed at a fixed depth).
- **Ray tracer**: the vectorized fan integrator (`trace_fan`) is checked
  bit-exact against the scalar single-ray integrator (`trace_ray`) for
  the same launch angle before being trusted for the real fan trace.
- **Shadow zone**: the near-surface-source-over-a-channel setup produces
  the textbook double-wedge shadow zone (see the propagation plot) — a
  physics regression guard in `tests/test_ray_tracing.py` fails loudly if
  this stops appearing.
- **Tonal peak-picker**: recovers all 4 known synthetic tonal frequencies
  (41.5/63.0/78.25/96.0 Hz) within one FFT bin, each 4+ dB above the
  adaptive noise floor, on top of real ocean background noise.
- **DEMON**: recovers the exact known synthetic blade rate (11.50 Hz)
  at **27 dB** above the noise floor — the strongest detection by a wide
  margin, as it should be for a deliberately-injected signal.

---

## Performance note

The first ray-fan implementation looped `trace_ray` once per launch angle
(161 rays) — 78 seconds, dominated by millions of scalar `np.interp`
calls. Rewritten to integrate the whole fan as arrays (one RK4 step
advances every ray at once): **0.87 seconds**, verified bit-exact against
the original scalar integrator first. See `ray_tracing.trace_fan`.

---

## CLI

```
python main.py profile                    # sound speed table + SOFAR axis
python main.py raytrace [--source-depth D] [--out PNG]
python main.py spectrogram [--out PNG] [--live]
python main.py demon [--live]
python main.py brief [--live]              # full pipeline -> PDF
python main.py demo                        # alias for brief, bundled data, no key/network needed
python main.py fetch-hydrophone --start ISO8601 --duration SECONDS --out FILE.npy
```

`DEMO_MODE=True` (default): brief uses a deterministic template, no
`ANTHROPIC_API_KEY` needed. Set `DEMO_MODE=False` with a real key for a
live Claude-written brief (falls back to the template on any API error,
same exception-handling doctrine as every Claude call site in this
portfolio: catch `Exception`, not `anthropic.APIError`).

---

## Honest limitations

- **Not a transmission-loss model.** The ray tracer produces a ray-density
  coverage map (where rays do/don't geometrically reach), not calibrated
  dB propagation loss. No absorption, no bottom-interaction attenuation,
  no frequency-dependent effects.
- **Synthetic ocean profile**, not a live WOA23 pull — grounded in
  published climatology but not the literal gridded dataset for an exact
  lat/lon/date.
- **Uncalibrated hydrophone amplitude** — raw OOI digitizer counts, not
  converted to physical pressure (Pa)/SPL.
- **No real submarine signature data exists publicly, ever** — the
  synthetic contact is illustrative, not validated against any real
  acoustic reference.
- **2D only** — no range-dependent bathymetry or 3D propagation effects.

---

## Tech stack

Python, NumPy, SciPy (`stft`, `hilbert`, `butter`/`sosfiltfilt`,
`find_peaks`, `binary_dilation`), Matplotlib, obspy (FDSN client),
Click, Rich, ReportLab, Anthropic Claude (Haiku). 41 tests, all passing.
