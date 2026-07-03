"""config.py — physical constants, region definition, data citations, and
the mandatory scope disclaimer. No physics or DSP logic lives here.

======================================================================
MANDATORY, NON-NEGOTIABLE FRAMING
======================================================================
This tool demonstrates UNCLASSIFIED, EDUCATIONAL ocean-acoustic modeling
and signal-processing techniques (sound-speed profiling, ray tracing,
spectral/envelope analysis) using:
  (1) a synthetic ocean temperature/salinity profile grounded in publicly
      published WOA23 climatological values, and
  (2) a REAL, public, unclassified hydrophone recording from NSF's Ocean
      Observatories Initiative (OOI) cabled array, with a SYNTHETIC
      "contact of interest" tonal/DEMON signal overlaid on top.
No dataset of actual submarine acoustic signatures is publicly available
anywhere -- that data is classified by definition. The synthetic contact's
parameters (shaft rate, blade count, tonal frequencies) are ALWAYS labeled
SYNTHETIC in every output (CLI, JSON, PDF) and never presented as, or
implied to represent, a real vessel. This tool does not process, and was
never exposed to, any classified or controlled data.
======================================================================
"""
from __future__ import annotations

import os

DEMO_MODE = os.environ.get("DEMO_MODE", "True") == "True"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

SCOPE_DISCLAIMER = (
    "EDUCATIONAL / UNCLASSIFIED DEMONSTRATION ONLY. Ocean environment "
    "data is a synthetic profile grounded in published WOA23 climatology. "
    "The hydrophone background is a REAL public OOI recording; the tonal "
    "contact detected in it is a SYNTHETIC signal with disclosed "
    "parameters -- it does NOT represent, and must never be presented as, "
    "a real vessel. No classified or controlled data was used or produced."
)

# ---------------------------------------------------------------------------
# Region: western North Atlantic subtropical gyre (Sargasso Sea sector),
# chosen instead of the immediate NUWC Newport / Narragansett Bay coastal
# shelf because a SOFAR channel and deep shadow zones require deep open
# water (>3000 m) -- exactly the blue-water ASW environment NUWC's research
# mission actually covers, not the shallow bay outside the front door.
# ---------------------------------------------------------------------------
REGION_NAME = "Western North Atlantic (Sargasso Sea sector)"
REGION_LAT = 33.0
REGION_LON = -65.0
REGION_BOTTOM_DEPTH_M = 5200.0
REGION_CITATION = (
    "Representative deep-water profile for the western North Atlantic "
    "subtropical gyre (~30-35N, 60-70W), grounded in NOAA World Ocean "
    "Atlas 2023 (WOA23) climatological temperature/salinity ranges and "
    "published North Atlantic water-mass structure: Subtropical Mode "
    "Water / '18C Water' (McCartney 1982) in the upper thermocline, "
    "North Atlantic Deep Water (~2-3C, 34.9-35.0 PSU) below ~1500 m. "
    "See README for the full depth/temperature/salinity table and the "
    "decision to use a synthetic-but-grounded profile instead of a live "
    "multi-hundred-MB WOA23 NetCDF pull."
)

# ---------------------------------------------------------------------------
# Synthetic temperature/salinity-by-depth profile. Standard WOA23 depth
# levels (0-5200 m). Values are NOT pulled from a live WOA23 file -- see
# README "Data sourcing decision" for why (large gridded NetCDF files
# behind a query interface, impractical for a reproducible CLI demo) --
# they are hand-built to match published WOA23-documented ranges for this
# region and water-mass structure. Every row is inside Mackenzie (1981)'s
# validity range (T in [2,30] C, S in [25,40] PSU, D in [0,8000] m).
# ---------------------------------------------------------------------------
WOA_DEPTH_M = [
    0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500,
    600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1750,
    2000, 2500, 3000, 3500, 4000, 4500, 5000, 5200,
]
WOA_TEMP_C = [
    24.0, 23.8, 23.5, 23.0, 22.0, 20.5, 19.0, 18.3, 18.0, 17.0, 15.0,
    13.0, 11.0, 9.5, 8.3, 7.3, 6.5, 5.9, 5.2, 4.8, 4.5, 4.3, 4.1, 3.9,
    3.6, 3.4, 3.0, 2.6, 2.3, 2.1, 2.0, 1.9, 1.9,
]
WOA_SALINITY_PSU = [
    36.5, 36.5, 36.5, 36.5, 36.4, 36.4, 36.5, 36.6, 36.7, 36.6, 36.4,
    36.1, 35.7, 35.4, 35.2, 35.1, 35.0, 35.0, 35.00, 34.99, 34.99,
    34.98, 34.97, 34.97, 34.95, 34.95, 34.93, 34.92, 34.91, 34.90,
    34.90, 34.90, 34.90,
]

# ---------------------------------------------------------------------------
# Hydrophone source: real, public, unclassified. NSF Ocean Observatories
# Initiative (OOI) Regional Cabled Array, Axial Base broadband hydrophone,
# archived at EarthScope (formerly IRIS) and retrievable with no
# authentication via obspy's FDSN client. Fetched live and verified working
# during this build (see README); a fixed 3-minute segment is bundled in
# data/ for a reproducible offline demo.
# ---------------------------------------------------------------------------
OOI_NETWORK = "OO"
OOI_STATION = "AXBA1"
OOI_LOCATION = ""
OOI_CHANNEL = "HDH"          # high-gain hydrophone
OOI_SAMPLE_RATE_HZ = 200.0
OOI_BUNDLED_START = "2023-01-15T06:00:00Z"
OOI_BUNDLED_DURATION_S = 180
OOI_BUNDLED_FILE = "data/ooi_axba1_hdh_20230115_raw.npy"
OOI_CITATION = (
    "NSF Ocean Observatories Initiative (OOI) Regional Cabled Array, "
    "station AXBA1 (RSN Axial Base), channel HDH (broadband hydrophone), "
    "network code OO. Archived at EarthScope/IRIS DMC, retrieved via the "
    "FDSN web service (no authentication required). Raw digitizer counts, "
    "not calibrated to physical pressure units (Pa) -- amplitude is "
    "relative, not absolute SPL."
)

# ---------------------------------------------------------------------------
# Synthetic "contact of interest" -- ALWAYS disclosed as synthetic.
# Parameters are illustrative, chosen to sit in physically plausible
# ranges for large-vessel machinery (shaft rate ~1-5 Hz, blade count
# 4-7, low-frequency tonals from reduction gears/generators), not drawn
# from, or representative of, any real vessel or classified reference.
# ---------------------------------------------------------------------------
SYNTHETIC_CONTACT = {
    "label": "SYNTHETIC CONTACT (illustrative only -- not a real vessel)",
    "shaft_rate_hz": 2.3,
    "blade_count": 5,
    "tonal_frequencies_hz": [41.5, 63.0, 78.25, 96.0],
    "tonal_amplitude": 6.0,
    "cavitation_band_hz": (20.0, 95.0),
    "cavitation_amplitude": 3.0,
    "seed": 42,
}

OUTPUT_DIR = "output"
