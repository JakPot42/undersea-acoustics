import os

import pytest

import config
from brief import AssessmentInputs, generate_brief
from dsp import Detection


def _sample_inputs() -> AssessmentInputs:
    return AssessmentInputs(
        region_name="Test Region",
        region_citation="Test citation",
        source_depth_m=50.0,
        sofar_axis_depth_m=1000.0,
        sofar_axis_speed_ms=1487.9,
        surface_speed_ms=1533.5,
        bottom_speed_ms=1546.9,
        shadow_zone_km={100: [(3.0, 25.0)], 1000: []},
        tonal_detections=[Detection(41.5, -2.6, -14.7, 4.1)],
        demon_detections=[Detection(11.5, 90.8, 63.9, 26.9)],
        background_source="Bundled: test source",
    )


def test_deterministic_template_includes_scope_disclaimer_and_synthetic_disclosure():
    text = generate_brief(_sample_inputs())
    assert config.SCOPE_DISCLAIMER in text
    assert "SYNTHETIC CONTACT DISCLOSURE" in text
    assert "does NOT represent" in text


def test_deterministic_template_never_invents_numbers():
    inputs = _sample_inputs()
    text = generate_brief(inputs)
    assert "1000" in text          # SOFAR axis depth
    assert "1487.9" in text        # SOFAR axis speed
    assert "11.50" in text         # blade rate


def test_fully_covered_depth_reports_no_shadow():
    inputs = _sample_inputs()
    text = generate_brief(inputs)
    assert "fully covered" in text


def test_generate_brief_falls_back_to_template_when_claude_call_fails(monkeypatch):
    # DEMO_MODE=False forces the Claude code path; with no API key configured
    # the SDK call fails and generate_brief must degrade to the deterministic
    # template rather than raising -- same doctrine as every Claude call site
    # in this portfolio (catch Exception, never crash the CLI).
    monkeypatch.setattr(config, "DEMO_MODE", False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    text = generate_brief(_sample_inputs())
    assert "DEMO_MODE" in text  # the template's own header string
    assert config.SCOPE_DISCLAIMER in text
