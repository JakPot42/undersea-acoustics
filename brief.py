"""brief.py — ASW environment brief synthesis.

Same doctrine as every Claude call site in this portfolio: every number in
the brief (SOFAR axis depth, shadow-zone ranges, detected frequencies) is
computed deterministically upstream by sound_speed.py / ray_tracing.py /
dsp.py. Claude's only job here is to turn those numbers into a one-page,
plain-English environment assessment -- it never invents a number, never
classifies or attributes the detected signal to a real vessel, and always
carries the SYNTHETIC-contact disclosure into the output verbatim.
DEMO_MODE (default) uses a deterministic template with the same numbers
and disclosure -- no API key needed, no wording variance between runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config


@dataclass
class AssessmentInputs:
    region_name: str
    region_citation: str
    source_depth_m: float
    sofar_axis_depth_m: float
    sofar_axis_speed_ms: float
    surface_speed_ms: float
    bottom_speed_ms: float
    shadow_zone_km: dict            # {depth_m: [(start_km, end_km), ...]}
    tonal_detections: list          # list of dsp.Detection
    demon_detections: list          # list of dsp.Detection
    background_source: str
    contact_params: dict = field(default_factory=lambda: dict(config.SYNTHETIC_CONTACT))


def _shadow_zone_lines(inputs: AssessmentInputs) -> list[str]:
    lines = []
    for depth_m, intervals in sorted(inputs.shadow_zone_km.items()):
        if not intervals:
            lines.append(f"  - {depth_m:.0f} m: fully covered by direct/refracted ray paths out to the modeled range.")
            continue
        span = "; ".join(f"{a:.1f}-{b:.1f} km" for a, b in intervals)
        lines.append(f"  - {depth_m:.0f} m: outside direct/refracted coverage at ranges {span} from the source.")
    return lines


def _deterministic_template(inputs: AssessmentInputs) -> str:
    top_tonals = ", ".join(f"{d.freq_hz:.1f} Hz ({d.margin_db:.1f} dB above local noise floor)" for d in inputs.tonal_detections[:6]) or "none above threshold"
    top_demon = inputs.demon_detections[0] if inputs.demon_detections else None
    demon_line = (
        f"strongest envelope-modulation line at {top_demon.freq_hz:.2f} Hz "
        f"({top_demon.margin_db:.1f} dB above local noise floor) -- a candidate blade-rate line"
        if top_demon else "no envelope-modulation line cleared the detection threshold"
    )

    return "\n".join([
        "ASW ENVIRONMENT & ACOUSTIC ASSESSMENT (deterministic template -- DEMO_MODE)",
        "=" * 72,
        config.SCOPE_DISCLAIMER,
        "",
        f"REGION: {inputs.region_name}",
        f"  {inputs.region_citation}",
        "",
        "ACOUSTIC PROPAGATION ENVIRONMENT:",
        f"  Sound speed at surface: {inputs.surface_speed_ms:.1f} m/s. "
        f"SOFAR channel axis at {inputs.sofar_axis_depth_m:.0f} m "
        f"({inputs.sofar_axis_speed_ms:.1f} m/s, the profile minimum). "
        f"Sound speed increases again to {inputs.bottom_speed_ms:.1f} m/s at the seafloor "
        "as pressure dominates over the (now roughly constant) deep-water temperature.",
        f"  A near-surface source at {inputs.source_depth_m:.0f} m depth was ray-traced "
        "across a +/-20 degree launch fan to map direct and refracted coverage.",
        "",
        "SHADOW ZONES (no direct/refracted ray coverage -- reduced detectability by a",
        "near-surface sensor at these depth/range combinations):",
        *_shadow_zone_lines(inputs),
        "",
        "ACOUSTIC DETECTIONS IN THE ANALYZED RECORDING:",
        f"  Background: {inputs.background_source}",
        f"  Tonal lines (LOFAR peak-picking, adaptive noise-floor threshold): {top_tonals}",
        f"  DEMON (envelope modulation on cavitation-band noise): {demon_line}",
        "",
        "*** SYNTHETIC CONTACT DISCLOSURE ***",
        f"  The tonal/DEMON signal detected above is a SYNTHETIC test signal "
        f"({inputs.contact_params['label']}), parameters: shaft rate "
        f"{inputs.contact_params['shaft_rate_hz']} Hz x {inputs.contact_params['blade_count']} blades "
        f"= {inputs.contact_params['shaft_rate_hz'] * inputs.contact_params['blade_count']:.2f} Hz blade rate, "
        f"tonals at {inputs.contact_params['tonal_frequencies_hz']} Hz. "
        "It does NOT represent, and must never be presented as, a real vessel. "
        "The ambient background it is embedded in is real, public, unclassified "
        "OOI hydrophone data; no classified or controlled data was used.",
    ])


_SYSTEM_PROMPT = (
    "You write a one-page, plain-English ASW (anti-submarine warfare) ACOUSTIC "
    "ENVIRONMENT ASSESSMENT from pre-computed, deterministic inputs. You are not "
    "given raw data to analyze -- you are given final numbers (sound speed "
    "profile summary, SOFAR channel axis, shadow-zone ranges, detected tonal and "
    "DEMON frequencies) and your only job is to explain what they mean in clear "
    "prose. Rules, all mandatory: "
    "(1) Never invent, adjust, or round a number differently than given. "
    "(2) Never classify, identify, or attribute the detected signal to any real "
    "vessel, hull, or vessel class -- describe only the frequencies and what "
    "detecting a tonal/DEMON line means technically. "
    "(3) You MUST include, verbatim, the synthetic-contact disclosure the user "
    "gives you -- do not paraphrase or soften it. "
    "(4) Explain shadow zones in plain operational language (why a sensor at the "
    "given source depth cannot see a contact at the given depth/range). "
    "Output plain text, no markdown headers, under 350 words."
)


def _claude_brief(inputs: AssessmentInputs) -> str:
    import anthropic

    template_numbers = _deterministic_template(inputs)
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=700,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": template_numbers}],
    )
    return msg.content[0].text.strip()


def generate_brief(inputs: AssessmentInputs) -> str:
    if config.DEMO_MODE:
        return _deterministic_template(inputs)
    try:
        return _claude_brief(inputs)
    except Exception:
        return _deterministic_template(inputs)
