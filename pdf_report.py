"""pdf_report.py — assembles the two-panel figure + Claude/template brief
into the "Acoustic Environment Brief" PDF (ReportLab)."""
from __future__ import annotations

import datetime as _dt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer

import config


def _banner(text: str, styles) -> Paragraph:
    style = ParagraphStyle(
        "Banner", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8,
        textColor=colors.white, alignment=TA_CENTER, backColor=colors.HexColor("#7a1f1f"),
        borderPadding=(4, 4, 4, 4), leading=11,
    )
    return Paragraph(text, style)


def build_asw_brief_pdf(
    output_path: str,
    figure_path: str,
    brief_text: str,
    region_name: str,
) -> str:
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("BriefTitle", parent=styles["Title"], fontSize=16, alignment=TA_CENTER)
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor("#475569"))
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, alignment=TA_LEFT, leading=13, fontName="Courier")

    story = [
        _banner(config.SCOPE_DISCLAIMER, styles),
        Spacer(1, 10),
        Paragraph("Undersea Acoustics (educational demo)", title_style),
        Paragraph("Acoustic Environment Assessment", styles["Heading2"]),
        Paragraph(
            f"Region: {region_name} &nbsp;|&nbsp; Generated: {_dt.datetime.now():%Y-%m-%d %H:%M} "
            f"&nbsp;|&nbsp; DEMO_MODE={config.DEMO_MODE}",
            meta_style,
        ),
        Spacer(1, 10),
        HRFlowable(width="100%", color=colors.HexColor("#1e293b")),
        Spacer(1, 10),
        Image(figure_path, width=7.3 * inch, height=7.3 * inch * (5.5 / 13.0)),
        Spacer(1, 12),
    ]

    for para in brief_text.split("\n\n"):
        safe = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe.replace("\n", "<br/>"), body_style))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 10))
    story.append(_banner(config.SCOPE_DISCLAIMER, styles))

    doc.build(story)
    return output_path
