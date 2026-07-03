import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pdf_report import build_asw_brief_pdf


def test_build_asw_brief_pdf_creates_nonempty_file(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    png_path = tmp_path / "fig.png"
    fig.savefig(png_path)
    plt.close(fig)

    pdf_path = tmp_path / "brief.pdf"
    out = build_asw_brief_pdf(
        str(pdf_path), str(png_path),
        "PARAGRAPH ONE.\n\nPARAGRAPH TWO.\n\n*** SYNTHETIC CONTACT DISCLOSURE ***\ntext",
        "Test Region",
    )
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000
    with open(out, "rb") as f:
        assert f.read(4) == b"%PDF"
