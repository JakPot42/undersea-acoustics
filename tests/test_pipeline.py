import os

import pipeline


def test_run_propagation_returns_shadow_summary_for_all_target_depths():
    result = pipeline.run_propagation(target_depths_m=[100, 1000])
    assert set(result.shadow_km.keys()) == {100, 1000}
    assert len(result.rays) == len(pipeline.DEFAULT_LAUNCH_ANGLES_DEG)


def test_run_signal_analysis_returns_structure():
    result = pipeline.run_signal_analysis(duration_s=20.0)
    assert result.recording.contact_present
    assert result.spectrogram.magnitude_db.shape[0] == len(result.spectrogram.freqs_hz)
    assert isinstance(result.tonal_detections, list)
    assert isinstance(result.demon.detections, list)


def test_run_full_pipeline_produces_all_outputs(tmp_path):
    out_dir = str(tmp_path / "output")
    prop, sig, brief_text, figure_path, pdf_path = pipeline.run_full_pipeline(output_dir=out_dir)
    assert os.path.exists(figure_path)
    assert os.path.exists(pdf_path)
    assert "SYNTHETIC CONTACT DISCLOSURE" in brief_text
    assert len(prop.rays) > 0
    assert sig.recording.contact_present
