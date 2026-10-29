from click.testing import CliRunner

from main import cli


def _run(*args):
    runner = CliRunner()
    return runner.invoke(cli, list(args))


def test_profile_command_runs_clean():
    result = _run("profile")
    assert result.exit_code == 0
    assert "SOFAR axis" in result.output


def test_raytrace_command_runs_clean(tmp_path):
    out = str(tmp_path / "prop.png")
    result = _run("raytrace", "--out", out)
    assert result.exit_code == 0
    assert "Shadow Zone Summary" in result.output


def test_spectrogram_command_runs_clean(tmp_path):
    out = str(tmp_path / "spec.png")
    result = _run("spectrogram", "--out", out)
    assert result.exit_code == 0
    assert "Tonal Lines" in result.output


def test_demon_command_runs_clean():
    result = _run("demon")
    assert result.exit_code == 0
    assert "SYNTHETIC CONTACT" in result.output


def test_demo_command_runs_full_pipeline_clean():
    result = _run("demo")
    assert result.exit_code == 0
    assert "Acoustic Environment Brief" in result.output
    assert "SYNTHETIC CONTACT DISCLOSURE" in result.output
