from typer.testing import CliRunner

from ml_baseline.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "train" in r.stdout
    assert "predict" in r.stdout
