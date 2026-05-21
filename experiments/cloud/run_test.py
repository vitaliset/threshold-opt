import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "run.sh"


@pytest.mark.skipif(os.name == "nt", reason="run.sh requires bash/Linux")
def test_run_rejects_unknown_study(tmp_path):
    """Unknown study name should exit with error."""
    result = subprocess.run(
        ["bash", str(SCRIPT), "fakestudy", "1"],
        cwd=str(tmp_path),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "unknown study" in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="run.sh requires bash/Linux")
def test_run_requires_study_arg(tmp_path):
    """Missing study arg should exit with error."""
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=str(tmp_path),
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0


@pytest.mark.skipif(os.name == "nt", reason="run.sh requires bash/Linux")
def test_run_escapes_parens_in_args(tmp_path):
    """Args with parentheses should be shell-escaped for tmux."""
    # We can't actually launch tmux in CI, but we can verify the script
    # builds the CMD correctly by checking it doesn't crash on parens.
    # The script will fail because uv/tmux aren't set up, but it should
    # get past the argument parsing stage without a syntax error.
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "binary",
            "1",
            "--models",
            "LGBMClassifier(),TabM",
        ],
        cwd=str(tmp_path),
        text=True,
        capture_output=True,
    )
    # Should NOT get "Syntax error: ( unexpected"
    assert "Syntax error" not in result.stderr
    assert "unexpected" not in result.stderr
