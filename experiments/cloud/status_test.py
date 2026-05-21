import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "status.sh"


@pytest.fixture
def fake_run(tmp_path):
    """Create a fake run directory with a known timestamp and parquets."""
    results = tmp_path / "results"
    run_dir = results / "run_results_binary_20260416_030935"
    run_dir.mkdir(parents=True)

    # Fake parquets
    (run_dir / "sonar_seed0.parquet").write_bytes(b"x")
    (run_dir / "ionosphere_seed0.parquet").write_bytes(b"x")

    # Fake log
    log = tmp_path / "run-binary.log"
    log.write_text("Running 6 total tasks (6 datasets x 1 seeds).\n")

    return tmp_path, run_dir


@pytest.mark.skipif(os.name == "nt", reason="status.sh requires bash/Linux")
def test_status_parses_timestamp_from_dirname(fake_run):
    """Elapsed time should be based on the dir name, not stat mtime."""
    tmp_path, run_dir = fake_run

    # Touch the dir to set mtime to NOW (which would give ~0s elapsed
    # if the script used stat instead of the dir-name timestamp).
    run_dir.touch()

    output = subprocess.check_output(
        ["bash", str(SCRIPT)],
        cwd=str(tmp_path),
        env={**os.environ, "RESULTS_DIR": str(tmp_path / "results")},
        text=True,
        stderr=subprocess.STDOUT,
    )

    assert "done:    2 / 6 tasks" in output
    # Elapsed should be large (since 20260416_030935 is in the past
    # relative to now), NOT ~0s as stat would give.
    assert "elapsed:" in output
    # Should NOT show 00h00m (which would mean it used stat mtime)
    elapsed_line = next(
        l for l in output.splitlines() if l.startswith("elapsed:")
    )
    assert "00h00m00s" not in elapsed_line


@pytest.mark.skipif(os.name == "nt", reason="status.sh requires bash/Linux")
def test_status_shows_tmux_line(fake_run):
    """Output should always include a tmux status line."""
    tmp_path, _ = fake_run

    output = subprocess.check_output(
        ["bash", str(SCRIPT)],
        cwd=str(tmp_path),
        env={**os.environ, "RESULTS_DIR": str(tmp_path / "results")},
        text=True,
        stderr=subprocess.STDOUT,
    )

    assert "tmux:" in output


@pytest.mark.skipif(os.name == "nt", reason="status.sh requires bash/Linux")
def test_status_rate_and_eta(fake_run):
    """With 2/6 tasks done and elapsed > 0, should show rate and eta."""
    tmp_path, _run_dir = fake_run

    output = subprocess.check_output(
        ["bash", str(SCRIPT)],
        cwd=str(tmp_path),
        env={**os.environ, "RESULTS_DIR": str(tmp_path / "results")},
        text=True,
        stderr=subprocess.STDOUT,
    )

    assert "rate:" in output
    assert "eta:" in output


@pytest.mark.skipif(os.name == "nt", reason="status.sh requires bash/Linux")
def test_status_no_run_dirs(tmp_path):
    """Should exit with error when no run dirs exist."""
    results = tmp_path / "results"
    results.mkdir()

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=str(tmp_path),
        env={**os.environ, "RESULTS_DIR": str(results)},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "no run folders" in result.stdout
