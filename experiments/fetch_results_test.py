"""Unit tests for experiments/fetch_results.py."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from experiments.fetch_results import fetch_results


def _write_archive(path: Path) -> str:
    """Write a two-file archive to *path* and return its SHA-256."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("results_binary_x.parquet", b"binary")
        archive.writestr("run_results_x/results_x.parquet", b"nested")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fetch_results_unpacks_verified_archive(tmp_path):
    archive = tmp_path / "archive.zip"
    sha256 = _write_archive(archive)
    results_dir = tmp_path / "results"

    written = fetch_results(
        url=archive.as_uri(), sha256=sha256, results_dir=results_dir
    )

    assert written == [
        results_dir / "results_binary_x.parquet",
        results_dir / "run_results_x" / "results_x.parquet",
    ]
    assert written[0].read_bytes() == b"binary"
    assert written[1].read_bytes() == b"nested"


def test_fetch_results_rejects_wrong_checksum(tmp_path):
    archive = tmp_path / "archive.zip"
    _write_archive(archive)
    results_dir = tmp_path / "results"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch_results(
            url=archive.as_uri(), sha256="0" * 64, results_dir=results_dir
        )

    assert not results_dir.exists()
