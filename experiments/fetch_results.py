"""Download the result files behind the dissertation's tables and figures.

The experiment runners write under ``experiments/results/``, which is not
versioned. The parquets that the table and figure generators read are
published as one archive attached to a GitHub release. This script
downloads the archive, checks its SHA-256 and unpacks it into
``experiments/results/``.

Usage:
    uv run python -m experiments.fetch_results
    uv run python -m experiments.fetch_results --url file:///path/to/copy.zip
"""

from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

RESULTS_URL = (
    "https://github.com/vitaliset/threshold-opt/releases/download/"
    "original-version/results-original-version.zip"
)
RESULTS_SHA256 = (
    "fd1004cb275c2fcc08426231833a086bbb8d37df63326f976d36776af14d98ed"
)


def fetch_results(
    url: str = RESULTS_URL,
    sha256: str = RESULTS_SHA256,
    results_dir: Path = RESULTS_DIR,
) -> list[Path]:
    """Download the archive at *url*, verify it and unpack it.

    The archive holds paths relative to *results_dir*. A download whose
    SHA-256 differs from *sha256* raises ``ValueError`` and nothing is
    unpacked.
    """
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url) as response:
        payload = response.read()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != sha256:
        raise ValueError(
            f"SHA-256 mismatch for {url}: expected {sha256}, got {digest}"
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(results_dir)
        written = [results_dir / name for name in archive.namelist()]
    for path in written:
        print(f"  Wrote {path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the published experiment result files.",
    )
    parser.add_argument(
        "--url",
        default=RESULTS_URL,
        help=(
            "Where to read the archive from (default: the GitHub release"
            " asset). The checksum is the same for any source."
        ),
    )
    args = parser.parse_args()
    fetch_results(url=args.url)
    print("Done.")


if __name__ == "__main__":
    main()
