"""Git provenance metadata for experiment parquet files.

Every parquet the experiment runners write carries embedded metadata
describing the git state at write time:

* ``git_sha`` - HEAD commit SHA when the runner wrote the parquet.
* ``tree_dirty_non_tex`` - ``"true"`` if any path outside ``tex/`` had
  uncommitted changes when the parquet was written.
* ``latest_non_tex_commit`` - the most recent commit touching any path
  outside ``tex/`` as of ``HEAD``.  On a clean tree this equals
  ``git_sha``; on a dirty tree it is still informative because the
  non-tex state of the repo at write time was *at least* this commit.

:func:`check_parquet_freshness` compares the embedded metadata against
the current working tree and flags parquets that were produced either
from a dirty tree or from a code state that predates the current latest
non-tex commit.  By design, changes confined to ``tex/`` (dissertation prose,
figures, captions) do not invalidate a parquet.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]

# git pathspec: everything except tex/
_NON_TEX_PATHSPEC: tuple[str, ...] = (".", ":(exclude)tex")


class GitMetadataError(RuntimeError):
    """Raised when a git subprocess call fails unexpectedly."""


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise GitMetadataError(f"git {' '.join(args)} failed: {exc}") from exc
    return result.stdout.strip()


def get_git_sha() -> str:
    """Return the SHA of ``HEAD``."""
    return _git("rev-parse", "HEAD")


def is_tree_dirty_non_tex() -> bool:
    """Return True if any path outside ``tex/`` has uncommitted changes."""
    out = _git("status", "--porcelain", "--", *_NON_TEX_PATHSPEC)
    return bool(out.strip())


def latest_non_tex_commit() -> str:
    """SHA of the most recent commit touching anything outside ``tex/``."""
    return _git("rev-list", "-n", "1", "HEAD", "--", *_NON_TEX_PATHSPEC)


def collect_runner_metadata() -> dict[str, str]:
    """Snapshot git state at the time a runner writes a parquet."""
    return {
        "git_sha": get_git_sha(),
        "tree_dirty_non_tex": "true" if is_tree_dirty_non_tex() else "false",
        "latest_non_tex_commit": latest_non_tex_commit(),
    }


def write_parquet_with_metadata(
    df: pd.DataFrame,
    path: Path | str,
    *,
    extra_metadata: dict[str, str] | None = None,
    preserve_index: bool = False,
) -> None:
    """Write *df* to *path* as parquet, embedding git provenance metadata.

    Behaves like ``df.to_parquet(path, index=preserve_index)`` but also
    stores :func:`collect_runner_metadata` under the parquet's schema
    key/value metadata.  Any keys in *extra_metadata* override the git
    keys.
    """
    table = pa.Table.from_pandas(df, preserve_index=preserve_index)
    md = dict(table.schema.metadata or {})
    provenance = {**collect_runner_metadata(), **(extra_metadata or {})}
    for k, v in provenance.items():
        md[k.encode("utf-8")] = v.encode("utf-8")
    table = table.replace_schema_metadata(md)

    # Atomic write. A crash/preemption mid-write must not leave a
    # truncated parquet at *path*: the resume logic treats any existing
    # per-task file as complete and merge_run_results reads every part,
    # so a partial file would be silently skipped on resume and then
    # abort the final merge. Write to a sibling temp file on the same
    # filesystem, then os.replace (atomic) into place. The temp name is
    # hidden and PID-tagged so a killed write leaves no *.parquet litter
    # and never collides with a concurrent writer.
    path = Path(path)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        pq.write_table(table, str(tmp))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


_INTERNAL_METADATA_PREFIXES: tuple[bytes, ...] = (b"pandas", b"ARROW:")


def read_parquet_metadata(path: Path | str) -> dict[str, str]:
    """Return the user-level key/value metadata embedded in *path*.

    Excludes pyarrow-/pandas-internal keys (``pandas``, ``ARROW:...``)
    that are added automatically during write.
    """
    md = pq.read_metadata(str(path)).metadata or {}
    return {
        k.decode("utf-8"): v.decode("utf-8")
        for k, v in md.items()
        if not any(k.startswith(p) for p in _INTERNAL_METADATA_PREFIXES)
    }


@dataclasses.dataclass(frozen=True)
class FreshnessStatus:
    fresh: bool
    reason: str


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def check_parquet_freshness(path: Path | str) -> FreshnessStatus:
    """Classify *path* as fresh or stale based on embedded git metadata.

    A parquet is fresh iff all of the following hold:

    1. It carries git metadata (runner embedded it at write time).
    2. The producing tree was clean (``tree_dirty_non_tex == "false"``).
    3. The current working tree has no uncommitted non-tex changes.
    4. The parquet's ``git_sha`` is a descendant of (or equal to) the
       current HEAD's ``latest_non_tex_commit`` - i.e. no non-tex code
       has been committed since the parquet was produced.
    """
    path = Path(path)
    md = read_parquet_metadata(path)

    git_sha = md.get("git_sha")
    if git_sha is None:
        return FreshnessStatus(
            False,
            f"{path.name}: no git metadata (predates staleness tracking); "
            f"re-run the producing experiment.",
        )

    if md.get("tree_dirty_non_tex") == "true":
        return FreshnessStatus(
            False,
            f"{path.name}: produced from a dirty working tree; "
            f"re-run after committing non-tex changes.",
        )

    if is_tree_dirty_non_tex():
        return FreshnessStatus(
            False,
            f"{path.name}: current working tree has uncommitted non-tex "
            f"changes; commit or stash before checking freshness.",
        )

    current_latest = latest_non_tex_commit()
    if not _is_ancestor(current_latest, git_sha):
        return FreshnessStatus(
            False,
            f"{path.name}: non-tex code has changed since this parquet "
            f"was produced (parquet SHA={git_sha[:8]}, current latest "
            f"non-tex commit={current_latest[:8]}); re-run the producing "
            f"experiment.",
        )

    return FreshnessStatus(True, "fresh")
