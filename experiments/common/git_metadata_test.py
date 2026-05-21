"""Unit tests for experiments/common/git_metadata.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import experiments.common.git_metadata as gm
from experiments.common.git_metadata import (
    check_parquet_freshness,
    collect_runner_metadata,
    read_parquet_metadata,
    write_parquet_with_metadata,
)


def _write_bare_parquet(
    path: Path, metadata: dict[str, str] | None = None
) -> None:
    """Write a minimal parquet with explicit key/value metadata (no git)."""
    table = pa.table({"a": [1, 2, 3]})
    md: dict[bytes, bytes] = {}
    if metadata:
        md = {k.encode(): v.encode() for k, v in metadata.items()}
    table = table.replace_schema_metadata(md)
    pq.write_table(table, str(path))


# --- read_parquet_metadata --------------------------------------------------


def test_read_parquet_metadata_extracts_custom_keys(tmp_path):
    path = tmp_path / "t.parquet"
    _write_bare_parquet(path, {"foo": "bar", "hello": "world"})
    assert read_parquet_metadata(path) == {"foo": "bar", "hello": "world"}


def test_read_parquet_metadata_empty_when_no_custom_keys(tmp_path):
    path = tmp_path / "t.parquet"
    _write_bare_parquet(path)
    assert read_parquet_metadata(path) == {}


def test_read_parquet_metadata_skips_pandas_key(tmp_path):
    # pandas roundtrip writes a 'pandas' kv metadata entry automatically
    df = pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "t.parquet"
    df.to_parquet(path, index=False)
    assert read_parquet_metadata(path) == {}


# --- collect_runner_metadata ------------------------------------------------


def test_collect_runner_metadata_has_required_keys():
    md = collect_runner_metadata()
    assert set(md) == {
        "git_sha",
        "tree_dirty_non_tex",
        "latest_non_tex_commit",
    }
    assert len(md["git_sha"]) == 40
    assert md["tree_dirty_non_tex"] in ("true", "false")
    assert len(md["latest_non_tex_commit"]) == 40


# --- write_parquet_with_metadata -------------------------------------------


def test_write_parquet_with_metadata_roundtrips_df(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    path = tmp_path / "t.parquet"
    write_parquet_with_metadata(df, path)
    assert pd.read_parquet(path).equals(df)


def test_write_parquet_with_metadata_embeds_git_provenance(tmp_path):
    df = pd.DataFrame({"x": [1]})
    path = tmp_path / "t.parquet"
    write_parquet_with_metadata(df, path)
    md = read_parquet_metadata(path)
    assert "git_sha" in md and len(md["git_sha"]) == 40
    assert md["tree_dirty_non_tex"] in ("true", "false")
    assert "latest_non_tex_commit" in md


def test_write_parquet_with_metadata_extra_overrides_git(tmp_path):
    df = pd.DataFrame({"x": [1]})
    path = tmp_path / "t.parquet"
    write_parquet_with_metadata(
        df, path, extra_metadata={"git_sha": "deadbeef", "note": "hello"}
    )
    md = read_parquet_metadata(path)
    assert md["git_sha"] == "deadbeef"
    assert md["note"] == "hello"


def test_write_parquet_with_metadata_leaves_no_temp_litter(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3]})
    path = tmp_path / "t.parquet"
    write_parquet_with_metadata(df, path)

    assert path.exists()
    assert pd.read_parquet(path).equals(df)  # final file is intact
    # The atomic-write temp (".<name>.<pid>.tmp") must be gone.
    leftovers = [p.name for p in tmp_path.iterdir() if p != path]
    assert leftovers == [], f"unexpected temp litter: {leftovers}"


def test_write_parquet_with_metadata_overwrites_atomically(tmp_path):
    path = tmp_path / "t.parquet"
    write_parquet_with_metadata(pd.DataFrame({"x": [1]}), path)
    new_df = pd.DataFrame({"x": [9, 9]})
    write_parquet_with_metadata(new_df, path)

    assert pd.read_parquet(path).equals(new_df)
    assert [p.name for p in tmp_path.iterdir()] == ["t.parquet"]


def test_write_parquet_with_metadata_cleans_temp_on_write_failure(
    tmp_path, monkeypatch
):
    """If the parquet write fails mid-way the destination must NOT be
    created (so resume re-runs the task) and no temp file is left."""

    def _boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(gm.pq, "write_table", _boom)

    path = tmp_path / "t.parquet"
    with pytest.raises(RuntimeError, match="disk full"):
        write_parquet_with_metadata(pd.DataFrame({"x": [1]}), path)

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []  # no temp litter


# --- check_parquet_freshness -----------------------------------------------


def test_check_parquet_freshness_missing_metadata(tmp_path):
    path = tmp_path / "t.parquet"
    _write_bare_parquet(path)
    status = check_parquet_freshness(path)
    assert not status.fresh
    assert "no git metadata" in status.reason


def test_check_parquet_freshness_producer_was_dirty(tmp_path):
    path = tmp_path / "t.parquet"
    _write_bare_parquet(
        path,
        {
            "git_sha": "deadbeef" * 5,
            "tree_dirty_non_tex": "true",
            "latest_non_tex_commit": "deadbeef" * 5,
        },
    )
    status = check_parquet_freshness(path)
    assert not status.fresh
    assert "dirty working tree" in status.reason


def test_check_parquet_freshness_stale_when_code_advanced(
    tmp_path, monkeypatch
):
    # Pretend current tree is clean, but the latest non-tex commit is newer
    # than the parquet's recorded SHA.
    monkeypatch.setattr(gm, "is_tree_dirty_non_tex", lambda: False)
    monkeypatch.setattr(gm, "latest_non_tex_commit", lambda: "f" * 40)
    monkeypatch.setattr(gm, "_is_ancestor", lambda a, d: False)

    path = tmp_path / "t.parquet"
    _write_bare_parquet(
        path,
        {
            "git_sha": "a" * 40,
            "tree_dirty_non_tex": "false",
            "latest_non_tex_commit": "a" * 40,
        },
    )
    status = check_parquet_freshness(path)
    assert not status.fresh
    assert "non-tex code has changed" in status.reason


def test_check_parquet_freshness_fresh_when_aligned(tmp_path, monkeypatch):
    monkeypatch.setattr(gm, "is_tree_dirty_non_tex", lambda: False)
    monkeypatch.setattr(gm, "latest_non_tex_commit", lambda: "a" * 40)
    monkeypatch.setattr(gm, "_is_ancestor", lambda a, d: True)

    path = tmp_path / "t.parquet"
    _write_bare_parquet(
        path,
        {
            "git_sha": "a" * 40,
            "tree_dirty_non_tex": "false",
            "latest_non_tex_commit": "a" * 40,
        },
    )
    status = check_parquet_freshness(path)
    assert status.fresh


def test_check_parquet_freshness_flags_dirty_current_tree(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gm, "is_tree_dirty_non_tex", lambda: True)
    path = tmp_path / "t.parquet"
    _write_bare_parquet(
        path,
        {
            "git_sha": "a" * 40,
            "tree_dirty_non_tex": "false",
            "latest_non_tex_commit": "a" * 40,
        },
    )
    status = check_parquet_freshness(path)
    assert not status.fresh
    assert "current working tree" in status.reason


# Deferred: raising GitMetadataError when git is unavailable requires
# patching subprocess.run; the happy path (git present) is covered by the
# write/collect tests above.


def test_gitmetadata_error_on_git_failure(tmp_path, monkeypatch):
    import subprocess

    def _raise(*_a, **_k):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    with pytest.raises(gm.GitMetadataError):
        gm.get_git_sha()
