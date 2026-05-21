"""Replicate-seed sampling and persistence.

Replicate seeds are drawn from a ``numpy.random.default_rng(master_seed)``
rather than taken as the contiguous range ``0..n_seeds-1``. Seeds are
persisted to ``<run_dir>/seeds.json`` on first sample so that resuming a
run reuses the original vector even if ``--master-seed`` or ``--n-seeds``
change on the resume invocation.

Sampled values are bounded to ``[0, 2**31 - 1)`` — wide enough to avoid
collisions at the replicate counts we use (<=1000) and narrow enough to
satisfy every downstream consumer (``sklearn.utils.check_random_state``
requires a 32-bit value).
"""

import json
from pathlib import Path
from typing import List

import numpy as np

SEEDS_FILENAME = "seeds.json"
_SEED_MAX = 2**31 - 1


def sample_seeds(n_seeds: int, master_seed: int) -> List[int]:
    """Draw ``n_seeds`` replicate seeds deterministically from the master."""
    rng = np.random.default_rng(master_seed)
    return rng.integers(0, _SEED_MAX, size=n_seeds, dtype=np.int64).tolist()


def load_or_sample_seeds(
    run_dir: Path, n_seeds: int, master_seed: int
) -> List[int]:
    """Return the seed vector for a run, loading from disk if present.

    On first call, samples ``n_seeds`` seeds from ``master_seed`` and
    writes them to ``<run_dir>/seeds.json``. Subsequent calls (including
    after ``--resume``) read the persisted vector and ignore the new
    ``n_seeds`` / ``master_seed`` — the original run's seeds win so the
    already-completed per-seed parquets stay consistent with the
    still-pending ones.
    """
    path = Path(run_dir) / SEEDS_FILENAME
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return [int(s) for s in data["seeds"]]
    seeds = sample_seeds(n_seeds, master_seed)
    path.write_text(
        json.dumps({"master_seed": master_seed, "seeds": seeds}, indent=2),
        encoding="utf-8",
    )
    return seeds
