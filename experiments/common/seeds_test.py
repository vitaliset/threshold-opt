import json

from experiments.common import seeds


def test_sample_seeds_is_reproducible():
    assert seeds.sample_seeds(5, master_seed=42) == seeds.sample_seeds(
        5, master_seed=42
    )


def test_sample_seeds_distinct_master_seeds_differ():
    assert seeds.sample_seeds(5, master_seed=42) != seeds.sample_seeds(
        5, master_seed=43
    )


def test_sample_seeds_length_matches_n():
    assert len(seeds.sample_seeds(7, master_seed=0)) == 7


def test_sample_seeds_in_range():
    vec = seeds.sample_seeds(100, master_seed=0)
    assert all(0 <= s < 2**31 - 1 for s in vec)


def test_sample_seeds_avoids_zero_domination():
    # Sanity: the default master should not produce 0..N-1.
    vec = seeds.sample_seeds(10, master_seed=42)
    assert vec != list(range(10))


def test_load_or_sample_seeds_persists_to_disk(tmp_path):
    vec = seeds.load_or_sample_seeds(tmp_path, 5, master_seed=42)
    persisted = json.loads((tmp_path / "seeds.json").read_text())
    assert persisted["master_seed"] == 42
    assert persisted["seeds"] == vec


def test_load_or_sample_seeds_resume_ignores_master(tmp_path):
    v1 = seeds.load_or_sample_seeds(tmp_path, 5, master_seed=42)
    # Second call with a different master+n should load the persisted
    # vector, not re-sample.
    v2 = seeds.load_or_sample_seeds(tmp_path, 99, master_seed=999)
    assert v1 == v2
