import os

import torch

from experiments.common import thread_limits


def test_env_vars_set():
    """Importing thread_limits should set BLAS thread env vars to 1."""
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("MKL_NUM_THREADS") == "1"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    assert os.environ.get("NUMEXPR_NUM_THREADS") == "1"


def test_torch_intra_op_threads():
    """torch intra-op threads should be 1 after import."""
    assert torch.get_num_threads() == 1


def test_torch_inter_op_threads():
    """torch inter-op threads should be 1 after import."""
    assert torch.get_num_interop_threads() == 1


def test_cap_torch_resets_intra_op():
    """cap_torch() should reset intra-op to 1 even if bumped."""
    torch.set_num_threads(4)
    assert torch.get_num_threads() == 4

    thread_limits.cap_torch()
    assert torch.get_num_threads() == 1
