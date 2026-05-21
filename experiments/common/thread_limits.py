"""Cap BLAS/torch thread pools to 1 per process.

Importing this module sets OMP/MKL/OpenBLAS/NumExpr env vars and pins
torch intra-op and inter-op thread counts to 1. Import it before any
other ML library so that library init reads the capped env vars and so
that ``torch.set_num_interop_threads`` runs before the first parallel
torch op (which can only be set once per process).

Rationale: on an ``n2-highcpu-80`` VM, joblib spawns one worker process
per task. With default thread pools each worker grabs ~80 BLAS threads
plus ~80 torch intra-op threads plus ~40 torch inter-op threads, which
collapses into a context-switch storm across the 80 cores. Capping to 1
per process leaves task-level parallelism (from joblib) as the sole
source of concurrency.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def cap_torch() -> None:
    """Reapply ``torch.set_num_threads(1)``.

    Some libraries (notably pytorch_lightning via pytabkit) reset the
    torch intra-op thread count at their own import time, overriding the
    module-level call above. Call this from inside a worker task — after
    all imports have run — to restore the cap before any model training.
    """
    torch.set_num_threads(1)
