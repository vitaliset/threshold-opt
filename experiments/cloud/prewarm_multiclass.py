"""Serial OpenML cache pre-warm for the multiclass dataset suite.

Run once on a fresh VM/pod before launching the parallel runner so the
worker pool reads datasets from the on-disk OpenML cache instead of all
cold-fetching simultaneously (the thundering-herd that triggers
SSL: UNEXPECTED_EOF / rate-limit GAVEUPs). The first 1-2 cold fetches
may still transiently fail if a prior burst rate-limited the IP; a
second pass picks them up once the transient clears.

ASCII-only by design (Windows plink chokes on non-ASCII heredocs).
"""

import time

from experiments.common.data_loader import load_dataset_openml
from experiments.multiclass.config import DATASETS

names = sorted(DATASETS)
print(f"prewarm: {len(names)} multiclass datasets", flush=True)
failed = []
for i, name in enumerate(names, 1):
    t0 = time.time()
    try:
        X, y = load_dataset_openml(name)
        shape = getattr(X, "shape", None)
        print(
            f"[{i}/{len(names)}] OK   {name:<18} shape={shape} "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )
    except Exception as e:
        failed.append(name)
        print(
            f"[{i}/{len(names)}] FAIL {name:<18} {type(e).__name__}: {e}",
            flush=True,
        )
print(f"PREWARM DONE (failed={failed})", flush=True)
