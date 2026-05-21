"""Serial OpenML cache pre-warm for the binary dataset suite.

Run once on a fresh VM/pod before launching the parallel runner so the
80/4-worker pool reads datasets from the on-disk OpenML cache instead of
all cold-fetching simultaneously (the thundering-herd that triggers
SSL: UNEXPECTED_EOF / rate-limit GAVEUPs).

ASCII-only by design (Windows plink chokes on non-ASCII heredocs).
"""

import time

from experiments.binary.config import DATASETS
from experiments.common.data_loader import load_dataset_openml

names = sorted(DATASETS)
print(f"prewarm: {len(names)} binary datasets", flush=True)
failed = []
for i, name in enumerate(names, 1):
    t0 = time.time()
    try:
        X, y = load_dataset_openml(name)
        shape = getattr(X, "shape", None)
        print(
            f"[{i}/{len(names)}] OK   {name:<16} shape={shape} "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )
    except Exception as e:
        failed.append(name)
        print(
            f"[{i}/{len(names)}] FAIL {name:<16} {type(e).__name__}: {e}",
            flush=True,
        )
print(f"PREWARM DONE (failed={failed})", flush=True)
