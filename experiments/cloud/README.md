# Running experiments on Google Cloud (GCP)

Step-by-step for renting GCP VMs, running experiment studies, pulling results back, and destroying the VMs. CPU-heavy models (sklearn, LGBM, XGBoost, CatBoost) run on an 80-vCPU N2 instance; NN models (TabM, RealMLP, TabPFN) run on a GPU instance with an NVIDIA L4. Results are merged on your laptop before generating tables.

**Caveats:**
- Setup is longer than a "click one button" provider (`gcloud` CLI + API enablement + GPU quota request).
- No built-in monthly price ceiling — you rely on budget alerts and habit. The CPU VM costs ~$2.80/hr and the GPU VM ~$0.70/hr; either left running for a month would be >$500, so **deletion discipline matters**.

---

## 1. Upgrade your expired trial to paid mode

If you used the $300 free trial before and it's exhausted, do **not** create a new account (against GCP's ToS). Flip the existing account into paid mode:

1. Go to <https://console.cloud.google.com>.
2. Sidebar → **Billing → Account management**.
3. Click **"Upgrade"** or **"Activate full account"**. Confirm.

Billing is now pay-as-you-go on the same email, project, and card.

## 2. Create a project and enable Compute Engine

- Top-left project dropdown → **New Project**. Name it `mestrado`, no organization. Select it.
- Sidebar → **Compute Engine → VM instances**. First visit shows "Compute Engine API is not enabled" — click **Enable**. Takes ~30 seconds.

## 3. Set a budget alert

GCP has no hard spending cap, so a budget alert is your safety net.

1. Sidebar → **Billing → Budgets & alerts → Create budget**.
2. **Scope:** your new project.
3. **Amount:** e.g. **R$200** (~$40).
4. **Alert thresholds:** 50%, 90%, 100%. Leave email on.
5. Save.

This won't stop compute automatically — GCP only emails you. Combined with the delete-VM discipline below, it's enough.

## 4. Install the `gcloud` CLI and authenticate

**Windows 11 (winget):**
```bash
winget install --id Google.CloudSDK -e --accept-source-agreements --accept-package-agreements
```

Accept the UAC prompt. Reopen your terminal so `gcloud` is on PATH. Fallback: <https://cloud.google.com/sdk/docs/install>.

```bash
gcloud auth login                              # opens browser
gcloud config set project <your-project-id>   # from the console top-left
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
```

Project IDs often look like `mestrado-123456`, not the display name.

**Why `us-central1`:** highest quota defaults, fastest approvals, every machine family available. Latency from Brazil is ~150 ms — fine for SSH and irrelevant for batch runs. `southamerica-east1` is lower-latency but has limited family availability.

## 5. Check your existing quotas

Paid-account upgrades often land with generous defaults already. List current CPU quotas for `us-central1`:

```bash
gcloud compute regions describe us-central1 --format=json \
    | python -c "import sys,json; d=json.load(sys.stdin); [print(f\"{q['metric']:20s} limit={q['limit']:>8}\") for q in d['quotas'] if 'CPUS' in q['metric']]"
```

You want to see `N2_CPUS` at 100+ and `CPUS` at 100+. Both are typically already set on paid accounts.

**Global cap to watch:** `CPUS-ALL-REGIONS-per-project` sits above regional quotas and is often 90 on new paid accounts. That's why the default VM below is 80 vCPU — it fits under the cap with no quota request.

## 6. Create the VM

N2 is Intel Ice Lake and 80 vCPU sits under the global 90-core cap. `us-central1-a` is usually in stock but occasionally returns `ZONE_RESOURCE_POOL_EXHAUSTED` — use [create-cpu.sh](create-cpu.sh) to cycle through `-a`, `-b`, `-c`, `-f` until a zone accepts the request (quota is regional, not per-zone):

```bash
bash experiments/cloud/create-cpu.sh
# optional: restrict to a subset of zones
bash experiments/cloud/create-cpu.sh us-central1-b,us-central1-c
```

The script exits immediately on non-stockout errors (auth, quota, bad flag) so real problems surface instead of looping silently. On success it prints the VM name and the zone it landed in — **note the zone**, you'll need it for SSH and for `make cloud-pull ZONE=...`.

If you'd rather run the raw `gcloud` command (e.g. you want a specific zone and are OK with one-shot failure):

```bash
gcloud compute instances create mestrado-runner \
    --zone=us-central1-a \
    --machine-type=n2-highcpu-80 \
    --image-family=ubuntu-2404-lts-amd64 \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-balanced
```

Either way, ~30 seconds later the VM is up. No manual firewall or SSH key work — `gcloud` handles it.

You'll see a harmless warning: *"Disk size: '100 GB' is larger than image size: '10 GB'. You might need to resize..."*. Ubuntu 24.04 autoexpands the root partition on first boot; `df -h /` will show ~96 GB free. Ignore it.

**Why 100 GB and not the 10 GB default:** `tabpfn` pulls `torch` + full NVIDIA CUDA wheels (~5 GB) as a transitive dep, even on a CPU-only VM. Installed uv cache + venv + OS easily hits 15 GB. 100 GB is comfortable headroom.

**Do not use the web console to create the VM.** The create-instance form hides the boot-disk size field behind an "ALTERAR" / "CHANGE" button that's easy to miss, and you'll end up with a 10 GB disk. `uv sync` then dies halfway through with *"No space left on device"* while extracting `nvidia-cudnn-cu12`. The `gcloud` one-liner above can't be misconfigured that way.

## 7. SSH in and run the experiment

```bash
gcloud compute ssh mestrado-runner --zone=us-central1-a
```

First time it offers to generate an SSH key — say yes, empty passphrase is fine. After ~10 seconds of key provisioning you're in as a non-root user with `sudo`.

For Claude commands without an interactive shell, append `--command=...`:

```bash
gcloud compute ssh mestrado-runner --zone=us-central1-a --command='cd threshold-opt && git pull --ff-only && uv sync'
```

On the VM:

```bash
git clone https://github.com/vitaliset/threshold-opt.git
cd threshold-opt
bash experiments/cloud/bootstrap.sh

# pre-warm the OpenML cache BEFORE any parallel run (see note below).
# Run the script for the study you are about to launch; run it twice --
# the 2nd pass retries the 1-2 datasets that the first cold burst may
# have had rate-limited.
uv run python -m experiments.cloud.prewarm_binary      # or prewarm_multiclass
uv run python -m experiments.cloud.prewarm_binary      # 2nd pass (retry)

# smoke run first (CPU models only)
make cloud-run STUDY=binary N_SEEDS=1 DEVICE=cpu
make cloud-status             # tmux session + task progress
tmux attach -t run-binary     # Ctrl+B then D to detach
```

> **Why pre-warm (do not skip this).** On a *fresh* VM/pod the OpenML
> cache is empty, so launching the parallel runner makes all workers
> fetch every dataset from `api.openml.org` simultaneously. That
> thundering herd gets rate-limited into `SSL: UNEXPECTED_EOF` on every
> dataset, every task fails to load, and the run aborts with
> `FileNotFoundError: No parquet files`. The `prewarm_{binary,multiclass}`
> scripts fetch each dataset once, serially, into the on-disk cache so the
> real run reads from disk. The first pass may still report 1-2 `FAIL`s if
> a prior burst rate-limited the IP; the second pass picks them up once
> the transient clears. A clean pass ends with `PREWARM DONE (failed=[])`.

If the smoke run finishes cleanly, launch the real thing:

```bash
make cloud-kill STUDY=binary
make cloud-run STUDY=binary N_SEEDS=10 DEVICE=cpu
```

Close your laptop. `tmux` keeps it running. Reconnect anytime:

```bash
gcloud compute ssh mestrado-runner --zone=us-central1-a
cd threshold-opt
make cloud-status
```

## 8. Monitor the run

From inside the VM, at the repo root:

```bash
make cloud-status
```

Prints the newest run folder, elapsed time, tasks completed (per-seed parquet count), throughput in tasks/min, and the tail of the tqdm progress bar from `run-<study>.log`. Run it anytime — it's read-only and safe.

Other useful checks from a second SSH session:
- `htop` — cores should be busy. With `DEVICE=cpu` and `N_SEEDS=10`, there are enough tasks to saturate all 80 cores.
- `df -h` / `free -h` — disk and RAM headroom. Neither should be tight on this VM.

## 9. Pull results home

On your **laptop**, in the repo directory:

```bash
cd threshold-opt
make cloud-pull
```

Wraps [pull-results.sh](pull-results.sh), which defaults to `INSTANCE=mestrado-runner ZONE=us-central1-a REMOTE_USER=$USER`. Override with `make cloud-pull INSTANCE=foo ZONE=us-central1-b REMOTE_USER=your_username`.

**Why not plain `gcloud compute scp --recurse`:** on Windows, `gcloud` shells out to PuTTY's `pscp`, which (a) doesn't expand `~`, and (b) refuses `.`/`..` entries in recursive mode with a "security violation" error. The script sidesteps both by tarring on the remote side and streaming the archive through `gcloud compute ssh` into a local `tar x`.

## 10. Kill or resume a run

Stop the tmux session without losing completed work:

```bash
make cloud-kill STUDY=binary
```

Parquet files already written are left in place. Resume later with the runner's `--resume` flag, which skips `(dataset, seed)` pairs that are already done:

```bash
uv run python -m experiments.binary.runner --resume experiments/results/run_binary_<timestamp>
```

## 11. Delete the VMs — not stop, delete

A **stopped** VM still bills you for the boot disk. Use **delete** to stop all billing:

```bash
gcloud compute instances delete mestrado-runner --zone=us-central1-a
gcloud compute instances delete mestrado-gpu --zone=us-central1-a   # if created
```

Type `y` to confirm each. Double-check in the console: **Compute Engine → VM instances** should be empty.

---

## Cost sanity check

- **CPU VM** (`n2-highcpu-80`): ~**$2.80/hr**. CPU models for the binary study finish in under an hour.
- **GPU VM** (`g2-standard-4` + L4): ~**$0.70/hr**. NN models are much faster on GPU than single-threaded CPU.
- Realistic cost for one full binary run (both VMs): **~$3–5**.
- Worst-case "forgot a VM running over the weekend": **~$200** (CPU) or **~$50** (GPU). This is why the budget alert and `gcloud compute instances delete` discipline matter.

## Appendix — GPU VM for NN models (TabM, RealMLP, TabPFN)

NN models (TabM, RealMLP, TabPFN) are much slower than boosting/tree models on CPU. Running them on an NVIDIA L4 GPU is faster and cheaper than burning CPU hours. Use the `DEVICE=gpu` flag to select only the NN model group.

### Request GPU quota

GPU quota is separate from CPU quota. You need **two** quotas:

1. **`GPUS_ALL_REGIONS`** — global cap, often 0 on new paid accounts.
2. **`NVIDIA_L4_GPUS`** in `us-central1` — per-region L4 quota.

In the console:

1. **IAM & Admin → Quotas & System Limits**.
2. Filter for each quota name above. Select, click **Edit Quotas**, request **1** (or higher if you want to scale out later — multi-GPU support in the runner is not wired yet, so 1 is enough for the current workflow).
3. Justification: *"GPU-accelerated deep learning for master's dissertation — TabPFN/TabM inference, no production traffic."*

Approval usually takes 2–6 hours. New paid accounts are often auto-denied on the first try (insufficient billing history); wait ~48 h and resubmit, or escalate via the Sales link in the denial email.

### Create the GPU VM

L4 stockouts are common, so the preferred path is [create-gpu.sh](create-gpu.sh), which mirrors [create-cpu.sh](create-cpu.sh) — it cycles through `-a`, `-b`, `-c`, `-f` and sleeps 30 s between full cycles when every zone is exhausted:

```bash
bash experiments/cloud/create-gpu.sh
# optional: restrict zones
bash experiments/cloud/create-gpu.sh us-central1-b,us-central1-c
```

Non-stockout errors (auth, quota, bad flag) break the loop immediately so they're not buried in retries.

Raw `gcloud` equivalent if you want a single shot in a fixed zone:

```bash
gcloud compute instances create mestrado-gpu \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=ubuntu-2404-lts-amd64 \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE
```

Cost: **~$0.70/hr** (L4 $0.53 + g2-standard-4 $0.17). The `--maintenance-policy=TERMINATE` flag is required for GPU instances. The harmless disk-size warning is the same as for the CPU VM — Ubuntu autoexpands the partition.

### Bootstrap and run

From your laptop:

```bash
gcloud compute ssh mestrado-gpu --zone=us-central1-a
```

On the GPU VM (copy-paste cheatsheet):

```bash
# 1. Clone and install drivers (triggers a reboot)
git clone https://github.com/vitaliset/threshold-opt.git
cd threshold-opt
bash experiments/cloud/bootstrap-gpu.sh
# Machine reboots here — wait ~30s, then SSH back in
```

After the reboot, SSH back in and finish setup:

```bash
gcloud compute ssh mestrado-gpu --zone=us-central1-a
```

```bash
# 2. Finish bootstrap (installs uv + Python deps, verifies GPU)
cd threshold-opt
bash experiments/cloud/bootstrap-gpu.sh
source ~/.bashrc

# 3. Sanity checks
nvidia-smi                    # GPU visible?
make test                     # unit tests pass?

# 3b. Pre-warm the OpenML cache (mandatory on a fresh pod; see the
#     "Why pre-warm" note in section 7). Run for the study you launch,
#     twice (2nd pass retries cold-burst rate-limit failures).
uv run python -m experiments.cloud.prewarm_binary      # or prewarm_multiclass
uv run python -m experiments.cloud.prewarm_binary      # 2nd pass (retry)

# 4. Smoke run (1 seed, quick)
make cloud-run STUDY=binary N_SEEDS=1 DEVICE=gpu
make cloud-status             # tmux session + task progress
tmux attach -t run-binary     # Ctrl+B then D to detach

# 5. Full run (once smoke is clean)
make cloud-kill STUDY=binary
make cloud-run STUDY=binary N_SEEDS=10 DEVICE=gpu
```

Close your laptop. `tmux` keeps it running. Reconnect anytime:

```bash
gcloud compute ssh mestrado-gpu --zone=us-central1-a
cd threshold-opt
make cloud-status
```

On the CPU VM, run the complementary set:

```bash
make cloud-run STUDY=binary N_SEEDS=10 DEVICE=cpu
```

The `DEVICE=cpu|gpu` flag selects from `CPU_MODELS` / `GPU_MODELS` defined in each study's `config.py`. `DEVICE=gpu` auto-sets `n_jobs=1` when `N_JOBS` is left unset (the runners apply this when `device=="gpu" and n_jobs==-1`). That serial default is safe but slow; the NN models use only a few GB of VRAM, so a small explicit override pays off. The optimum is per-GPU — set it explicitly:

- **L4 (GCP `mestrado-gpu`): `N_JOBS=2`** — near-linear throughput (~2× faster), both workers sharing the L4 at 100 % util. Tested on full binary runs.
- **RTX 5090 (RunPod): `N_JOBS=4`** — empirically the optimal value found on the 5090; more workers thrash on the single GPU without further speedup.
- Going higher risks VRAM contention on the larger datasets (`spambase`, `MiniBooNE`) where TabM's full-data fit allocates more headroom; tune per GPU.

> **RunPod caveat:** RunPod containers expose the **host's** full core count via `nproc` / `os.cpu_count()` (e.g. 128), *not* the pod's allocated vCPU (e.g. 15). The auto-`n_jobs=1` protects a run that leaves `N_JOBS` unset, but **never pass `N_JOBS=-1` on RunPod** — joblib would launch ~128 workers on a ~15-vCPU pod and thrash. Always set an explicit small `N_JOBS` (4 on the 5090).

You can also combine with `MODELS=` to narrow further, or omit both flags to run everything.

### Pull and merge with CPU results

From your laptop:

```bash
make cloud-pull INSTANCE=mestrado-gpu REMOTE_USER=$USER    # GPU results
make cloud-pull REMOTE_USER=$USER                          # CPU results (if not pulled yet)

uv run python experiments/cloud/merge_results.py \
    experiments/results/run_results_binary_<cpu_ts> \
    experiments/results/run_results_binary_<gpu_ts> \
    -o experiments/results/results_binary_merged.parquet

make tables
```

### Delete the GPU VM

```bash
gcloud compute instances delete mestrado-gpu --zone=us-central1-a
```

Verify in the console: **Compute Engine → VM instances** should show only the CPU VM (or nothing).

## Appendix — Alternative: RunPod (GPU)

Use this when GCP GPU quota is denied or you want bigger cards than the L4. RunPod accepts most credit cards, has no quota gates, and gives you an H100 in minutes. Container-based (not a VM), but SSH works directly so the workflow is close to GCP's.

### Pricing (as of 2026-04, Secure Cloud tier)

| GPU                   | Price     | Use case                         |
|-----------------------|-----------|----------------------------------|
| RTX 4000 Ada (20 GB)  | ~$0.30/hr | Connection smoke only            |
| RTX 4090 (24 GB)      | ~$0.50/hr | Cheap smoke                      |
| A100 80 GB            | ~$1.19/hr | Full run, budget-conscious       |
| **H100 80 GB HBM3**   | ~$2.50/hr | Full run, fastest                |

### Launch a pod

1. Sign up at <https://runpod.io> and add a payment method.
2. **Settings → SSH Public Keys**: paste your full `id_ed25519.pub`, **including the `ssh-ed25519 ` prefix**. Missing the prefix causes silent auth failure at SSH time.
3. Deploy a pod:
   - **Template**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (CUDA 12.8, Ubuntu 24.04, PyTorch 2.8). `uv sync` installs a fresh venv anyway, but the template gives you the right driver + toolchain.
   - **Container disk**: 30 GB (default, ephemeral at `/`).
   - **Volume disk**: 50 GB (persistent at `/workspace`; our venv alone is ~8 GB).
   - **GPU**: H100 PCIe for real runs; a cheap RTX for a connection smoke first.

Skip the Network Volume unless you need persistence across *termination* (not just stop/start).

### Connect

The pod's **Connect** panel gives an SSH string like:

```bash
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

RunPod's SSH gateway forces a PTY, so non-interactive invocations need `-tt` or stdin-piping:

```bash
echo 'nvidia-smi; exit' | ssh -tt <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**"Permission denied (publickey)"** after launch usually means the key was added *after* the pod started — stop and restart the pod (not just reboot) to re-inject it.

### Bootstrap — no `bootstrap-gpu.sh` needed

RunPod's template ships with `uv`, `python3`, `git`, `tmux`, `make`, and the NVIDIA stack preinstalled. Skip [bootstrap-gpu.sh](bootstrap-gpu.sh) entirely:

```bash
cd /workspace                 # persistent; / is ephemeral
git clone https://github.com/vitaliset/threshold-opt.git
cd threshold-opt
uv sync                       # ~1 min on RunPod's fast network
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Run the experiment

`make cloud-run` is GCP-specific (it shells out to `gcloud compute ssh`), so invoke the runner directly inside `tmux`:

```bash
cd /workspace/threshold-opt
tmux new-session -d -s smoke \
  "RESULTS_DIR=/workspace/results uv run python -m experiments.binary.runner \
     --n-seeds 1 --device gpu --n-jobs 4 \
     2>&1 | tee /workspace/smoke.log"
```

Reattach with `tmux a -t smoke`; detach with `Ctrl-B d`.

### Tuning `--n-jobs` on H100

H100 has 80 GB VRAM — plenty of headroom — but our binary workload is **GPU-compute-bound** once one worker pegs the card. Observed on a 1-seed smoke:

- `--n-jobs 4`: GPU 100 %, ~4.7 GB VRAM, baseline.
- `--n-jobs 8`: GPU still 100 %, ~10 GB VRAM, per-fit time up ~25 % on small datasets due to kernel-queue contention. Aggregate throughput *drops*.

**`--n-jobs 4` is the sweet spot on H100** for this workload. Free VRAM ≠ free parallelism.

### Resume gotcha

`--resume <run_dir>` works like on GCP, but **you must re-pass `--n-seeds`** — the runner re-reads CLI defaults on resume, and without it `--n-seeds` falls back to 3 even if the original run was `--n-seeds 1`.

```bash
uv run python -m experiments.binary.runner \
  --resume /workspace/results/run_results_binary_<ts> \
  --n-seeds 1 --device gpu --n-jobs 4
```

### Pull results home

From your laptop (RunPod's SSH gateway supports `scp`):

```bash
scp -r -i ~/.ssh/id_ed25519 \
  <pod-id>@ssh.runpod.io:/workspace/results/run_results_binary_<ts> \
  experiments/results/
```

### Stop vs Terminate

- **Stop**: `/workspace` persists (billed at $0.20/GB/mo while stopped). Restart later on the same volume.
- **Terminate**: kills the pod *and* its Volume disk. Use a Network Volume if you need survive-termination persistence across pods.

## Troubleshooting

- **`gcloud auth login` hangs in WSL:** try Git Bash or PowerShell instead.
- **"Quota 'CPUS' exceeded" when creating the VM:** either the global `CPUS-ALL-REGIONS` cap kicked in (drop to 80 vCPU), or a per-family quota is still 0 (switch to N2).
- **`uv sync` fails with "No space left on device" extracting `nvidia-cudnn-cu12`:** your boot disk is 10 GB — you created the VM via the web console and missed the "ALTERAR" / "CHANGE" button. Delete and recreate with the `gcloud` one-liner in step 6.
- **`gcloud compute ssh` returns "Permission denied":** first-time SSH can take 10–30 seconds to provision keys. Wait and retry once.
- **`make cloud-status` prints "no run folders under experiments/results":** you're not at the repo root, or no run has started yet. `cd threshold-opt` and try again.
- **All cores idle during run:** verify `--n-jobs` is unset or `-1`. The binary study has ~110 tasks, enough to saturate 80 cores.
- **Forgot the VM name:** `gcloud compute instances list` shows everything running in the project.
