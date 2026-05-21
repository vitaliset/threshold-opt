# Benchmark Validation Reference

Reference for sanity-checking the **full-run model ROC AUC** against published
numbers. Compiled 2026-05-16 from a literature sweep (OpenML same-data run
distributions, AMLB, Grinsztajn 2022, TabArena, TabPFN-v2 *Nature*,
RealMLP/TabM/TabICL papers, Sheng & Ling 2006, Elkan 2001, O'Brien 2008) plus
internal-consistency checks.

## How to use this after the final run

1. Extract per-`(dataset, model)` mean test ROC AUC from the merged parquet
   (dedupe to `(dataset, model, seed)` on `model_roc_auc_full`; it is a
   model-level diagnostic duplicated across strategy/objective rows). The
   extraction snippet is in the last section.
2. For each dataset, check the best-model AUC against the **Published band**
   column. Inside the band (or at/just above the OpenML same-data ceiling) =
   consistent.
3. Confirm the **Expected behaviors** below still hold (they are the
   suspicious-looking-but-correct cases).
4. Confirm the **post-recode MiniBooNE** numbers landed (the only pipeline
   change since this reference was written).

Methodology note: the recorded metric is held-out test **ROC AUC** (binary:
positive class; multiclass: one-vs-rest, class-frequency-weighted),
80/20 stratified split, 5 seeds. OVR-weighted AUC is structurally far higher
than accuracy, so high AUC with moderate accuracy is normal. OpenML public
maxima are 10-fold-CV; an 80/20-split number modestly above the OpenML p95/max
is split variance, not leakage.

---

## Baseline numbers (pre-final-run, from `results_*_merged.parquet`, Apr 2026)

These are the numbers validated below. The final run should reproduce them
(±seed noise) for every model except GaussianNB/LogReg on **MiniBooNE**, which
change by design (see the MiniBooNE recode section).

### Binary — mean test ROC AUC (full-train model)

| dataset | LogReg | GaussNB | RF | ExtraTrees | LGBM | XGB | CatBoost | RealMLP | TabM | TabPFN | TabPFN25 | TabICL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sonar | 0.824 | 0.800 | 0.928 | 0.959 | 0.925 | 0.921 | 0.955 | 0.934 | 0.929 | 0.965 | 0.965 | – |
| ionosphere | 0.890 | 0.944 | 0.982 | 0.989 | 0.974 | 0.962 | 0.977 | 0.978 | 0.989 | 0.988 | 0.988 | – |
| wdbc | 0.995 | 0.981 | 0.993 | 0.996 | 0.995 | 0.995 | 0.996 | 0.995 | 0.996 | 0.997 | 0.997 | 0.997 |
| pima_diabetes | 0.834 | 0.810 | 0.821 | 0.826 | 0.817 | 0.805 | 0.836 | 0.777 | 0.791 | 0.844 | 0.844 | 0.836 |
| tic_tac | 0.991 | 0.747 | 0.999 | 1.000 | 0.999 | 0.999 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 0.999 |
| german_credit | 0.779 | 0.729 | 0.794 | 0.784 | 0.766 | 0.778 | 0.797 | 0.745 | 0.771 | 0.792 | 0.792 | 0.796 |
| pc3 | 0.822 | 0.754 | 0.845 | 0.851 | 0.825 | 0.815 | 0.848 | 0.777 | 0.812 | 0.857 | 0.857 | 0.873 |
| ozone | 0.915 | 0.832 | 0.905 | 0.917 | 0.920 | 0.914 | 0.919 | 0.845 | 0.913 | 0.934 | 0.934 | 0.937 |
| spambase | 0.971 | 0.937 | 0.988 | 0.987 | 0.991 | 0.989 | 0.989 | 0.977 | 0.983 | 0.993 | 0.993 | 0.993 |
| phoneme | 0.821 | 0.823 | 0.970 | 0.974 | 0.962 | 0.961 | 0.960 | 0.941 | 0.961 | 0.974 | 0.974 | 0.980 |
| mammography | 0.899 | 0.908 | 0.931 | 0.938 | 0.938 | 0.942 | 0.950 | 0.924 | 0.935 | 0.952 | 0.952 | 0.950 |
| bank_marketing | 0.905 | 0.809 | 0.928 | 0.909 | 0.935 | 0.931 | 0.937 | 0.931 | 0.925 | – | – | – |
| adult | 0.906 | 0.821 | 0.902 | 0.878 | 0.929 | 0.929 | 0.930 | 0.917 | 0.904 | – | – | – |
| MiniBooNE | 0.937 | **0.503** | 0.980 | 0.980 | 0.983 | 0.984 | 0.986 | 0.987 | 0.984 | – | – | – |

`–` = model skipped by documented size/feature/class/min-sample gate (see
TabPFN/TabICL caps below). MiniBooNE GaussNB **0.503 is the pre-recode value**;
post-recode expected ~0.86–0.92.

### Binary suite — 2 hard additions (2026-05-16)

15 → 17 → 16: added `madelon`, `jm1` so discrimination does not lean
only on pima/german; `numerai28` (near-random ~0.52 AUC) and the
considered-but-dropped `philippine` were removed.
No results yet — validate on **error rate** (= Argmax under Symmetric
Cost ⇒ `1 − objective_value`):

| dataset (OpenML) | Expected best ACC / AUC | Source | Note |
|---|---|---|---|
| jm1 (1053) | acc ~0.80–0.82, AUC ~0.70–0.76 | OpenML 1053 runs; NASA PROMISE | defect prediction; extends pc3 line, 25 NaN → median-imputed |
| madelon (1485) | acc ~0.60–0.78, AUC ~0.65–0.88 | Guyon NeurIPS-2003 feature-selection challenge; OpenML 1485 | synthetic, 500 feat (20 informative); model-dependent — LogReg near-random (~0.5–0.6), trees/boosting much higher |

jm1's AUC sits *below* its accuracy band's naive read because the
positive (defect) class is rare (IR~4.2) and hard to rank — ~0.80 acc is
close to the majority-class base rate, so check AUC (~0.73), not acc, for
jm1. madelon is the designed-hard tripwire: if LogReg scores ≫0.6 AUC on
madelon, preprocessing is leaking informative structure.

### Multiclass — mean test ROC AUC (OVR class-weighted, full-train model)

> Suite rebuilt over several 2026-05-16 revisions (see "Multiclass suite
> rebuild"). Only **4 original datasets survive** with Apr-run validated
> AUC (below); the final run should reproduce these ±seed noise. The 11
> hard members have **no results yet** — their measured/target OVR-AUC is
> in the "Hard members — OVR-AUC targets" table. Judge the suite on
> **measured OVR-weighted AUC**, not accuracy: OVR-AUC is structurally
> inflated, so "hard" here means OVR-AUC well below ~0.95.

| dataset | LogReg | GaussNB | RF | ExtraTrees | LGBM | XGB | CatBoost | RealMLP | TabM | TabPFN | TabPFN25 | TabICL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iris | 0.997 | 0.995 | 0.996 | 0.998 | 0.983 | 0.975 | 0.992 | 0.997 | 0.996 | 0.998 | 0.998 | – |
| vehicle | 0.946 | 0.767 | 0.929 | 0.925 | 0.931 | 0.928 | 0.935 | 0.959 | 0.965 | 0.977 | 0.977 | 0.981 |
| satimage | 0.980 | 0.959 | 0.991 | 0.992 | 0.993 | 0.992 | 0.993 | 0.990 | 0.994 | 0.996 | 0.996 | 0.997 |
| jannis | 0.807 | 0.738 | 0.861 | 0.852 | 0.871 | 0.871 | 0.875 | 0.880 | 0.891 | – | – | – |

### Multiclass suite rebuild (2026-05-16)

OVR-weighted AUC is structurally ~1 on classic multiclass benchmarks
(one-vs-rest ranking stays easy even when argmax is hard), so a
decision-strategy comparison cannot discriminate on them. The suite was
rebuilt around **measured RF/LGBM OVR-weighted AUC**: keep only datasets
where ranking itself is hard. Final = **15 datasets**, all in OVR-AUC
~0.65–0.92 except `iris` (~1.0, deliberate near-separable baseline).

- **Dropped (near-separable / redundant):** `wine`, `car`, `pendigits`,
  then `digits`, `segment`, `vowel`, `letter` (all ~0.99–1.0 OVR-AUC).
  (`digits` stays in `data_loader.DATASET_IDS` only because the profiler
  defaults to it; it is **not** in the study.)
- **Hard members (11) + iris + vehicle/satimage/jannis = 15.**
- **Not added:** `robert` (41165) — OpenML serves an ARFF whose md5
  fails its own metadata checksum; sklearn rejects it for every user.
  Unreproducible, excluded.

### Hard members — OVR-AUC targets to check after the final run

**Measured** = RF/LGBM OVR-weighted AUC under pipeline preprocessing,
80/20, seed 42 (single-shot scout, not 5-seed). **Est.** = literature
band, validate post-run. Tripwire: if a "hard" dataset returns OVR-AUC
≫ its band, the wrong dataset version loaded or there is leakage.

| dataset (OpenML, K) | OVR-AUC target | Basis |
|---|---|---|
| cmc (23, 3) | **~0.65–0.70** (RF 0.65 / LGBM 0.69) | measured; hardest. Contraceptive Method Choice |
| abalone (1557, 3) | **~0.83** (RF 0.83 / LGBM 0.83) | measured; balanced, ordinal age groups |
| yeast (181, 10) | **~0.81–0.85** (RF 0.85 / LGBM 0.81) | measured; classic, IR~93 |
| wine_quality_red (40691, 6) | **~0.85–0.87** (RF 0.87 / LGBM 0.85) | measured; ordinal quality |
| fot_proving (1475, 6) | ~0.86 (RF 0.86 / LGBM 0.86) | measured; AMLB |
| jannis (41168, 4) | ~0.85 (RF 0.85 / LGBM 0.86) | measured; AMLB (OpenML AMLB RF AUC 0.869) |
| helena (41169, 100) | RF ~0.86; **LGBM ~0.5 (collapses at K=100 — expected, not a bug)** | measured; AMLB; K=100 → all foundation models skip |
| volkert (41166, 10) | ~0.91 (RF 0.91 / LGBM 0.92) | measured; AMLB; N=58k → TabPFN/ICL skip |
| gesture_phase (4538, 5) | 0.904 (measured); acc ~0.60–0.70 | OpenML 4538 |
| vehicle (54, 4) | ~0.91 (measured) | kept for K=4 mid-difficulty |

**Dropped per the 1-seed smoke (2026-05-17):** `cardiotocography` (OVR-AUC
**1.000**, predicted ~0.88–0.92) and `artificial_characters` (**0.996**,
predicted ~0.93–0.96) — their labels are near-deterministic from
features, so they do not discriminate strategies (same reasoning that
removed digits/segment/letter). Multiclass suite → **13**. The genuinely
low-OVR-AUC carriers are `cmc`/`abalone`/`yeast`/`wine_quality_red`/`fot_proving`.

---

## Binary — per-dataset external validation

Primary evidence is the OpenML run-evaluation distribution for the **exact
dataset IDs** (hundreds–thousands of public runs each). Verdicts: all
CONSISTENT.

| Dataset (OpenML id) | Our best (model) | Published band | Source (OpenML task / paper) |
|---|---|---|---|
| sonar (40) | 0.959 (ET) | p50 0.83, p95 0.93, max 0.964 | OpenML task 39 (n=881); Sheng & Ling 2006 |
| ionosphere (59) | 0.989 (ET) | p50 0.96, p95 0.99, max 0.999 | OpenML task 14964 (n=2000) |
| wdbc (1510) | 0.996 (ET/CB) | p50 0.986, p95 0.990, max 0.995 | OpenML task 9946 (n=2000) |
| pima_diabetes (37) | 0.836 (CB) | p95 0.836, max 0.840; lit ~0.82–0.84 | OpenML task 37; Sheng & Ling 2006 (Diabetes) |
| tic_tac (50) | 1.000 (ET/CB) | p95 0.998, max 1.000 | OpenML task 49; Sheng & Ling 2006. Deterministic concept, perfectly learnable by trees — not leakage |
| german_credit (31) | 0.797 (CB) | p75 0.780, p95 0.788, max 0.799 | OpenML task 31; Sheng & Ling 2006 (Credit-g) |
| pc3 (1050) | 0.851 (ET) | p95 0.847, max 0.857; NASA-defect lit ~0.78–0.85 | OpenML task 3903 |
| ozone (1487) | 0.920 (LGBM) | p95 0.917, max 0.928 | OpenML task 9978 |
| spambase (44) | 0.991 (LGBM) | max 0.989; RF 0.9894 / XGB 0.9892 baseline | OpenML task 43; Appl. Sci. 13(3):1971 |
| phoneme (1489) | 0.974 (ET) | p95 0.962, max 0.972 | OpenML task 9952 (slightly above CV max = split variance) |
| mammography (310) | 0.950 (CB) | p95 0.931, max 0.950 (n=154 runs) | OpenML task 3483 |
| bank_marketing (1461) | 0.937 (CB) | p95 0.933, max 0.936 | OpenML task 14965 |
| adult (1590) | 0.930 (CB) | p95 0.920, max 0.928; XGB lit ~0.92–0.93 | OpenML task 7592; Elkan 2001 (adult) |
| MiniBooNE (41150) | 0.986 (CB) | max 0.987; Grinsztajn GBDT AUC ~0.98 | OpenML task 168335; Grinsztajn 2022 |

Several land **exactly at the OpenML same-data ceiling** (pima 0.836 vs 0.840;
german 0.797 vs 0.799; mammography 0.950 vs 0.950; bank 0.937 vs 0.936) — the
expected outcome for strong boosting on an 80/20 split vs the 10-fold-CV public
history, not a problem.

---

## Multiclass — per-dataset external validation

Reasoned via documented **accuracy** (AUC must be consistent with it). All
CONSISTENT.

Surviving classic datasets (the 11 hard members are in the "Hard members
— OVR-AUC targets" table above; validate those on measured OVR-AUC):

| Dataset (OpenML id, K) | Our best AUC | Published accuracy / AUC | Source |
|---|---|---|---|
| iris (61, 3) | 0.998 (ET) | RF acc ~0.95–0.96; AUC ~0.99 | OpenML 61; O'Brien 2008 (used iris) |
| vehicle (54, 4) | 0.946 (LogReg) | RF acc ~0.74–0.78; Statlog best ~0.80–0.85 → AUC ~0.93–0.96 | OpenML 54; UCI Statlog Vehicle |
| satimage (182, 6) | 0.993 (LGBM/CB) | RF acc ~0.91–0.92 → AUC ~0.99 | OpenML 182; UCI Statlog Landsat |
| jannis (41168, 4) | 0.875 (CB) | **RF AUC 0.8688 / acc 0.7113**, AutoML ~0.876 / 0.722 | OpenML AMLB task 168330 (direct API) |

`jannis` is the tightest external anchor: OpenML's own AMLB run records
RandomForest OVR-weighted AUC **0.8688** / accuracy **0.7113**; our RF 0.861,
GBMs 0.871–0.875 sit exactly on the published RF→AutoML band. Strong evidence
the pipeline is calibrated and not leaking.

`O'Brien, Gupta & Gray (2008)` overlap is now only via `iris` (their
`wine`/`car`/`segment` overlaps were dropped as near-separable). It
reports *relative misclassification-cost improvement*, not raw
accuracy/AUC, so it supplies no point estimates; the RiskPartitioning
strategy implements O'Brien's method regardless of dataset overlap.

---

## Modern tabular models — published caps and rankings

### Documented size/feature/class gates (explain every `–` in the tables)

| Model | Published cap | Source |
|---|---|---|
| TabPFN v1 | 1000 samples / 100 feat / 10 classes | Hollmann et al. 2022, arXiv:2207.01848 |
| TabPFN v2 (`TabPFN`) | **10k samples / 500 feat / 10 classes** | Hollmann et al. 2025 *Nature* s41586-024-08328-6 |
| TabPFN-2.5 (`TabPFN25`) | code cap 10k / 2000 feat / 10 classes | TabPFN-2.5 (scale-up of v2) |
| TabICL | min ~300 train samples, max 10k / 500 feat / 10 classes | Qu et al. 2025, arXiv:2502.05564; TabArena arXiv:2506.16791 |

These are raised as an explicit `SkippedModelError` in
[`experiments/common/model_wrappers.py`](common/model_wrappers.py) — a
documented skip, not a silent failure. Explains the `–` cells on the
>10k datasets — binary MiniBooNE (130k) / adult (48k) / bank (45k) / jm1
(~11k); multiclass volkert (58k) / helena (65k) / jannis (84k); plus
TabPFN/25/ICL `–` on **helena**
(K=100 > 10-class cap); TabICL `–` on **sonar** (N=208) and
**ionosphere** (N=351) — 80/20 train below the ~300 floor.

### Ranking verdict (all CONSISTENT)

Observed in-cap ordering **TabPFN/TabPFN25 ≳ TabICL ≳ TabM > default-GBM ≳
RealMLP**, GBMs strongest on the large `–`-only sets. Matches:

- **TabPFN v2 ≥ tuned GBM on ≤10k data**: *Nature* paper reports normalized
  ROC AUC 0.952 (TabPFN) vs 0.822 (tuned CatBoost) on 29 datasets
  (PMC11711098). Ye et al. 2025 (arXiv:2502.17361): TabPFN v2 top-ranked over
  XGB/LGBM/CatBoost on 261 datasets.
- **TabM strong, > GBM on larger sets** (MiniBooNE 0.984, letter 0.9999,
  jannis 0.891 > GBM 0.875): Gorishniy et al. ICLR 2025, arXiv:2410.24210.
- **TabICL competitive with tuned GBMs**: Qu et al. 2025; TabArena.
- **RealMLP roughly GBM-competitive** (Holzmüller et al. NeurIPS 2024,
  arXiv:2407.04491; Grinsztajn 2022 arXiv:2207.08815).

### Expected non-blocking caveat — RealMLP weak on tiny/imbalanced sets

RealMLP loses to GBMs on the smallest sets (pima 0.777, pc3 0.777, ozone
0.845, german 0.745, sonar 0.934). **Expected, not a misconfiguration**:
Holzmüller et al. meta-learned RealMLP's defaults on a benchmark that
*excludes datasets <1000 samples* and explicitly caveat small-data
generalization. Worth one sentence in the thesis citing that caveat. It would
only be a red flag if RealMLP also lost on medium/large sets — it does not
(MiniBooNE 0.987 ≈ best, jannis 0.880 > GBM).

---

## Expected behaviors (suspicious-looking but correct — re-confirm these)

1. **tic_tac ≈ 1.000.** Deterministic logical concept; OpenML public max is
   also 1.000. NOT leakage. (`numerai28`, the near-random ~0.52-AUC
   tripwire, has been dropped from the suite.)
2. **GaussianNB ≈ 0.50 on MiniBooNE — only PRE-recode.** Caused by the −999
   sentinel surviving StandardScaler. **Fixed**; see next section. Post-recode
   it should be ~0.86–0.92.
3. **GaussianNB weak on vehicle (0.767).** Correlated features violate NB
   conditional independence; textbook NB weakness, not a bug. NB stays
   strong (~0.96–0.99) on the near-separable `iris`.
4. **RealMLP < GBM on tiny sets** — see caveat above.

---

## MiniBooNE −999 sentinel recode (pipeline change 2026-05-16)

OpenML serves MiniBooNE's documented "PID variable not computable" sentinel
as a literal `-999`, so the data had **0 NaN**, median imputation never fired,
and `-999` survived StandardScaler as a `-999 / -17256` outlier — collapsing
GaussianNB to 0.503. Verified: exactly **468 rows have all 50 features
= -999**; no other cell equals -999; genuine extremes (e.g. -17256) are real
values. A MiniBooNE-only `replace(-999.0 → NaN)` was added to
[`data_loader.py`](common/data_loader.py) so those 468 rows get
median-imputed. (Mainstream benchmarks, e.g. Grinsztajn, leave it raw and rely
on tree robustness; recoding was a deliberate choice for a clean preprocessing
story.)

**Post-recode MiniBooNE expected (seed 191664963 spot check):**

| model | pre-recode | post-recode | |
|---|---|---|---|
| GaussianNB | 0.503 | **0.916** | rescued |
| LogisticRegression | 0.937 | 0.962 | modest gain |
| RandomForest | 0.980 | 0.980 | unchanged |
| LGBM | 0.983 | 0.983 | unchanged |

After the final run, confirm MiniBooNE GaussNB landed ~0.86–0.92 (5-seed
mean), LogReg ~0.95–0.96, and trees/boosting still ~0.98–0.99. Post-preprocess
MiniBooNE must have **0 NaN** and a sane standardized range (≈ [−33, 360]).

---

## Internal-consistency checks (re-run on the final parquet)

1. **split vs full AUC gap** small for every config dataset (the only large
   gaps in the Apr run were on `credit_fraud`, since dropped). RealMLP on
   ozone/pc3 has higher seed std (~0.07–0.09) — plausible for a deep net on
   small imbalanced data, watch but not a blocker.
2. **Argmax + Symmetric Cost** ⇒ implied accuracy `= 1 − objective_value`
   should track AUC and match textbook levels:
   - adult GBMs ~0.87 acc / ~0.93 AUC
   - spambase ~0.95 acc / ~0.99 AUC
   - wdbc ~0.98 acc / ~0.99 AUC
   - german ~0.75 acc / ~0.78 AUC
   - jm1 ~0.81 acc / ~0.73 AUC (acc near base rate; check AUC)

## Re-validation snippet

```python
import pandas as pd
df = pd.read_parquet("experiments/results/results_binary_merged.parquet")
sub = (df[["dataset_name","model_name","random_seed","model_roc_auc_full"]]
       .dropna().drop_duplicates(["dataset_name","model_name","random_seed"]))
auc = (sub.groupby(["dataset_name","model_name"])["model_roc_auc_full"]
          .mean().unstack("model_name").round(4))
print(auc)
# best model per dataset
for ds, row in auc.iterrows():
    print(f"{ds:16s} best={row.idxmax():14s} {row.max():.4f}")
```

## Source index

- OpenML run evaluations (gold standard, same data): tasks
  39, 31, 9946, 37, 49, 3903, 43, 9978, 3483, 14964, 9952, 14965, 7592,
  167120, 168335 (binary); 61, 187, 54, 307, 21, 28, 36, 182, 32, 6 and
  AMLB task 168330 (jannis) — <https://www.openml.org>
- AMLB / Gijsbers et al. 2024 JMLR — <https://www.jmlr.org/papers/volume25/22-0493/22-0493.pdf>
- Grinsztajn et al. 2022, tree-based vs DL on tabular — arXiv:2207.08815
- TabArena 2025 — arXiv:2506.16791
- TabPFN v1 — arXiv:2207.01848 · TabPFN v2 — *Nature* s41586-024-08328-6 / PMC11711098
- Ye et al. 2025, "A Closer Look at TabPFN v2" — arXiv:2502.17361
- RealMLP / "Better by Default", Holzmüller et al. NeurIPS 2024 — arXiv:2407.04491
- TabM, Gorishniy et al. ICLR 2025 — arXiv:2410.24210
- TabICL, Qu et al. 2025 — arXiv:2502.05564
- FT-Transformer (jannis accuracy ref), Gorishniy et al. 2021 — arXiv:2106.11959
- Sheng & Ling 2006, "Thresholding for Making Classifiers Cost-sensitive", AAAI
  (used Diabetes/pima, Credit-g/german, sonar, tic-tac-toe)
- Elkan 2001, "The Foundations of Cost-Sensitive Learning" (used adult)
- O'Brien, Gupta & Gray 2008, "Cost-sensitive multi-class classification from
  probability estimates", ICML (used iris, wine, car, image-segmentation)
- spambase ensemble baseline — Appl. Sci. 13(3):1971, <https://www.mdpi.com/2076-3417/13/3/1971>
- UCI: Statlog Vehicle / Statlog Landsat (satimage) / Letter (Frey & Slate
  1991) / Deterding Vowel / optdigits — <https://archive.ics.uci.edu>
