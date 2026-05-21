"""Loads datasets and computes basic statistics.

Prints feature counts, preprocessing shape, and class distribution for each
configured dataset.

Usage:
    uv run python -m experiments.analysis.data_stats
"""

import dataclasses
from collections.abc import Sequence

import numpy as np
import pandas as pd

from experiments.binary.config import DATASETS as BINARY_DATASETS
from experiments.common import data_loader
from experiments.multiclass.config import DATASETS as MULTICLASS_DATASETS

DATASETS = list(BINARY_DATASETS) + list(MULTICLASS_DATASETS)

# Maps each dataset to the BibTeX key of its original source, i.e. the
# primary reference attached to the dataset's OpenML page (paper,
# technical report, or source repository). The OpenML platform itself is
# cited separately via vanschoren2014openml in the experiment chapters.
# Entries are defined in tex/references.bib; the dataset-summary tables
# emit a \citep next to each dataset name. A dataset absent from this map
# (e.g. the profiler-only `digits`) simply gets no citation.
DATASET_CITATIONS = {
    # Binary suite
    "sonar": "gorman1988sonar",
    "ionosphere": "sigillito1989ionosphere",
    "wdbc": "street1993wdbc",
    "pima_diabetes": "smith1988diabetes",
    "german_credit": "hofmann1994germancredit",
    "pc3": "sayyadshirabad2005promise",
    "ozone": "zhang2008ozone",
    "madelon": "guyon2004madelon",
    "spambase": "hopkins1999spambase",
    "phoneme": "elena1993phoneme",
    "jm1": "sayyadshirabad2005promise",
    "mammography": "woods1993mammography",
    "bank_marketing": "moro2011bank",
    "adult": "kohavi1996adult",
    # Multiclass suite
    "iris": "fisher1936iris",
    "vehicle": "siebert1987vehicle",
    "cmc": "lim2000cmc",
    "yeast": "horton1996yeast",
    "wine_quality_red": "cortez2009wine",
    "abalone": "nash1994abalone",
    "fot_proving": "bridge2014fol",
    "satimage": "srinivasan1993satimage",
    "gesture_phase": "madeo2013gesture",
    "volkert": "guyon2015automl",
    "jannis": "guyon2015automl",
}


@dataclasses.dataclass(frozen=True)
class _DatasetStatsRow:
    dataset: str
    samples: int
    n_classes: int
    num_features: int
    cat_features: int
    after_prep: int
    imbalance_ratio: str


_COLUMN_NAMES = {
    "dataset": "Dataset",
    "samples": "Samples",
    "n_classes": "Classes",
    "num_features": "Numerical",
    "cat_features": "Categorical",
    "after_prep": "Encoded",
    "imbalance_ratio": "IR",
}


def main():
    """Loads datasets and computes basic statistics."""
    for name in DATASETS:
        print(f"--- Dataset: {name} ---")
        try:
            X_raw, y = data_loader.load_raw_dataset(name)

            num_features_raw = X_raw.shape[1]
            num_numerical = X_raw.select_dtypes(include=np.number).shape[1]
            num_categorical = X_raw.select_dtypes(
                include=["object", "category"]
            ).shape[1]

            print(f"Features before preprocessing: {num_features_raw}")
            print(f"  Numerical features: {num_numerical}")
            print(f"  Categorical features: {num_categorical}")

            preprocessor = data_loader.create_preprocessor(X_raw)
            X_proc = preprocessor.fit_transform(X_raw)
            print(f"Shape after preprocessing: {X_proc.shape}")

            print("Class Distribution:")
            class_counts = pd.Series(y).value_counts().sort_index()
            class_props = (
                pd.Series(y).value_counts(normalize=True).sort_index()
            )
            dist_df = pd.DataFrame(
                {
                    "Count": class_counts,
                    "Proportion": class_props.map("{:.2%}".format),
                }
            )
            print(f"{dist_df}\n")
        except Exception as e:
            print(f"Could not load or process dataset {name}: {e}\n")


def build_dataset_stats_latex(
    datasets: Sequence[str] | None = None,
    label: str = "dataset-summary",
    caption: str = "Summary of datasets used in the experiments.",
    include_class_count: bool = False,
) -> str:
    """Build a LaTeX table summarizing dataset characteristics.

    Returns
    -------
    str
        Complete LaTeX table environment with booktabs formatting.
    """
    if datasets is None:
        datasets = DATASETS
    rows: list[_DatasetStatsRow] = []
    for name in datasets:
        try:
            X_raw, y = data_loader.load_raw_dataset(name)
            n_num = X_raw.select_dtypes(include=np.number).shape[1]
            n_cat = X_raw.select_dtypes(include=["object", "category"]).shape[
                1
            ]
            preprocessor = data_loader.create_preprocessor(X_raw)
            X_proc = preprocessor.fit_transform(X_raw)
            class_counts = pd.Series(y).value_counts()
            ir = class_counts.max() / class_counts.min()
            openml_id = data_loader.DATASET_IDS.get(name, "")
            dataset_label = name.replace("_", "\\_")
            if openml_id:
                dataset_label = (
                    f"{dataset_label}\\textsuperscript{{\\tiny {openml_id}}}"
                )
            cite_key = DATASET_CITATIONS.get(name)
            if cite_key:
                dataset_label = f"{dataset_label}~\\citep{{{cite_key}}}"
            rows.append(
                _DatasetStatsRow(
                    dataset=dataset_label,
                    samples=X_raw.shape[0],
                    n_classes=len(class_counts),
                    num_features=n_num,
                    cat_features=n_cat,
                    after_prep=X_proc.shape[1],
                    imbalance_ratio=f"{ir:.1f}",
                )
            )
        except Exception as exc:
            print(f"Warning: skipping {name}: {exc}")
            continue

    df = pd.DataFrame([dataclasses.asdict(r) for r in rows]).rename(
        columns=_COLUMN_NAMES
    )
    if not include_class_count:
        df = df.drop(columns=[_COLUMN_NAMES["n_classes"]])
    column_format = "lrrrrrl" if include_class_count else "lrrrrl"
    body = df.to_latex(index=False, escape=False, column_format=column_format)

    # Group the three feature-count columns under a spanning "Features"
    # header so it is clear they are all feature counts. The leading
    # columns are Dataset, Samples, (Classes); the trailing column is IR.
    n_lead = 3 if include_class_count else 2
    first_feat_col = n_lead + 1
    last_feat_col = first_feat_col + 2
    cells = [""] * n_lead + ["\\multicolumn{3}{c}{Features}", ""]
    group_row = " & ".join(cells) + " \\\\"
    cmidrule = f"\\cmidrule(lr){{{first_feat_col}-{last_feat_col}}}"
    body = body.replace(
        "\\toprule\n",
        f"\\toprule\n{group_row}\n{cmidrule}\n",
        1,
    )

    # Superscripts on dataset names are OpenML dataset IDs; the \citep next
    # to each row points to that dataset's original source. We do not spell
    # that out in the caption itself to keep it focused on the column
    # definitions.
    full_caption = (
        f"{caption}"
        " Superscripts denote OpenML dataset IDs."
        " Numerical and Categorical are the counts of raw numerical and"
        " categorical input features; Encoded is the model input"
        " dimensionality after one-hot encoding of the categorical"
        " features."
        " The imbalance ratio (IR) is the size of the majority class"
        " divided by that of the minority class."
    )
    return (
        "\\begin{table}[H]\n"
        "  \\centering\n"
        f"  \\caption{{{full_caption}}}%\n"
        f"  \\label{{tab:{label}}}\n"
        "  \\resizebox{\\textwidth}{!}{%\n"
        f"  {body}"
        "  }\n"
        "\\end{table}\n"
    )


if __name__ == "__main__":
    main()
