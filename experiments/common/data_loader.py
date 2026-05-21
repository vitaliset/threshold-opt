"""Data ingestion pipeline. Handles OpenML fetching and basic preprocessing."""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import compose, datasets, impute, pipeline, preprocessing

logger = logging.getLogger(__name__)

# Map internal names to OpenML dataset IDs
DATASET_IDS = {
    # Binary datasets
    "german_credit": 31,
    "pima_diabetes": 37,
    "adult": 1590,
    "bank_marketing": 1461,
    "phoneme": 1489,
    "spambase": 44,
    "mammography": 310,
    "ozone": 1487,
    "sonar": 40,
    "ionosphere": 59,
    "pc3": 1050,
    "wdbc": 1510,
    "jm1": 1053,  # NASA PROMISE software defects (extends pc3)
    "madelon": 1485,  # Guyon NeurIPS-2003 designed-hard synthetic
    # Multiclass datasets
    "iris": 61,
    "vehicle": 54,
    "satimage": 182,
    "jannis": 41168,
    "digits": 28,  # not in the study suite; kept for the profiler default
    # Hard multiclass. Selection rationale and measured per-dataset
    # OVR-AUC live in the experiments/multiclass/config.py block.
    "fot_proving": 1475,  # first-order-theorem-proving
    "gesture_phase": 4538,  # GesturePhaseSegmentationProcessed
    "volkert": 41166,
    "cmc": 23,  # Contraceptive Method Choice
    "abalone": 1557,  # 3-class grouped (ordinal age groups)
    "yeast": 181,  # protein localization
    "wine_quality_red": 40691,  # ordinal quality
}


def create_preprocessor(X: pd.DataFrame) -> compose.ColumnTransformer:
    """Creates a preprocessing pipeline for the given data schema."""
    cat_cols = X.select_dtypes(include=["object", "category", "str"]).columns
    num_cols = X.select_dtypes(include=["number"]).columns

    transformers = []

    if len(num_cols) > 0:
        num_transformer = pipeline.Pipeline(
            steps=[
                ("imputer", impute.SimpleImputer(strategy="median")),
                ("scaler", preprocessing.StandardScaler()),
            ]
        )
        transformers.append(("num", num_transformer, num_cols))

    if len(cat_cols) > 0:
        cat_transformer = pipeline.Pipeline(
            steps=[
                ("imputer", impute.SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    preprocessing.OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False
                    ),
                ),
            ]
        )
        transformers.append(("cat", cat_transformer, cat_cols))

    return compose.ColumnTransformer(transformers=transformers)


def load_dataset_openml(
    name: str,
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Fetches and cleans a supported dataset from OpenML."""
    if name not in DATASET_IDS:
        raise ValueError(
            f"Dataset {name} not supported."
            f" Available: {list(DATASET_IDS.keys())}"
        )

    logger.info("Loading dataset: %s (ID: %s)", name, DATASET_IDS[name])

    bunch = datasets.fetch_openml(
        data_id=DATASET_IDS[name], as_frame=True, parser="auto"
    )
    X_raw = bunch.data
    y_raw = bunch.target

    le = preprocessing.LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    if name == "german_credit":
        pos_label_idx = np.where(le.classes_ == "bad")[0][0]
        y_enc = (y_enc == pos_label_idx).astype(int)

    return X_raw, y_enc.astype(int)


def load_raw_dataset(name: str) -> tuple[pd.DataFrame, npt.ArrayLike]:
    """Loads raw features and encoded target from OpenML."""
    return load_dataset_openml(name)


def load_dataset(
    name: str,
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Fetches and preprocesses a supported dataset."""
    X_raw, y_enc = load_raw_dataset(name)
    preprocessor = create_preprocessor(X_raw)
    X_proc = preprocessor.fit_transform(X_raw)
    X_proc = np.asarray(X_proc).astype(np.float64)
    return X_proc, y_enc
