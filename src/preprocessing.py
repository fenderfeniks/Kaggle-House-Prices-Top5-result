
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pandas as pd


CLIP_COLS = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1",
    "TotalBsmtSF", "1stFlrSF", "WoodDeckSF", "OpenPorchSF",
]


def fit_preprocessing_stats(df_train: pd.DataFrame) -> Dict:
    """Fit preprocessing statistics on train only."""
    stats = {}

    stats["lot_frontage_medians"] = (
        df_train.groupby("Neighborhood")["LotFrontage"].median().to_dict()
    )

    stats["global_lotfrontage_median"] = df_train["LotFrontage"].median()

    stats["cat_modes"] = {
        col: df_train[col].mode(dropna=True)[0]
        for col in [
            "MSZoning",
            "Utilities",
            "Exterior1st",
            "Exterior2nd",
            "KitchenQual",
            "Functional",
            "Electrical",
        ]
    }

    stats["num_zero_cols"] = [
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "GarageCars",
        "GarageArea",
    ]

    return stats


def preprocessing(df: pd.DataFrame, stats: Dict, is_train: bool = True) -> pd.DataFrame:
    """Apply preprocessing using train-fitted statistics."""
    df = df.copy()

    if is_train:
        df = df[df["GrLivArea"] <= 4000].copy()

    df["LotFrontage"] = df["Neighborhood"].map(
        stats["lot_frontage_medians"]
    ).where(df["LotFrontage"].isna(), df["LotFrontage"])

    df["LotFrontage"] = df["LotFrontage"].fillna(
        stats["global_lotfrontage_median"]
    )

    na_cols = [
        "Alley",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "Fence",
        "MiscFeature",
    ]
    for col in na_cols:
        df[col] = df[col].fillna("NA")

    for col in stats["num_zero_cols"]:
        df[col] = df[col].fillna(0)

    df["MasVnrType"] = df["MasVnrType"].fillna("None")
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

    for col, mode_val in stats["cat_modes"].items():
        df[col] = df[col].fillna(mode_val)

    df["SaleType"] = df["SaleType"].fillna("Oth")
    return df


def quantile_scorer(series: pd.Series) -> Tuple[float, float]:
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    return q25 - 1.5 * iqr, q75 + 1.5 * iqr


def make_borders(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    borders = {}
    for col in cols:
        low, high = quantile_scorer(df[col])
        borders[col] = (low, high)
    return borders


def apply_borders(df: pd.DataFrame, borders: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    df = df.copy()
    for col, (low, high) in borders.items():
        df[col] = df[col].clip(lower=low, upper=high)
    return df


def prepare_datasets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    clip_cols: Iterable[str] = CLIP_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline for train and test."""
    df_train_for_stats = df_train.copy()
    df_train_for_stats = df_train_for_stats[df_train_for_stats["GrLivArea"] <= 4000]

    stats = fit_preprocessing_stats(df_train_for_stats)

    df_train_processed = preprocessing(df_train, stats=stats, is_train=True)
    df_test_processed = preprocessing(df_test, stats=stats, is_train=False)

    borders = make_borders(df_train_processed, clip_cols)
    df_train_processed = apply_borders(df_train_processed, borders)
    df_test_processed = apply_borders(df_test_processed, borders)

    return df_train_processed, df_test_processed
