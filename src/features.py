
from __future__ import annotations

import numpy as np
import pandas as pd


def quality_to_num(series: pd.Series) -> pd.Series:
    mapping = {
        "Ex": 5,
        "Gd": 4,
        "TA": 3,
        "Fa": 2,
        "Po": 1,
        "NA": 0,
    }
    return series.map(mapping).fillna(0)


def new_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["OverallScore"] = df["OverallQual"] * df["OverallCond"]

    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    df["TotalHouseSquare"] = df["TotalBsmtSF"] + df["GrLivArea"]

    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
        + df["WoodDeckSF"]
    )

    df["TotalBath"] = (
        df["FullBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["HalfBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    df["TotalLivingArea"] = df["GrLivArea"] + df["TotalBsmtSF"]

    df["QualityArea"] = df["OverallQual"] * df["GrLivArea"]
    df["GarageAreaPerCar"] = df["GarageArea"] / (df["GarageCars"] + 1)

    quality_cols = [
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageQual",
        "GarageCond",
        "PoolQC",
    ]

    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasWoodDeck"] = (df["WoodDeckSF"] > 0).astype(int)
    df["HasPorch"] = (
        (
            df["OpenPorchSF"]
            + df["EnclosedPorch"]
            + df["3SsnPorch"]
            + df["ScreenPorch"]
        ) > 0
    ).astype(int)
    df["HasFence"] = (df["Fence"] != "NA").astype(int)
    df["HasAlley"] = (df["Alley"] != "NA").astype(int)

    df["TotalQualityScore"] = 0
    for col in quality_cols:
        df[f"{col}_num"] = quality_to_num(df[col])
        df["TotalQualityScore"] += df[f"{col}_num"]

    df.drop(columns=["HasBasement", "TotalLivingArea"], inplace=True, errors="ignore")
    return df


def add_log_target(df: pd.DataFrame, target_col: str = "SalePrice") -> pd.DataFrame:
    df = df.copy()
    df[target_col] = np.log1p(df[target_col])
    return df
