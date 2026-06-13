
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def quality_to_num(series: pd.Series) -> pd.Series:
    """
               Функция преобразует категориальные фичи качества с определенными метками в числовые фичи

               Examples
               --------
               >>>  df[f"{col}_num"] = quality_to_num(df[col])
                        """

    series = series.copy()
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
    """
                   Функция добавляет фичи в датафрейм возвращая итоговый + новые фичи (вместе)

                   Parameters
                   ----------
                   df : pandas.DataFrame

                   Returns
                   -------
                   df : pandas.DataFrame

                   Examples
                   --------
                   >>>  df = new_features(X)
                            """

    df = df.copy()

    # Base
    df["OverallScore"] = df["OverallQual"] * df["OverallCond"]
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]


    # Square
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
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['LotRatio'] = df['GrLivArea'] / df['LotArea']
    df["TotalLivingArea"] = df["GrLivArea"] + df["TotalBsmtSF"]

    # Bool Features
    df['IsNew'] = (df['YearBuilt'] == df['YrSold']).astype(int)
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
    df["HasRemode"] = ((df["HouseAge"] - df["RemodAge"]) > 0).astype(int)
    # df["Has2ndExterion"] = (df['Exterion2nd'] != "NA").astype(int)
    # df["Has2ndBsmtFinType"] = (df["BsmtFinType2"] != "NA").astype(int)


    # Преобразования с качеством ремонта и т.п.
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

    df["TotalQualitySum"] = 0
    for col in quality_cols:
        df[f"{col}_num"] = quality_to_num(df[col])
        df["TotalQualitySum"] += df[f"{col}_num"]

    df["TotalQualityMulti"] = 1
    for col in quality_cols:
        df["TotalQualityMulti"] *= df[f"{col}_num"].apply(lambda x: x if x != 0 else 1)
    df["TotalQualityMulti"] = df["TotalQualityMulti"].clip(upper=350000)
    df['Quality*Square'] = df["TotalHouseSquare"] * df["TotalQualitySum"]
    df['QualityMulty*Square'] = df["TotalHouseSquare"] * df["TotalQualityMulti"]
    df["QualityArea"] = df["OverallQual"] * df["GrLivArea"]
    df["GarageAreaPerCar"] = df["GarageArea"] / (df["GarageCars"] + 1)

    # нелинейность по качеству
    df['OverallQual_sq'] = df['OverallQual'] ** 2
    df['OverallQual_cube'] = df['OverallQual'] ** 3

    # качество * разные площади
    df['QualBsmt'] = df['OverallQual'] * df['TotalBsmtSF']
    df['QualGarage'] = df['OverallQual'] * df['GarageArea']

    # df.drop(columns=["HasBasement", "TotalLivingArea"], inplace=True, errors="ignore")
    return df


def add_log_target(df: pd.DataFrame, target_col: str = "SalePrice") -> pd.DataFrame:
    """
                       Функция принимает весь DF логарифмирует Таргет и возвращает весь DF
                                """
    df = df.copy()
    df[target_col] = np.log1p(df[target_col])
    return df

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
           Класс для встраивания в пайплайн
                    """

    def fit(self, X, y=None):
        return self  # ничего не считаем, просто возвращаем self

    def transform(self, X, y=None):
        df = new_features(X)
        self.feature_names_out_ = df.columns.tolist()  # ПОСЛЕ new_features
        return df

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
