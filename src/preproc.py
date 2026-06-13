
from typing import Dict, Iterable, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


CLIP_COLS = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1",
    "TotalBsmtSF", "1stFlrSF", "WoodDeckSF", "OpenPorchSF",
]

def optimize_dtypes(df):
    for col in df.select_dtypes('integer').columns:
        df[col] = pd.to_numeric(df[col], downcast='integer') # до int32
    for col in df.select_dtypes('float').columns:
        df[col] = pd.to_numeric(df[col], downcast='float')  # до float32
    return df

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


class HousePreprocessor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        df = X.copy()
        df = df[df["GrLivArea"] <= 4000]  # фильтрация только здесь

        self.stats_ = fit_preprocessing_stats(df)
        self.borders_ = make_borders(
            preprocessing(df, self.stats_),
            CLIP_COLS
        )
        return self

    def transform(self, X, y=None):
        df = preprocessing(X, self.stats_)
        df = apply_borders(df, self.borders_)
        self.feature_names_out_ = df.columns.tolist()
        return df

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_