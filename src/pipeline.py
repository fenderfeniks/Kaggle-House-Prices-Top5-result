from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder, StandardScaler

from preproc import HousePreprocessor
from features import FeatureEngineer



TARGET_ENC_COLS = [
    'Neighborhood', 'MSSubClass', 'MSZoning', 'Condition1',
    'HouseStyle', 'RoofMatl', 'Exterior1st', 'BsmtFinType1',
    'Functional', 'Fence', 'SaleType', 'SaleCondition',
]

def get_cat_num_cols(X_train, y_train):
    """Получить списки колонок после preprocessing и feature engineering."""
    temp_pipe = Pipeline([
        ("preprocessor", HousePreprocessor()),
        ("feature_engineer", FeatureEngineer()),
    ])
    temp_pipe.fit(X_train, y_train)
    X_temp = temp_pipe.transform(X_train)

    cat_cols = X_temp.select_dtypes(exclude='number').columns.tolist()
    num_cols = X_temp.select_dtypes(include='number').columns.tolist()
    return cat_cols, num_cols


def build_pipeline(model, X_train, y_train):
    """Собрать полный pipeline для заданной модели."""
    cat_cols, num_cols = get_cat_num_cols(X_train, y_train)
    cat_cols_ordinal = [col for col in cat_cols if col not in TARGET_ENC_COLS]

    column_transformer = ColumnTransformer(
        transformers=[
            ('target_enc', TargetEncoder(target_type='continuous'), TARGET_ENC_COLS),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols_ordinal),
            ('scaler', StandardScaler(), num_cols),
        ],
        remainder='drop'
    ).set_output(transform="pandas")

    return Pipeline([
        ("preprocessor", HousePreprocessor()),
        ("feature_engineer", FeatureEngineer()),
        ("encoder_scaler", column_transformer),
        ("model", model),
    ])