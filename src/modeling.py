
from __future__ import annotations

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor


def fit_autogluon(
    df_train_processed: pd.DataFrame,
    hyperparameters: dict,
    label: str = "SalePrice",
    time_limit: int = 1800,
    presets: str = "best_quality",
    drop_id: bool = True,
    use_bag_holdout: bool = False,
    dynamic_stacking = False,
    num_stack_levels: int = 1,
    refit_full: bool | None = None,
) -> TabularPredictor:
    train_data = df_train_processed.copy()
    if drop_id and "Id" in train_data.columns:
        train_data = train_data.drop(columns="Id")

    fit_kwargs = dict(
        train_data=train_data,
        hyperparameters=hyperparameters,
        presets=presets,
        time_limit=time_limit,
        dynamic_stacking=dynamic_stacking,
        num_stack_levels=num_stack_levels,
    )

    if use_bag_holdout:
        fit_kwargs["use_bag_holdout"] = True

    if refit_full is not None:
        fit_kwargs["refit_full"] = refit_full

    predictor = TabularPredictor(
        label=label,
        eval_metric="root_mean_squared_error",
    ).fit(**fit_kwargs)

    return predictor


def predict_submission(
    predictor: TabularPredictor,
    df_test_processed: pd.DataFrame,
    submission_path: str | None = None,
    id_col: str = "Id",
    target_col: str = "SalePrice",
    drop_id_for_predict: bool = True,
    target_is_logged: bool = True,
) -> pd.DataFrame:
    test_data = df_test_processed.copy()

    ids = test_data[id_col].copy() if id_col in test_data.columns else pd.Series(range(len(test_data)))

    if drop_id_for_predict and id_col in test_data.columns:
        test_data = test_data.drop(columns=id_col)

    preds = predictor.predict(test_data)

    if target_is_logged:
        preds = np.expm1(preds)

    submission = pd.DataFrame({
        id_col: ids,
        target_col: preds,
    })

    if submission_path is not None:
        submission.to_csv(submission_path, index=False)

    return submission
