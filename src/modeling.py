#pipeline
from pipeline import build_pipeline
from sklearn.model_selection import KFold, cross_val_predict
#mlflow
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("file:///C:/mlflow_runs")
mlflow.set_experiment("house-prices")
# metrics
from sklearn.metrics import root_mean_squared_error
#base
import numpy as np
import pandas as pd
#model
from sklearn.linear_model import Ridge
#optuna
from optuna_tun import tune_meta_model

RANDOM_SEED = 42

def model_train(model, X_train, y_train, notes: str = None, feature_engineering: str = None, preprocessing: str = None, cv_folds: int = 5):
    """
                       Функция полностью выполняет процесс обучения модели с логированием в mlflow

                       # Логирование:
                       Tag "notes" : заметка
                       Tag "feature_engineering" : заметка о fe
                       Tag "preprocessing" : заметка о предобработке

                       params = model.get_params()

                       log_metric = ("rmse_cv", rmse)

                       log_model(pipe, "pipeline")

                       # Описание:
                       Создает пайплайн
                       Разбивает на фолды
                       обучает модель с помощью cross_val_predict
                       (модель обучается на логарифме таргета)
                       считает метрику (для подсчета метрики преобразуем обратно таргет и считаем rmse)

                       Parameters
                       ----------
                       model : модель поддерживающая fir/predict обученная(будет обучаться на новых данных)/необученная

                       X_train: pd.DataFrame Матрица параметров/объектов

                       y_train: pd.Series вектор ответов(таргетов)

                       notes: str = None,
                       feature_engineering: str = None,
                       preprocessing: str = None  параметры для заметок

                       cv_folds: int =5  количество фолдов

                       Returns
                       -------
                       model обученую модель

                       pipe весь пайплайн с шагами и обученной моделью

                       y_pred: pd.Series предсказанный ответ

                       residuals: pd.Series  разница между реальным и предсказанным значением

                       Examples
                       --------
                       >>>  model, pipe, y_pred, residuals = model_train(model=catboost,
                                              X_train=X_train,
                                              y_train=y_train,
                                              notes="catboost after optuna",
                                              feature_engineering="v9 = TE + Base + Bool + Multiply + NoLinear Quality",
                                              preprocessing="fillna + clip",
                                              cv_folds=10
                                             )
                                """

    with mlflow.start_run(run_name=model.__class__.__name__):
        #Tags
        ## Main Tag
        mlflow.set_tag("notes", f"{notes}")
        ## Others
        mlflow.set_tag("feature_engineering", f"{feature_engineering}")
        mlflow.set_tag("preprocessing", f"{preprocessing}")

        pipe = build_pipeline(model=model, X_train=X_train, y_train=y_train)

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)

        y_pred_log = cross_val_predict(pipe, X_train, y_train, cv=kf)

        y_pred = pd.Series(np.expm1(y_pred_log), index=X_train.index)
        y_true = np.expm1(y_train)
        residuals = y_true - y_pred

        rmse = root_mean_squared_error(y_train, y_pred_log)
        print(f"{model.__class__.__name__} RMSE: {rmse:.4f}")

        # логирование модели
        mlflow.log_params(model.get_params())
        # логирование метрики
        mlflow.log_metric("rmse_cv", rmse)

        # логирование pipeline целиком
        pipe.fit(X_train, y_train)
        mlflow.sklearn.log_model(pipe, "pipeline")

    return model, pipe, y_pred, residuals


def make_oof_predictions(models, X_train, y_train, X_test, cv=10):
    """Генерирует OOF предсказания для каждой L1 модели."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    oof_preds = np.zeros((len(X_train), len(models)))
    test_preds = np.zeros((len(X_test), len(models)))

    for i, (name, model, params) in enumerate(models):
        with mlflow.start_run(run_name=f"L1_{name}", nested=True):
            mlflow.set_tag("level", "L1")
            mlflow.set_tag("model", name)
            mlflow.log_params(params)

            pipe = build_pipeline(model=model, X_train=X_train, y_train=y_train)

            # OOF предсказания
            oof_preds[:, i] = cross_val_predict(pipe, X_train, y_train, cv=kf)

            rmse = root_mean_squared_error(y_train, oof_preds[:, i])
            mlflow.log_metric("rmsle_oof", rmse)
            print(f"L1 {name} OOF RMSLE: {rmse:.4f}")

            # обучаем на всём трейне для предсказания теста
            pipe.fit(X_train, y_train)
            test_preds[:, i] = pipe.predict(X_test)

            mlflow.sklearn.log_model(pipe, f"pipeline_{name}")

    return oof_preds, test_preds


def run_stacking(l1_models, X_train, y_train, X_test, cv=10, tune_meta=True, n_trials=50):

    with mlflow.start_run(run_name="stacking"):
        mlflow.set_tag("method", "2-level stacking + optuna meta")

        # L1
        print("=== L1: обучение базовых моделей ===")
        oof_preds, test_preds = make_oof_predictions(
            l1_models, X_train, y_train, X_test, cv=cv
        )

        # тюнинг мета-модели
        if tune_meta:
            print("=== Тюнинг мета-модели ===")
            with mlflow.start_run(run_name="L2_meta_tuning", nested=True):
                study = tune_meta_model(oof_preds, y_train, n_trials=n_trials, cv=cv)
                mlflow.log_params(study.best_params)
                mlflow.log_metric("best_rmse_meta", study.best_value)
                meta_model = Ridge(**study.best_params)
        else:
            meta_model = Ridge(alpha=1.0)

        # L2
        print("=== L2: финальное обучение мета-модели ===")
        with mlflow.start_run(run_name="L2_meta", nested=True):
            mlflow.set_tag("level", "L2")

            kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
            oof_meta = cross_val_predict(meta_model, oof_preds, y_train, cv=kf)
            rmse_meta = root_mean_squared_error(y_train, oof_meta)

            mlflow.log_metric("rmse_oof", rmse_meta)
            print(f"L2 meta OOF RMSE: {rmse_meta:.4f}")

            meta_model.fit(oof_preds, y_train)
            mlflow.sklearn.log_model(meta_model, "meta_model")

        y_test_pred = np.expm1(meta_model.predict(test_preds))
        mlflow.log_metric("best_rmse", rmse_meta)
        print(f"\n=== Итог: Stacking RMSE: {rmse_meta:.4f} ===")

    return y_test_pred, meta_model, oof_preds, test_preds
