#base
import numpy as np
#Optuna
import optuna
#pipeline
from pipeline import build_pipeline
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.linear_model import Ridge
from xgboost.callback import EarlyStopping
#CV
from sklearn.model_selection import KFold, cross_val_predict
#Metrics
from sklearn.metrics import root_mean_squared_error
#mlflow
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("file:///C:/mlflow_runs")
mlflow.set_experiment("house-prices")

RANDOM_SEED = 42

PARAM_SPACES = {
    'LGBMRegressor': lambda trial: {
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': RANDOM_SEED,
        'verbose': -1,
    },
    'CatBoostRegressor': lambda trial: {
        'iterations': trial.suggest_int('iterations', 300, 2000),
        'depth': trial.suggest_int('depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_seed': RANDOM_SEED,
        'verbose': 0,
    },
    'Ridge': lambda trial: {
        'alpha': trial.suggest_float('alpha', 1e-3, 100.0, log=True),
    },
    'ElasticNet': lambda trial: {
        'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
        'max_iter': 10000,  # фиксируем чтобы не было ConvergenceWarning
    },
    'XGBRegressor': lambda trial: {
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'random_state': RANDOM_SEED,
        'verbosity': 0,
    },
}

def objective(trial, model_class, X_train, y_train, cv_folds=10):
    """
    
    :param trial:   количество триалов
    :param model_class:   класс модели (например CatBoostRegressor, LGBMRegressor, XGBRegressor)
    :param X_train: матрица параметров тренировочной выборки
    :param y_train:  вектор ответов тренировочной выборки
    :param cv_folds:   количество фолдов для кроссвалидации
    
    :return: 
    rmse(metric)
    """

    # получаем параметры для конкретной модели
    model_name = model_class.__name__
    params = PARAM_SPACES[model_name](trial)

    model = model_class(**params)
    pipe = build_pipeline(model=model, X_train=X_train, y_train=y_train)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    y_pred_log = cross_val_predict(pipe, X_train, y_train, cv=kf)

    return root_mean_squared_error(y_train, y_pred_log)


def run_optuna(model_class, X_train, y_train, n_trials=100, cv_folds=10):
    """

    :param model_class: класс модели (например CatBoostRegressor, LGBMRegressor, XGBRegressor)
    :param X_train: матрица параметров тренировочной выборки
    :param y_train: вектор ответов тренировочной выборки
    :param n_trials: количество триалов
    :param cv_folds: количество фолдов для кроссвалидации
    :return:
    study
    :Example:
    study_catboost = run_optuna(CatBoostRegressor, X_train, y_train, n_trials=20)
    """
    model_name = model_class.__name__
    supports_early_stopping = model_name in ('LGBMRegressor', 'CatBoostRegressor', 'XGBRegressor')

    def objective_with_mlflow(trial):
        """

        :param trial: количество триалов
        :return:
        rmse(метрика)
        """

        with mlflow.start_run(run_name=f"{model_name}_trial_{trial.number}", nested=True):
            mlflow.set_tag("tuning", "optuna")
            params = PARAM_SPACES[model_name](trial)
            mlflow.log_params(params)
            mlflow.set_tag("model", model_name)

            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
            scores = []

            if supports_early_stopping:
                for train_idx, val_idx in kf.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    model = model_class(**params)
                    pipe = build_pipeline(model=model, X_train=X_tr, y_train=y_tr)

                    # трансформируем данные через все шаги кроме модели
                    pipe_no_model = Pipeline(pipe.steps[:-1])
                    pipe_no_model.fit(X_tr, y_tr)
                    X_tr_transformed = pipe_no_model.transform(X_tr)
                    X_val_transformed = pipe_no_model.transform(X_val)

                    if model_name == 'LGBMRegressor':
                        model.fit(
                            X_tr_transformed, y_tr,
                            eval_set=[(X_val_transformed, y_val)],
                            callbacks=[lgb.early_stopping(50, verbose=False)]
                        )
                    elif model_name == 'CatBoostRegressor':
                        model.fit(
                            X_tr_transformed, y_tr,
                            eval_set=(X_val_transformed, y_val),
                            early_stopping_rounds=50,
                        )
                    elif model_name == 'XGBRegressor':
                        model.fit(
                            X_tr_transformed, y_tr,
                            eval_set=[(X_val_transformed, y_val)],
                            verbose=False,
                        )

                    pred = model.predict(X_val_transformed)
                    scores.append(root_mean_squared_error(y_val, pred))

                rmse = float(np.mean(scores))

            else:
                # для Ridge/Lasso/ElasticNet — обычный cross_val_predict
                model = model_class(**params)
                pipe = build_pipeline(model=model, X_train=X_train, y_train=y_train)
                y_pred_log = cross_val_predict(pipe, X_train, y_train, cv=kf)
                rmse = root_mean_squared_error(y_train, y_pred_log)

            mlflow.log_metric("rmse_cv", rmse)
            return rmse

    with mlflow.start_run(run_name=f"{model_name}_optuna"):
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )
        study.optimize(objective_with_mlflow, n_trials=n_trials, show_progress_bar=True)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_rmse", study.best_value)

        print(f"Best RMSE: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

    return study


def tune_meta_model(oof_preds, y_train, n_trials=50, cv=10):
    """

    :param oof_preds: редсказание l1 моделей
    :param y_train: вектров ответов
    :param n_trials: количество триалов
    :param cv: количество фолдов для кроссвалидации
    :return:
    study
    :Example:
    study = tune_meta_model(oof_preds, y_train, n_trials=n_trials, cv=cv)
    """

    def objective(trial):
        params = {
            'alpha': trial.suggest_float('alpha', 1e-4, 100.0, log=True),
        }
        model = Ridge(**params)
        kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        oof_meta = cross_val_predict(model, oof_preds, y_train, cv=kf)
        return root_mean_squared_error(y_train, oof_meta)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best meta RMSLE: {study.best_value:.4f}")
    print(f"Best meta params: {study.best_params}")
    return study