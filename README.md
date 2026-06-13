# Kaggle House Prices — Top 5% Solution

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-Pipeline-orange?logo=scikitlearn)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-00BFFF)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow)
![Kaggle](https://img.shields.io/badge/Kaggle-Top%205%25-20BEFF?logo=kaggle&logoColor=white)
![Score](https://img.shields.io/badge/RMSE-0.11898-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Результат

**RMSE: 0.11898 — Top 5% на Kaggle Leaderboard**

![Место в рейтинге](output/Place.PNG)

---

## Бизнес-контекст

Задача — предсказать рыночную стоимость жилой недвижимости на основе характеристик объекта. Подобные модели применяются в:

- **Оценке недвижимости** — автоматизация работы оценщиков
- **Банковском секторе** — скоринг залогового имущества при выдаче ипотеки
- **Риелторских платформах** — рекомендации справедливой цены продавцам и покупателям
- **Инвестиционном анализе** — поиск недооценённых объектов

**Ключевая сложность задачи** — крайне малый объём данных (всего 1 460 объектов), что требует особого внимания к:
- качеству признаков
- устойчивости валидационной схемы
- контролю переобучения на каждом этапе

**Что могло бы существенно улучшить качество модели при наличии в данных:**
- Координаты объектов (адрес, геолокация) — позволили бы создать дополнительные признаки:
  - расстояние до остановок общественного транспорта
  - расстояние до торговых центров и значимых городских объектов
  - наличие метро в районе и расстояние до ближайшей станции
  - плотность инфраструктуры в радиусе N км

---

## Структура проекта

```
├── data/
│   ├── raw/                    # Исходные данные (не включены в репозиторий)
│   └── preprocessed/           # Обработанные данные
├── models/                     # Сохранённые модели
├── notebooks/
│   ├── EDA/                    # Разведочный анализ данных
│   ├── FeatureEngineering/     # Feature Engineering
│   └── Modeling/               # Обучение и оценка моделей
├── output/
│   ├── distributions.png       # Графики распределений (EDA)
│   ├── Place.PNG               # Результат на Kaggle
│   └── Submission.csv          # Финальный сабмит
├── src/
│   ├── __init__.py
│   ├── eda.py                  # Функции разведочного анализа
│   ├── features.py             # Feature Engineering (FeatureEngineer)
│   ├── modeling.py             # Обучение моделей и логирование
│   ├── optuna_tun.py           # Тюнинг гиперпараметров через Optuna
│   ├── pipeline.py             # Сборка sklearn Pipeline
│   ├── preproc.py              # Препроцессинг (HousePreprocessor)
│   └── visual.py               # Визуализация ошибок и метрик
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Data Preprocessing

Весь препроцессинг реализован в виде sklearn-совместимого трансформера `HousePreprocessor`, что гарантирует отсутствие утечки данных при кросс-валидации — все статистики вычисляются только на трейн-фолде.

### Стратегии заполнения пропусков

| Тип признака | Стратегия | Примеры |
|---|---|---|
| Категориальные (отсутствие объекта) | `"NA"` | `GarageType`, `BsmtQual`, `PoolQC` |
| Числовые (отсутствие объекта) | `0` | `GarageArea`, `BsmtFinSF1`, `MasVnrArea` |
| Категориальные (неизвестное значение) | Мода по трейну | `MSZoning`, `KitchenQual`, `SaleType` |
| `LotFrontage` | Медиана по `Neighborhood` → fallback глобальная медиана | — |
| `GarageYrBlt` | Значение `YearBuilt` | — |
| `MasVnrType` | `"None"` | — |

### Обработка выбросов

Для числовых признаков с высоким skew применялось клипирование по IQR-границам (метод Тьюки), вычисленным только на трейне:

```
lower = Q25 - 1.5 × IQR
upper = Q75 + 1.5 × IQR
```

Признаки для клипирования: `LotFrontage`, `LotArea`, `MasVnrArea`, `BsmtFinSF1`, `TotalBsmtSF`, `1stFlrSF`, `WoodDeckSF`, `OpenPorchSF`.

Дополнительно из трейна удалялись объекты с `GrLivArea > 4000` — явные выбросы, нарушающие общую зависимость.

---

## Feature Engineering

Feature Engineering реализован в классе `FeatureEngineer` внутри pipeline.

### Ключевые группы признаков

**Агрегированные площади**
- `TotalHouseSquare` = `TotalBsmtSF` + `GrLivArea`
- `TotalPorchSF` — суммарная площадь всех типов веранд
- `TotalBath` = полные + 0.5 × неполные ванные комнаты

**Качество**
- `TotalQualitySum` — сумма числовых оценок качества по 10 признакам
- `TotalQualityMulti` — произведение (с клипированием выбросов)
- Каждый качественный признак (`ExterQual`, `BsmtQual`, `KitchenQual` и др.) переводился в числовую шкалу: `Ex=5, Gd=4, TA=3, Fa=2, Po=1, NA=0`

**Взаимодействия качество × площадь** *(наибольший вклад в feature importance)*
- `Quality*Square` = `TotalHouseSquare` × `TotalQualitySum`
- `QualityArea` = `OverallQual` × `GrLivArea`
- `QualityMulty*Square` = `TotalHouseSquare` × `TotalQualityMulti`

**Временны́е признаки**
- `HouseAge` = `YrSold` − `YearBuilt`
- `RemodAge` = `YrSold` − `YearRemodAdd`
- `HasRemod` — бинарный признак наличия ремонта

**Бинарные признаки наличия объектов**
`HasGarage`, `HasPool`, `HasFireplace`, `HasWoodDeck`, `HasPorch`, `HasFence`, `HasAlley`

### Target Encoding

Значительный прирост качества дал **Target Encoder** для признака `Neighborhood` и ряда других категориальных признаков с высокой кардинальностью.

`Neighborhood` — один из сильнейших предикторов цены, однако обычный `OrdinalEncoder` не передаёт информацию о связи района с целевой переменной. Target Encoding заменяет категорию на среднее значение таргета по группе, вычисленное без утечки данных (только по трейн-фолду кросс-валидации).

Признаки с Target Encoding: `Neighborhood`, `MSSubClass`, `MSZoning`, `Condition1`, `HouseStyle`, `RoofMatl`, `Exterior1st`, `BsmtFinType1`, `Functional`, `Fence`, `SaleType`, `SaleCondition`.

### Feature Importance (топ признаков)

По результатам анализа importance наибольший вклад вносят:
1. `Quality*Square` — взаимодействие качества и площади
2. `QualityArea` — качество × жилая площадь
3. `Neighborhood` (target-encoded)
4. `QualityMulty*Square`
5. `OverallScore`, `TotalHouseSquare`, `OverallQual`

---

## Моделирование

Весь процесс построен на **кастомном sklearn Pipeline** с полным контролем над каждым шагом. Все эксперименты логировались через **MLflow**.

### Архитектура Pipeline

```
HousePreprocessor          # fillna + clip (без утечки)
       ↓
FeatureEngineer            # новые признаки
       ↓
ColumnTransformer
  ├── TargetEncoder        # Neighborhood и др.
  ├── OrdinalEncoder       # остальные категориальные
  └── StandardScaler       # числовые признаки
       ↓
Model
```

---

### Этап 1 — Baseline: линейные модели

Линейные модели использовались как baseline для оценки нижней границы качества.

| Модель | RMSE (CV, 10 фолдов) |
|---|---|
| Ridge | ~0.1255 |
| Lasso | ~0.1270 |
| ElasticNet | ~0.1270 |

![Ridge residuals](outputs/distributions.png)

Графики остатков показали характерную для линейных моделей проблему — систематическое занижение цен на дорогих домах (правый хвост), что ожидаемо при нелинейной природе данных.

---

### Этап 2 — Бустинговые модели (без тюнинга)

| Модель | RMSE (CV, 10 фолдов) |
|---|---|
| LightGBM | ~0.1300 |
| XGBoost | ~0.1387 |
| CatBoost | **0.1176** |

CatBoost показал наилучший результат без каких-либо изменений гиперпараметров. Визуальный анализ остатков показал, что все три модели совершают похожие ошибки — это важный сигнал для стекинга (высокая корреляция ошибок снижает эффект ансамбля).

---

### Этап 3 — Тюнинг гиперпараметров (Optuna)

Тюнинг проводился через `Optuna` с логированием каждого trial в MLflow. Для бустинговых моделей применялся early stopping.

| Модель | До тюнинга | После тюнинга | Δ |
|---|---|---|---|
| Ridge | 0.1255 | 0.1253 | −0.0002 |
| ElasticNet | 0.1270 | 0.1246 | −0.0024 |
| LightGBM | 0.1300 | 0.1224 | −0.0076 |
| XGBoost | 0.1387 | 0.1206 | −0.0181 |
| CatBoost | 0.1176 | 0.1178 | +0.0002 |

**Выводы по тюнингу:**
- Линейные модели дали минимальный прирост — для Ridge и ElasticNet единственные тюнируемые параметры это коэффициенты регуляризации (`alpha`, `l1_ratio`), пространство поиска невелико
- XGBoost показал наибольший прирост от тюнинга
- CatBoost практически не улучшился — вероятно из-за малого числа trials (20). Это точка роста для дальнейшей работы

---

### Этап 4 — Двухуровневый стекинг

Стекинг реализован вручную через OOF-предсказания (Out-of-Fold), что гарантирует отсутствие утечки данных при формировании мета-признаков.

**Схема:**
```
L1 базовые модели → OOF предсказания → L2 мета-модель (Ridge)
```

**Выбор Ridge в качестве L2:**  
На входе мета-модели всего 3–5 числовых признаков (OOF предсказания базовых моделей). Любая нелинейная модель (LightGBM, XGBoost) на таком малом числе признаков немедленно переобучится. Ridge находит оптимальные веса для каждой базовой модели с регуляризацией.

| Конфигурация L1 | RMSE (CV) |
|---|---|
| 5 моделей: CatBoost + XGB + LGBM + ElasticNet + Ridge | **0.1160** |
| 3 модели: CatBoost + XGB + Ridge | 0.1162 |

Добавление LightGBM и ElasticNet в стек не дало значимого улучшения — их ошибки коррелируют с остальными моделями.

**Веса мета-модели (коэффициенты Ridge):**

| Модель | Коэффициент |
|---|---|
| Ridge | 0.627 |
| CatBoost | 0.560 |
| XGBoost | 0.204 |
| LightGBM | 0.022 |
| ElasticNet | −0.401 |

Ridge и CatBoost несут основной сигнал. Отрицательный коэффициент ElasticNet означает, что мета-модель использует его предсказания как корректирующий сигнал в обратную сторону.

**Итоговый результат на Kaggle: RMSE = 0.11898 (Top 5%)**

---

## Эксперименты и логирование

Все эксперименты логировались через **MLflow**:
- параметры модели (`mlflow.log_params`)
- метрики (`rmse_cv`, `best_rmse`)
- теги (`feature_engineering`, `preprocessing`, `notes`)
- артефакты (pipeline, графики остатков)
- каждый Optuna trial — вложенный run с параметрами и метрикой

---

## Документация

Проект содержит docstring-документацию в формате NumPy для всех публичных функций и классов:

```python
def build_pipeline(model, X_train, y_train):
    """
    Собирает sklearn Pipeline с препроцессингом и feature engineering.

    Parameters
    ----------
    model : sklearn estimator
    X_train : pd.DataFrame
    y_train : pd.Series  # в лог-масштабе (log1p)

    Returns
    -------
    pipe : sklearn.pipeline.Pipeline
    """
```

---

## Возможные улучшения

- **Feature Engineering**
  - логарифм и квадратный корень для признаков с высоким skew (`log1p(GrLivArea)`)
  - нелинейные взаимодействия: качество × площадь конкретных объектов (подвал, гараж)
  - полиномиальные признаки для `OverallQual` (`**2`, `**3`)

- **Стекинг**
  - добавить KNN в L1 — на небольших датасетах KNN хорошо улавливает локальные паттерны и слабо коррелирует с бустингом
  - добавить второй уровень стека с несколькими мета-моделями
  - передать на L2 не только OOF предсказания, но и исходные числовые признаки

- **Тюнинг**
  - увеличить число trials для CatBoost (сейчас 20 — недостаточно)
  - добавить тюнинг мета-модели L2

- **Данные**
  - добавление геопространственных признаков (координаты объекта):
    - расстояние до транспортной инфраструктуры
    - расстояние до значимых городских объектов
    - плотность инфраструктуры в радиусе

---

## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone https://github.com/your-username/house-prices-top5pct.git

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Скачать данные с Kaggle
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

# 4. Запустить MLflow UI
mlflow ui --backend-store-uri file:///C:/mlflow_runs

# 5. Запустить ноутбуки по порядку:
# notebooks/EDA/ → notebooks/FeatureEngineering/ → notebooks/Modeling/
```

> ⚠️ Данные не включены в репозиторий в соответствии с правилами Kaggle.

---

## 💡 Ключевой вывод

> На табличных данных малого объёма качество модели определяется не сложностью алгоритма,  
> а качеством признаков, корректностью валидационной схемы и устойчивостью пайплайна.  
>
> Feature Engineering (особенно взаимодействия качество × площадь и Target Encoding для Neighborhood)  
> дал больший прирост, чем переход от линейных моделей к бустингу.
