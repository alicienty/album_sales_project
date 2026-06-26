import pandas as pd
import numpy as np
import logging
import time
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test):
    """Вычисляет метрики, возвращает словарь"""
    y_pred = model.predict(X_test)
    return {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }


def train_model(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    """
    Обучает модель с подбором гиперпараметров (RandomizedSearchCV),
    возвращает обученную модель и метрики.
    """
    logger.info(f"Начало обучения {model_name}")
    start = time.time()
    
    if param_grid:
        # Используем RandomizedSearchCV для оптимизации времени подбора лучших параметров
        rs = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3, 
                                scoring='neg_mean_squared_error', 
                                random_state=42, n_jobs=-1, verbose=0)
        rs.fit(X_train, y_train)
        best_model = rs.best_estimator_
        logger.info(f"{model_name} лучшие параметры: {rs.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train, y_train)
    
    train_time = time.time() - start
    metrics = evaluate_model(best_model, X_test, y_test)
    logger.info(f"{model_name} обучена за {train_time:.2f} сек, R2 = {metrics['R2']:.4f}")
    return best_model, metrics


def run_experiment(X_train, X_test, y_train, y_test, experiment_name):
    """
    Запускает обучение всех моделей на предобработанных данных,
    возвращает словарь с результатами.
    """
    logger.info(f"\n\nЭксперимент: {experiment_name}\n")
    
    # Определяем модели и гиперпараметры для поиска лучшей комбинации
    models = [
        ('LinearRegression', LinearRegression(), {'fit_intercept': [True, False]}),
        ('RandomForest', RandomForestRegressor(random_state=42), 
         {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 
          'min_samples_split': [2, 5, 10]}),
        ('SVM', SVR(), {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 0.5], 'kernel': ['rbf', 'linear']}),
        ('XGBoost', XGBRegressor(random_state=42, verbosity=0), 
         {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 9], 
          'learning_rate': [0.01, 0.05, 0.1]}),
        ('CatBoost', CatBoostRegressor(random_seed=42, verbose=0), 
         {'iterations': [100, 200, 300], 'depth': [4, 6, 8], 
          'learning_rate': [0.01, 0.05, 0.1]})
    ]
    
    results = {}
    trained_models = {}
    
    for name, model, param_grid in models:
        trained_model, metrics = train_model(model, param_grid, X_train, y_train, X_test, y_test, name)
        results[name] = metrics
        trained_models[name] = trained_model
    
    return results, trained_models


def main(data_path, use_text=False):
    """
    Загружает и предобрабатывает данные, обучает модели, сохраняет результаты и лучшую модель
    """
    from preprocess import preprocess_data
    
    # Загрузка данных
    df = pd.read_csv(data_path, index_col=0)
    logger.info(f"Загружено {len(df)} строк, {df.shape[1]} колонок")
    
    # Предобработка (с текстом или без)
    final_data = preprocess_data(df, use_text=use_text, test_size=0.2, random_state=42)

    X_train = final_data['X_train']
    X_test = final_data['X_test']
    y_train = final_data['y_train']
    y_test = final_data['y_test']
    scaler = final_data['scaler']
    embedder = final_data['embedder']
    feature_names = final_data['feature_names']
    
    # Запуск эксперимента
    results, models = run_experiment(X_train, X_test, y_train, y_test, 
                                     "с текстами" if use_text else "только числовые признаки")
    
    # Таблица результатов
    comparison = pd.DataFrame(results).T.round(4)
    logger.info("\n" + comparison.to_string())
    comparison.to_csv(f"results_{'with_text' if use_text else 'numeric_only'}.csv")
    
    # Выбор лучшей модели по R2
    best_name = comparison['R2'].idxmax()
    best_model = models[best_name]
    joblib.dump(best_model, f"best_model_{'text' if use_text else 'numeric'}.pkl")
    joblib.dump(scaler, "scaler.pkl")
    if use_text and embedder:
        joblib.dump(embedder, "embedder.pkl")
    logger.info(f"Лучшая модель ({best_name}) сохранена.")
    
    return results, models


if __name__ == "__main__":
    DATA_PATH = "lyrics_11k.csv"
    # Эксперимент с числовыми признаками
    main(DATA_PATH, use_text=False)
    # Эксперимент с числовыми + текстовыми признаками
    main(DATA_PATH, use_text=True)